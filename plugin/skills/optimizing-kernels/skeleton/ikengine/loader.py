"""Checkpoint reading: config, generation defaults, the quantization scheme and weights as stored.

Nothing here changes a stored value. `Checkpoint.linear(prefix)` returns a `QWeight` holding the stored
tensors (packed values and every scale) so that `qlinear.qlinear` can compute exactly what the checkpoint
declares. Tensors are read lazily, one at a time, from the safetensors shards.

Formats read (the kind of each linear layer is decided by the tensors stored for it, so mixed checkpoints work):
- 16-bit or fp32 weights (no scales).
- FP8 with block scales (`quant_method: fp8`, `weight_block_size`, tensors `weight` e4m3 + `weight_scale_inv`),
  as Qwen's and DeepSeek's FP8 checkpoints ship; activations are quantized dynamically per token and group.
- NVIDIA ModelOpt NVFP4 (`hf_quant_config.json`, or `quantization_config` with `quant_method: modelopt` and
  `config_groups` in config.json): `weight` uint8 (two e2m1 per byte), `weight_scale` e4m3 per 16 along K,
  `weight_scale_2` fp32, `input_scale` fp32 (static activation scale: W4A4 as shipped).
- ModelOpt FP8 per tensor (`quant_algo: FP8`, or a MIXED_PRECISION layer): `weight` e4m3 + fp32 `weight_scale`
  and a static fp32 `input_scale`.
- KV-cache scales (`k_scale` / `v_scale`, or one `kv_scale`) when the checkpoint stores them.
compressed-tensors checkpoints (`quant_method: compressed-tensors`) store the reciprocal global scales under
other names; they raise rather than load with a silently different meaning.
"""
from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from typing import Any

QUANT_NONE, QUANT_FP8_BLOCK, QUANT_NVFP4 = "none", "fp8_block", "nvfp4"
QUANT_FP8_TENSOR, QUANT_MIXED = "fp8_tensor", "mixed"
DENSE_KINDS = ("bf16", "fp16", "fp32")
_DENSE = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}


@dataclass
class QWeight:
    """One linear layer's weight as stored. Logical shape is [out_features, in_features] = [N, K].

    kind "bf16" | "fp16" | "fp32": weight is the dense tensor.
    kind "fp8_block": weight is float8_e4m3fn [N, K]; scale is weight_scale_inv [ceil(N/bn), ceil(K/bk)] as stored
        (BF16 or FP32), widened to fp32 in qlinear; block is (bn, bk); activations are quantized dynamically
        per token in groups of bk (as shipped).
    kind "fp8_tensor": weight is float8_e4m3fn [N, K]; scale is the fp32 weight scale (one value, or one per
        output row); input_scale is the fp32 static activation scale, or None (dynamic, one scale per tensor).
    kind "nvfp4": weight is uint8 [N, K/2] (two e2m1 values per byte, low nibble first); scale is the
        float8_e4m3fn per-16 block scale [N, K/16]; scale_2 is the fp32 global weight scale;
        input_scale is the fp32 static activation scale (W4A4 as shipped), or None (W4A16 as shipped).

    Every tensor may carry one extra leading dimension (a stack of experts, see moe.py); `shape` stays [N, K].
    """
    kind: str
    weight: Any
    shape: tuple[int, int]
    scale: Any = None
    scale_2: Any = None
    input_scale: Any = None
    block: tuple[int, int] | None = None
    bias: Any = None


def _read_json(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _bits(group: dict, key: str) -> tuple[int | None, str | None, bool, int | None]:
    """(num_bits, type, dynamic, group_size) of `weights` or `input_activations` in one config group."""
    g = group.get(key) or {}
    return g.get("num_bits"), g.get("type"), bool(g.get("dynamic", False)), g.get("group_size")


class Checkpoint:
    """A Hugging Face checkpoint directory (config.json, *.safetensors, optional hf_quant_config.json)."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.dir = model_dir
        self.device = device
        self.config = _read_json(os.path.join(model_dir, "config.json"))
        self.text_config = self.config.get("text_config") or self.config
        self.generation_config = _read_json(os.path.join(model_dir, "generation_config.json"))
        self.quant = self._quant_scheme()
        self._index = self._build_index()
        self._handles: dict[str, Any] = {}

    # --- scheme -------------------------------------------------------------------------------------
    def _quant_scheme(self) -> dict:
        """Normalised: {"method", "block", "act", "kv", "exclude", "layers", "source"}. Unknown schemes raise.

        "act" is the declared activation scheme of the quantized layers: "nvfp4_static" (W4A4 with the stored
        input_scale), "fp8_dynamic_group" (per token, per block-K group), "fp8_static" (per tensor), or None
        (16-bit activations). "kv" is "fp8" when the checkpoint declares an FP8 KV cache, else None.
        "layers" maps module names to their algorithm for MIXED_PRECISION checkpoints.
        """
        hq_file = _read_json(os.path.join(self.dir, "hf_quant_config.json"))
        hq = hq_file.get("quantization") or {}
        qc = self.config.get("quantization_config") or self.text_config.get("quantization_config") or {}
        method = str(qc.get("quant_method") or "").lower()
        if hq or method == "modelopt":
            return self._modelopt_scheme(hq, qc)
        if method == "fp8":
            block = qc.get("weight_block_size")
            dynamic = qc.get("activation_scheme", "dynamic") == "dynamic"
            if block:
                if not dynamic:
                    raise NotImplementedError("block FP8 with static activation scales is not supported by the skeleton")
                return {"method": QUANT_FP8_BLOCK, "block": list(block), "act": "fp8_dynamic_group", "kv": None,
                        "exclude": list(qc.get("modules_to_not_convert") or qc.get("ignored_layers") or []),
                        "layers": {}, "source": "config.json quantization_config (fp8, block)"}
            return {"method": QUANT_FP8_TENSOR, "block": None, "act": "fp8_dynamic_tensor" if dynamic else "fp8_static",
                    "kv": None, "exclude": list(qc.get("modules_to_not_convert") or qc.get("ignored_layers") or []),
                    "layers": {}, "source": "config.json quantization_config (fp8, per tensor)"}
        if method == "compressed-tensors":
            raise NotImplementedError(
                "compressed-tensors checkpoints store reciprocal global scales under other tensor names "
                f"(format {qc.get('format')!r}); the skeleton reads ModelOpt and block-FP8 checkpoints only")
        if qc:
            raise NotImplementedError(f"quantization_config not supported by the skeleton: {json.dumps(qc)[:200]}")
        return {"method": QUANT_NONE, "block": None, "act": None, "kv": None, "exclude": [], "layers": {},
                "source": "none"}

    def _modelopt_scheme(self, hq: dict, qc: dict) -> dict:
        """ModelOpt keys are read from hf_quant_config.json first, then from config.json's quantization_config
        (an export may carry either file, and config.json may use the same key names or config_groups)."""
        algo = str(hq.get("quant_algo") or qc.get("quant_algo") or "").upper()
        exclude = list(hq.get("exclude_modules") or qc.get("ignore") or qc.get("exclude_modules") or [])
        layers = {k: str((v or {}).get("quant_algo", "")).upper()
                  for k, v in (hq.get("quantized_layers") or qc.get("quantized_layers") or {}).items()}
        kv_algo = str(hq.get("kv_cache_quant_algo") or qc.get("kv_cache_quant_algo") or "").upper()
        kv_cfg = qc.get("kv_cache_scheme") or {}
        kv = "fp8" if kv_algo == "FP8" or (kv_cfg.get("num_bits") == 8 and kv_cfg.get("type") == "float") else None
        groups = list((qc.get("config_groups") or {}).values())
        source = "hf_quant_config.json" if hq else "config.json quantization_config (modelopt)"
        if algo == "NVFP4" or (not algo and any(_bits(g, "weights")[:2] == (4, "float") for g in groups)):
            gsize = (hq.get("group_size") or qc.get("group_size")
                     or next((_bits(g, "weights")[3] for g in groups if _bits(g, "weights")[3]), 16))
            if int(gsize) != 16:
                raise NotImplementedError(f"NVFP4 with group size {gsize}: the skeleton reads group 16")
            act = "nvfp4_static"
            for g in groups:
                bits, typ, dynamic, _ = _bits(g, "input_activations")
                if g.get("input_activations") is None:
                    act = None                      # weight-only (W4A16) as shipped
                elif (bits, typ) != (4, "float") or dynamic:
                    raise NotImplementedError(f"NVFP4 with input activations {g.get('input_activations')}")
            return {"method": QUANT_NVFP4, "block": [1, 16], "act": act, "kv": kv, "exclude": exclude,
                    "layers": layers, "source": source}
        if algo == "FP8":
            return {"method": QUANT_FP8_TENSOR, "block": None, "act": "fp8_static", "kv": kv, "exclude": exclude,
                    "layers": layers, "source": source}
        if algo == "MIXED_PRECISION":
            unknown = sorted({a for a in layers.values() if a not in ("NVFP4", "FP8")})
            if unknown:
                raise NotImplementedError(f"MIXED_PRECISION layers with {unknown}: the skeleton reads NVFP4 and FP8")
            return {"method": QUANT_MIXED, "block": [1, 16], "act": "per layer", "kv": kv, "exclude": exclude,
                    "layers": layers, "source": source}
        raise NotImplementedError(f"ModelOpt quant_algo {algo or '(none)'!r} is not supported by the skeleton")

    def is_excluded(self, prefix: str) -> bool:
        """True when the scheme leaves this module unquantized (stored in 16-bit)."""
        pats = self.quant.get("exclude") or []
        return any(prefix == p or fnmatch.fnmatch(prefix, p) or prefix.startswith(p + ".") for p in pats)

    def layer_algo(self, prefix: str) -> str:
        """The declared format of one linear layer: "none", "fp8_block", "fp8_tensor" or "nvfp4"."""
        q = self.quant
        if q["method"] == QUANT_NONE or self.is_excluded(prefix):
            return QUANT_NONE
        if q["method"] == QUANT_MIXED:
            return {"NVFP4": QUANT_NVFP4, "FP8": QUANT_FP8_TENSOR}.get(q["layers"].get(prefix, ""), QUANT_NONE)
        return q["method"]

    @property
    def dtype_name(self) -> str:
        """The checkpoint's 16-bit compute dtype ("bfloat16" unless config.json says otherwise)."""
        for c in (self.text_config, self.config):
            d = c.get("torch_dtype") or c.get("dtype")
            if d in ("bfloat16", "float16", "float32"):
                return d
        return "bfloat16"

    # --- tensors ------------------------------------------------------------------------------------
    def _build_index(self) -> dict[str, str]:
        idx = _read_json(os.path.join(self.dir, "model.safetensors.index.json")).get("weight_map")
        if idx:
            return {k: os.path.join(self.dir, v) for k, v in idx.items()}
        out: dict[str, str] = {}
        from safetensors import safe_open
        for fn in sorted(os.listdir(self.dir)):
            if fn.endswith(".safetensors"):
                path = os.path.join(self.dir, fn)
                with safe_open(path, framework="pt") as f:
                    for k in f.keys():
                        out[k] = path
        return out

    def names(self) -> list[str]:
        return sorted(self._index)

    def has(self, name: str) -> bool:
        return name in self._index

    def tensor(self, name: str, device: str | None = None, dtype: Any = None) -> Any:
        """One stored tensor, unchanged unless `dtype` is given."""
        from safetensors import safe_open
        path = self._index[name]
        h = self._handles.get(path)
        if h is None:
            h = self._handles[path] = safe_open(path, framework="pt")
        t = h.get_tensor(name)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device or self.device)

    def maybe(self, name: str, device: str | None = None) -> Any:
        """The stored tensor, or None when the checkpoint has no tensor of that name."""
        return self.tensor(name, device) if self.has(name) else None

    def linear(self, prefix: str, bias: bool | None = None) -> QWeight:
        """The weight (and scales) of `prefix` (e.g. "model.layers.0.mlp.gate_proj") as stored.

        The kind follows the stored tensors: e4m3 + weight_scale_inv is block FP8; uint8 + weight_scale +
        weight_scale_2 is NVFP4; e4m3 + weight_scale is per-tensor FP8; a 16-bit tensor alone is dense.
        bias None loads `prefix.bias` when stored (a stored bias is never dropped); True requires it.
        """
        import torch
        w = self.tensor(prefix + ".weight")
        if bias is True or (bias is None and self.has(prefix + ".bias")):
            b = self.tensor(prefix + ".bias")
        else:
            b = None
        if w.dtype == torch.float8_e4m3fn and self.has(prefix + ".weight_scale_inv"):
            block = tuple(self.quant.get("block") or (128, 128))
            return QWeight("fp8_block", w, tuple(w.shape), scale=self.tensor(prefix + ".weight_scale_inv"),
                           block=(int(block[0]), int(block[1])), bias=b)
        if w.dtype == torch.uint8 and self.has(prefix + ".weight_scale") and self.has(prefix + ".weight_scale_2"):
            n, k2 = w.shape
            return QWeight("nvfp4", w, (n, k2 * 2), scale=self.tensor(prefix + ".weight_scale"),
                           scale_2=self.tensor(prefix + ".weight_scale_2"), input_scale=self.maybe(prefix + ".input_scale"),
                           block=(1, 16), bias=b)
        if w.dtype == torch.float8_e4m3fn and self.has(prefix + ".weight_scale"):
            return QWeight("fp8_tensor", w, tuple(w.shape), scale=self.tensor(prefix + ".weight_scale"),
                           input_scale=self.maybe(prefix + ".input_scale"), bias=b)
        kind = _DENSE.get(str(w.dtype))
        if kind is None:
            raise NotImplementedError(f"{prefix}: stored dtype {w.dtype} with no known scales")
        return QWeight(kind, w, tuple(w.shape), bias=b)

    def kv_scales(self, attn_prefix: str) -> tuple[Any, Any]:
        """(k_scale, v_scale) fp32 tensors for one attention module, as stored, or (None, None).

        Names seen in ModelOpt and vLLM-style exports: `<attn>.k_scale`, `<attn>.k_proj.k_scale`,
        `<attn>.attn.k_scale` (and the same for v), and the older single `<attn>.kv_scale` (or
        `<attn>.k_proj.kv_scale`), which serves as both. As vLLM does: when only one of k and v is stored,
        it is used for both (vLLM's BaseKVCacheMethod copies it); when neither is, the scale is 1.0.
        """
        def find(*names: str):
            for name in names:
                if self.has(name):
                    return self.tensor(name)
            return None
        p = attn_prefix
        k = find(f"{p}.k_scale", f"{p}.k_proj.k_scale", f"{p}.attn.k_scale")
        v = find(f"{p}.v_scale", f"{p}.v_proj.v_scale", f"{p}.attn.v_scale")
        if k is None and v is None:
            k = v = find(f"{p}.kv_scale", f"{p}.k_proj.kv_scale", f"{p}.attn.kv_scale")
        return (v if k is None else k), (k if v is None else v)
