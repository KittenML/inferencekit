"""Gemma 4 text decoder (Gemma4ForConditionalGeneration's text path, and Gemma4ForCausalLM): correct first.

Every size comes from config.json. One decoder layer, as the Hugging Face reference computes it:

    r = x;  x = r + post_attention_norm(attention(input_norm(x)))
    r = x;  y = mlp(pre_ffn_norm(r))                                      # the dense MLP, every layer
            if MoE: y = post_ffn_norm_1(y) + post_ffn_norm_2(experts(pre_ffn_norm_2(r), router(r)))
            x = r + post_ffn_norm(y)
            x = x + per_layer_input_block(x)                              # only when the config has one
            x = x * layer_scalar

- Attention. Each layer is "sliding_attention" (sees the last sliding_window positions, itself included)
  or "full_attention"; the two types can differ in head size, KV-head count and RoPE. q and k get a
  per-head RMSNorm with a weight, v a per-head RMSNorm without one, and the score scale is 1.0. With
  attention_k_eq_v, full layers have no v_proj: the values are the k_proj output (before k_norm and RoPE).
  "proportional" RoPE rotates only the first partial_rotary_factor share of each half of the head (its
  frequencies use the full head size) and leaves the rest unrotated.
- Router: RMSNorm without a weight, times `scale` and hidden_size^-0.5, a linear, softmax in fp32, top-k,
  renormalise to sum 1, then times per_expert_scale of each chosen expert. Experts: down(act(gate) * up).
- Embedding: the table times sqrt(hidden_size), the factor rounded to the table's dtype first.
- Output: final RMSNorm, the tied embedding (or lm_head), then logits = cap * tanh(logits / cap).
- Per-layer inputs and KV-shared layers (configs with hidden_size_per_layer_input or num_kv_shared_layers)
  follow the same reference. Vision and audio towers are not loaded: text only. Image/audio placeholder ids
  in a prompt are embedded as ordinary tokens.

Numerics. Each linear goes through qlinear.qlinear with its weight as stored (NVFP4 W4A4 with the stored
input_scale for quantized modules, 16-bit for excluded ones); act="a16" keeps activations 16-bit and says
so when the checkpoint quantizes them (a weight-only NVFP4 module, W4A16, ships 16-bit activations).
`numerics` describes the activations from the scales each linear stores. Activations run in the
checkpoint's dtype; RMSNorm, the router softmax, the MoE weighted sum and the logits run in fp32. KV cache:
FP8 e4m3 when the checkpoint declares it, with the checkpoint's k/v scales (1.0 when it has none, as vLLM
does): stored = fp8(x / scale), read = stored * scale. kv="bf16" keeps the cache 16-bit and says so in
`numerics`.

Cache layout: full layers keep max_seq positions; sliding layers keep a ring of sliding_window slots
(slot = position mod window). Decode is CUDA-graph safe: static shapes, the position stays on the device,
and each chosen expert's stored weights are gathered with index_select (no host sync). Prefill is not
captured: it runs in chunks and loops over the experts that its tokens were routed to.
Each call decodes that expert's FP4 weights again: slow first tokens (seconds) by design in this reference, not a hang.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..loader import QWeight
from ..qlinear import ACT_A16, ACT_AS_SHIPPED, qlinear
from . import register

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # the registry imports every model module; only building a model needs torch
    torch = F = None

SLIDING, FULL = "sliding_attention", "full_attention"
KV_AS_SHIPPED, KV_BF16, KV_FP8 = "as_shipped", "bf16", "fp8"
FP8_E4M3_MAX = 448.0
_FIELDS = ("weight", "scale", "scale_2", "input_scale", "bias")
_DENSE = ("bf16", "fp16", "fp32")
_DENSE_KIND = {} if torch is None else {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}

# Hugging Face's values (Gemma4TextConfig) for keys a config.json may leave out. The sizes that define the
# network (hidden_size, layers, heads, KV heads, vocab, experts) have no default: a missing one is an error.
_HF_DEFAULTS: dict[str, Any] = {
    "head_dim": 256, "global_head_dim": 512, "sliding_window": 512, "rms_norm_eps": 1e-6,
    "hidden_activation": "gelu_pytorch_tanh", "final_logit_softcapping": None, "attention_bias": False,
    "attention_k_eq_v": False, "num_kv_shared_layers": 0, "enable_moe_block": False,
    "hidden_size_per_layer_input": 256, "use_bidirectional_attention": None,
    "rope_parameters": {SLIDING: {"rope_type": "default", "rope_theta": 10_000.0},
                        FULL: {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                               "rope_theta": 1_000_000.0}},
}


def _cfg(tc: dict, key: str) -> Any:
    return tc[key] if key in tc else _HF_DEFAULTS[key]


# --- configuration ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class AttnSpec:
    """How one layer attends."""
    kind: str               # SLIDING or FULL
    head_dim: int
    n_kv_heads: int
    k_eq_v: bool            # no v_proj: the values are the k_proj output
    window: int | None      # positions a query sees, itself included (sliding layers only)
    kv_layer: int           # the layer whose cache this layer reads: itself unless KV-shared


def _per_layer_overrides(tc: dict, types: list[str]) -> dict[int, dict]:
    """Per-layer head_dim / num_key_value_heads. transformers >= 5.19 writes `per_layer_config`; earlier
    exports write global_head_dim and num_global_key_value_heads, which apply to the full layers."""
    plc = tc.get("per_layer_config")
    if plc is None:
        o = {"head_dim": _cfg(tc, "global_head_dim")}
        if _cfg(tc, "attention_k_eq_v") and tc.get("num_global_key_value_heads") is not None:
            o["num_key_value_heads"] = tc["num_global_key_value_heads"]
        return {i: o for i, t in enumerate(types) if t == FULL}
    out: dict[int, dict] = {}
    for key, over in plc.items():
        unknown = set(over) - {"head_dim", "num_key_value_heads"}
        if unknown:
            raise NotImplementedError(f"per_layer_config overrides {sorted(unknown)}: not supported")
        idx = [int(key)] if str(key).isdigit() else [i for i, t in enumerate(types) if t == key]
        for i in idx:
            out.setdefault(i, {}).update(over)
    return out


def layer_plan(tc: dict) -> list[AttnSpec]:
    """The attention shape of every layer, resolved from the text config as Hugging Face resolves it."""
    n = int(tc["num_hidden_layers"])
    types = list(tc.get("layer_types") or [FULL if (i + 1) % 6 == 0 else SLIDING for i in range(n)])
    if len(types) != n or set(types) - {SLIDING, FULL}:
        raise ValueError(f"layer_types must list {n} of {SLIDING!r}/{FULL!r}: {types}")
    types[-1] = FULL                                   # the reference forces the last layer to full attention
    over = _per_layer_overrides(tc, types)
    n_shared = int(_cfg(tc, "num_kv_shared_layers") or 0)
    first_shared = n - n_shared
    if n_shared and first_shared <= 0:
        raise NotImplementedError(f"num_kv_shared_layers={n_shared} leaves no layer that owns a cache")
    plan = []
    for i, t in enumerate(types):
        o = over.get(i, {})
        src = i
        if n_shared and i >= first_shared:             # reads the last earlier layer of its type
            before = types[:first_shared]
            if t not in before:
                raise ValueError(f"layer {i} shares KV but no earlier layer is {t}")
            src = first_shared - 1 - before[::-1].index(t)
        plan.append(AttnSpec(
            kind=t, head_dim=int(o.get("head_dim", _cfg(tc, "head_dim"))),
            n_kv_heads=int(o.get("num_key_value_heads", tc["num_key_value_heads"])),
            k_eq_v=bool(_cfg(tc, "attention_k_eq_v")) and t == FULL,
            window=int(_cfg(tc, "sliding_window")) if t == SLIDING else None, kv_layer=src))
    return plan


def rope_inv_freq(params: dict, head_dim: int):
    """Inverse frequencies [head_dim // 2] (fp32) of one layer type, computed as Hugging Face does.
    "proportional": the first int(partial_rotary_factor * head_dim // 2) pairs rotate with exponents over
    the full head_dim; the remaining pairs get frequency 0 (cos 1, sin 0: left as they are)."""
    kind, theta = params.get("rope_type", "default"), float(params["rope_theta"])
    factor = float(params.get("factor", 1.0))
    if kind in ("default", "linear"):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        return inv / factor if kind == "linear" else inv
    if kind == "proportional":
        n_rot = int(params.get("partial_rotary_factor", 1.0) * head_dim // 2)
        inv = 1.0 / (theta ** (torch.arange(0, 2 * n_rot, 2, dtype=torch.int64).float() / head_dim))
        inv = torch.cat([inv, torch.zeros(head_dim // 2 - n_rot, dtype=torch.float32)])
        return inv / factor
    raise NotImplementedError(f"rope_type {kind!r} is not supported by this module")


def _activation(name: str):
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    fns = {"gelu": F.gelu, "silu": F.silu, "swish": F.silu, "relu": F.relu}
    if name not in fns:
        raise NotImplementedError(f"hidden_activation {name!r} is not supported by this module")
    return fns[name]


def _declared_dtype(ck) -> Any:
    """The top-level config's dtype first: Hugging Face's dtype="auto" reads it and copies it onto every
    sub-config (a saved text_config can keep a stale one)."""
    for c in (ck.config, ck.text_config):
        name = c.get("dtype") or c.get("torch_dtype")
        if name in ("bfloat16", "float16", "float32"):
            return getattr(torch, name)
    return torch.bfloat16


def _declares_fp8_kv(ck) -> bool:
    if (ck.quant.get("kv") or "") == "fp8":
        return True
    qc = ck.config.get("quantization_config") or ck.text_config.get("quantization_config") or {}
    scheme = qc.get("kv_cache_scheme") or {}
    return (str(qc.get("kv_cache_quant_algo") or "").upper() == "FP8"
            or (scheme.get("num_bits") == 8 and scheme.get("type") == "float"))


def _text_prefix(ck) -> str:
    for p in ("model.language_model.", "model.", "language_model.model.", ""):
        if ck.has(p + "embed_tokens.weight"):
            return p
    raise KeyError("no embed_tokens.weight under model.language_model., model. or language_model.model.")


# --- small ops ----------------------------------------------------------------------------------------
def _rms(x, w, eps: float):
    """RMSNorm in fp32, as the reference: x * (mean(x^2) + eps)^-0.5 [* w], back to x's dtype."""
    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (y if w is None else y * w).to(x.dtype)


def _apply_rope(x, cos, sin):
    """x [C, heads, D]; cos/sin [C, D]. Rotates pairs (i, i + D/2)."""
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def _attend(q, k, v, mask):
    """softmax(q k^T, masked) v with score scale 1.0. q [C, Hq, D]; k, v [S, Hkv, D]; mask [C, S] bool.
    The query heads of one KV head are folded into rows, so k and v are not copied per query head."""
    c, hq, d = q.shape
    hkv = k.shape[1]
    g = hq // hkv
    qg = q.reshape(c, hkv, g, d).permute(1, 2, 0, 3).reshape(hkv, g * c, d)
    o = F.scaled_dot_product_attention(qg, k.transpose(0, 1), v.transpose(0, 1),
                                       attn_mask=mask.repeat(g, 1), scale=1.0)
    return o.reshape(hkv, g, c, d).permute(2, 0, 1, 3).reshape(c, hq * d)


def _raw(t):
    """A 1-byte float tensor as uint8: index_copy_ and friends are not defined for float8 on every backend."""
    return t.view(torch.uint8) if t.element_size() == 1 and t.dtype != torch.uint8 else t


def _gather(t, idx):
    """t[idx[0]] through index_select (graph safe)."""
    return _raw(t).index_select(0, idx).view(t.dtype)[0]


def _name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _to(w: QWeight, device) -> QWeight:
    for f in _FIELDS:
        t = getattr(w, f)
        if t is not None:
            setattr(w, f, t.to(device))
    return w


# --- weights ------------------------------------------------------------------------------------------
@dataclass
class Experts:
    """One projection of every expert: the stored tensors stacked on a leading expert dimension, so one
    expert is a view (`one(e)`, prefill) or an index_select with a device index (`pick(idx)`, decode)."""
    kind: str
    shape: tuple[int, int]
    block: Any
    t: dict

    def one(self, e: int) -> QWeight:
        return QWeight(self.kind, shape=self.shape, block=self.block, **{f: v[e] for f, v in self.t.items()})

    def pick(self, idx) -> QWeight:
        return QWeight(self.kind, shape=self.shape, block=self.block,
                       **{f: _gather(v, idx) for f, v in self.t.items()})

    @classmethod
    def stack(cls, load, n: int, device) -> "Experts":
        """Stack n per-expert linears (load(e) -> QWeight), copying each into place as it is read."""
        first = load(0)
        t = {f: torch.empty((n, *getattr(first, f).shape), dtype=getattr(first, f).dtype, device=device)
             for f in _FIELDS if getattr(first, f) is not None}
        for e in range(n):
            w = first if e == 0 else load(e)
            stored = {f for f in _FIELDS if getattr(w, f) is not None}
            if w.kind != first.kind or w.shape != first.shape or stored != set(t):
                raise ValueError(f"expert {e} is stored differently from expert 0")
            for f, buf in t.items():
                buf[e].copy_(getattr(w, f))
        return cls(first.kind, first.shape, first.block, t)


@dataclass
class MoE:
    router: QWeight
    router_scale: Any        # [H], activation dtype
    expert_scale: Any        # [E], fp32
    pre_norm_2: Any
    post_norm_1: Any
    post_norm_2: Any
    gate: Experts
    up: Experts
    down: Experts
    top_k: int


@dataclass
class Layer:
    spec: AttnSpec
    q: QWeight
    k: QWeight | None        # None for KV-shared layers
    v: QWeight | None        # None for KV-shared layers and k_eq_v layers
    o: QWeight
    q_norm: Any
    k_norm: Any
    input_norm: Any
    post_attn_norm: Any
    pre_ffn_norm: Any
    post_ffn_norm: Any
    gate: QWeight
    up: QWeight
    down: QWeight
    moe: MoE | None
    ple: tuple | None        # (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm)
    scalar: Any
    k_scale: Any             # fp32 scalars used by an FP8 cache
    v_scale: Any


@dataclass
class Gemma4State:
    """One sequence: per-layer caches (None for KV-shared layers), the logits buffer, position ramps."""
    max_seq: int
    k: list
    v: list
    logits: Any
    ramp: dict               # n -> arange(n) int64 on the device


# --- the model ----------------------------------------------------------------------------------------
@register("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")
class Gemma4:
    """engine_api.Model for the Gemma 4 text decoder."""

    @classmethod
    def from_checkpoint(cls, ck, device: str = "cuda", max_seq: int = 4096, act: str = ACT_AS_SHIPPED,
                        kv: str = KV_AS_SHIPPED, dtype: Any = None, prefill_chunk: int = 1024) -> "Gemma4":
        """kv: "as_shipped" (FP8 e4m3 if the checkpoint declares it, else 16-bit), "bf16" (16-bit, a declared
        change when the checkpoint ships FP8) or "fp8" (FP8 e4m3 even if not declared, also declared).
        dtype: activation dtype (default: the checkpoint's). prefill_chunk: prompt positions per step."""
        return cls(ck, device, max_seq, act, kv, dtype, prefill_chunk)

    def __init__(self, ck, device, max_seq, act, kv, dtype, prefill_chunk):
        if torch is None:
            raise ImportError("ikengine.models.gemma4 needs torch")
        if act not in (ACT_AS_SHIPPED, ACT_A16):
            raise ValueError(f"act must be {ACT_AS_SHIPPED!r} or {ACT_A16!r}, got {act!r}")
        if kv not in (KV_AS_SHIPPED, KV_BF16, KV_FP8):
            raise ValueError(f"kv must be {KV_AS_SHIPPED!r}, {KV_BF16!r} or {KV_FP8!r}, got {kv!r}")
        tc = ck.text_config
        if _cfg(tc, "use_bidirectional_attention") == "all":
            raise NotImplementedError("use_bidirectional_attention='all' is not a causal decoder")
        self.device = torch.device(device)
        self.dtype = dtype or _declared_dtype(ck)
        self.act, self.max_seq, self.prefill_chunk = act, int(max_seq), max(1, int(prefill_chunk))
        self.eps = float(_cfg(tc, "rms_norm_eps"))
        self.act_fn = _activation(_cfg(tc, "hidden_activation"))
        self.softcap = _cfg(tc, "final_logit_softcapping")
        self.plan = layer_plan(tc)
        self.vocab_size = int(tc["vocab_size"])
        hidden = int(tc["hidden_size"])
        self.router_root = hidden ** -0.5
        eos: set[int] = set()
        for src in (ck.generation_config, ck.config, tc):
            e = src.get("eos_token_id")
            eos |= set(e if isinstance(e, list) else [] if e is None else [e])
        self.eos_token_ids = frozenset(int(e) for e in eos)

        # RoPE per layer type, sized by that type's head
        rp = _cfg(tc, "rope_parameters")
        self.inv_freq = {}
        for kind in sorted({a.kind for a in self.plan}):
            dims = {a.head_dim for a in self.plan if a.kind == kind}
            if len(dims) != 1:
                raise NotImplementedError(f"{kind} layers with different head sizes {sorted(dims)}")
            params = rp if "rope_theta" in rp else rp.get(kind)
            if params is None:
                raise ValueError(f"rope_parameters has no entry for {kind}")
            self.inv_freq[kind] = rope_inv_freq(params, dims.pop()).to(self.device)

        # KV cache dtype
        declared_fp8 = _declares_fp8_kv(ck)
        self.kv_fp8 = kv == KV_FP8 or (kv == KV_AS_SHIPPED and declared_fp8)
        self.kv_dtype = torch.float8_e4m3fn if self.kv_fp8 else self.dtype

        p = _text_prefix(ck)
        kinds: Counter = Counter()             # (kind, stores an input_scale) -> linears
        attn_bias = bool(_cfg(tc, "attention_bias"))

        def lin(prefix: str, bias: bool = False) -> QWeight:
            w = _to(ck.linear(prefix, bias=bias), self.device)
            kinds[(w.kind, w.input_scale is not None)] += 1
            return w

        def norm(prefix: str):
            return ck.tensor(prefix + ".weight", device=self.device, dtype=torch.float32)

        def tensor(name: str, dt):
            return ck.tensor(name, device=self.device, dtype=dt)

        self.embed = ck.tensor(p + "embed_tokens.weight", device=self.device)
        self.embed_scale = torch.tensor(hidden ** 0.5).to(self.embed.dtype).to(self.device)
        self.final_norm = norm(p + "norm")
        tie = bool(ck.config.get("tie_word_embeddings", tc.get("tie_word_embeddings", True)))
        if tie:
            self.lm_head = QWeight(_DENSE_KIND[self.embed.dtype], self.embed, tuple(self.embed.shape))
        else:
            self.lm_head = lin("lm_head")

        self.ple_dim = int(_cfg(tc, "hidden_size_per_layer_input") or 0)
        if self.ple_dim:
            self.ple_embed = ck.tensor(p + "embed_tokens_per_layer.weight", device=self.device)
            self.ple_embed_scale = torch.tensor(self.ple_dim ** 0.5).to(self.ple_embed.dtype).to(self.device)
            self.ple_proj = lin(p + "per_layer_model_projection")
            self.ple_proj_scale = hidden ** -0.5
            self.ple_norm = norm(p + "per_layer_projection_norm")

        use_moe = bool(_cfg(tc, "enable_moe_block"))
        ones = torch.ones(1, dtype=self.dtype, device=self.device)
        kv_scales_found = False
        self.layers: list[Layer] = []
        self._kv_sources = {a.kv_layer for i, a in enumerate(self.plan) if a.kv_layer != i}
        for i, a in enumerate(self.plan):
            b, at = f"{p}layers.{i}.", f"{p}layers.{i}.self_attn."
            owns = a.kv_layer == i
            ks = _kv_scale(ck, at, "k", self.device) if owns else None
            vs = _kv_scale(ck, at, "v", self.device) if owns else None
            kv_scales_found |= ks is not None or vs is not None
            one32 = torch.ones((), dtype=torch.float32, device=self.device)
            self.layers.append(Layer(
                spec=a, q=lin(at + "q_proj", attn_bias), o=lin(at + "o_proj", attn_bias),
                k=lin(at + "k_proj", attn_bias) if owns else None,
                v=lin(at + "v_proj", attn_bias) if owns and not a.k_eq_v else None,
                q_norm=norm(at + "q_norm"), k_norm=norm(at + "k_norm") if owns else None,
                input_norm=norm(b + "input_layernorm"), post_attn_norm=norm(b + "post_attention_layernorm"),
                pre_ffn_norm=norm(b + "pre_feedforward_layernorm"),
                post_ffn_norm=norm(b + "post_feedforward_layernorm"),
                gate=lin(b + "mlp.gate_proj"), up=lin(b + "mlp.up_proj"), down=lin(b + "mlp.down_proj"),
                moe=self._load_moe(ck, tc, b, lin, norm, tensor, kinds) if use_moe else None,
                ple=(lin(b + "per_layer_input_gate"), lin(b + "per_layer_projection"),
                     norm(b + "post_per_layer_input_norm")) if self.ple_dim else None,
                scalar=tensor(b + "layer_scalar", self.dtype) if ck.has(b + "layer_scalar") else ones,
                k_scale=one32 if ks is None else ks, v_scale=one32 if vs is None else vs))

        expert_scales = _expert_input_scales(self.layers)
        self.numerics = _numerics(kinds, _name(self.dtype), _name(self.embed.dtype), act, self.kv_fp8,
                                  declared_fp8, kv_scales_found, expert_scales)
        self.resolved = {
            "architecture": (ck.config.get("architectures") or [None])[0], "text_prefix": p,
            "layer_types": [a.kind for a in self.plan],
            "head_dim": {a.kind: a.head_dim for a in self.plan},
            "kv_heads": {a.kind: a.n_kv_heads for a in self.plan},
            "k_eq_v_full_layers": bool(_cfg(tc, "attention_k_eq_v")),
            "sliding_window": int(_cfg(tc, "sliding_window")),
            "rope": {k: (rp if "rope_theta" in rp else rp[k]) for k in self.inv_freq},
            "moe": use_moe, "experts": tc.get("num_experts") if use_moe else 0,
            "top_k_experts": tc.get("top_k_experts") if use_moe else 0,
            "per_layer_input_dim": self.ple_dim,
            "kv_shared_layers": int(_cfg(tc, "num_kv_shared_layers") or 0),
            "expert_input_scales": expert_scales,
            "final_logit_softcapping": self.softcap, "tie_word_embeddings": tie,
            "dtype": _name(self.dtype), "kv_cache_dtype": _name(self.kv_dtype), "prefill_chunk": self.prefill_chunk,
        }

    def _load_moe(self, ck, tc, b, lin, norm, tensor, kinds) -> MoE:
        n, top_k, hidden = int(tc["num_experts"]), int(tc["top_k_experts"]), int(tc["hidden_size"])
        ex = b + "experts."
        if ck.has(ex + "gate_up_proj"):            # Hugging Face layout: gate_up [E, 2I, H], down [E, H, I]
            gu = ck.tensor(ex + "gate_up_proj", device=self.device)
            dn = ck.tensor(ex + "down_proj", device=self.device)
            kind = _DENSE_KIND.get(gu.dtype)
            if kind is None or gu.shape[0] != n or gu.shape[2] != hidden or dn.shape[:2] != (n, hidden):
                raise NotImplementedError(f"{ex}gate_up_proj {tuple(gu.shape)} {gu.dtype}: unexpected layout")
            i = gu.shape[1] // 2
            gate = Experts(kind, (i, hidden), None, {"weight": gu[:, :i]})
            up = Experts(kind, (i, hidden), None, {"weight": gu[:, i:]})
            down = Experts(kind, (hidden, dn.shape[2]), None, {"weight": dn})
            kinds[(kind, False)] += 3 * n
        elif ck.has(f"{ex}0.gate_proj.weight"):    # one linear per expert (quantized exports)
            gate, up, down = (Experts.stack(lambda e, m=m: _to(ck.linear(f"{ex}{e}.{m}"), self.device),
                                            n, self.device) for m in ("gate_proj", "up_proj", "down_proj"))
            for x in (gate, up, down):
                kinds[(x.kind, "input_scale" in x.t)] += n
        else:
            raise KeyError(f"no experts under {ex} (expected gate_up_proj/down_proj or <e>.gate_proj)")
        inter = tc.get("moe_intermediate_size")
        if inter is not None and (gate.shape != (inter, hidden) or down.shape != (hidden, inter)):
            raise ValueError(f"{ex}: expert shapes {gate.shape}/{down.shape} do not match the config")
        return MoE(router=lin(b + "router.proj"), router_scale=tensor(b + "router.scale", self.dtype),
                   expert_scale=tensor(b + "router.per_expert_scale", torch.float32),
                   pre_norm_2=norm(b + "pre_feedforward_layernorm_2"),
                   post_norm_1=norm(b + "post_feedforward_layernorm_1"),
                   post_norm_2=norm(b + "post_feedforward_layernorm_2"),
                   gate=gate, up=up, down=down, top_k=top_k)

    # --- engine_api.Model -------------------------------------------------------------------------------
    def alloc_state(self, max_seq: int | None = None) -> Gemma4State:
        s = int(max_seq or self.max_seq)
        if s > self.max_seq:
            raise ValueError(f"max_seq {s} > the model's {self.max_seq}")
        ks, vs = [], []
        for i, a in enumerate(self.plan):
            if a.kv_layer != i:
                ks.append(None)
                vs.append(None)
                continue
            n = s if a.window is None else min(a.window, s)
            ks.append(torch.zeros(n, a.n_kv_heads, a.head_dim, dtype=self.kv_dtype, device=self.device))
            vs.append(torch.zeros(n, a.n_kv_heads, a.head_dim, dtype=self.kv_dtype, device=self.device))
        sizes = {s} | {c.shape[0] for c in ks if c is not None}
        return Gemma4State(s, ks, vs, torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device),
                           {n: torch.arange(n, device=self.device) for n in sizes})

    def reset(self, state: Gemma4State) -> None:
        for c in state.k + state.v:
            if c is not None:
                _raw(c).zero_()
        state.logits.zero_()

    def prefill(self, input_ids, state: Gemma4State):
        ids = input_ids.reshape(-1).to(device=self.device, dtype=torch.int64)
        n = ids.shape[0]
        if not 0 < n <= state.max_seq:
            raise ValueError(f"prompt of {n} tokens; this state holds 1..{state.max_seq}")
        for start in range(0, n, self.prefill_chunk):
            chunk = ids[start:start + self.prefill_chunk]
            x = self._forward(chunk, state.ramp[state.max_seq][start:start + chunk.shape[0]], start, state)
        state.logits.copy_(self._logits(x[-1:]))
        return state.logits

    def decode(self, token, pos, state: Gemma4State):
        x = self._forward(token.reshape(1), pos.reshape(1).to(torch.int64), None, state)
        state.logits.copy_(self._logits(x))
        return state.logits

    # --- forward ----------------------------------------------------------------------------------------
    def _forward(self, ids, pos, start, state):
        """Hidden states after the last layer for one chunk at positions `pos` [C] (device, int64).
        start: the chunk's first position (prefill) or None (decode: graph-safe path)."""
        x = (F.embedding(ids, self.embed) * self.embed_scale).to(self.dtype)
        ple = self._per_layer_inputs(ids, x) if self.ple_dim else None
        ropes = {}
        for kind, inv in self.inv_freq.items():
            f = pos.float()[:, None] * inv[None, :]
            e = torch.cat([f, f], -1)
            ropes[kind] = (e.cos().to(self.dtype), e.sin().to(self.dtype))
        shared: dict[int, tuple] = {}
        for i, lw in enumerate(self.layers):
            x = self._layer(i, lw, x, pos, start, state, ropes, shared, ple)
        return x

    def _per_layer_inputs(self, ids, x):
        """[C, layers, D]: (norm(proj(x) / sqrt(hidden)) + embed_per_layer(ids)) / sqrt(2)."""
        n, d = len(self.layers), self.ple_dim
        tok = (F.embedding(ids, self.ple_embed) * self.ple_embed_scale).reshape(-1, n, d)
        proj = (qlinear(x, self.ple_proj, self.act) * self.ple_proj_scale).reshape(-1, n, d)
        proj = _rms(proj, self.ple_norm, self.eps)
        return (proj + tok.to(proj.dtype)) * 2.0 ** -0.5

    def _layer(self, i, lw: Layer, x, pos, start, state, ropes, shared, ple):
        r = x
        h = self._attention(i, lw, _rms(x, lw.input_norm, self.eps), pos, start, state, ropes, shared)
        x = r + _rms(h, lw.post_attn_norm, self.eps)
        r = x
        h = self._mlp(_rms(r, lw.pre_ffn_norm, self.eps), lw.gate, lw.up, lw.down)
        if lw.moe is not None:
            m = lw.moe
            w, idx = self._route(m, r)
            xe = _rms(r, m.pre_norm_2, self.eps)
            he = self._experts_decode(m, xe, w, idx) if start is None else self._experts_prefill(m, xe, w, idx)
            h = _rms(h, m.post_norm_1, self.eps) + _rms(he, m.post_norm_2, self.eps)
        x = r + _rms(h, lw.post_ffn_norm, self.eps)
        if ple is not None:
            gate, proj, norm = lw.ple
            h = self.act_fn(qlinear(x, gate, self.act)) * ple[:, i, :]
            x = x + _rms(qlinear(h, proj, self.act), norm, self.eps)
        return x * lw.scalar

    def _attention(self, i, lw: Layer, h, pos, start, state, ropes, shared):
        a = lw.spec
        c = h.shape[0]
        cos, sin = ropes[a.kind]
        q = _apply_rope(_rms(qlinear(h, lw.q, self.act).view(c, -1, a.head_dim), lw.q_norm, self.eps), cos, sin)
        if a.kv_layer == i:
            k_raw = qlinear(h, lw.k, self.act).view(c, -1, a.head_dim)
            v_raw = k_raw if lw.v is None else qlinear(h, lw.v, self.act).view(c, -1, a.head_dim)
            k = _apply_rope(_rms(k_raw, lw.k_norm, self.eps), cos, sin)
            v = _rms(v_raw, None, self.eps)
            kv = self._cache(i, lw, k, v, pos, start, state)
            if i in self._kv_sources:
                shared[i] = kv
        else:
            kv = shared[a.kv_layer]                  # the same keys, values and mask as the source layer
        return qlinear(_attend(q, *kv), lw.o, self.act)

    def _store(self, x, scale):
        if not self.kv_fp8:
            return x.to(self.kv_dtype)
        return (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    def _load(self, t, scale):
        return (t.float() * scale).to(self.dtype) if self.kv_fp8 else t

    def _cache(self, i, lw: Layer, k, v, pos, start, state):
        """Write this chunk's k/v into layer i's cache and return (keys, values, mask [C, S]) to attend to.
        Prefill reads what the cache would hold, so the chunk sees its own k/v after the FP8 round trip."""
        a, kc, vc = lw.spec, state.k[i], state.v[i]
        kq, vq = self._store(k, lw.k_scale), self._store(v, lw.v_scale)
        c, n = k.shape[0], kc.shape[0]
        if a.window is None:                         # full layer: slot = position
            if start is None:
                _raw(kc).index_copy_(0, pos, _raw(kq))
                _raw(vc).index_copy_(0, pos, _raw(vq))
                keys, vals, kpos = kc, vc, state.ramp[n]
            else:
                _raw(kc)[start:start + c] = _raw(kq)
                _raw(vc)[start:start + c] = _raw(vq)
                keys, vals, kpos = kc[:start + c], vc[:start + c], state.ramp[n][:start + c]
            mask = kpos[None, :] <= pos[:, None]
        else:                                        # sliding layer: ring of n slots, slot = position mod n
            s = state.ramp[n]
            if start is None:
                _raw(kc).index_copy_(0, pos % n, _raw(kq))
                _raw(vc).index_copy_(0, pos % n, _raw(vq))
                keys, vals = kc, vc
                kpos = pos - (pos - s) % n          # the position each slot now holds (< 0: never written)
            else:
                held = (start - 1) - ((start - 1) - s) % n
                keys = torch.cat([_raw(kc), _raw(kq)]).view(kc.dtype)
                vals = torch.cat([_raw(vc), _raw(vq)]).view(vc.dtype)
                kpos = torch.cat([held, pos])
                m = min(c, n)
                _raw(kc).index_copy_(0, pos[-m:] % n, _raw(kq[-m:]))
                _raw(vc).index_copy_(0, pos[-m:] % n, _raw(vq[-m:]))
            kp, qp = kpos[None, :], pos[:, None]
            mask = (kp >= 0) & (kp <= qp) & (kp > qp - a.window)
        return self._load(keys, lw.k_scale), self._load(vals, lw.v_scale), mask

    def _mlp(self, x, gate: QWeight, up: QWeight, down: QWeight):
        return qlinear(self.act_fn(qlinear(x, gate, self.act)) * qlinear(x, up, self.act), down, self.act)

    def _route(self, m: MoE, r):
        """Top-k expert weights [C, k] (fp32) and ids [C, k] from the pre-norm residual r [C, H]."""
        h = _rms(r, None, self.eps) * m.router_scale * self.router_root
        probs = torch.softmax(qlinear(h, m.router, self.act), dim=-1, dtype=torch.float32)
        w, idx = torch.topk(probs, m.top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        return w * m.expert_scale[idx], idx

    def _experts_prefill(self, m: MoE, x, w, idx):
        """Loop over the experts the chunk's tokens chose (host sync: prefill only)."""
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in torch.unique(idx).tolist():
            tok, slot = torch.where(idx == e)
            y = self._mlp(x[tok], m.gate.one(e), m.up.one(e), m.down.one(e))
            out.index_add_(0, tok, y.float() * w[tok, slot, None])
        return out.to(self.dtype)

    def _experts_decode(self, m: MoE, x, w, idx):
        """One token: gather each chosen expert's stored weights by a device index (graph safe)."""
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for j in range(m.top_k):
            e = idx[0, j:j + 1]
            y = self._mlp(x, m.gate.pick(e), m.up.pick(e), m.down.pick(e))
            out = out + y.float() * w[:, j:j + 1]
        return out.to(self.dtype)

    def _logits(self, x):
        y = qlinear(_rms(x, self.final_norm, self.eps), self.lm_head, self.act).float().reshape(-1)
        if self.softcap:
            y = torch.tanh(y / self.softcap) * self.softcap
        return y


def _expert_input_scales(layers: list[Layer]) -> dict | None:
    """How the stored expert activation scales vary, for comparing with a serving engine: this module uses each
    expert's own input_scale as stored, while an engine that quantizes the activation once for all experts (or
    once for gate and up together) uses one value. None when the experts have no input scales."""
    moes = [lw.moe for lw in layers if lw.moe is not None]
    if not moes or any(x.t.get("input_scale") is None for m in moes for x in (m.gate, m.up, m.down)):
        return None
    same = {name: all(bool((t == t.reshape(-1)[0]).all()) for t in (getattr(m, name).t["input_scale"] for m in moes))
            for name in ("gate", "up", "down")}
    return {"one_value_per_layer": same,
            "gate_equals_up": all(bool(torch.equal(m.gate.t["input_scale"], m.up.t["input_scale"])) for m in moes)}


# --- numerics -----------------------------------------------------------------------------------------
def _act_scheme(kind: str, static_input: bool) -> str | None:
    """How a linear of this stored kind quantizes its input as shipped; None for a 16-bit input (16-bit
    weights, and weight-only NVFP4: W4A16)."""
    if kind in _DENSE or (kind == "nvfp4" and not static_input):
        return None
    if kind == "nvfp4":
        return "nvfp4 with each module's static input_scale (W4A4)"
    if kind == "fp8_tensor":
        return ("fp8 e4m3 per tensor with each module's static input_scale" if static_input
                else "fp8 e4m3 per tensor, one dynamic scale per tensor")
    if kind == "fp8_block":
        return "fp8 e4m3, dynamic per-token groups"
    return f"{kind} as the checkpoint declares"


def _expert_scales_text(s: dict, act: str) -> str:
    """numerics["expert_input_scales"]: the first thing to check when greedy ids differ from a serving engine."""
    one, gu = s["one_value_per_layer"], s["gate_equals_up"]
    facts = (f"stored: gate equals up {'yes' if gu else 'no'}; one value across a layer's experts: "
             + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in one.items()))
    if act == ACT_A16:
        return "unused (a16 keeps the expert inputs 16-bit); " + facts
    return ("each expert's gate, up and down use their own stored input_scale; " + facts + "; an engine that "
            "shares one scale per fused group (vLLM takes the max of the stored ones) "
            + ("computes the same" if gu and all(one.values()) else
               "quantizes some expert inputs differently: check this first when greedy ids differ"))


def _numerics(kinds: Counter, dt: str, embed_dt: str, act: str, kv_fp8: bool, declared_fp8: bool,
              kv_scales_found: bool, expert_scales: dict | None) -> dict:
    """The `numerics` block for ENGINE.json. kinds: (stored kind, stores an input_scale) -> linears. The
    activations are described from the scales each linear stores, so a weight-only export reads as W4A16 and
    act="a16" is a declared change only when some linear quantizes its input as shipped."""
    weights, shipped, w4a16 = Counter(), Counter(), 0
    for (kind, static), n in kinds.items():
        weights[kind] += n
        s = _act_scheme(kind, static)
        if s is not None:
            shipped[s] += n
        elif kind not in _DENSE:
            w4a16 += n
    dense = sum(n for k, n in weights.items() if k in _DENSE)
    quant = [f"{s} for {n} linears" for s, n in sorted(shipped.items())]
    rest = ([f"{dt} into {w4a16} weight-only nvfp4 linears (W4A16)"] if w4a16 else []) \
        + ([f"{dt} into {dense} 16-bit linears"] if dense else [])
    changed = []
    if not quant and not w4a16:
        acts = f"{dt} (the checkpoint declares no activation quantization)"
    elif not quant:
        acts = "as shipped: " + "; ".join(rest)
    elif act == ACT_A16:
        acts = f"{dt} into every linear (a16)"
        changed.append("activations: a16 instead of " + "; ".join(quant))
    else:
        acts = "as shipped: " + "; ".join(quant + rest)
    if kv_fp8:
        kvs = "fp8 e4m3, " + ("k/v scales from the checkpoint" if kv_scales_found
                               else "scale 1.0 (the checkpoint has no k/v scales; vLLM uses 1.0 too)")
    else:
        kvs = dt
    if kv_fp8 != declared_fp8:
        changed.append(f"kv: {kvs} instead of the declared " + ("fp8 e4m3" if declared_fp8 else dt))
    else:
        kvs += " (as shipped)"
    out = {
        "weights": "as stored: " + ", ".join(f"{weights[k]} {k} linears" for k in sorted(weights))
                   + f"; embedding {embed_dt}",
        "activations": acts,
        "kv": kvs,
        "compute": f"{dt} activations; fp32 RMSNorm, router softmax, MoE weighted sum and logits; "
                   "attention by torch SDPA",
    }
    if expert_scales is not None:
        out["expert_input_scales"] = _expert_scales_text(expert_scales, act)
    out.update(as_shipped=not changed, changed=changed)
    return out


def _kv_scale(ck, attn: str, which: str, device):
    """The checkpoint's static k or v cache scale of one layer, or None. Exporters name it differently."""
    names = (f"{attn}{which}_scale", f"{attn}{which}_proj.{which}_scale", f"{attn}attn.{which}_scale",
             f"{attn}k_proj.{which}_scale")
    for name in dict.fromkeys(names):
        if ck.has(name):
            return ck.tensor(name, device=device, dtype=torch.float32).reshape(())
    return None
