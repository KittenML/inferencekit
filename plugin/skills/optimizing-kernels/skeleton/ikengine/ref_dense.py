"""Config-driven dense decoder reference: Llama, Mistral, Qwen2 and Qwen3 checkpoints (16-bit, block FP8, NVFP4).

    embed -> L x [RMSNorm -> attention (GQA, RoPE, optional q/k-norm, optional sliding window) -> residual
                  RMSNorm -> gated MLP (SwiGLU) -> residual] -> RMSNorm -> output layer (tied or not)

Everything the checkpoint's config.json says is honoured: head_dim, grouped KV heads, attention biases (a
stored bias is always used), RoPE with default, linear, llama3 or yarn scaling (the frequencies follow the
Hugging Face formulas), q/k-norm when the checkpoint stores q_norm/k_norm, a tied output layer, and sliding
windows per layer (`layer_types`, Mistral's `sliding_window`, or Qwen's `use_sliding_window` + `max_window_layers`).
Dynamic NTK and LongRoPE depend on the sequence length and raise rather than run with other frequencies.

Numerics: 16-bit tensors in the checkpoint dtype, as Hugging Face computes them (norms in fp32 and rounded
back, RoPE tables in fp32 and rounded); every linear goes through qlinear with its weight as stored; attention
scores, softmax and the weighted sum run in fp32 and round once. The KV cache is the model dtype, or e4m3
with the stored k_scale/v_scale when the checkpoint declares an FP8 KV cache (kv="bf16" keeps it 16-bit, a
declared change). `numerics["changed"]` lists only real changes (act="a16" on a 16-bit checkpoint, or kv="bf16"
on a 16-bit KV cache, changes nothing) and `numerics["as_shipped"]` is True when that list is empty.
Attention always reads keys and values back from the cache, so prefill and decode see the same cached values
(as vLLM does with an FP8 cache).

Decode is CUDA-graph capturable: static shapes (attention reads the whole cache and masks by position on the
device), no host synchronisation, logits written into the state's static buffer. It is slow on purpose: to
make it fast, replace these internals with kernels behind the same Model contract and keep the proofs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .loader import QWeight
from .models import register
from .qlinear import ACT_A16, ACT_AS_SHIPPED, kv_dequant, kv_quant, qlinear

ARCHITECTURES = ("LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM")
KV_AS_SHIPPED, KV_BF16 = "as_shipped", "bf16"
_Q_CHUNK = 512          # prefill queries per attention chunk (bounds the fp32 score matrix)


# --- config ---------------------------------------------------------------------------------------------
def rope_parameters(cfg: dict) -> dict:
    """RoPE settings from either config layout: transformers 5 `rope_parameters`, or the older `rope_theta`
    plus `rope_scaling`. Per-layer-type nesting (used by some multimodal families) is not a dense model's."""
    rp = {**(cfg.get("rope_scaling") or {}), **(cfg.get("rope_parameters") or {})}
    if any(isinstance(v, dict) for v in rp.values()):
        raise NotImplementedError("RoPE parameters per layer type are not a dense decoder's; use a model module")
    rp.setdefault("rope_theta", cfg.get("rope_theta", 10000.0))
    rp["rope_type"] = rp.get("rope_type") or rp.get("type") or "default"
    if "partial_rotary_factor" not in rp and cfg.get("partial_rotary_factor") is not None:
        rp["partial_rotary_factor"] = cfg["partial_rotary_factor"]
    if rp["rope_type"] in ("llama3", "yarn"):
        rp.setdefault("original_max_position_embeddings",
                      cfg.get("original_max_position_embeddings") or cfg.get("max_position_embeddings"))
    return rp


def rope_inv_freq(cfg: dict, head_dim: int):
    """(inv_freq fp32 [rotary_dim/2] on the CPU, attention_scaling) for the config's RoPE type.

    Written to give the same fp32 values as transformers' ROPE_INIT_FUNCTIONS (default, linear, llama3, yarn).
    """
    import torch
    rp = rope_parameters(cfg)
    kind, base = rp["rope_type"], float(rp["rope_theta"])
    dim = int(head_dim * float(rp.get("partial_rotary_factor", 1.0)))
    exps = torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float) / dim
    inv = 1.0 / (base ** exps)
    if kind == "default":
        return inv, 1.0
    if kind == "linear":
        return inv / float(rp["factor"]), 1.0
    if kind == "llama3":
        factor, lo, hi = float(rp["factor"]), float(rp["low_freq_factor"]), float(rp["high_freq_factor"])
        old = float(rp["original_max_position_embeddings"])
        wavelen = 2 * math.pi / inv
        out = torch.where(wavelen > old / lo, inv / factor, inv)                    # long waves: interpolate
        smooth = (old / wavelen - lo) / (hi - lo)
        mid = ~(wavelen < old / hi) * ~(wavelen > old / lo)
        return torch.where(mid, (1 - smooth) * out / factor + smooth * out, out), 1.0
    if kind == "yarn":
        factor = rp.get("factor")
        orig = float(rp["original_max_position_embeddings"])
        if factor is None:
            factor = float(cfg["max_position_embeddings"]) / orig
        factor = float(factor)

        def mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
        attn = rp.get("attention_factor")
        if attn is None:
            m, m_all = rp.get("mscale"), rp.get("mscale_all_dim")
            attn = mscale(factor, m) / mscale(factor, m_all) if (m and m_all) else mscale(factor)
        beta_fast, beta_slow = float(rp.get("beta_fast") or 32), float(rp.get("beta_slow") or 1)

        def corr_dim(rot):
            return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))
        low, high = corr_dim(beta_fast), corr_dim(beta_slow)
        if rp.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        pos_freqs = base ** (torch.arange(0, dim, 2).to(torch.float) / dim)
        extra = 1 - ramp                                                            # share kept unscaled
        return (1.0 / (factor * pos_freqs)) * (1 - extra) + (1.0 / pos_freqs) * extra, float(attn)
    raise NotImplementedError(f"RoPE type {kind!r} (it depends on the sequence length) is not in the reference")


@dataclass
class DenseConfig:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab: int
    eps: float
    act: str
    windows: list            # per layer: None (full attention) or the window length in tokens
    max_pos: int

    @classmethod
    def from_dict(cls, cfg: dict, arch: str) -> "DenseConfig":
        heads = int(cfg["num_attention_heads"])
        n = int(cfg["num_hidden_layers"])
        sw = cfg.get("sliding_window")
        if cfg.get("layer_types"):
            win = [sw if t == "sliding_attention" else None for t in cfg["layer_types"]]
        elif arch == "MistralForCausalLM":
            win = [sw] * n
        elif cfg.get("use_sliding_window") and sw:
            first = int(cfg.get("max_window_layers", n))
            win = [sw if i >= first else None for i in range(n)]
        else:
            win = [None] * n
        return cls(hidden=int(cfg["hidden_size"]), heads=heads,
                   kv_heads=int(cfg.get("num_key_value_heads") or heads),
                   head_dim=int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads), layers=n,
                   vocab=int(cfg["vocab_size"]), eps=float(cfg.get("rms_norm_eps", 1e-6)),
                   act=str(cfg.get("hidden_act") or cfg.get("hidden_activation") or "silu"),
                   windows=[int(w) if w else None for w in win],
                   max_pos=int(cfg.get("max_position_embeddings") or 1 << 20))


# --- building blocks ------------------------------------------------------------------------------------
def rms_norm(x, weight, eps: float):
    """Hugging Face RMSNorm: normalise in fp32, round to x's dtype, then multiply by the stored weight."""
    import torch
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def apply_rope(x, cos, sin):
    """x [..., T, D]; cos/sin [T, R] with R <= D (partial rotary leaves the last D - R dims as they are)."""
    import torch
    r = cos.shape[-1]
    xr, rest = x[..., :r], x[..., r:]
    half = r // 2
    rot = torch.cat((-xr[..., half:], xr[..., :half]), dim=-1)
    out = xr * cos + rot * sin
    return out if rest.shape[-1] == 0 else torch.cat((out, rest), dim=-1)


def activation(name: str):
    import torch.nn.functional as F
    if name in ("silu", "swish"):
        return F.silu
    if name in ("gelu_pytorch_tanh", "gelu_tanh", "gelu_new"):
        return lambda t: F.gelu(t, approximate="tanh")
    if name == "gelu":
        return F.gelu
    raise NotImplementedError(f"hidden_act {name!r}")


def attend(q, k, v, qpos, kpos, window: int | None, scale: float):
    """Causal (optionally sliding-window) attention in fp32. q [H, Tq, D]; k, v [KVH, Tk, D] (any dtype);
    qpos [Tq], kpos [Tk] int64 positions. A key is visible when kpos <= qpos and, with a window W,
    kpos > qpos - W (W keys including the current one, as Hugging Face and vLLM). Returns [H, Tq, D] fp32."""
    import torch
    h, tq, d = q.shape
    kvh = k.shape[0]
    qf = q.float().reshape(kvh, h // kvh, tq, d)
    kf, vf = k.float(), v.float()
    mask = kpos.unsqueeze(0) <= qpos.unsqueeze(1)                                   # [Tq, Tk]
    if window:
        mask = mask & (kpos.unsqueeze(0) > qpos.unsqueeze(1) - window)
    s = torch.einsum("hgqd,hkd->hgqk", qf, kf) * scale
    p = s.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return torch.einsum("hgqk,hkd->hgqd", p, vf).reshape(h, tq, d)


# --- the model ------------------------------------------------------------------------------------------
@dataclass
class Layer:
    q: QWeight
    k: QWeight
    v: QWeight
    o: QWeight
    gate: QWeight
    up: QWeight
    down: QWeight
    ln_attn: Any
    ln_mlp: Any
    q_norm: Any
    k_norm: Any
    k_scale: Any             # fp32 0-d, or None (1.0) for an FP8 KV cache
    v_scale: Any


@dataclass
class DenseState:
    k: list                  # per layer [KVH, max_seq, D] (model dtype or float8_e4m3fn)
    v: list
    logits: Any              # [vocab] fp32, the static output buffer
    kpos: Any                # [max_seq] int64 positions (static)
    max_seq: int


def _eos_ids(ck) -> frozenset[int]:
    out: set[int] = set()
    for src in (ck.generation_config, ck.config):
        e = src.get("eos_token_id")
        out.update(int(t) for t in (e if isinstance(e, list) else [e] if e is not None else []))
    return frozenset(out)


@register(*ARCHITECTURES)
class DenseDecoder:
    """engine_api.Model for Llama-style dense decoders. Build with `from_checkpoint`."""

    def __init__(self, cfg: DenseConfig, embed, layers: list[Layer], norm, head: QWeight, rope, device,
                 max_seq: int, act: str, kv_dtype, numerics: dict, eos: frozenset[int]):
        import torch
        self.cfg, self.embed, self.layers, self.norm, self.head = cfg, embed, layers, norm, head
        self.device = torch.device(device)
        self.dtype = embed.dtype
        self.max_seq = int(max_seq)
        self.vocab_size = int(head.shape[0])
        self.act = act
        self.kv_dtype = kv_dtype
        self.numerics = numerics
        self.eos_token_ids = eos
        self.scale = cfg.head_dim ** -0.5
        self.mlp_act = activation(cfg.act)
        inv, attn_scaling = rope
        pos = torch.arange(self.max_seq, device=self.device, dtype=torch.float32)
        freqs = pos.unsqueeze(1) * inv.to(self.device).unsqueeze(0)                # [S, R/2], fp32 as HF
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = (emb.cos() * attn_scaling).to(self.dtype)
        self.sin = (emb.sin() * attn_scaling).to(self.dtype)

    @classmethod
    def from_checkpoint(cls, ck, device: str = "cuda", max_seq: int = 4096, act: str = ACT_AS_SHIPPED,
                        kv: str = KV_AS_SHIPPED) -> "DenseDecoder":
        import torch
        if act not in (ACT_AS_SHIPPED, ACT_A16):
            raise ValueError(f"act must be {ACT_AS_SHIPPED!r} or {ACT_A16!r}")
        if kv not in (KV_AS_SHIPPED, KV_BF16):
            raise ValueError(f"kv must be {KV_AS_SHIPPED!r} or {KV_BF16!r}")
        arch = (ck.config.get("architectures") or [""])[0]
        tc = ck.text_config
        cfg = DenseConfig.from_dict(tc, arch)
        dtype = getattr(torch, ck.dtype_name)

        def t(name):                        # 16-bit tensors compute in the config dtype, as Hugging Face loads them
            return ck.tensor(name, device=device, dtype=dtype)

        def lin(prefix):
            w = ck.linear(prefix)
            for f in ("weight", "scale", "scale_2", "input_scale", "bias"):   # the Checkpoint may load elsewhere
                v = getattr(w, f)
                if v is not None:
                    setattr(w, f, v.to(device))
            if w.kind in ("bf16", "fp16", "fp32") and w.weight.dtype != dtype:
                w.weight = w.weight.to(dtype)
                w.kind = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, "fp32")
            if w.bias is not None and w.bias.dtype != dtype:
                w.bias = w.bias.to(dtype)
            return w
        fp8_kv = kv == KV_AS_SHIPPED and (ck.quant.get("kv") or "") == "fp8"
        layers = []
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            ks, vs = ck.kv_scales(p + "self_attn") if fp8_kv else (None, None)
            layers.append(Layer(
                q=lin(p + "self_attn.q_proj"), k=lin(p + "self_attn.k_proj"), v=lin(p + "self_attn.v_proj"),
                o=lin(p + "self_attn.o_proj"), gate=lin(p + "mlp.gate_proj"), up=lin(p + "mlp.up_proj"),
                down=lin(p + "mlp.down_proj"), ln_attn=t(p + "input_layernorm.weight"),
                ln_mlp=t(p + "post_attention_layernorm.weight"),
                q_norm=t(p + "self_attn.q_norm.weight") if ck.has(p + "self_attn.q_norm.weight") else None,
                k_norm=t(p + "self_attn.k_norm.weight") if ck.has(p + "self_attn.k_norm.weight") else None,
                k_scale=None if ks is None else ks.float().reshape(()).to(device),
                v_scale=None if vs is None else vs.float().reshape(()).to(device)))
        embed = t("model.embed_tokens.weight")
        tied = bool(ck.config.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)))
        if not tied and ck.has("lm_head.weight"):
            head = lin("lm_head")
        else:
            head = QWeight({torch.bfloat16: "bf16", torch.float16: "fp16"}.get(embed.dtype, "fp32"), embed,
                           tuple(embed.shape))
        q = ck.quant
        declared_act = {"fp8_dynamic_group": f"fp8 e4m3, dynamic per token in groups of {(q.get('block') or [0, 128])[1]}",
                        "nvfp4_static": "nvfp4 e2m1, per-16 e4m3 scales under the static input_scale",
                        "fp8_static": "fp8 e4m3, static input_scale", "fp8_dynamic_tensor": "fp8 e4m3, dynamic per tensor",
                        "per layer": "per layer, as declared"}.get(q.get("act"), "16-bit")
        name = str(dtype).replace("torch.", "")
        kv_text = "fp8 e4m3 (stored k_scale/v_scale, 1.0 when absent)" if fp8_kv else name
        if kv == KV_BF16 and (q.get("kv") or "") == "fp8":
            kv_text += " (changed from the declared fp8 e4m3)"
        changed = []                        # only what differs from the checkpoint: a16 on 16-bit weights is not
        if act == ACT_A16 and q.get("act"):
            changed.append(f"activations: 16-bit (a16) instead of the declared {declared_act}")
        if kv == KV_BF16 and (q.get("kv") or "") == "fp8":
            changed.append(f"kv: {name} instead of the declared fp8 e4m3")
        numerics = {
            "weights": "16-bit as stored" if q["method"] == "none" else f"{q['method']} as stored ({q['source']})",
            "activations": (f"16-bit (changed from {declared_act})" if act == ACT_A16 and q.get("act")
                            else "16-bit" if act == ACT_A16 else declared_act),
            "kv": kv_text, "compute": name, "attention": "fp32 scores, softmax and sum",
            "as_shipped": not changed, "changed": changed,
        }
        max_seq = min(int(max_seq), cfg.max_pos)
        return cls(cfg, embed, layers, t("model.norm.weight"), head, rope_inv_freq(tc, cfg.head_dim), device,
                   max_seq, act, torch.float8_e4m3fn if fp8_kv else dtype, numerics, _eos_ids(ck))

    # --- engine_api.Model ---------------------------------------------------------------------------
    def alloc_state(self, max_seq: int) -> DenseState:
        import torch
        s = min(int(max_seq), self.max_seq)
        c = self.cfg
        shape = (c.kv_heads, s, c.head_dim)
        return DenseState(k=[torch.zeros(shape, dtype=self.kv_dtype, device=self.device) for _ in range(c.layers)],
                          v=[torch.zeros(shape, dtype=self.kv_dtype, device=self.device) for _ in range(c.layers)],
                          logits=torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device),
                          kpos=torch.arange(s, dtype=torch.long, device=self.device), max_seq=s)

    def reset(self, state: DenseState) -> None:
        for t in state.k + state.v:
            t.zero_()

    def prefill(self, input_ids, state: DenseState):
        """Run the prompt from position 0; queries are processed in chunks so long prompts stay bounded."""
        n = int(input_ids.shape[0])
        if n > state.max_seq:
            raise ValueError(f"prompt of {n} tokens exceeds the cache ({state.max_seq})")
        pos = state.kpos[:n]
        h = self._forward(input_ids, pos, state, kv_len=n)
        state.logits.copy_(self._logits(h[-1:])[0])
        return state.logits

    def decode(self, token, pos, state: DenseState):
        h = self._forward(token.reshape(1), pos.reshape(1).long(), state, kv_len=None)
        state.logits.copy_(self._logits(h)[0])
        return state.logits

    # --- forward --------------------------------------------------------------------------------------
    def _logits(self, h):
        return qlinear(rms_norm(h, self.norm, self.cfg.eps), self.head, self.act).float()

    def _forward(self, ids, pos, state: DenseState, kv_len: int | None):
        """ids [T], pos [T] int64. kv_len: the number of cached positions to attend over (prefill), or None
        to attend over the whole cache with the position mask (decode: static shapes)."""
        x = self.embed.index_select(0, ids)
        for i, L in enumerate(self.layers):
            x = x + self._attention(i, L, rms_norm(x, L.ln_attn, self.cfg.eps), pos, state, kv_len)
            h = rms_norm(x, L.ln_mlp, self.cfg.eps)
            x = x + qlinear(self.mlp_act(qlinear(h, L.gate, self.act)) * qlinear(h, L.up, self.act), L.down, self.act)
        return x

    def _attention(self, i: int, L: Layer, h, pos, state: DenseState, kv_len: int | None):
        import torch
        c = self.cfg
        t = h.shape[0]
        q = qlinear(h, L.q, self.act).view(t, c.heads, c.head_dim)
        k = qlinear(h, L.k, self.act).view(t, c.kv_heads, c.head_dim)
        v = qlinear(h, L.v, self.act).view(t, c.kv_heads, c.head_dim)
        if L.q_norm is not None:
            q = rms_norm(q, L.q_norm, c.eps)
        if L.k_norm is not None:
            k = rms_norm(k, L.k_norm, c.eps)
        cos, sin = self.cos.index_select(0, pos), self.sin.index_select(0, pos)
        q = apply_rope(q.transpose(0, 1), cos, sin)                                 # [H, T, D]
        k = apply_rope(k.transpose(0, 1), cos, sin)
        v = v.transpose(0, 1)
        kc, vc = state.k[i], state.v[i]
        if kc.dtype == self.dtype:
            kc.index_copy_(1, pos, k)
            vc.index_copy_(1, pos, v)
        else:                               # e4m3 bytes, written through a uint8 view
            kc.view(torch.uint8).index_copy_(1, pos, kv_quant(k, L.k_scale).view(torch.uint8))
            vc.view(torch.uint8).index_copy_(1, pos, kv_quant(v, L.v_scale).view(torch.uint8))
        n = state.max_seq if kv_len is None else kv_len
        keys, vals = kc[:, :n], vc[:, :n]
        if kc.dtype != self.dtype:
            keys, vals = kv_dequant(keys, L.k_scale), kv_dequant(vals, L.v_scale)
        kpos = state.kpos[:n]
        outs = [attend(q[:, s:s + _Q_CHUNK], keys, vals, pos[s:s + _Q_CHUNK], kpos, c.windows[i], self.scale)
                for s in range(0, t, _Q_CHUNK)]
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        o = o.to(self.dtype).transpose(0, 1).reshape(t, c.heads * c.head_dim)
        return qlinear(o, L.o, self.act)
