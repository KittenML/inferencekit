"""Qwen3.5 / Qwen3.8 hybrid family, text path: Qwen3_5ForConditionalGeneration and Qwen3_5ForCausalLM.

Qwen3.8-27B is this family: 64 layers, three Gated DeltaNet ("linear_attention") layers per gated softmax
attention ("full_attention") layer, as config `layer_types` lists them. The vision tower (model.visual.*) and the
multi-token-prediction head (mtp.*) are not loaded, so there is no speculative decoding here.

Correct first, written from the config and the Hugging Face reference (transformers 5.19, modeling_qwen3_5.py):
- RMSNorm is zero-centred: fp32 x / rms(x) * (1 + w), cast back. The DeltaNet output norm is not: w * x / rms(x),
  then times silu(z) in fp32.
- Attention: q_proj emits [query | gate] per head; RMSNorm on q and k per head; RoPE (rotate-half) on the first
  partial_rotary_factor of each head (text tokens have equal positions on all three M-RoPE axes, so this is plain
  RoPE); the output is multiplied by sigmoid(gate) before o_proj. Query head h reads KV head h // (heads / kv_heads).
- Gated DeltaNet: in_proj_qkv -> causal depthwise conv1d (kernel 4) -> silu -> q, k, v. q and k are L2-normalised
  (q also scaled by k_dim^-0.5) and repeated to the value heads; beta = sigmoid(in_proj_b);
  g = -exp(A_log) * softplus(in_proj_a + dt_bias). Per value head, with the state S [k_dim, v_dim] in fp32:
      S <- exp(g) * S;   S <- S + k (beta * (v - S^T k))^T;   o = S^T q
  then the gated output norm and out_proj.
- Prefill runs the delta rule a chunk (64 tokens) at a time, one triangular solve per chunk; decode runs the
  one-token update in place. tests/test_skeleton_qwen35.py checks one against the other and both against HF.
  Prefill attention runs 512 queries at a time, so its score memory grows with the prompt, not its square.

State (alloc_state): the KV cache of the attention layers in the activation dtype (BF16 for the shipped
checkpoints; FP8 e4m3 only when the checkpoint declares it), the conv state (the last kernel-1 inputs) in the
activation dtype, the recurrent state in fp32 (config mamba_ssm_dtype; HF keeps both the same way).

What the shipped checkpoints store (read from their safetensors headers): Qwen/Qwen3.8-27B-FP8 keeps every
attention, DeltaNet and MLP projection in FP8 e4m3 with 128x128 block scales (stored BF16) and dynamic activations;
lm_head, embed_tokens, the norms, conv1d, A_log, dt_bias, in_proj_a and in_proj_b are BF16; no KV-cache scheme.
nvidia/Qwen3.8-27B-NVFP4 is mixed: attention and DeltaNet projections in per-tensor FP8 (static input scale), MLP
and lm_head in NVFP4 W4A4; the same small tensors BF16; no KV-cache scheme. Each projection (lm_head included) goes
through `loader.Checkpoint.linear` and `qlinear.qlinear`, so a stored format that those two handle needs no change
here.

Speed is not the point of this file: decode reads the whole KV cache under a mask, every projection goes through
the reference qlinear, and the delta rule is plain torch ops. Profile, then replace.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..loader import Checkpoint, QWeight
from ..qlinear import ACT_A16, ACT_AS_SHIPPED, qlinear
from . import register

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # the registry imports every model module; listing them must not need torch
    torch = F = None

KV_AS_SHIPPED, KV_BF16 = "as_shipped", "bf16"
LINEAR, FULL = "linear_attention", "full_attention"
CHUNK = 64          # prefill delta-rule chunk length (tokens)
Q_CHUNK = 512       # prefill queries per attention call: bounds the score matrix to heads x 512 x T
FP8_MAX = 448.0     # largest finite float8 e4m3fn value
_DENSE = ("bf16", "fp16", "fp32")


@dataclass(frozen=True)
class Qwen35Config:
    """The text-config fields the forward uses. Absent keys take Hugging Face's Qwen3_5TextConfig defaults."""
    hidden: int
    inter: int
    vocab: int
    eps: float
    layer_types: tuple[str, ...]
    heads: int
    kv_heads: int
    head_dim: int
    rot_dim: int            # RoPE dims per head: head_dim * partial_rotary_factor
    rope_theta: float
    attn_gate: bool         # q_proj also emits the per-head output gate
    attn_bias: bool
    k_heads: int            # DeltaNet q/k heads
    v_heads: int            # DeltaNet value heads (a multiple of k_heads)
    k_dim: int
    v_dim: int
    conv_kernel: int
    tie: bool

    @property
    def conv_dim(self) -> int:
        return 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim

    @classmethod
    def from_checkpoint(cls, ck: Checkpoint) -> "Qwen35Config":
        tc = ck.text_config
        act = tc.get("hidden_act", "silu")
        if act != "silu":
            raise NotImplementedError(f"hidden_act {act!r}: this module implements silu")
        gate_act = tc.get("output_gate_type") or "silu"         # the DeltaNet output-norm gate ("swish" = silu)
        if gate_act not in ("silu", "swish"):
            raise NotImplementedError(f"output_gate_type {gate_act!r}: this module implements silu")
        ssm = str(tc.get("mamba_ssm_dtype") or "float32")
        if ssm not in ("float32", "fp32"):
            raise NotImplementedError(f"mamba_ssm_dtype {ssm!r}: this module keeps the DeltaNet state in float32")
        rope = dict(tc.get("rope_parameters") or tc.get("rope_scaling") or {})
        rtype = rope.get("rope_type", rope.get("type", "default"))
        if rtype not in ("default", "mrope"):
            raise NotImplementedError(f"rope_type {rtype!r}: this module implements plain RoPE (text positions)")
        n = int(tc.get("num_hidden_layers", 32))
        every = int(tc.get("full_attention_interval", 4))
        types = tuple(tc.get("layer_types") or [FULL if (i + 1) % every == 0 else LINEAR for i in range(n)])
        if len(types) != n or set(types) - {LINEAR, FULL}:
            raise NotImplementedError(f"layer_types {sorted(set(types))} for {n} layers: expected {LINEAR}/{FULL}")
        head_dim = int(tc.get("head_dim", 256))
        k_heads, v_heads = int(tc.get("linear_num_key_heads", 16)), int(tc.get("linear_num_value_heads", 32))
        if v_heads % k_heads:
            raise NotImplementedError(f"{v_heads} DeltaNet value heads do not split over {k_heads} key heads")
        heads, kv_heads = int(tc.get("num_attention_heads", 16)), int(tc.get("num_key_value_heads", 4))
        if heads % kv_heads:
            raise NotImplementedError(f"{heads} attention heads do not split over {kv_heads} KV heads")
        prf = float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
        top = ck.config if "text_config" in ck.config else tc
        return cls(
            hidden=int(tc.get("hidden_size", 4096)), inter=int(tc.get("intermediate_size", 12288)),
            vocab=int(tc.get("vocab_size", 248320)), eps=float(tc.get("rms_norm_eps", 1e-6)), layer_types=types,
            heads=heads, kv_heads=kv_heads, head_dim=head_dim, rot_dim=int(head_dim * prf),
            rope_theta=float(rope.get("rope_theta", tc.get("rope_theta", 10000.0))),
            attn_gate=bool(tc.get("attn_output_gate", True)), attn_bias=bool(tc.get("attention_bias", False)),
            k_heads=k_heads, v_heads=v_heads, k_dim=int(tc.get("linear_key_head_dim", 128)),
            v_dim=int(tc.get("linear_value_head_dim", 128)), conv_kernel=int(tc.get("linear_conv_kernel_dim", 4)),
            tie=bool(top.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))))


@dataclass
class _Attention:
    q: QWeight
    k: QWeight
    v: QWeight
    o: QWeight
    q_norm: Any
    k_norm: Any
    slot: int               # index into the KV cache
    k_scale: float = 1.0    # FP8 KV cache only
    v_scale: float = 1.0


@dataclass
class _DeltaNet:
    qkv: QWeight
    z: QWeight
    b: QWeight
    a: QWeight
    out: QWeight
    conv_w: Any             # fp32 [conv_dim, 1, kernel] (widened once at load; exact)
    conv_b: Any             # fp32 [conv_dim] or None (the shipped checkpoints have none)
    neg_a: Any              # fp32 -exp(A_log) [v_heads]
    dt_bias: Any            # fp32 [v_heads]
    norm_w: Any             # [v_dim] as stored
    slot: int               # index into the conv and recurrent state


@dataclass
class _Layer:
    kind: str
    mixer: Any
    ln_in: Any
    ln_post: Any
    gate: QWeight
    up: QWeight
    down: QWeight


@dataclass
class Qwen35State:
    """One sequence. Shapes are fixed by alloc_state; decode only writes in place (CUDA-graph safe)."""
    k: Any                  # [attention layers, kv_heads, max_seq, head_dim]
    v: Any
    conv: Any               # [DeltaNet layers, conv_dim, kernel - 1]: the last kernel-1 conv inputs
    rec: Any                # [DeltaNet layers, v_heads, k_dim, v_dim] fp32
    positions: Any          # int64 [max_seq] = 0..max_seq-1, for decode's attention mask
    logits: Any             # fp32 [vocab]: what prefill and decode return
    max_seq: int


# --- building blocks (pure functions; the tests call them directly) -----------------------------------------
def rms_norm(x, w, eps):
    """Zero-centred RMSNorm (HF Qwen3_5RMSNorm): fp32 x / rms(x) * (1 + w), cast back to x's dtype."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


def gated_rms_norm(x, z, w, eps):
    """The DeltaNet output norm (HF Qwen3_5RMSNormGated): w * x / rms(x) in x's dtype, then * silu(z) in fp32."""
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return ((w.to(x.dtype) * y).float() * F.silu(z.float())).to(x.dtype)


def l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rope(x, cos, sin, rot: int):
    """Rotate-half RoPE on the first `rot` dims of each head, in fp32. x [T, heads, d]; cos, sin [T, 1, rot/2]."""
    half = rot // 2
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:rot]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, xf[..., rot:]), -1).to(x.dtype)


def delta_rule_step(q, k, v, g, beta, S):
    """One token of the gated delta rule, in place on S [H, dk, dv] (fp32). q, k [H, dk]; v [H, dv]; g, beta [H].
    Returns o [H, dv]."""
    S.mul_(torch.exp(g)[:, None, None])
    pred = torch.einsum("hkv,hk->hv", S, k)                     # what the decayed state already gives for k
    S.add_(k[:, :, None] * (beta[:, None] * (v - pred))[:, None, :])
    return torch.einsum("hkv,hk->hv", S, q)


def delta_rule_chunked(q, k, v, g, beta, S, chunk: int = CHUNK):
    """The gated delta rule over a sequence, `chunk` tokens at a time; equal to delta_rule_step token by token.
    q, k [T, H, dk]; v [T, H, dv]; g, beta [T, H]; S [H, dk, dv] the state before token 0. All fp32.
    Returns (o [T, H, dv], the state after the last token).

    In a chunk, let G_t = g_1 + ... + g_t (log of the decay since the chunk start) and S_0 the incoming state.
    The value written at step t, u_t = beta_t (v_t - k_t^T S_{t-1} exp(g_t)), satisfies
        u_t + sum_{j<t} beta_t exp(G_t - G_j) (k_t . k_j) u_j = beta_t (v_t - exp(G_t) S_0^T k_t),
    a unit lower-triangular system in u. Then
        o_t   = exp(G_t) S_0^T q_t + sum_{j<=t} exp(G_t - G_j) (q_t . k_j) u_j
        S_end = exp(G_C) S_0 + sum_j exp(G_C - G_j) k_j u_j^T
    Every exponent is <= 0 (g <= 0), so nothing overflows.
    """
    T = q.shape[0]
    q, k, v, g, beta = (x.transpose(0, 1) for x in (q, k, v, g, beta))      # heads first
    out = []
    for s in range(0, T, chunk):
        qc, kc, vc, bc = q[:, s:s + chunk], k[:, s:s + chunk], v[:, s:s + chunk], beta[:, s:s + chunk]
        G = g[:, s:s + chunk].cumsum(-1)                                    # [H, C]
        n = G.shape[-1]
        lower = torch.ones(n, n, dtype=torch.bool, device=G.device).tril()
        decay = torch.where(lower, G[:, :, None] - G[:, None, :], float("-inf")).exp()   # exp(G_t - G_j), j <= t
        A = (bc[:, :, None] * (kc @ kc.transpose(-1, -2)) * decay).tril(-1)
        rhs = bc[:, :, None] * (vc - G.exp()[:, :, None] * (kc @ S))
        u = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)   # diagonal taken as 1
        out.append(G.exp()[:, :, None] * (qc @ S) + ((qc @ kc.transpose(-1, -2)) * decay) @ u)
        tail = (G[:, -1:] - G).exp()[:, :, None]                            # exp(G_C - G_j)
        S = G[:, -1:].exp()[:, :, None] * S + (kc * tail).transpose(-1, -2) @ u
    return torch.cat(out, 1).transpose(0, 1), S


def attend(q, k, v, qpos, kpos, scale: float):
    """Causal attention: q [heads, Tq, d] at positions qpos [Tq] over k, v [kv_heads, Tk, d] at positions kpos [Tk].
    Query head h reads KV head h // (heads / kv_heads). The query heads of one KV head are stacked along the query
    axis ([kv_heads, group * Tq, d]), so K and V are never repeated; the inputs are 4-D so a GPU can run a fused
    kernel instead of materialising the scores. Returns [heads, Tq, d]."""
    h, tq, d = q.shape
    kvh = k.shape[0]
    group = h // kvh
    mask = kpos[None, :] <= qpos.repeat(group)[:, None]                     # [group * Tq, Tk]; row gi * Tq + t
    o = F.scaled_dot_product_attention(q.reshape(kvh, group * tq, d)[None], k[None], v[None], attn_mask=mask,
                                       scale=scale)
    return o[0].reshape(h, tq, d)


def act_scheme(w: QWeight, dt: str) -> tuple[str, bool]:
    """What qlinear does to the input of `w` as shipped, read from the stored tensors: (label for ENGINE.json,
    whether the activations are quantized). dt is the activation dtype's name."""
    if w.kind in _DENSE:
        return dt, False
    if w.kind == "fp8_block":
        return f"fp8 e4m3, dynamic per token in groups of {(w.block or (128, 128))[1]}", True
    if w.kind == "fp8_tensor":
        if w.input_scale is None:
            return "fp8 e4m3, dynamic per tensor", True
        return "fp8 e4m3, static per-tensor input_scale", True
    if w.kind == "nvfp4":
        if w.input_scale is None:                       # weight-only NVFP4: qlinear keeps the input 16-bit
            return f"{dt} (weight-only nvfp4)", False
        return "nvfp4 e2m1, static input_scale, groups of 16", True
    return f"as the checkpoint declares for {w.kind}", True


# --- the model ---------------------------------------------------------------------------------------------
@register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
class Qwen35Model:
    """engine_api.Model for the Qwen3.5 / Qwen3.8 hybrid family (see the module docstring)."""

    @classmethod
    def from_checkpoint(cls, ck: Checkpoint, device: str, max_seq: int, act: str = ACT_AS_SHIPPED,
                        kv: str = KV_AS_SHIPPED) -> "Qwen35Model":
        return cls(ck, device, max_seq, act, kv)

    def __init__(self, ck: Checkpoint, device: str, max_seq: int, act: str = ACT_AS_SHIPPED,
                 kv: str = KV_AS_SHIPPED):
        if torch is None:
            raise ImportError("the Qwen3.5 model module needs torch")
        if act not in (ACT_AS_SHIPPED, ACT_A16):
            raise ValueError(f"act {act!r}: expected {ACT_AS_SHIPPED!r} or {ACT_A16!r}")
        if kv not in (KV_AS_SHIPPED, KV_BF16):
            raise ValueError(f"kv {kv!r}: expected {KV_AS_SHIPPED!r} or {KV_BF16!r}")
        self.cfg = c = Qwen35Config.from_checkpoint(ck)
        self.device, self.max_seq, self.act, self.vocab_size = torch.device(device), int(max_seq), act, c.vocab
        self._kinds: Counter = Counter()
        self._schemes: set[tuple[str, str, bool]] = set()      # (kind, activation label, quantized), from _linear
        p = self._prefix(ck)

        def t(name, dtype=None):
            return ck.tensor(p + name, device=device, dtype=dtype)

        def lin(name, bias=False):
            return self._linear(ck, p + name, bias)

        self.embed = t("embed_tokens.weight")
        if not self.embed.is_floating_point():
            raise NotImplementedError(f"embed_tokens stored as {self.embed.dtype}: expected a 16/32-bit table")
        self.dtype = self.embed.dtype                         # activation dtype: BF16 for the shipped checkpoints
        self.norm = t("norm.weight")
        if c.tie:
            self.lm_head = QWeight(_kind(self.embed), self.embed, tuple(self.embed.shape))
        elif ck.has("lm_head.weight"):
            self.lm_head = self._linear(ck, "lm_head")
        else:
            raise KeyError("lm_head.weight is missing and tie_word_embeddings is false")

        declared_fp8_kv = (ck.quant.get("kv") or "") == "fp8"
        self.kv_fp8 = kv == KV_AS_SHIPPED and declared_fp8_kv
        self.kv_dtype = torch.float8_e4m3fn if self.kv_fp8 else torch.bfloat16 if kv == KV_BF16 else self.dtype
        scales_found = False
        self.layers: list[_Layer] = []
        n_attn = n_lin = 0
        for i, kind in enumerate(c.layer_types):
            L = f"layers.{i}."
            if kind == FULL:
                a = L + "self_attn."
                mixer = _Attention(lin(a + "q_proj", c.attn_bias), lin(a + "k_proj", c.attn_bias),
                                   lin(a + "v_proj", c.attn_bias), lin(a + "o_proj", c.attn_bias),
                                   t(a + "q_norm.weight"), t(a + "k_norm.weight"), slot=n_attn)
                if self.kv_fp8:
                    ks, vs = _kv_scales(ck, p + a)
                    scales_found |= ks is not None
                    mixer.k_scale, mixer.v_scale = (1.0, 1.0) if ks is None else (ks, vs)
                n_attn += 1
            else:
                a = L + "linear_attn."
                if ck.has(p + a + "in_proj_qkvz.weight"):
                    raise NotImplementedError("fused in_proj_qkvz (the Qwen3-Next layout) is not this family's")
                conv = t(a + "conv1d.weight", torch.float32)
                mixer = _DeltaNet(
                    lin(a + "in_proj_qkv"), lin(a + "in_proj_z"), lin(a + "in_proj_b"), lin(a + "in_proj_a"),
                    lin(a + "out_proj"), conv_w=conv.reshape(conv.shape[0], 1, -1),
                    conv_b=t(a + "conv1d.bias", torch.float32) if ck.has(p + a + "conv1d.bias") else None,
                    neg_a=-t(a + "A_log", torch.float32).exp(), dt_bias=t(a + "dt_bias", torch.float32),
                    norm_w=t(a + "norm.weight"), slot=n_lin)
                n_lin += 1
            self.layers.append(_Layer(kind, mixer, t(L + "input_layernorm.weight"),
                                      t(L + "post_attention_layernorm.weight"),
                                      lin(L + "mlp.gate_proj"), lin(L + "mlp.up_proj"), lin(L + "mlp.down_proj")))
        self.n_attn, self.n_lin = n_attn, n_lin
        self.scale = c.head_dim ** -0.5

        # RoPE tables, fp32 [max_seq, rot_dim / 2], computed as HF does (fp32 positions times fp32 inverse freqs).
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.rot_dim, 2, dtype=torch.float32) / c.rot_dim))
        ang = torch.arange(self.max_seq, dtype=torch.float32)[:, None] * inv[None, :]
        self.cos, self.sin = ang.cos().to(self.device), ang.sin().to(self.device)

        eos: set[int] = set()
        for src in (ck.generation_config, ck.text_config, ck.config):
            e = src.get("eos_token_id")
            eos.update([e] if isinstance(e, int) else (e or []))
        self.eos_token_ids = frozenset(int(x) for x in eos)
        self.numerics = self._describe(kv, declared_fp8_kv, scales_found)

    # --- loading helpers ----------------------------------------------------------------------------------
    @staticmethod
    def _prefix(ck: Checkpoint) -> str:
        """Where the text model's tensors live: model.language_model. (multimodal save) or model. (text-only)."""
        for p in ("model.language_model.", "model.", "language_model.model."):
            if ck.has(p + "embed_tokens.weight"):
                return p
        raise KeyError("no embed_tokens.weight under model.language_model., model. or language_model.model.")

    def _linear(self, ck: Checkpoint, name: str, bias: bool = False) -> QWeight:
        w = ck.linear(name, bias=bias)
        for f in ("weight", "scale", "scale_2", "input_scale", "bias"):     # the checkpoint may load elsewhere
            v = getattr(w, f)
            if v is not None and hasattr(v, "to"):
                setattr(w, f, v.to(self.device))
        self._kinds[w.kind] += 1
        self._schemes.add((w.kind, *act_scheme(w, _dtype_name(self.dtype))))
        return w

    def _describe(self, kv: str, declared_fp8_kv: bool, scales_found: bool) -> dict:
        """The numerics block for ENGINE.json: what this instance computes, and what differs from as shipped."""
        dt = _dtype_name(self.dtype)
        acts, kept_16bit = {}, []
        for k in sorted(self._kinds):
            schemes = sorted((label, quant) for kind, label, quant in self._schemes if kind == k)
            if self.act == ACT_A16 and any(quant for _, quant in schemes):
                acts[k] = f"{dt} (a16)"
                kept_16bit.append(k)
            else:                               # one label per kind, unless the stored scales differ by layer
                acts[k] = "; ".join(label for label, _ in schemes)
        if self.kv_fp8:
            kv_label = "fp8 e4m3" + (" with the stored k/v scales" if scales_found else " (no stored scales: 1.0)")
        else:
            kv_label = _dtype_name(self.kv_dtype)
        changed = []
        if kept_16bit:
            changed.append(f"activations kept 16-bit (a16) for the {', '.join(kept_16bit)} layers")
        if kv == KV_BF16 and (declared_fp8_kv or self.dtype != torch.bfloat16):
            changed.append(f"KV cache bf16; as shipped it is {'fp8 e4m3' if declared_fp8_kv else dt}")
        return {
            "weights": "as stored: " + ", ".join(f"{k} x{n}" for k, n in sorted(self._kinds.items())),
            "activations": acts,
            "kv": kv_label,
            "state": "fp32 (Gated DeltaNet recurrent state)",
            "conv_state": dt,
            "lm_head": self.lm_head.kind + (" (tied to embed_tokens)" if self.cfg.tie else ""),
            "not_loaded": "vision tower; mtp head (no speculative decoding)",
            "changed": changed,
        }

    # --- engine_api.Model ---------------------------------------------------------------------------------
    def alloc_state(self, max_seq: int) -> Qwen35State:
        c, dev, s = self.cfg, self.device, int(max_seq)
        if s > self.max_seq:
            raise ValueError(f"max_seq {s} > {self.max_seq} (the model's RoPE tables)")
        return Qwen35State(
            k=torch.zeros(self.n_attn, c.kv_heads, s, c.head_dim, dtype=self.kv_dtype, device=dev),
            v=torch.zeros(self.n_attn, c.kv_heads, s, c.head_dim, dtype=self.kv_dtype, device=dev),
            conv=torch.zeros(self.n_lin, c.conv_dim, c.conv_kernel - 1, dtype=self.dtype, device=dev),
            rec=torch.zeros(self.n_lin, c.v_heads, c.k_dim, c.v_dim, dtype=torch.float32, device=dev),
            positions=torch.arange(s, device=dev), logits=torch.zeros(c.vocab, dtype=torch.float32, device=dev),
            max_seq=s)

    def reset(self, state: Qwen35State) -> None:
        for buf in (state.k, state.v, state.conv, state.rec, state.logits):
            buf.zero_()

    def prefill(self, input_ids, state: Qwen35State):
        T = int(input_ids.shape[0])
        if not 0 < T <= state.max_seq:
            raise ValueError(f"prompt of {T} tokens; the state holds 1..{state.max_seq}")
        return self._forward(input_ids, self.cos[:T, None, :], self.sin[:T, None, :], state, None)

    def decode(self, token, pos, state: Qwen35State):
        p = pos.long()
        return self._forward(token, self.cos.index_select(0, p)[:, None, :], self.sin.index_select(0, p)[:, None, :],
                             state, p)

    # --- forward ------------------------------------------------------------------------------------------
    def _forward(self, ids, cos, sin, state: Qwen35State, pos):
        """pos None: prefill of ids [T] from position 0. Else one decode step of ids [1] at position pos [1]."""
        eps = self.cfg.eps
        x = F.embedding(ids, self.embed)
        for L in self.layers:
            h = rms_norm(x, L.ln_in, eps)
            if L.kind == FULL:
                x = x + self._attention(L.mixer, h, cos, sin, state, pos)
            else:
                x = x + self._deltanet(L.mixer, h, state, pos)
            h = rms_norm(x, L.ln_post, eps)
            x = x + qlinear(F.silu(qlinear(h, L.gate, self.act)) * qlinear(h, L.up, self.act), L.down, self.act)
        h = rms_norm(x[-1:], self.norm, eps)
        state.logits.copy_(qlinear(h, self.lm_head, self.act)[0])      # logits in the activation dtype, then fp32
        return state.logits

    def _attention(self, at: _Attention, h, cos, sin, state: Qwen35State, pos):
        c, T = self.cfg, h.shape[0]
        q = qlinear(h, at.q, self.act).view(T, c.heads, -1)
        if c.attn_gate:
            q, gate = q[..., :c.head_dim], q[..., c.head_dim:].reshape(T, -1)
        k = qlinear(h, at.k, self.act).view(T, c.kv_heads, c.head_dim)
        v = qlinear(h, at.v, self.act).view(T, c.kv_heads, c.head_dim).transpose(0, 1)
        q = rope(rms_norm(q, at.q_norm, c.eps), cos, sin, c.rot_dim).transpose(0, 1)        # [heads, T, d]
        k = rope(rms_norm(k, at.k_norm, c.eps), cos, sin, c.rot_dim).transpose(0, 1)        # [kv_heads, T, d]
        kc, vc = state.k[at.slot], state.v[at.slot]                                         # [kv_heads, S, d]
        if pos is None:
            kc[:, :T].copy_(self._kv_in(k, at.k_scale))
            vc[:, :T].copy_(self._kv_in(v, at.v_scale))
            K, V = self._kv_out(kc[:, :T], at.k_scale), self._kv_out(vc[:, :T], at.v_scale)  # read the cache, as decode
            qpos = state.positions[:T]
            # Q_CHUNK queries at a time against the keys up to the chunk's end: the scores stay heads x Q_CHUNK x T.
            o = torch.cat([attend(q[:, s:s + Q_CHUNK], K[:, :s + Q_CHUNK], V[:, :s + Q_CHUNK], qpos[s:s + Q_CHUNK],
                                  qpos[:s + Q_CHUNK], self.scale) for s in range(0, T, Q_CHUNK)], 1)
        else:
            _put(kc, pos, self._kv_in(k, at.k_scale))
            _put(vc, pos, self._kv_in(v, at.v_scale))
            o = attend(q, self._kv_out(kc, at.k_scale), self._kv_out(vc, at.v_scale), pos, state.positions, self.scale)
        o = o.transpose(0, 1).reshape(T, -1)
        if c.attn_gate:
            o = o * torch.sigmoid(gate)
        return qlinear(o, at.o, self.act)

    def _kv_in(self, x, scale: float):
        if self.kv_fp8:
            return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return x.to(self.kv_dtype)

    def _kv_out(self, x, scale: float):
        if self.kv_fp8:
            return (x.float() * scale).to(self.dtype)
        return x.to(self.dtype)

    def _deltanet(self, d: _DeltaNet, h, state: Qwen35State, pos):
        c, T = self.cfg, h.shape[0]
        mixed = qlinear(h, d.qkv, self.act)                                     # [T, conv_dim]
        z = qlinear(h, d.z, self.act).view(T, c.v_heads, c.v_dim)
        beta = torch.sigmoid(qlinear(h, d.b, self.act)).float()                 # sigmoid in the activation dtype, as HF
        g = d.neg_a * F.softplus(qlinear(h, d.a, self.act).float() + d.dt_bias)   # [T, v_heads], <= 0

        # Causal depthwise conv over [the last kernel-1 inputs, these T inputs], in fp32. Prefill starts from zeros.
        conv = state.conv[d.slot]                                               # [conv_dim, kernel - 1]
        seq = F.pad(mixed.t(), (c.conv_kernel - 1, 0)) if pos is None else torch.cat([conv, mixed.t()], 1)
        y = F.conv1d(seq.float()[None], d.conv_w, d.conv_b, groups=c.conv_dim)[0]   # [conv_dim, T]
        conv.copy_(seq[:, T:])
        mixed = F.silu(y).t().to(self.dtype).float()                            # rounded as HF rounds it

        kd = c.k_heads * c.k_dim
        q = l2norm(mixed[:, :kd].reshape(T, c.k_heads, c.k_dim)) * c.k_dim ** -0.5
        k = l2norm(mixed[:, kd:2 * kd].reshape(T, c.k_heads, c.k_dim))
        v = mixed[:, 2 * kd:].reshape(T, c.v_heads, c.v_dim)
        r = c.v_heads // c.k_heads
        if r > 1:                                       # key head i serves value heads i*r .. i*r + r-1
            q, k = q.repeat_interleave(r, 1), k.repeat_interleave(r, 1)
        S = state.rec[d.slot]                                                   # [v_heads, k_dim, v_dim] fp32
        if pos is None:
            o, S_end = delta_rule_chunked(q, k, v, g, beta, torch.zeros_like(S))
            S.copy_(S_end)
        else:
            o = delta_rule_step(q[0], k[0], v[0], g[0], beta[0], S)[None]
        y = gated_rms_norm(o.to(self.dtype), z, d.norm_w, c.eps).reshape(T, -1)
        return qlinear(y, d.out, self.act)


def _put(cache, pos, x) -> None:
    """cache[:, pos] = x, in place. FP8 caches are written as bytes: index_copy_ has no float8 kernel."""
    if cache.element_size() == 1:
        cache, x = cache.view(torch.uint8), x.view(torch.uint8)
    cache.index_copy_(1, pos, x)


def _kind(t) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[t.dtype]


def _dtype_name(dtype) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32",
            torch.float8_e4m3fn: "fp8 e4m3"}.get(dtype, str(dtype).replace("torch.", ""))


def _kv_scales(ck: Checkpoint, attn: str) -> tuple[float | None, float | None]:
    """The per-tensor FP8 (k_scale, v_scale) of one attention module as modelopt and vLLM name them, or (None, None)
    when the checkpoint stores none (1.0). As vLLM loads them: when only one of k and v is stored, or only the older
    shared kv_scale, that one serves both."""
    def find(*names: str) -> float | None:
        for name in names:
            if ck.has(name):
                s = ck.tensor(name, device="cpu", dtype=torch.float32).reshape(-1)
                if s.numel() != 1:
                    raise NotImplementedError(f"{name}: {s.numel()} KV scales; this module reads one per tensor")
                return float(s[0])
        return None
    k = find(f"{attn}k_scale", f"{attn}k_proj.k_scale", f"{attn}attn.k_scale")
    v = find(f"{attn}v_scale", f"{attn}v_proj.v_scale", f"{attn}attn.v_scale")
    if k is None and v is None:
        k = v = find(f"{attn}kv_scale", f"{attn}k_proj.kv_scale", f"{attn}attn.kv_scale")
    return (v if k is None else k), (k if v is None else v)
