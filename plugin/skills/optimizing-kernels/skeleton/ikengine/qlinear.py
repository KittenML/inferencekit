"""Reference linear layers for every stored format: correct first, not fast.

`qlinear(x, w, act)` computes y = x @ W^T (+ bias) with the numerics the checkpoint declares, accumulating in
fp32 and rounding once to x's dtype at the end:
- "bf16"/"fp16"/"fp32": the stored dense weight.
- "fp8_block": per-token, per-group dynamic e4m3 activation quantization (group = block[1], scale = amax/448),
  then the block-scaled product: each K block's fp32 partial sum is multiplied by its activation scale and its
  weight scale before the blocks are added (as vLLM's and SGLang's block-FP8 GEMMs do). Weight x scale is
  never pre-rounded into a 16-bit weight.
- "fp8_tensor": e4m3 activations with the static input_scale (or one dynamic scale per tensor), fp32 product,
  times input_scale x weight_scale.
- "nvfp4": W4A4 as shipped: activations quantized to e2m1 with per-16 e4m3 block scales under the static
  global `input_scale`; weights decoded from e2m1 with their per-16 e4m3 scales; the global scales
  (input_scale x weight_scale_2) multiply the fp32 result, as the GEMM epilogue's alpha does.

act="as_shipped" (default) applies the declared activation scheme. act="a16" keeps activations in
16-bit (x times the fp32 dequantized weight): a numerics change the model must declare in its `numerics`.

The NVFP4 activation recipe follows vLLM's reference `ref_nvfp4_quant` (vllm/model_executor/layers/quantization/
utils/nvfp4_emulation_utils.py) and its CUDA kernel `scaled_fp4_quant` (csrc/quantization/fp4/nvfp4_quant_kernels.cu),
called with global_scale = 1 / input_scale (ModelOptNvFp4LinearMethod: `input_scale_inv`):
    s_block = e4m3(global_scale * amax_block / 6)          (clamped to +-448, rounded to nearest even)
    x_q     = e2m1(x / (s_block / global_scale))            (clamped to +-6, rounded to nearest even)
so x ~= x_q * s_block * input_scale. The CUDA kernel uses an approximate reciprocal, so a GPU kernel can differ
from this reference in the last bit of a scaled value; ties and saturation are the same.
The input_scale is per stored layer here. An engine that fuses layers (q|k|v, gate|up, or a stack of experts)
quantizes the shared input once, with one scale per fused group: vLLM takes the max of the stored ones. A model
module that fuses does the same and says so in its `numerics`.
The block-FP8 activation recipe follows vLLM's `per_token_group_quant_fp8` (scale = max(amax, 1e-10) / 448,
x_q = e4m3(clamp(x / scale, -448, 448))).
"""
from __future__ import annotations

from .loader import DENSE_KINDS, QWeight

ACT_AS_SHIPPED, ACT_A16 = "as_shipped", "a16"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # code 0..7; bit 3 is the sign
E2M1_MAX, E4M3_MAX = 6.0, 448.0
FP8_GROUP_EPS = 1e-10
_E2M1_TIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)     # midpoints between neighbouring e2m1 magnitudes
_CHUNK_ELEMS = 1 << 26                                     # fp32 temporaries per block-FP8 chunk (256 MB)


# --- e2m1 (FP4) and e4m3 (FP8) element formats ------------------------------------------------------------
_CONSTS: dict = {}


def _const(values: tuple, device):
    """A small fp32 constant tensor, made once per device. Building it from a Python list on every call would
    be a host-to-device copy, which CUDA-graph capture rejects; the warm-up before a capture makes it here."""
    import torch
    key = (values, str(device))
    t = _CONSTS.get(key)
    if t is None:
        t = _CONSTS[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return t


def e2m1_unpack(packed):
    """uint8 [..., K/2] -> 4-bit codes uint8 [..., K] (low nibble is the even element)."""
    import torch
    lo, hi = packed & 0x0F, packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def e2m1_pack(codes):
    """4-bit codes uint8 [..., K] (K even) -> uint8 [..., K/2], low nibble first."""
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return c[..., 0] | (c[..., 1] << 4)


def e2m1_decode(codes):
    """4-bit codes -> fp32 values (sign bit 3, magnitude from E2M1_VALUES)."""
    import torch
    mag = _const(E2M1_VALUES, codes.device)[(codes & 0x07).long()]
    return torch.where((codes & 0x08) != 0, -mag, mag)


def e2m1_encode(x):
    """fp32 -> 4-bit codes: round to nearest, ties to even, saturating at +-6 (as cvt.rn.satfinite.e2m1x2).

    A tie between two magnitudes goes to the one with an even code (0.25->0, 0.75->1, 1.25->1, 1.75->2,
    2.5->2, 3.5->4, 5->4), which is round-half-to-even on the e2m1 mantissa.
    """
    import torch
    a = x.float().abs().clamp(max=E2M1_MAX)
    ties = _const(_E2M1_TIES, x.device)
    lo = torch.bucketize(a, ties, right=False)           # a == tie -> the lower code
    hi = torch.bucketize(a, ties, right=True)            # a == tie -> the upper code
    code = torch.where((lo != hi) & (hi % 2 == 0), hi, lo).to(torch.uint8)
    return code | ((x < 0).to(torch.uint8) << 3)


def e2m1_round(x):
    """fp32 values rounded onto the e2m1 grid (RNE, saturating); the value form of e2m1_encode."""
    return e2m1_decode(e2m1_encode(x))


def e4m3_round(x):
    """fp32 -> nearest float8_e4m3fn value (RNE) after clamping to +-448, returned as fp32."""
    import torch
    return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def _recip(t):
    """1/t with 1/0 = 0 (vLLM's get_reciprocal)."""
    import torch
    return torch.where(t == 0, torch.zeros_like(t), 1.0 / t)


# --- activation quantizers (each returns fp32 values that are exactly representable in the target format) --
def quant_act_nvfp4(x, input_scale, block: int = 16):
    """NVFP4 activation quantization with a static global scale (see the module docstring).

    x [..., K] (K % 16 == 0); input_scale: fp32 scalar or a tensor broadcastable to x[..., :1].
    Returns (x_q [..., K] fp32 on the e2m1 grid, s [..., K/16] fp32 e4m3 block scales).
    The value x_q * s (per block) * input_scale approximates x.
    """
    import torch
    *lead, k = x.shape
    if k % block:
        raise ValueError(f"NVFP4 activations need K % {block} == 0, got K={k}")
    xf = x.float().reshape(*lead, k // block, block)
    gs = _recip(torch.as_tensor(input_scale, dtype=torch.float32, device=x.device))   # vLLM: input_scale_inv
    gs = gs.reshape(*gs.shape, 1) if gs.dim() else gs
    amax = xf.abs().amax(dim=-1, keepdim=True)
    s = e4m3_round(gs * (amax * (1.0 / E2M1_MAX)))
    out_scale = _recip(s * _recip(gs))
    xq = e2m1_round((xf * out_scale).clamp(-E2M1_MAX, E2M1_MAX))
    return xq.reshape(*lead, k), s.squeeze(-1)


def quant_act_fp8_group(x, group: int = 128):
    """Dynamic per-token, per-group e4m3 quantization. x [..., K] -> (x_q [..., K] fp32 e4m3 values,
    s [..., ceil(K/group)] fp32). A last partial group is quantized on its own elements."""
    import torch
    *lead, k = x.shape
    kt = -(-k // group)
    xf = x.float()
    if kt * group != k:
        xf = torch.nn.functional.pad(xf, (0, kt * group - k))
    xf = xf.reshape(*lead, kt, group)
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=FP8_GROUP_EPS) / E4M3_MAX
    xq = e4m3_round(xf / s)
    return xq.reshape(*lead, kt * group)[..., :k], s.squeeze(-1)


def quant_act_fp8_tensor(x, input_scale=None):
    """Per-tensor e4m3 quantization: static `input_scale` (a scalar, or one per row as [rows, 1]), or one
    dynamic amax/448 scale for the whole tensor when it is None. Returns (x_q fp32 e4m3 values, scale fp32)."""
    import torch
    xf = x.float()
    if input_scale is None:
        s = (xf.abs().amax().clamp(min=FP8_GROUP_EPS) / E4M3_MAX).reshape(())
    else:
        s = torch.as_tensor(input_scale, dtype=torch.float32, device=x.device)
    return e4m3_round(xf / s), s


# --- KV cache (FP8 e4m3 with a per-tensor scale, as vLLM's reshape_and_cache: x / scale, saturating) ---------
def kv_quant(x, scale=None):
    """K or V values -> float8_e4m3fn for an FP8 KV cache. scale None means 1.0 (vLLM's default)."""
    import torch
    xf = x.float() if scale is None else x.float() / scale.float()
    return xf.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def kv_dequant(q, scale=None, dtype=None):
    """FP8 KV-cache values -> fp32 (or `dtype`)."""
    y = q.float() if scale is None else q.float() * scale.float()
    return y if dtype is None else y.to(dtype)


# --- weights ------------------------------------------------------------------------------------------------
def _expand_rows(s, n: int, bn: int):
    """Block scales [..., ceil(N/bn), kt] -> per-row scales [..., N, kt]."""
    return s.repeat_interleave(bn, dim=-2)[..., :n, :]


def dequant(w: QWeight, dtype=None):
    """The stored weight decoded to a dense tensor (fp32 unless `dtype`). Dense kinds return as is.

    fp8_block: e4m3 x weight_scale_inv of its (bn, bk) block. fp8_tensor: e4m3 x weight_scale.
    nvfp4: e2m1 x e4m3 block scale (per 16 along K) x weight_scale_2. All products are formed in fp32.
    """
    import torch
    if w.kind in DENSE_KINDS:
        return w.weight if dtype is None else w.weight.to(dtype)
    n, k = w.shape
    if w.kind == "fp8_block":
        bn, bk = w.block
        s = _expand_rows(w.scale.float(), n, bn).repeat_interleave(bk, dim=-1)[..., :k]
        out = w.weight.float() * s
    elif w.kind == "fp8_tensor":                    # one scale per tensor, or one per output row
        out = w.weight.float() * w.scale.float().reshape(*w.weight.shape[:-2], -1, 1)
    elif w.kind == "nvfp4":
        vals = e2m1_decode(e2m1_unpack(w.weight))                                   # [..., N, K]
        blk = vals.reshape(*vals.shape[:-1], k // 16, 16) * w.scale.float().unsqueeze(-1)
        out = blk.reshape(vals.shape) * w.scale_2.float().reshape(*w.weight.shape[:-2], 1, 1)
    else:
        raise NotImplementedError(f"dequant for {w.kind}")
    return out if dtype is None else out.to(dtype)


def _mm(a, b):
    """a [..., M, K] (or [B, K] with b [B, N, K]) times b^T in fp32. A 3-D `b` is a per-row stack (moe.py)."""
    import torch
    if b.dim() == 2:
        return a @ b.t()
    return torch.matmul(a.unsqueeze(-2), b.transpose(-1, -2)).squeeze(-2)


def _scalar_rows(t, rows: int | None):
    """A global scale: one value (-> 0-d), or one per row of a stacked weight (-> [rows, 1])."""
    return t.float().reshape(rows, 1) if rows is not None else t.float().reshape(())


def _fp8_block_mm(x2, w: QWeight):
    """Block-FP8 W8A8: dynamic per-token, per-K-group activation scales; fp32 partial sums per K block."""
    import torch
    n, k = w.shape
    bn, bk = w.block
    kt = -(-k // bk)
    xq, xs = quant_act_fp8_group(x2, bk)                        # [M, K], [M, kt]
    ws = _expand_rows(w.scale.float(), n, bn)                   # [N, kt] or [B, N, kt]
    wf = w.weight.float()
    if kt * bk != k:
        xq = torch.nn.functional.pad(xq, (0, kt * bk - k))
        wf = torch.nn.functional.pad(wf, (0, kt * bk - k))
    m = xq.shape[0]
    if wf.dim() == 3:                                           # one weight per row (gathered experts)
        part = torch.einsum("btk,bntk->btn", xq.reshape(m, kt, bk), wf.reshape(m, n, kt, bk))
        return (part * xs.unsqueeze(-1) * ws.transpose(1, 2)).sum(dim=1)
    wt = wf.reshape(n, kt, bk).permute(1, 2, 0)                 # [kt, bk, N]
    out = []
    step = max(1, _CHUNK_ELEMS // (kt * n))
    for i in range(0, m, step):                                 # bounded fp32 temporaries for long prompts
        xc = xq[i:i + step].reshape(-1, kt, bk).permute(1, 0, 2)                        # [kt, m, bk]
        part = torch.bmm(xc, wt)                                                       # [kt, m, N]
        out.append((part * xs[i:i + step].t().unsqueeze(-1) * ws.t().unsqueeze(1)).sum(dim=0))
    return torch.cat(out, dim=0)


def qlinear(x, w: QWeight, act: str = ACT_AS_SHIPPED):
    """y = x @ W^T (+ bias); x is [..., K] (bf16, fp16 or fp32). Returns x's dtype.

    A weight whose tensors carry a leading stack dimension (moe.py's gathered experts) takes x [B, K] and
    applies row b's own weight and scales to row b.
    """
    import torch
    if act not in (ACT_AS_SHIPPED, ACT_A16):
        raise ValueError(f"act must be {ACT_AS_SHIPPED!r} or {ACT_A16!r}, got {act!r}")
    if w.kind in DENSE_KINDS and w.weight.dim() == 2:
        y = torch.nn.functional.linear(x.to(w.weight.dtype), w.weight, w.bias)
        return y.to(x.dtype)
    n, k = w.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    stacked = w.weight.dim() == 3
    rows = x2.shape[0] if stacked else None
    if w.kind in DENSE_KINDS:
        y = _mm(x2.float(), w.weight.float())
    elif act == ACT_A16:
        y = _mm(x2.float(), dequant(w))
    elif w.kind == "fp8_block":
        y = _fp8_block_mm(x2, w)
    elif w.kind == "fp8_tensor":
        if stacked and w.input_scale is None:
            raise NotImplementedError("stacked fp8_tensor weights need static input scales")
        in_s = None if w.input_scale is None else _scalar_rows(w.input_scale, rows)
        xq, s_in = quant_act_fp8_tensor(x2, in_s)
        ws = w.scale.float().reshape(rows if stacked else 1, -1)       # [.., 1] per tensor or [.., N] per row
        y = _mm(xq, w.weight.float()) * ws * s_in
    elif w.kind == "nvfp4":
        if w.input_scale is None:                               # weight-only NVFP4 as shipped (W4A16)
            y = _mm(x2.float(), dequant(w))
        else:
            in_s = _scalar_rows(w.input_scale, rows)
            xq, xs = quant_act_nvfp4(x2, in_s)                  # e2m1 values, e4m3 block scales
            xd = (xq.reshape(-1, k // 16, 16) * xs.unsqueeze(-1)).reshape(-1, k)        # exact in fp32
            wv = e2m1_decode(e2m1_unpack(w.weight))
            wd = (wv.reshape(*wv.shape[:-1], k // 16, 16) * w.scale.float().unsqueeze(-1)).reshape(wv.shape)
            y = _mm(xd, wd) * (in_s * _scalar_rows(w.scale_2, rows))                   # alpha, as the epilogue
    else:
        raise NotImplementedError(f"qlinear for {w.kind}")
    if w.bias is not None:
        y = y + w.bias.float()
    return y.reshape(*lead, n).to(x.dtype)
