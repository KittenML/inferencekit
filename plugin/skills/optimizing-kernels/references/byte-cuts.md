# Exact byte cuts: lossless 16-bit re-encoding and a bound-pruned output layer

At batch 1 the ceiling is measured bandwidth ÷ bytes per token. An exact byte cut reads fewer bytes and leaves the model unchanged (same greedy tokens, same sampling distribution): it stays lossless (the "Exact byte cut" row of [proving-correctness](../../proving-correctness/SKILL.md)) and moves the ceiling, so start early, next to kernel work. Size it first: the byte-floor estimator ([bytes_per_token.py](../../optimizing-inference/scripts/bytes_per_token.py), keys `packable_16bit_bytes`, `lm_head_share`, `floor_bytes_per_token`) says what each recipe can save.

Terms: **BF16** has 1 sign bit, 8 exponent bits and 7 mantissa bits. A **binade** is the range of one exponent value, [2^e, 2^(e+1)). Exponent field 0 holds zero and **subnormals**; 255 holds Inf and NaN.

## A. Lossless re-encoding of 16-bit weights

Within a small group of one weight row, the exponents sit within a few binades of the group's largest, so the 8-bit exponent field carries only 3-4 bits of information. Store each exponent as a small delta.

**Layout** (per row, groups of `G` along K, deltas of `d` bits):
- per group, one byte: the base `E`, the largest exponent among the group's normal elements (finite, non-zero, not subnormal);
- per element, one byte: `sign << 7 | mantissa`;
- per element, `d` bits: `E - exponent`, from 0 to `2^d - 2`; the code `2^d - 1` is the **escape**;
- escapes: zeros, subnormals, Inf, NaN and anything more than `2^d - 2` binades below `E`. Each escape's raw 16-bit value goes in the row's exception list, in column order.

Bits per weight = 8 + d + 8/G, plus 16 per escape. With d = 4 and G = 64 that is 12.125 bits. On synthetic Gaussian rows (observed in this plugin's tests) d = 4 escapes about 1 element in 10,000; d = 3 escapes about 2 in 100, which still totals fewer bits but puts escapes in nearly every warp stripe, so the slow path runs everywhere. Real rows have heavier tails: measure your checkpoint, and choose d and G by total bytes and decode cost together.

**Never store an exception as an fp32 correction** added after the multiply (`w_decoded * x + delta * x`): it rounds differently from `w * x`, so it is not exact. Put the raw 16-bit value into the register before the multiply.

**Decode inside the GEMV.** Load the sign/mantissa bytes and the deltas with 16-byte loads. Rebuild each BF16 with integer operations (`exp = E - delta`, then `bits = (sm & 0x80) << 8 | exp << 7 | (sm & 0x7f)`), then multiply exactly as the unpacked kernel does. Once the scalar version is proven, byte-wise SIMD (`__vsub4`) and `__byte_perm` handle 4 elements per instruction. Escapes are rare, so test a whole warp stripe with `__any_sync` first; only then find each escape's slot in the row's list with a warp prefix sum of per-thread counts, plus the count already consumed in the row (a running counter, or a stored offset per row segment when several warps split a row). Keep the unpacked kernel's accumulation order so the logits stay byte-equal.

Reference packer and decoder (torch, run at load time). The kernel layout packs the deltas into d bits and interleaves them for 16-byte loads; same values.

```python
import torch

def pack_bf16(w, G=64, d=4):
    """w: bfloat16 [N, K], K % G == 0. Returns base [N, K/G], sm [N, K], delta [N, K] (uint8, one value
    per byte here) and exc: the raw int16 values of the escapes, in row-major order."""
    esc_code = (1 << d) - 1
    b = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF       # raw bit patterns 0..65535
    exp = (b >> 7) & 0xFF
    normal = (exp > 0) & (exp < 255)                                     # not zero, subnormal, Inf or NaN
    base = torch.where(normal, exp, 0).view(b.shape[0], -1, G).amax(-1)   # largest normal exponent per group
    delta = base.repeat_interleave(G, -1) - exp
    esc = ~normal | (delta >= esc_code)                                  # outside the window: raw 16-bit
    delta = torch.where(esc, esc_code, delta)
    sm = ((b >> 8) & 0x80) | (b & 0x7F)                                  # sign << 7 | 7-bit mantissa
    exc = w.contiguous().view(torch.int16).flatten()[esc.flatten()]
    return base.to(torch.uint8), sm.to(torch.uint8), delta.to(torch.uint8), exc

def unpack_bf16(base, sm, delta, exc, d=4):
    esc_code = (1 << d) - 1
    s, dl = sm.to(torch.int32), delta.to(torch.int32)
    exp = base.to(torch.int32).repeat_interleave(s.shape[1] // base.shape[1], -1) - dl
    b = ((s & 0x80) << 8) | ((exp & 0xFF) << 7) | (s & 0x7F)
    b[dl == esc_code] = exc.to(torch.int32) & 0xFFFF                     # the exact stored value
    return (b - ((b >> 15) << 16)).to(torch.int16).view(torch.bfloat16)
```

**Proof recipe (every step must pass):**
1. Round trip on every packed tensor: `unpack_bf16(*pack_bf16(w))` equals `w` as 16-bit patterns. Compare `.view(torch.int16)`, not values, because a value comparison hides -0 vs +0 and NaN payloads. Run it also on a synthetic tensor holding all 65,536 BF16 bit patterns (both zeros, subnormals, Inf, NaN) and on groups made only of escapes.
2. Kernel: the packed GEMV is byte-equal to the unpacked GEMV over a shape sweep, outputs pre-filled with NaN.
3. End to end: logits byte-equal to the unpacked build on fixed prompts, in eager mode and in graph replay; greedy ids identical.
4. Recompute the ceiling for the new bytes per token (packed bytes, group bytes, exceptions, unchanged tensors) and report % of the new ceiling. Above 100% of the old ceiling is expected.
5. At the next milestone: the "Exact byte cut" row of proving-correctness (`rare_tokens` PASS, then equivalence).

## B. Exact bound-pruned output layer

The output layer (V rows × H) is often the largest single read of a decode step. Pass 1 streams a cheap proxy (int8 or FP8 with per-group scales, about half the bytes of BF16) and bounds every row's logit. Pass 2 rescores only the rows that can still be chosen, with the full head's own arithmetic.

**Bounds.** At pack time, per row v and group g, store the proxy q and its scale s. Also compute, exactly in float64 and rounded up, `e[v,g] = max |w - q*s|` and `a[v,g] = max |w|`. Per token, compute `n1[g] = ||x_g||_1` once for all rows. Then each proxy logit `p[v]` satisfies:
- `|z[v] - p[v]| <= B[v] = sum_g e[v,g]*n1[g] + 2*gamma*sum_g (a[v,g] + e[v,g])*n1[g]`, so `L = p - B` and `U = p + B`;
- `gamma = n*2^-24 / (1 - n*2^-24)` bounds the rounding of an fp32 sum of n terms (here n = H + H/G + 1), for both passes. It is loose but rigorous. A pass on tensor cores or with lower-precision accumulation needs that path's own error bound.
- On the GPU, compute n1 and B rounding up (`__fmaf_ru`, `__fadd_ru`) and L, U rounding outward; the reference below uses float64 instead.

**Greedy.** Let f be everything the full path does after the GEMV: scaling, a soft cap `c*tanh(z/c)`, rounding the logits to the output dtype. f must be monotone non-decreasing, also as implemented in floating point (check it with a sweep). Round L and U outward to the dtype f runs in and let `w* = max f(L)`. The candidates are `C = {v : f(U[v]) >= w*}`. Rescore C with the full head's per-row arithmetic (same kernel and reduction order, so the values are bit-identical), apply f, and take the argmax, ties to the lowest token id. Why it is exact: the winner and every row tied with it have `f(U) >= w*`, and every pruned row is strictly below some candidate. f can turn distinct logits into ties (rounding, saturation), so decide after f, never before.

**Sampling** (temperature, then top-k, then top-p, the order vLLM and Hugging Face use):
- top-k set: the candidates are the rows with f(U) at least the k-th largest f(L). They contain the exact top k, including every row tied at the k-th value (make the tie rule lowest id first in both paths). Rescore them, take the top k and run the unchanged sampler on them: the same distribution.
- min-p: a row with f(z) below max f(z) − T·ln(1/min_p) can never be drawn; keep the rows with f(U) at least max f(L) − T·ln(1/min_p).
- top-p without top-k: membership depends on the normaliser over all rows. Keep the rows with f(U) at least max f(L) − T·ln(V/(1 − top_p)); all the others together hold less than 1 − top_p of the mass, so none of them is in the nucleus. Bound the pruned rows' mass; if the nucleus boundary differs between the two ends of that bound, use the full head.
- Temperature only (no top-k, top-p = 1): any row can be drawn, so always use the full head.
- The same token for the same seed (not only the same distribution) needs random numbers keyed by row id (a counter-based generator), identical in both paths.

**Fill pruned rows with their f(U) or f(L), never -inf:** vLLM's Triton top-k/top-p kernel (seeded requests) guesses its cut-off from the finite values in the first 8,192-token tile, and -inf rows make it re-sweep the whole vocabulary (measured: 295 vs 91 us per step; 1.056x greedy, 1.019x sampled). Still exact: a filled row is below the k-th largest kept logit, so top-k and top-p drop it either way and it never ranks in a top 20 (k >= 20); logprob requests still take the full head (the fill moves the normaliser).

**Always the full head** when logprobs are requested (the log-softmax needs every row), when the candidates overflow their fixed buffer (CUDA graphs need static sizes), when the bounds cannot decide, when the proxy pass sees NaN or Inf, and when the sampling settings are outside what the pruned path implements.

Reference bounds and candidate selection (torch, CPU). Use them as the oracle in your kernel tests.

```python
import torch

def head_proxy(W, G=64):
    """W: [V, H] 16-bit head -> int8 q, fp32 scales s, and e, a per (row, group of G), rounded up."""
    V, H = W.shape
    w = W.double().view(V, H // G, G)
    a = w.abs().amax(-1)
    s = (a / 127).float().clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round(w / s.double()[..., None]).clamp(-127, 127)
    e = (w - q * s.double()[..., None]).abs().amax(-1)
    up = lambda t: torch.nextafter(t.float(), torch.tensor(float("inf")))
    return q.to(torch.int8), s, up(e), up(a)

def head_bounds(x, q, s, e, a, G=64):
    """float64 L <= z <= U for each row's fp32 logit z = sum_k w[v,k] * x[k], in any fp32 summation order."""
    H = x.numel()
    xg = x.double().view(-1, G)
    n1 = xg.abs().sum(-1)                                           # ||x_g||_1, once per token
    p = ((q.double() * xg).sum(-1) * s.double()).sum(-1)            # proxy logits (fp32 on the GPU: covered)
    n = H + H // G + 1
    gamma = n * 2.0 ** -24 / (1 - n * 2.0 ** -24)
    B = (e.double() * n1).sum(-1) + 2 * gamma * ((a.double() + e.double()) * n1).sum(-1)
    return p - B, p + B

def candidates(L, U, f=lambda z: z, k=1):
    """Rows that can still be in the top k after the monotone fp32 transform f (k=1: greedy)."""
    inf = torch.tensor(float("inf"))
    l, u = L.float(), U.float()
    l = torch.where(l.double() > L, torch.nextafter(l, -inf), l)    # round outward to fp32
    u = torch.where(u.double() < U, torch.nextafter(u, inf), u)
    fl, fu = f(l), f(u)
    return (fu >= fl.topk(k).values[-1]).nonzero().squeeze(1)      # ascending row ids

def pruned_argmax(rows, z_rows, f=lambda z: z):
    """z_rows: the candidates' fp32 logits from the full head's own arithmetic. Ties go to the lowest id."""
    v = f(z_rows)
    return int(rows[(v == v.max()).nonzero()[0, 0]])
```

**Proof recipe (every step must pass):**
1. A debug switch that runs both heads every step: assert that the pruned token equals the full token, greedy, over many prompts and thousands of steps (the whole dev set at full length, plus long and code prompts). Log the candidate count and the fallback rate (median, p99): each rescored row costs H × 2 bytes.
2. Constructed cases: duplicate rows (exact ties), near-ties inside the bound, all-equal logits, logits large enough that the soft cap or the output rounding merges them, and NaN in the hidden state (it must fall back).
3. Sampling: at every step the full sampler's support (the rows with non-zero probability after top-k/top-p) is inside C, and with top-k the pruned path hands the sampler the same (id, logit) pairs in the same order as the full path, so its probabilities are bit-identical. At the next milestone, when sampling is timed: `inferencekit quality sampling` on the pruned build with `--baseline-url` the full-head build: both PASS. The reported TV between their implied distributions is a readout, not a pass rule; equal probabilities are already required by step 4. It samples only the first token, so it counts only when the first token also goes through the pruned head.
4. A logprob request returns the same logprobs as the full-head build; `rare_tokens` PASS at the next milestone.
5. Recompute the ceiling with the proxy, its scales and the rescored rows in place of the full head.
