# FP8 block-scaled GEMV at M = 1

## Terms

- **GEMM / GEMV:** matrix x matrix / matrix x vector. Decode at batch 1 multiplies an `[1, K]` activation by a `[K, N]` weight: M = 1, a GEMV.
- **FP8 E4M3:** 8-bit float with 4 exponent and 3 mantissa bits (max magnitude 448).
- **Block scaling (128x128):** one scale per 128x128 block of the weight matrix (stored as BF16 in `Qwen/Qwen3-8B-FP8`, FP32 in some other checkpoints: check the safetensors header); the real weight is `fp8_value x scale[n / 128][k / 128]`. Fine-grained scales keep the error of each block small.
- **Split-K:** split the reduction dimension K across several blocks; each produces a partial sum; a final step adds the partials.
- **Bytes in flight:** loads issued but not yet returned. DRAM only delivers full bandwidth when enough are outstanding.

## Why library GEMM tiles waste bandwidth at M = 1

GEMM kernels (cuBLAS, CUTLASS) tile the output into blocks of BM x BN with BM typically 64 or 128, because tensor-core instructions work on 16- or 64-row tiles and large tiles reuse each loaded weight across many rows. At M = 1:

1. **Rows are padding.** 63 of 64 tile rows are empty. The maths is wasted, which is harmless (decode is not compute-bound), but the tile's pipeline (multi-stage shared-memory buffers, TMA, synchronisation) is sized for reuse that never happens.
2. **Too few blocks.** The grid is about `N / BN` blocks. For N = 4,096 and BN = 128 that is 32 blocks on a GPU with 132-170 SMs: most SMs sit idle, and idle SMs issue no loads.
3. **Not enough bytes in flight.** By Little's law, bytes in flight needed = bandwidth x latency. With ~1.8 TB/s and a loaded DRAM latency of several hundred nanoseconds (say ~600 ns, illustrative), you need about 1.1 MB outstanding across the GPU, about 6-7 KB per SM on 170 SMs. A grid that occupies a fifth of the SMs cannot get there.
4. **Split-K heuristics** in libraries may not choose enough splits for M = 1, or may add a separate reduction kernel (one more hand-off).

Result: a GEMM that is "memory-bound" on paper runs at a fraction of measured bandwidth. ncu shows it as latency-bound: low DRAM %, low SM %, waves below 1, long-scoreboard stalls.

## Design of a bandwidth-bound GEMV

1. **Many blocks.** Assign a few output rows per warp and split K so the grid covers every SM several times (for example N = 4,096, K = 12,288: 16 rows per block and 4 K-splits gives 1,024 blocks).
2. **16-byte vector loads.** Each thread loads 16 FP8 weights at once (`uint4`, SASS `LDG.E.128`). Fewer load instructions per byte, more bytes in flight per thread.
3. **Several loads in flight per thread.** Unroll so a thread issues the next loads before consuming the first.
4. **Activation loaded once.** The `[K]` activation is small (8 KB in BF16 for K = 4,096); keep it in shared memory or rely on L1/L2.
5. **Scales per block, not per element.** Accumulate a partial dot product over a chunk inside one 128-wide K block in FP32, then multiply once by that block's scale.
6. **Warp-shuffle reduction** (`__shfl_xor_sync`) across the warp's 32 threads; then combine K-splits.
7. **Deterministic split-K reduction** if you need reproducibility: write partials, and have a second tiny pass or the last-arriving block add them in a fixed order. `atomicAdd` on floats is faster but order-dependent.
8. **W8A16 vs W8A8.** At M = 1 the maths is cheap, so dequantising weights in registers and multiplying by a BF16/FP32 activation (W8A16) avoids a separate activation-quantisation kernel. W8A8 (FP8 activations with per-token, per-128 scales) is what tensor-core FP8 GEMMs need; it adds a quantise step unless fused into the previous kernel. Either way, prove correctness against the reference.
9. **E8M0 scales are no blocker:** take the requantised weights and power-of-two scales vLLM hands DeepGEMM (the baseline's numerics) and apply each 128-block's scale in DeepGEMM's order: the GEMV can then match it bit for bit. Check `torch.equal` on real layers before timing; a mismatch makes it a numerics change.

## Sketch (CUDA, untuned, for learning)

```cuda
#include <cuda_fp8.h>
// W: [N, K] FP8 E4M3, row-major. x: [K] FP32 activation. sw: [N/128, K/128] FP32 block scales.
// Grid: (N / (ROWS * WARPS), K / KSPLIT); block: 32 * WARPS threads. Assumes N % 128 == 0, K % KSPLIT == 0,
// KSPLIT % 512 == 0. Writes one partial per (K-split, row); a second pass sums the partials in order.
template <int ROWS, int WARPS, int KSPLIT>
__global__ void fp8_gemv_splitk(const __nv_fp8_e4m3* __restrict__ W, const float* __restrict__ x,
                                const float* __restrict__ sw, float* __restrict__ partial, int N, int K) {
  const int lane = threadIdx.x % 32;
  const int row0 = (blockIdx.x * WARPS + threadIdx.x / 32) * ROWS;   // this warp's first output row
  const int k_begin = blockIdx.y * KSPLIT;                            // this block's K-slice
  float acc[ROWS] = {};
  for (int k = k_begin + lane * 16; k < k_begin + KSPLIT; k += 32 * 16) {  // each lane: 16 values per step
    float xs[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) xs[i] = x[k + i];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int n = row0 + r;
      const uint4 raw = *reinterpret_cast<const uint4*>(W + (size_t)n * K + k);  // one 16-byte load
      const __nv_fp8_e4m3* w = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
      float dot = 0.f;
      #pragma unroll
      for (int i = 0; i < 16; ++i) dot += float(w[i]) * xs[i];             // FP8 -> FP32, FMA
      acc[r] += dot * sw[(n / 128) * (K / 128) + k / 128];                 // 16 values never straddle a 128 block
    }
  }
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);  // sum over 32 lanes
    if (lane == 0) partial[(size_t)blockIdx.y * N + row0 + r] = v;
  }
}
```

Line by line, the parts that matter:
- `row0`: each warp owns `ROWS` consecutive output rows; `blockIdx.y` picks the K-slice, which is the split-K.
- The loop strides by `32 * 16`: the warp's 32 threads each take a different 16-byte chunk of the same row, so a warp reads 512 contiguous bytes per row per step (coalesced).
- `uint4` is 16 bytes: the compiler emits one 128-bit load. Reinterpreting it as 16 FP8 values is free.
- `float(w[i])` converts FP8 to FP32 (hardware conversion on recent GPUs). If ncu later shows the ALU pipes saturated, the conversion is the bottleneck: convert pairs with packed instructions.
- The scale is applied once per 16 values, using the scale of the 128x128 block that contains them.
- `__shfl_xor_sync` is a butterfly reduction: after 5 steps every thread in the warp holds the warp's sum.

Not shown: loading `x` into shared memory, double-buffering, the reduction pass, PDL, fused epilogues (SiLU*up for the gate-up GEMV). Tune split count and rows per warp *inside the real decode step*. Under PDL, drop `__restrict__` on `x` ([why](launch-overhead.md#pdl-placement)).

## lm_head at batch 1

For Qwen3-8B the lm_head is 151,936 x 4,096. In BF16 that is 1.245 GB per token, ~15% of the FP8 model's bytes. Options, each with a correctness bar:
- FP8 lm_head: saves ~0.62 GB per token (~8% ceiling gain), changes logits directly; check token agreement and the logit margin at divergences.
- Fuse argmax/top-k into the lm_head GEMV epilogue: avoids writing and re-reading 151,936 logits (0.6 MB in FP32), and one kernel boundary.
- Split the vocabulary across many blocks: N is huge, so the lm_head is the one GEMV that fills the GPU easily.
- Exact bound pruning (a cheap proxy pass, then only the rows that can still win) and lossless re-encoding of a 16-bit head: [byte-cuts.md](byte-cuts.md).
