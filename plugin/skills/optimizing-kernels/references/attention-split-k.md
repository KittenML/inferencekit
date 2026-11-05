# Decode attention: split-K (flash-decoding)

## The problem at batch 1

One new query token attends to every cached key and value. With grouped-query attention (GQA) in Qwen3-8B, 32 query heads share 8 KV heads (4 query heads per KV head). A natural kernel launches one block per KV head (or per query head): 8 to 32 blocks. On a GPU with 132-170 SMs, most SMs are idle, so the kernel cannot pull full bandwidth while it streams the KV cache.

## Split-K over the sequence

Split each head's keys into chunks; one block per (KV head, chunk). Each block computes a partial attention with the online-softmax recurrence and writes three things:
- `m`: the running maximum score in its chunk;
- `l`: the sum of `exp(score - m)` over its chunk;
- `o`: the unnormalised weighted sum of values, `sum(exp(score - m) * v)`.

A combine step merges the chunks:

```
M = max_i m_i
L = sum_i l_i * exp(m_i - M)
O = sum_i o_i * exp(m_i - M) / L
```

This is exact (up to floating-point rounding): it is the same softmax, computed in pieces and rescaled to a common maximum.

## Details that matter

- **GQA reuse:** load each K/V tile once and use it for all query heads in the group (4 here). Why: K/V bytes dominate; reading them four times quadruples the traffic.
- **Chunk size:** enough chunks to cover the SMs several times, but large enough that each block streams a few KB. At 8K context with 8 KV heads, 256-key chunks give 32 x 8 = 256 blocks.
- **Graph-safety:** the chunk count must not depend on host-side values that change per step, or the CUDA graph would bake in one value. Read the sequence length from device memory and size the grid for the maximum context; blocks beyond the current length exit early.
- **Deterministic combine:** merge chunks in a fixed order (not "first to finish") if you need reproducible outputs.
- **Combine cost:** the combine is another kernel (or the last-arriving block of each head does it). At short context it can cost as much as the attention itself.

## When it matters (derived for Qwen3-8B, BF16 KV)

KV bytes read per layer per token = 2 x 8 x 128 x 2 bytes x context = 4 KB x context.

| Context | KV per layer | At a measured 1,600 GB/s | All 36 layers |
|---|---|---|---|
| 1,024 | 4 MB | ~2.6 us | ~0.09 ms (~2% of a 4.7 ms token) |
| 8,192 | 32 MB | ~21 us | ~0.75 ms |
| 32,768 | 128 MB | ~84 us | ~3.0 ms |

At short context, attention is small and dominated by fixed costs; split-K mostly helps fill the GPU. At long context it becomes a first-order cost, and an FP8 KV cache (a precision change, with its own correctness proof) halves it.

## Evidence that it worked

- ncu: the attention kernel's grid covers several waves; DRAM throughput near the measured peak at long context.
- nsys: attention's share of the step falls at long context; step wall time falls in an interleaved A/B at the context lengths you claim (and does not regress at short context).
