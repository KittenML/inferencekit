# Prefill and time to first token

Prefill multiplies every prompt token at once: M (the number of tokens) is large, so the GEMMs are compute-bound for long prompts, and a short prompt's handful of tiny kernels is usually host-bound. TTFT = queueing + prefill + the first decode step + client overhead; measure it with the client timer, per bucket.

Everything below comes from the custom Qwen3-8B FP8 engine built during early inferencekit development, on an RTX 5090 (from its development notes, not shipped; the engine is in the companion inferencekit-engines repository, folder `qwen3-8b-fp8_rtx-5090`). Every step was checked bitwise against the previous version.

## 1. Short prompts: remove the host from the critical path

- **Symptom:** at 128 tokens, ~11 ms of GPU work behind ~45 ms of eager launches.
- **Fix:** capture prefill as a CUDA graph per prompt length (the early engine captured on the second sighting of a length, up to 4 lengths), then replay. Real traffic needs padded length buckets instead (not implemented there).
- **Report both TTFTs:** a repeated length (graph replay) and a first-seen length (eager, plus any kernel compilation). The early engine (engine-timed): 8.1 ms repeated vs 19.5 ms first-seen at 128 tokens. Publishing only the replay number is a reward hack when the harness repeats one prompt.

## 2. Long prompts: kernel count around the GEMMs

- **Symptom (nsys, prefill graph at 1,024 tokens):** 60.5 ms of kernels, 2,068 kernels. FP8 GEMMs 71.8%, but ~1,600 PyTorch elementwise/copy/cat kernels took 14.7%: each RMSNorm was 8 kernels, RoPE 5 per tensor.
- **Fix:** four fused kernels per layer (residual + RMSNorm + activation quantisation; q/k norm + RoPE + KV write; SwiGLU + quantisation; attention-output quantisation) that reproduce PyTorch's arithmetic bit for bit: round to BF16 where PyTorch does, no FMA contraction where PyTorch rounds first, and PyTorch's exact reduction order for `mean(x^2)`. Getting the reduction order right was the difference between identical and "1 ulp off in a third of the rows".
- **Result:** 60.5 -> 49.0 ms (2,068 -> 377 kernels).

## 3. GEMM tiles for wave quantisation

- **Symptom (ncu, M = 1,024):** DRAM at 7-9% of peak (not memory-bound); 128x128 tiles with 8 warps fit one CTA per SM; tensor pipe at 29-36% of peak; 256 tiles on 170 SMs run as one full wave plus a half-empty one (1.51 waves).
- **Fix:** smaller tiles (128x64 or 64x64, 4 warps) so 2-3 CTAs fit per SM: finer tail granularity and a second resident CTA hiding each tile's pipeline fill and epilogue. Tile shape does not change any output's K accumulation order, so it stays bitwise.
- **Result:** GEMM time per layer 1,207.7 -> 1,023.0 us (-15.3%); prefill graph 49.0 -> 42.3 ms.
- **Refuted:** bigger tiles (128x256, 256x128) never faster; 32x64 tiles 25-40% slower; split-K and stream-K not pursued because they change the K order (not bitwise).

## 4. The end-to-end proof

Interleaved A/B through the early in-process harness (engine-timed), a fresh process per engine per round, 3 rounds: TTFT 11.47 -> 8.16 ms at 128 tokens and 60.77 -> 42.75 ms at 1,024 (per-round ratios 0.70-0.72), decode tok/s unchanged (ratios 0.9997-1.0006), first 64 generated tokens identical. Logits and the whole KV cache memcmp-identical to the previous engine at 128-2,048 tokens on every path (eager, capture, replay).

## Trade-offs to state

- The early engine kept a row-major copy of the weights for prefill (+6.4 GiB). Without it (same bits), TTFT rose 1.15x at 1,024 tokens. The memory-vs-TTFT choice belongs in the report.
- Under load, prefill competes with other users' decode steps: chunked prefill and scheduling knobs trade TTFT against TPOT. Measure that trade with `bench rounds --mode load` (open-loop `bench serve` under the hood), not with single-request TTFT.
