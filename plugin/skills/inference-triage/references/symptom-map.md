# Symptom map: from complaint to first measurement

Each row: what people say, what it usually means, the cheapest evidence that separates the hypotheses, and the skill that owns the next step.

## Terms

- **Decode step:** one forward pass that produces one new token. At batch 1 it reads every weight once.
- **TTFT (time to first token):** prompt processing (prefill) plus the first decode step. Prefill is usually compute-bound; decode is usually memory-bound.
- **Ceiling (speed of light):** the tok/s you would get if the GPU did nothing but stream the bytes one token needs at the measured DRAM bandwidth.
- **% of ceiling:** measured tok/s divided by ceiling tok/s. A derived number.
- **Launch-bound / latency-bound:** the GPU spends its time on fixed per-kernel costs (launch, ramp-up, tail, hand-off between dependent kernels), not on moving bytes or doing maths.

## Symptom table

| What people say | Usual meaning | Separating evidence | Owner |
|---|---|---|---|
| "Make it fast" / "which engine should I use" | A whole flow, not one measurement | intake: model, hardware, goal (single-user or capacity), SLO, budget | optimizing-inference |
| "Decode is slow" | Unknown until compared to a ceiling | `inferencekit roofline` + single-user `bench rounds` -> % of ceiling | benchmarking |
| "GPU utilisation is only 40%" | `nvidia-smi` utilisation is "a kernel was running", not "SMs were busy"; low values point at host gaps | nsys per-step busy/idle and gap histogram | profiling |
| "GPU is 100% busy, still slow" | Busy is not used: tiny kernels that fill a fraction of the SMs | nsys kernel count and median kernel duration, then ncu on the top kernels | profiling (nsys, then ncu) |
| "vLLM is faster than ours" | Could be precision, flags that did not take effect, CUDA graphs, context length, timing boundary, or a real kernel gap | `bench rounds` with both engines, same precision, resolved configs; the paired ratio CI | benchmarking |
| "Our new kernel is 2x faster" | Measured in isolation; end-to-end effect is Amdahl-limited | the kernel's critical-path share of the step from nsys, then an interleaved A/B | profiling, benchmarking |
| "First token takes forever" | Prefill (compute-bound), compilation, CUDA graph capture or cold caches in the first request | TTFT per bucket; first-seen vs repeated prompt length; round 1 vs later rounds | benchmarking (warmup rules), optimizing-kernels (prefill) |
| "Gets slower over a long generation" | KV cache bytes grow with context; attention share rises | roofline at several context lengths; nsys attention share early vs late | benchmarking (ceilings), profiling |
| "Results changed after the optimisation" | Numerics (expected small drift) or a bug | trajectory score vs the A/A noise floor; margin at disagreements | proving-correctness |
| "Numbers differ run to run" | Noise: clocks, thermal state, other tenants, CPU contention | rounds, CI width, `doctor` clocks, host load during the run; fresh process per round | benchmarking |
| "Profile says kernel X is 30% of time" | A ranking, not a speedup | nothing more needed for ranking; a speedup needs an A/B | reporting-perf-claims |
| "How many users can one GPU serve?" | Capacity under an SLO, not peak throughput | SLO targets, traffic shape, then an open-loop `bench rounds --mode load` sweep -> `N*` | benchmarking (capacity) |
| "p95 latency explodes under load" | Queueing past the knee, prefill stalling decode, KV cache full | latency-vs-concurrency curve; engine logs for preemption or KV-cache limits | benchmarking (capacity), then engine tuning |
| "We set the flag and nothing changed" | The setting was ignored or overridden | the engine's resolved config and log warnings | benchmarking (prove each knob took effect) |
| "Is the faster config still the same model?" | Numerics changed: kernel, precision, backend, speculation | quality trajectory + score vs the noise floor; gates (spec-decode exactness) | proving-correctness, benchmarking (gates) |

## % of ceiling bands (heuristic)

The bands are the ones in [inference-triage](../SKILL.md#ceiling-bands): rules of thumb for single-user (batch-1) decode on current NVIDIA GPUs. Treat them as a prompt for the next measurement, not a verdict.

| Band | Likely state | Next evidence |
|---|---|---|
| > 100% | Impossible: the ceiling is wrong (weights partly in L2, wrong dtype, speculative decoding, wrong token count) | Recheck bytes/token and the token count in the benchmark record (after an exact byte cut, for the new format) |
| 85-100% | Bandwidth-bound and near the floor; only fewer bytes per token helps | Lower precision or an exact byte cut (lossless re-encoding, bound-pruned lm_head), with a correctness proof; or stop |
| 50-85% | Mixed: launch hand-offs, under-filled GEMVs, attention at long context, sampling on the host | nsys per-step timeline; also the exact byte floor (byte cuts work in every band) |
| < 50% | Structural: eager mode, CUDA graphs off, host sync every token, CPU-bound scheduler | nsys: look for idle gaps and `cudaStreamSynchronize`/`.item()`-style syncs |

## Questions to ask the user before measuring

1. Which goal leads: single-user speed (TTFT, per-token latency, decode tok/s) or capacity (concurrent streams per GPU within an SLO)?
2. Which comparison is the decision: engine A vs B, before vs after a change, or measured vs ceiling?
3. Is the GPU shared with anything else during the measurement window?
4. Which precision is in production? Benchmarks run at that precision, and comparisons stay within it.
5. How does input arrive (complete or streaming) and how is output consumed? Which input buckets matter?
6. Is GPU time approved, with a budget and an hourly price?
