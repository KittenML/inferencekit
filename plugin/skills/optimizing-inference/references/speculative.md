# Speculative decoding: a standard lever at low concurrency

At batch 1 each decode step reads every weight it uses to make one token, while compute sits mostly idle. Speculative decoding proposes several tokens cheaply and checks them all in one pass of the full model; each accepted token skips its own weight pass. It is headroom estimate 3 ([stop-point.md](stop-point.md), "Headroom").

## On or off
- **`auto` (default):** try it when the lead is `single_user` or N\* is well below B_crit ([capacity.md](capacity.md), section 4); at high concurrency draft and verify work competes with real tokens (inferred).
- **Off** when the task or intake forbids it (`speculation: off`, or a task rule): for the baseline too, and no headroom estimate.
- **Baselines:** engine rankings keep it off unless every engine runs the same lossless method ([engine-configs.md](../../benchmarking/references/engine-configs.md)). Quote a speedup with it against a baseline the task or person fixed (e.g. one run without speculation); otherwise each baseline engine with exact speculation for this checkpoint also gets a row at its best documented speculative config, and the speedup is against the faster of its two rows (evidence rule 9).

## Draft sources, in order (stop at the first that is faster; log each in `experiments.jsonl`)
1. **Multi-token prediction (MTP) heads in the checkpoint:** MTP tensors in the headers (often the `skipped` group of `scripts/bytes_per_token.py`) and the config keys declaring them.
2. **EAGLE-style heads published for this exact checkpoint,** pinned, same tokenizer; heads for another revision or precision accept less.
3. **A small same-family model** with the identical tokenizer and chat template (compare token ids, not only the vocabulary size).
4. **N-gram or prompt lookup:** no weights; pays on copy-heavy work (code edits, answers over retrieved text), little on free text.

## Measure, keep only if faster
- **Acceptance on the workload:** the run's prompts, at the target concurrency, with the timed sampling (it usually accepts less than greedy); report mean accepted tokens per verify step from `/metrics` or the engine log.
- **Estimate first (derived):** speedup ≈ tokens per verify step ÷ (verify + draft time, in plain-step units). A dense batch-1 verify step reads the same weights as a plain step, so it costs about one step (inferred). **MoE gains less:** the verified tokens route to different experts, so a verify step reads their union.
- **Screen draft lengths 1 to about 5**, then keep it only if `bench rounds` says faster at the workload's concurrency, against the same baseline row, prompts and sampling.
- **Bytes:** the MTP head or draft model is read during decode: add it back ([ceilings.md](ceilings.md)); report the ceiling per verify step.

## Exactness (the lossless track)
- **Greedy:** speculation on and off give the same tokens: `spec_exact` (`inferencekit gates run --spec-off-url <same model, no speculation>`, or `inferencekit gates spec-exact`).
- **Sampling:** standard verification (accept with probability min(1, p/q), else resample from the residual) keeps the distribution in theory. `inferencekit quality sampling` sees only the first token, which comes from prefill before any draft: it proves the sampler and head, not speculation. Also check the accept/resample step: run the verify code (yours, or the engine's rejection sampler called directly) on fixed target and draft distributions over many seeds, and compare the emitted-token histogram with the target distribution inside the floor of as many direct draws ([sampled exactness](../../proving-correctness/SKILL.md#sampled-exactness)). Either missing: sampled exactness UNPROVEN, never PASS.
- **Lossy acceptance** (typical acceptance, relaxed thresholds) only under `fidelity: lossy`, inside its budget ([fidelity.md](fidelity.md)).

## In a custom engine
Verify the drafts in one forward pass, drop rejected positions' KV, capture one CUDA graph per draft length, reuse the plain path's sampler and seeds, take logprobs from the full model, and keep an off switch for the exactness checks. Report source, draft length, acceptance and where it was measured, speed with and without (paired CI), and both exactness verdicts.
