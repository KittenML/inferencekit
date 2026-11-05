# Global evidence rules, with reasons

Every skill follows these. The numbers match the repository's `AGENTS.md` (which drops the reasons) and never change, since skills cite them; the reasons are here because rules without reasons get bent under deadline pressure.

"The early engine" in these skills means the custom Qwen3-8B FP8 decode engine built on one RTX 5090 during early inferencekit development (now in the companion inferencekit-engines repository, folder `qwen3-8b-fp8_rtx-5090`). Its numbers quoted here are engine-timed (rule 18) unless they say otherwise.

## Claim classes

| Class | Definition | Must include | Example |
|---|---|---|---|
| observed | A tool measured it in this environment | evidence id, statistic (median, p95), n | "decode 171.4 tok/s median, client-timed, 6 rounds (ev_...)" (placeholder values) |
| derived | Arithmetic on observed values or published constants | the formula and its inputs' ids | "85% of ceiling = 171.4 / 201.6 (ev_..., ev_...)" (placeholder values) |
| inferred | Reasoning from evidence; not measured | what measurement would confirm or refute it | "likely latency-bound; confirm with ncu waves and stalls" |

Why: observed numbers can be noisy, derived numbers can have wrong formulas, inferred statements can be wrong entirely. Readers need to know which failure mode applies.

## Verdict vocabularies

- **Comparison:** `improved`, `regressed`, `within_noise`, `inconclusive`: from `inferencekit compare` (`bench decode` records), or `bench rounds`' `<metric>_ratio_to_best` paired CI read the same way (served engines).
- **Claim:** PASS (every required check ran, coverage complete, evidence supports the pre-registered claim); FAIL (a required check contradicts it, including a regression, a quality failure or a gate failure); UNPROVEN (checks ran, evidence insufficient); NOT RUN (a required check never executed).

## The rules

1. **Label every number observed / derived / inferred.** Why: each class fails in a different way.
2. **Claim verdicts are PASS / FAIL / UNPROVEN / NOT RUN.** Why: a fixed vocabulary stops "probably faster".
3. **Missing data is UNPROVEN or NOT RUN, never zero, blank or PASS.** Why: an empty column displayed as 0 looks like a measurement.
4. **Combine checks with FAIL > UNPROVEN > PASS.** A check that does not apply is NOT RUN and skipped; one that applies but produced nothing counts as UNPROVEN; if nothing ran, the whole is NOT RUN. Per check, [reporting-perf-claims](../../reporting-perf-claims/SKILL.md) uses the finer steps FAIL > UNDERPOWERED > INCOMPLETE > NO-DATA > PASS, which map onto these verdicts. Why: one failing gate must never be averaged away by passing ones.
5. **Pre-register the claim before measuring.** Why: choosing the comparison after seeing results inflates false positives.
6. **A speedup needs interleaved rounds and a paired ratio CI95 that excludes 1** (`bench rounds` for served engines, `compare` for `bench decode`). Why: two medians can differ by noise alone; interleaving cancels drift.
7. **Comparison verdicts come only from the tools:** `bench rounds`' `<metric>_ratio_to_best` for served engines (ratio of medians, paired bootstrap CI95 over rounds); `inferencekit compare` for two `bench decode` or two single-user `bench serve` records (load mode refused). Why: they apply interleaving and the paired bootstrap the same way every time; eyeballed medians do not.
8. **Never compare speed across precisions** (a llama.cpp Q8_0 row next to FP8 engines is labelled cross-precision). Why: a lower-precision row reads fewer bytes, so the ratio measures the format, not the engine. Exception: a format trade-off ([evaluating-quantization](../../evaluating-quantization/SKILL.md)) shows its speed ratio with CI95 only next to its quality verdict against the full-precision reference, labelled "cross-precision: format trade-off, not an engine comparison", no `improved` / `regressed`: the format buys speed with quality, so only the pair is a result. A numerics change the task allows is ranked, declared and labelled "numerics changed"; a lossy intake row has its own block.
9. **Compare against each baseline's best documented configuration for the goal.** Why: beating a misconfigured baseline proves nothing, and the best single-user config is not the best serving config.
10. **Quote the settings actually in effect; prove each knob took effect.** Why: engines ignore settings silently. In an earlier version, a setting passed through a channel the engine version no longer read was dropped, and the A/B that set it was an accidental A/A.
11. **Use a measured ceiling:** `inferencekit ceiling bw` on NVIDIA; elsewhere a vendor or STREAM-style read-bandwidth measurement saved with the run ([hardware.md](hardware.md)). Why: no kernel reaches datasheet bandwidth, so datasheet ceilings make every engine look worse and hide whether a gap is real.
12. **Profiles rank; they never prove a speedup.** Why: profilers serialise, replay and perturb; removing a kernel can move work elsewhere.
13. **Every published claim and the final result ships with a quality proof, an interleaved A/B and, for each claimed change, a mechanism profile; the final candidate and a scored commit get the full proof, other milestones a light check; each step inside the change loop gets one fast correctness check and a screen.** Why: speed without correctness is a bug; speed without mechanism may be luck; a full proof on every step spends the budget on proof instead of search, and the milestones catch drift.
14. **Bitwise identity only for pure refactors within one engine.** Why: floating-point addition is not associative; different valid engines diverge after one near-tie flip.
15. **Compute the Amdahl bound first, with critical-path shares, not summed kernel time.** Why: a 2x kernel at 4% of the step is ~1.02x end to end, often below what the benchmark can resolve; and under programmatic dependent launch one kernel family was 8.7% of summed time but 2.5% of the critical path, and removing a third of it gave +0.8%.
16. **Publish losses and refuted ideas.** Why: a list of only wins hides the base rate, and the next engineer retries the losers.
17. **Never extrapolate.** Why: decode and serving behaviour change with GPU, precision, context, bucket and load.
18. **Published numbers come only from `bench serve`.** Why: an engine that times itself chooses its own boundaries; one client timer outside every engine is the only like-for-like boundary.
19. **Fresh process per engine per round, balanced order, bootstrap CIs over rounds** (`inferencekit bench rounds`; R pre-registered, default the smallest multiple of the Williams period that is at least 3: 4 for 2 engines, 6 for 3; never below 3; or an adaptive plan pre-registered as `--ci-target`, `--min-rounds` and a maximum `--rounds`, stopping only at whole Williams periods; kernel-loop milestones time a fixed 2 or 3, unbalanced with 2 engines: [statistics.md](../../benchmarking/references/statistics.md)). Why: warm state and a fixed neighbour land on one engine; requests within a round are not independent samples.
20. **Run the gates, per engine, against its live server and the run directory; inside a change loop, at milestones only (rule 13).** Why: caching, short outputs, miscounted tokens, repeated prompts under load, lossy speculation, benchmark special-casing and lossy pruned vocabularies all make the number better without making the engine better.
21. **Quality against the A/A noise floor or a pre-registered margin, on the exact speed configuration.** Why: invented tolerances drift toward the candidate, and debug or batch-invariant modes run different kernels from the ones timed.
22. **Capacity is `N*` from open-loop load.** Why: a closed loop slows its own arrivals under overload, so latency looks bounded while real users would queue; a pass at one level after a fail at a lower level is noise.
23. **Commit ledger.** Why: every kept change becomes a revertible checkpoint carrying its numbers, and nothing tried is lost.

## Safety rules (with reasons)

- **GPU work only with recorded approval and a stated cost.** Why: it occupies shared hardware and spends money. Run nothing else on the machine while timing: contention on "spare" CPUs or a shared GPU silently changes results.
- **Profile under `env -i` with an allowlisted environment.** Why: `.nsys-rep` files embed the full process environment, including any secret in it.
- **Rented GPUs: destroy after use and verify none is left.** Why: forgotten instances bill by the hour; stopped ones often still bill for storage.

## Workload identity (what must match for a comparison)

GPU model and count, driver, CUDA, clock policy, model id and revision, weight dtype and quantisation scheme, KV cache dtype, lm_head dtype, prompt set (name and sha256) and bucket, output length, mode (single-user or load) and concurrency or rate, SLO, sampling settings, seed, engine version and resolved settings. `inferencekit compare` and `bench rounds` check what they can from `identity` and the manifest; you are responsible for the rest.
