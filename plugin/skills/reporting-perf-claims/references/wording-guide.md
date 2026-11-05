# Claim layout and wording guide

## Run report (`REPORT.md`)

Explain results before flags or ids. Preserve verdicts and qualifiers.

```markdown
# <Model: result on the measured hardware>

## Result
<Plain-English outcome: what was delivered, what improved, whether the goal was met.>

| What matters | Result |
|---|---|
| Model and hardware | <model, precision, device and count> |
| Performance | <named baseline value → candidate value, units and timer boundary> |
| Improvement | <ratio, CI95 and conservative bound, or no verified gain> |
| Quality | <passed, failed or inconclusive checks; accepted trade-offs> |
| Scope | <workload, concurrency, important limits and missed targets> |

## Run it
<Prerequisites, working directory, install, launch and example request commands.>

## Files and outputs
<Links to delivered code/configuration, scripts, saved outputs and review page.>

## Details
### What changed
<Each optimization: what it does, why it helps, profile evidence.>
### Performance and target
<Claim/metric tables, baseline selection, environment, workload and statistics.>
### Quality and correctness
<Checks, scope, margins and verdicts.>
### What did not work and where this applies
<Rejected approaches, regressions, missing measurements and limits.>
### Reproduce and inspect
<Benchmark/quality/profile commands, manifest, evidence and commit ledger.>
### Time and cost
<Recorded agent time, tokens/API cost and hardware time/cost; missing totals stated.>
```

- **Result:** explain unfamiliar units; distinguish kernel and request timings. Show target misses, failed/inconclusive quality, accepted trade-offs and other retained options. Use descriptive links ("Timing comparison"); ids, observed / derived / inferred labels and formulas go in Details or its linked table.
- **Run it:** give copyable commands, a minimal request and output location for the delivered configuration. State which install/launch/request checks ran; never imply untested steps passed. Shared reports use portable paths and credential environment-variable names, not secrets or private host details.
- **Files and outputs:** link actual saved text, playable audio, images/video and the review page. Check targets; state when an expected artifact was not retained.
- **Details:** retain versions/settings, model revision/precision, hardware, timer boundaries, counts, round order, CIs and quality criteria. Profiles support mechanisms, not speedups. Cost accounting follows [stop-point.md](../../optimizing-inference/references/stop-point.md), "Summary"; do not estimate absent usage.

`inferencekit report <ids...> --format md` is a fixed technical export: link it or include its tables in Details. Add quality, mechanism and gates from their own records.

## Final reply

Link `REPORT.md` (absolute path locally), state the result against the named baseline and material quality/target limits, include install/launch/example request commands, and link the delivered engine and samples/review page. Say when there is no verified improvement or runnable artifact. Keep technical depth in the report.

## Wording guide

| Do not write | Write instead | Why |
|---|---|---|
| "12% faster" | "1.12x [1.08, 1.16] decode tok/s, Qwen3-8B FP8, p512/g256, batch 1, one RTX 5090" | ratio, CI and workload make it checkable |
| "up to 2x faster" | the median ratio for each workload measured, including the worst | "up to" reports the best case as if typical |
| "slightly faster" | "UNPROVEN: CI [0.98, 1.05] includes 1" | noise is not an effect |
| "beats vLLM" | "1.08x [1.04, 1.11] vs vLLM x.y.z at its documented low-latency config, same FP8 precision" | baseline version and configuration matter |
| "the kernel is 2x faster, so decode is faster" | "kernel 2x in ncu replay; step 1.02x [1.00, 1.04], UNPROVEN" | Amdahl; ncu is not end-to-end |
| "memory-bound" | "DRAM at 88% of measured peak (ncu, ev_k): memory-bound (inferred)" | label plus counter |
| "0 tok/s on H100" (a missing run) | "H100: NOT RUN" | missing data is not zero |
| "should also be faster on H100" | "H100: NOT RUN; expected direction inferred from bandwidth ratio" | extrapolation is inference, labelled |
| "near optimal" | "84% of the measured-bandwidth ceiling (ev_roof, ev_bench)" | quantify against a ceiling |
| "handles 40 users" | "N* = 40 streams per GPU at p95 TTFT <= 300 ms and p95 TPOT <= 50 ms, 0 failures, open-loop Poisson arrivals (ev_cap)" | capacity without its SLO and arrival process is unfalsifiable |
| "cheapest engine" | "$X per 1M output tokens at N* (derived: price / (tok/s at N* x 3,600) x 1e6, at $Y/h)" | cost depends on the price and the operating point |
| "our engine: 190 tok/s" (from `bench decode`) next to served engines | "190 tok/s engine-timed (in-process); client-timed: NOT RUN" | different timing boundary |
| "1.11x faster than llama.cpp" (Q8_0 vs FP8) | "llama.cpp Q8_0: cross-precision; 81% of its own ceiling" | a format difference is not an engine result |
| "lossless", "the same model" | "greedy tokens within the A/A floor (ev_q); `spec_exact` PASS (ev_g); sampled first tokens follow the engine's own log-probabilities (`quality sampling`, ev_s)", or "sampled: UNPROVEN" | each part of "the same model" has its own check |
| "1.4x faster" (16-bit activations, checkpoint 8-bit) | "1.4x [1.37, 1.43]; numerics changed: 16-bit activations, allowed by the task; quality no worse than the baseline (ev_q)" | an allowed change is still declared |
| "2x with speculation" | "1.9x [1.85, 1.95], prompt lookup, 2.6 tokens accepted per step on the dev prompts at concurrency 1, default sampling; `spec_exact` PASS; sampled: UNPROVEN (`quality sampling` PASS, accept/resample check NOT RUN)" | acceptance depends on prompts, concurrency and sampling |
| "AWQ is 2.2x faster than BF16" | "vLLM AWQ: decode at least 2.1x the vLLM BF16 row (CI [2.1, 2.3]); mean KL +0.012 nats over the floor (margin 0.03: PASS); GSM8K -0.8 points: PASS; cross-precision: format trade-off, not an engine comparison" | a format trade-off is a speed-and-quality pair, never speed alone (evaluating-quantization) |

## Audit checklist for someone else's claim

Use the claims lint, line by line: [claims-lint.md](claims-lint.md). Any failing line downgrades the claim per the precedence rules; say which line failed.
