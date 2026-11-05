# Claims lint: line by line

Run this over the final text of any report, PR, README table or agent summary: what to check, a failing example, the fix (placeholder values). A line that cannot be fixed downgrades the claim per the verdict precedence (say which line failed). Early-engine numbers quoted in these skills are engine-timed (rule 18) and carry that label (line 6) wherever repeated.

| # | Check | Fails | Passes |
|---|---|---|---|
| 1 | Every number has an evidence id, or a formula whose inputs have ids | "decode runs at 171 tok/s" | "171.4 tok/s median, client-timed (bench rounds), 6 rounds (ev_...)" |
| 2 | Every number is labelled observed / derived / inferred | "85% of the ceiling" | "85% of the measured-bandwidth ceiling (derived: 171.4 / 201.6; ev_rounds, ev_roof)" |
| 3 | Claim verdicts use only PASS / FAIL / UNPROVEN / NOT RUN | "looks good", "probably faster", "PASS-ish" | "UNPROVEN (underpowered): CI [0.98, 1.05] includes 1" |
| 4 | Comparison verdicts come from the tools (`bench rounds` ratio CI for served engines, `compare` for `bench decode`) | "B is faster: 152 vs 149" | "improved, 1.02x [1.01, 1.03] (ev_rounds, decode_tok_s_ratio_to_best)" |
| 5 | No speed claim across precisions, except a format trade-off paired with its quality verdict ([evaluating-quantization](../../evaluating-quantization/SKILL.md)) | "1.11x faster than llama.cpp" (Q8_0 vs FP8); "AWQ is 2.2x faster than BF16" (no quality, no CI, no label); a format trade-off ratio called `improved` | "llama.cpp Q8_0: cross-precision, reported against its own ceiling; no ratio verdict"; and for a format trade-off: "vLLM AWQ vs vLLM BF16: decode at least 2.1x (CI [2.1, 2.3], ev_rounds); mean KL +0.012 nats over the A/A floor, 4-bit margin 0.03: PASS (ev_q); GSM8K -0.8 points [-2.0, +0.4]: PASS (ev_g8k); cross-precision: format trade-off, not an engine comparison" |
| 6 | Engine-timed numbers labelled and kept out of published tables | a `bench decode` number in the headline table | "engine-timed (in-process); not comparable with client-timed rows" |
| 7 | Every speedup has a CI95 and quotes the conservative bound | "up to 2x faster", "12% faster" | "at least 1.09x (CI [1.09, 1.16]), medium bucket" |
| 8 | Each baseline is pinned, tuned for this goal, settings proven | "beats vLLM" | "vs vLLM <pinned version> at its documented config for this goal, attention backend resolved <X> (ev_...)" |
| 9 | Capacity claims state SLO, arrivals, `N*`, price | "serves 40 users" | "N* = 40 at p95 TTFT <= 300 ms, p95 TPOT <= 50 ms, 0 failures, open-loop Poisson (ev_cap); $X per 1M tokens at $Y/h (derived)" |
| 10 | Missing data is NOT RUN, never 0 or blank | an empty H100 column; "0 tok/s" | "H100: NOT RUN" |
| 11 | Profiles explain, never prove | "ncu shows the kernel is 2x faster, so decode is faster" | "kernel 2x in ncu replay (not end-to-end); step 1.02x [1.00, 1.04]: UNPROVEN" |
| 12 | Quality verdict present for every speed claim, from the right path | speed table only | "quality: decode-path top-1 agreement within the A/A floor, engine minus floor [-0.4, +0.1] pp inside the pre-registered 0.5 pp margin: PASS (ev_q)" |
| 13 | Gates ran and passed for every engine shown | no gates record | "gates: PASS (ev_g)" or the failing gate named |
| 14 | Losses and refuted ideas listed with the same columns | only wins | "split-K x8: 1.00x [0.99, 1.01], within_noise at a 2% effect; not adopted" |
| 15 | No extrapolation | "should also hold on H100" | "H100: NOT RUN; expected direction inferred from the bandwidth ratio" |
| 16 | Session dependence disclosed | one session's absolute tok/s presented as the engine's speed | "vLLM measured 1-3% slower in this session than in four earlier ones; cause not identified" |
| 17 | The pre-registered claim is quoted with its timestamp | the claim appears only after the results | "C1 (written <timestamp>, before any candidate run)" |
| 18 | The review page is linked | outputs never shown | "outputs: runs/<run-id>/review/index.html" |

## Auditing someone else's claim

Go down the table, quote each failing sentence, and state the verdict the evidence actually supports. A claim that fails line 1, 5 or 11 has no verdict better than UNPROVEN until it is re-measured; a claim that fails line 12 or 13 cannot PASS.
