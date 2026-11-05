---
description: Benchmark inference speed with inferencekit - single-user TTFT/TPOT/decode tok/s per input bucket, or capacity (N*, goodput, $/1M tokens) under an SLO, with gates and % of ceiling. Needs GPU approval.
argument-hint: "<engine endpoint(s) or launch line(s)> <model> [--capacity] [--slo-ttft-p95-ms N] [--vs <other engine or evidence id>]"
---

Benchmark inference speed for: $ARGUMENTS

Follow the benchmarking skill. Read `${CLAUDE_PLUGIN_ROOT}/skills/benchmarking/SKILL.md` (in a checkout of this repo: `plugin/skills/benchmarking/SKILL.md`) and its references before running anything.

Steps:
1. Restate the workload in one sentence: model + revision, precision (weights, activations, KV cache, lm_head), GPU, input buckets (default: the dev set's four, said so), output length, goal (single-user, or capacity with its SLO).
2. Approval: `intake.yaml` with `budget.approved: true`, or the user's explicit yes to the GPU, $/h x hours and a cap. Without it, stop.
3. Write the claim before measuring (metric, bucket, threshold, decision rule); with no comparison, "report the metrics and % of ceiling".
4. `inferencekit doctor` (stop if anything else uses the GPU or the host is busy), then `inferencekit ceiling bw` and `inferencekit roofline` at each context you report. Not NVIDIA: `ceiling bw` is NOT RUN; follow `${CLAUDE_PLUGIN_ROOT}/skills/inference-triage/references/hardware.md` (hardware card, bandwidth, `--serve-args "--gpu-name <hardware>"`).
5. Per engine: its best documented configuration for this goal, with launch line, env, port and ready check in `runs/<id>/manifest.json` (`inferencekit manifest validate`); prove each setting in effect from the resolved config or log.
6. `inferencekit prompts verify --set dev` (or `--set <path>`); pass the printed path as `--prompts`.
7. Single-user: `inferencekit bench rounds --manifest runs/<id>/manifest.json --prompts <set.jsonl> --rounds <R> --mode single` (R pre-registered: default 5, never below 3; a fresh engine per round, Williams order), never a bare `bench serve` with its default `--rounds`. Capacity: `--mode load` with the skill's `--serve-args` grid, SLO and price.
8. Gates per engine, its server relaunched from its manifest line: `inferencekit gates run --base-url <its url> --engine <label> --run-dir runs/<id> --reference-engine <baseline label>`, then stop it. A FAIL voids that engine's numbers (say which gate); a live gate NOT RUN is not a pass.
9. Comparisons: the `bench rounds` record's `<metric>_ratio_to_best` per bucket, with its paired CI95. `inferencekit compare` accepts matching decode, single-user serve, clip, image or audio records; use `bench rounds`' own results for its aggregate, and do not pass load-mode records. Never across precisions, except an evaluating-quantization format trade-off (with its quality verdict, labelled, no `improved` / `regressed`).
10. Engines without a server only: `inferencekit bench decode` (`--rounds 1 --extend <ev>` chains alternated between engines, then `compare`), labelled "engine-timed" everywhere.
11. Finish with the skill's fixed tables, the claim verdict (PASS / FAIL / UNPROVEN / NOT RUN), every number observed / derived / inferred with its evidence id, and `inferencekit report <ids...> --format md`.

No secrets in commands, logs or the report. No suitable GPU: "Getting a GPU" in the inference-triage skill; rent nothing without an explicit yes to the offer and its cost.
