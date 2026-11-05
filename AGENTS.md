# AGENTS.md: always-on rules for inference performance work

These rules apply to any agent doing inference performance work with this repository or the inferencekit tools, including agents that read only this file. Procedures live in `plugin/skills/<name>/SKILL.md`; each rule's reason is in `plugin/skills/inference-triage/references/evidence-rules.md` (same numbers).

## Tools

- CLI `inferencekit <command>`; MCP server `inferencekit mcp serve` (stdio; every command is also a tool with the same arguments; launcher `uvx --from 'inferencekit-cli[mcp]==0.1.0' inferencekit mcp serve`). Commands and flags: `inferencekit --help` and `inferencekit <command> --help`.
- Measure: `doctor`, `ceiling bw`, `roofline`, `bench rounds` (runs `bench serve` per engine per round, fresh process, balanced order: published numbers), `bench serve` (one client-timed call), `bench decode` (engine-timed; engines without a server), `bench image`, `bench audio` (client-timed media screening; contracts in `docs/media-benchmarks.md`), `compare` (matching benchmark kinds; load records are refused).
- Check: `gates run`, `prompts verify|show`, `quality trajectory|score|gsm8k|sampling`, `correctness`.
- Diagnose: `nsys capture|analyze`, `ncu capture|summarize`.
- Record and report: `intake template|validate`, `manifest validate`, `evidence list|show`, `ledger trailers`, `history`, `review build`, `report`.
- Evidence lands in `runs/`; `runs/`, `*.nsys-rep`, `*.ncu-rep` and `*.sqlite` are git-ignored. Never commit them.
- Hardware routing starts with `doctor`: NVIDIA uses Nsight Systems then Nsight Compute; AMD ROCm uses rocprofv3 with architecture-specific counter support; Apple Silicon uses Instruments, Metal/Xcode and MLX captures. `doctor` reads AMD DRM sysfs and Apple system facts; unknown details stay unknown. `ceiling bw` and nsys/ncu capture remain NVIDIA-only. Retrieve current official docs when architecture, runtime, flags or profiler support is uncertain; see `plugin/skills/inference-triage/references/hardware.md`.
- Subagents (`plugin/agents/`): `engine-scout` (one engine's support, sourced), `quant-scout` (one model's quantized checkpoints, bytes per token and fit from headers), `profile-analyst` (one profile's bottleneck), `claims-skeptic` (refutes a report's claims), `log-reader` (one question about a big log or doc, answered in at most 40 lines). They do GPU-free work only; any agent can follow the files as checklists.

## Which skill

| Need | Skill |
|---|---|
| Make a model fast end to end; which engine; hit a target | `optimizing-inference` |
| Find current optimizations and turn sources into experiments | `researching-optimizations` |
| Not sure where to start | `inference-triage` |
| tok/s, TTFT, capacity `N*`, ceilings, engine vs engine, before vs after | `benchmarking` |
| Where the time goes; why one kernel is slow | `profiling` |
| Custom engine, compiler inspection or kernel work for the task | `optimizing-kernels` |
| Is it still the same model; did quality regress | `proving-correctness` |
| Fit consumer memory while meeting latency and quality; minimize memory | `optimizing-inference` (affordability reference) |
| Should I quantize; which quant; will it fit; a big model on a small GPU | `evaluating-quantization` |
| Writing up or auditing a claim | `reporting-perf-claims` |

## Evidence rules (numbers are stable: skills and evals cite them)

1. Label every number **observed** (cite the evidence id), **derived** (show the formula) or **inferred** (name the measurement that would confirm it).
2. Claim verdicts are **PASS / FAIL / UNPROVEN / NOT RUN**.
3. Missing data is UNPROVEN (attempted, insufficient) or NOT RUN (never attempted); never zero, blank or PASS.
4. Combine checks with the precedence FAIL > UNPROVEN > PASS: any FAIL fails the whole; else any UNPROVEN makes it UNPROVEN. A check that does not apply is NOT RUN and is skipped; one that applies but ended NOT RUN counts as UNPROVEN; if nothing ran, the whole is NOT RUN.
5. Write the claim (metric, threshold, workload, bucket, decision rule) before measuring.
6. A speedup needs interleaved rounds and a paired ratio CI95 that excludes 1 (`bench rounds` `<metric>_ratio_to_best` for served engines, `compare` for `bench decode`); quote the conservative bound.
7. Comparison verdicts `improved | regressed | within_noise | inconclusive` come only from the tools: `bench rounds`' paired ratio CI for served engines, or `inferencekit compare` for two records of the same kind and workload (`bench decode`, single-user `bench serve`, `bench clip`, `bench image` or `bench audio`). Media timing alone leaves quality and lifecycle proof UNPROVEN.
8. Never compare speed across precisions (weights, activations, KV cache, lm_head). A format trade-off (`evaluating-quantization`) reports speed only paired with quality, labelled cross-precision. A numerics change the task explicitly allows is ranked, declared and labelled "numerics changed".
9. Compare against each baseline's best documented configuration for the goal measured, at a pinned version.
10. Quote the settings actually in effect (the engine's resolved config and logs), not the intended config; prove each knob took effect.
11. Report decode against a measured ceiling, not the datasheet: `inferencekit ceiling bw` on NVIDIA; on other hardware, a vendor or STREAM-style read-bandwidth measurement saved with the run (`plugin/skills/inference-triage/references/hardware.md`).
12. Profiles rank where time went; they never prove a speedup. Nsight Compute durations are never end-to-end evidence.
13. Every published claim and the final result needs a correctness or quality proof, an interleaved A/B with a CI and, for each claimed change, a profile showing the mechanism; the final candidate and any commit you hand over need the full proof, once (`bench rounds`, all gates, quality, all of GSM8K for LLMs, `quality sampling` when token sampling is timed; image and speech need their pre-registered modality checks, never LLM proxy gates). Inside the change loop each candidate gets one fast correctness check (greedy token identity, or logprob agreement for a numerics change, vs the previous build) and a screen; the first candidate that screens at >= 1.2x and the first at >= 1.5x of the baseline (or the task's target) get a light check (`plugin/skills/optimizing-inference/references/state.md`, "Best so far"). Between milestones a best is screened, not verified.
14. Bitwise identity is expected only for pure refactors within one engine; otherwise judge quality against the A/A noise floor, with margins at divergence.
15. Compute the Amdahl bound before optimising a kernel. Size it with critical-path time per step (`critical_pct_of_step`), not summed kernel durations; `kernel_sum_us / span_us > 1` means kernels overlap.
16. Publish losses, refuted ideas and NOT RUN items with the wins.
17. Never extrapolate to GPUs, precisions, contexts, buckets or concurrency levels you did not measure.
18. Published LLM numbers come only from `inferencekit bench serve` (client-side timer); label every `bench decode` number "engine-timed". Images and speech require a pre-registered client-timed modality adapter including final image decoding or first playable audio. A token endpoint is not proof of image latency or speech TTFA.
19. Each round starts every engine as a fresh process, in a balanced order (`inferencekit bench rounds`); R is pre-registered (default: the smallest multiple of the Williams period that is at least 3, i.e. 4 for 2 engines, 6 for 3, 4 for 4; the period is the shortest run of rounds in which each engine follows every other one equally often; never below 3), or an adaptive plan is (`bench rounds --ci-target`, `--min-rounds`, a maximum `--rounds`; it stops at whole Williams periods); CIs are bootstrapped over rounds; warmups never count.
20. Run `inferencekit gates run --base-url <url> --engine <label> --run-dir runs/<id>` for every engine while its server is up; a gate FAIL voids that engine's speed numbers; a gate NOT RUN is not a pass; inside a change loop, at milestones only (rule 13).
21. Quality passes only with no regression beyond the A/A noise floor or a pre-registered margin, with CIs, measured on the exact speed configuration. Identity and quality checks run greedy; speed is timed at the workload's sampling, the same for every engine; a sampled-exactness claim needs `inferencekit quality sampling` (first tokens only), and speculation also a check of its accept/resample step; else UNPROVEN.
22. Capacity is `N*` from open-loop load: the highest concurrency at which every SLO holds there and at every lower level. Closed loop is a stress test only.
23. Every kept change is a commit with Perf-* trailers from `inferencekit ledger trailers` and its screen numbers; failed experiments go to `experiments.jsonl`, not the branch.

## Safety rules

- GPU work (benchmarks, captures, microbenchmarks, quality runs, rentals) runs only with approval recorded in `intake.yaml` (`budget.approved: true`) or explicit user authorization to the stated scope and spending policy. Default policy uses a stated cost and cap; an explicitly uncapped run uses `budget.spend_cap: none` with a finite deadline, never a large numeric sentinel. Existing session authorization persists. Run nothing else on the device while timing.
- Begin with simple quality, objective, budget, deadline and check-in questions; reuse known answers. No plan or plan approval unless requested. Record authorization in intake/state; preserve existing session scope. Separate permission for model changes from permission for numerical approximations.
- With a stated deadline, work until the deadline minus the report reserve: "target unreachable" is a verdict, never a stop. Explicit per-run `budget.stage_timeboxes: false` disables stage cutoffs; `budget.report_reserve_h: 0` disables early reserve/freeze cutoffs. Keep testing and packaging continuously through that run’s deadline; defaults remain unchanged for other runs.
- Parallelize independent research, analysis, review and kernel writing in isolated files/branches. One coordinator integrates changes sequentially. One owner/lock per device serializes warmup, quality, captures and timing; avoid shared-host contention. Independent authorized devices may run concurrently.
- No GPU counters on the host (`inferencekit doctor` `profiling_counters.status: admin_only`, common on rented and cloud GPUs): profile with nsys only (or the engine's PyTorch profiler, which needs no counters), mark ncu NOT RUN, and ask for the person's own profiles (`nsys analyze` / `ncu summarize` read them without a GPU).
- On NVIDIA, profile only through `inferencekit nsys capture` / `inferencekit ncu capture`; AMD and Apple vendor captures must use the same `env -i` allowlist and downloaded offline models. Retrieve official tool docs for unfamiliar capture options. Every profiler output is sensitive; `.nsys-rep` files embed the process environment, secrets included. Treat `.nsys-rep`, `.ncu-rep` and `.sqlite` files as sensitive anyway (hostnames, paths, command lines).
- Never print, log, commit or pass on a command line any API key or token; read keys from environment variables by name.
- Rented GPUs: destroy after use and confirm none is left running; never name a provider in shared files.
- Never push without asking, except for a post-run [engine contribution](plugin/skills/optimizing-inference/references/engine-contribution.md) already explicitly authorized by the user. That opt-in covers only the contribution's reviewed branch push and PR, not unrelated publication.

## Output shape

End performance work with `REPORT.md` using the shared [report format](plugin/skills/reporting-perf-claims/references/wording-guide.md): **Result → Run it → Files and outputs → Details**. Start with the outcome in plain English, the named baseline and absolute before/after values, ratio [CI95], quality and material limitations or missed targets. Provide install, launch and example request commands and links to actual outputs. The final reply links the report and repeats the runnable commands and key result.

Details retains the fixed claim table (claim, workload, baseline, candidate, ratio [CI95], quality, mechanism, gates, verdict, evidence ids), findings and losses, limitations, reproduction commands and time/cost accounting. Use descriptive evidence links up front; ids, numeric classifications and formulas belong in Details or its linked table. `inferencekit report <ids...> --format md` remains a technical export: preserve its verdict, environment and results tables there or link the export. It has no quality or gates columns, so add those verdicts from their own records (`quality score`, `gates run`) and link the review page (`inferencekit review build <run-dir>`).
