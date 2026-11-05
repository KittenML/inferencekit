---
# Generated: edit plugin/commands/perf-report.md
description: Turn inferencekit evidence into a performance claim with verdicts (PASS / FAIL / UNPROVEN / NOT RUN) and a review page, or audit an existing claim. Needs no GPU.
---
Write or audit a performance claim using: $ARGUMENTS

Follow the reporting-perf-claims skill. Read the `reporting-perf-claims` skill (load it with the skill tool, which also prints its base directory) and its references.

If a run directory or evidence ids were given (or none: use `inferencekit evidence list` and ask which records belong to the claim):
1. Read each record with `inferencekit evidence show <id>`: identity, coverage and limitations. Find its pre-registered claim; a post-hoc claim is at best UNPROVEN.
2. Apply the skill's comparability, gate, quality and verdict rules, including its evaluating-quantization exception. A speedup needs an `improved` comparison from an interleaved A/B with CI95 excluding 1. Missing data is UNPROVEN or NOT RUN, never zero or PASS.
3. Generate `inferencekit report <ids...> --format md` and `inferencekit review build <run-dir>`. Add quality/gates verdicts from their own records; the export has no such columns.
4. Use `references/wording-guide.md` in reporting-perf-claims for `REPORT.md` and the final reply. Start with the plain-English summary specified in optimizing-inference's `references/stop-point.md`, "Summary". Keep mechanisms, losses, refuted ideas, ledger, NOT RUN items and costs in Details. Put ids and observed / derived / inferred labels there or in linked evidence; label `bench decode` values "engine-timed" wherever quoted.
5. Run the skill's claims lint; fix or downgrade every failing line.

If a claim text or file was given to audit: run the claims lint line by line, quote each failing sentence, and state the verdict the evidence actually supports.

## OpenCode notes

- **Authorization:** reuse the approved spending/scope; ask only for new decisions. Plans are opt-in. Respect the client's current execution permissions.
- **Subagents** (engine-scout, quant-scout, profile-analyst, claims-skeptic, log-reader) are read-only. Independent kernel writing may use implementation-capable workers with isolated files. Serialize integration and device measurements.
- **Long steps** (`bench rounds`, `ceiling bw`, quality runs, captures) **use the CLI, detached with a log:** OpenCode ends an MCP call at its `timeout`; this server sends no progress.
