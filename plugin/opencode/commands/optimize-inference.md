---
# Generated: edit plugin/commands/optimize-inference.md
description: Make a model faster, use less memory or run on cheaper hardware. Ask a few simple questions, research and implement optimizations, and prove the result against a tuned baseline within the approved budget.
---
Optimize inference for: $ARGUMENTS

Run the optimizing-inference skill in the main conversation: read the `optimizing-inference` skill (load it with the skill tool, which also prints its base directory) and follow it.

- Start with the optimizing-inference skill's `references/starting-questions.md`: reuse prior answers; ask only missing quality, goal, budget, deadline, check-in and model-change choices in plain language.
- No plan or plan approval by default. Show/create a plan only when requested. Existing spending and scope authorization persists; new paid work outside it needs approval.
- Research current implementations early and when progress stalls. Allow independent kernel writing in parallel; serialize integration and measurements on each device.
- **Work to the deadline:** reserve final validation/report time as specified in intake. An estimated unreachable target is a verdict, never a stop by itself. One `claims-skeptic` reviews at freeze (no freeze: on the draft, before the host is released).
- Finish with reproducible code, baseline/quality evidence, verdicts, total time/tokens/cost and a plain-English report.
- Then follow the skill's optional engine-contribution workflow: prepare a qualifying contribution, and publish its PR only within explicit user authorization. Reuse a recorded opt-in; without one, show the proposal before asking (unattended runs leave it awaiting permission).

## OpenCode notes

- **Authorization:** reuse the approved spending/scope; ask only for new decisions. Plans are opt-in. Respect the client's current execution permissions.
- **Subagents** (engine-scout, quant-scout, profile-analyst, claims-skeptic, log-reader) are read-only. Independent kernel writing may use implementation-capable workers with isolated files. Serialize integration and device measurements.
- **Long steps** (`bench rounds`, `ceiling bw`, quality runs, captures) **use the CLI, detached with a log:** OpenCode ends an MCP call at its `timeout`; this server sends no progress.
