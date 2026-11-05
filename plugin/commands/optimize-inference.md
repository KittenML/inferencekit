---
description: Make a model faster, use less memory or run on cheaper hardware. Ask a few simple questions, research and implement optimizations, and prove the result against a tuned baseline within the approved budget.
argument-hint: "<model or intake> [hardware] [goal] [budget] [deadline] [quality preference]"
---

Optimize inference for: $ARGUMENTS

Run the optimizing-inference skill in the main conversation: read `${CLAUDE_PLUGIN_ROOT}/skills/optimizing-inference/SKILL.md` (in a checkout of this repo: `plugin/skills/optimizing-inference/SKILL.md`) and follow it.

- Start with `${CLAUDE_PLUGIN_ROOT}/skills/optimizing-inference/references/starting-questions.md`: reuse prior answers; ask only missing quality, goal, budget, deadline, check-in and model-change choices in plain language.
- No plan or plan approval by default. Show/create a plan only when requested. Existing spending and scope authorization persists; new paid work outside it needs approval.
- Research current implementations early and when progress stalls. Allow independent kernel writing in parallel; serialize integration and measurements on each device.
- **Work to the deadline:** reserve final validation/report time as specified in intake. An estimated unreachable target is a verdict, never a stop by itself. One `claims-skeptic` reviews at freeze (no freeze: on the draft, before the host is released).
- Finish with reproducible code, baseline/quality evidence, verdicts, total time/tokens/cost and a plain-English report.
- Then follow the skill's optional engine-contribution workflow: prepare a qualifying contribution, and publish its PR only within explicit user authorization. Reuse a recorded opt-in; without one, show the proposal before asking (unattended runs leave it awaiting permission).
