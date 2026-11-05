# Starting questions

Reuse conversation, brief, intake and safe inspection. Ask only missing choices, two or three at a time. Never require engine, compiler or numerical-metric knowledge.

## Quality: always reuse an existing answer

**“How much can the results change to make this faster?”**

| Choice | Explain it this way | Agent mapping |
|---|---|---|
| **Exactly the same** | Keep the same results. I'll focus on how the model runs. | `quality.preference: exact`, `fidelity: lossless`; pin deterministic/seeded output contract and verify equality. |
| **Same quality, small differences** (recommended) | Small differences are fine if the results remain just as useful, accurate, or natural. | `quality.preference: balanced`, `fidelity: lossy`; task/perceptual noninferiority with agreed bounds. |
| **Prioritize speed** | Try stronger compression and shortcuts. Some quality loss is okay, within a limit we agree on. | `quality.preference: speed`, `fidelity: lossy`; aggressive quantization is eligible, never automatically accepted. |

For speed, ask what must remain good: readable image text, complete speech, speaker identity or answer accuracy. Use examples to agree tradeoffs, then register `quality.budget` and protocol before evaluation. Do not invent permission or treat a recommendation as an answer.

Exactly the same is stronger than the legacy lossless distribution contract. Fix seeds and define which outputs must match. If the baseline itself varies, report that and ask whether the small-differences choice is acceptable; never silently weaken exactness.

## Other decisions

| Ask when missing | Simple choices or prompt |
|---|---|
| Main goal | **“What matters most?”** Faster individual results / More results at once / Less memory or cheaper hardware. Ask “Do you have a target, or should I find the best improvement within your budget?” only if needed. |
| Spending | **“What may I spend?”** Use only hardware already available / Up to a stated amount / Ask before renting anything. Include API/model charges if applicable. Confirm an actual amount/currency and rental rate before new spending, not a fresh approval for every benchmark. |
| Deadline | **“When should I stop and hand over the results?”** A date/time and timezone / After a stated duration / No deadline. No deadline does not grant unlimited spending or change the budget's stop rule. |
| Check-ins | **“When should I pause for you?”** Only when a new decision is needed / Also let me review quality before accepting a result / At each major milestone. |
| Model changes, separately | **“May I simplify or replace parts of the model itself?”** Keep the model as it is (recommended) / Yes, if it still meets my quality requirements / Ask me when you have a specific change to suggest. |

Map model changes to `model.architecture_changes: forbid|allow|ask`. Compression alone never authorizes pruning, distillation, skipped layers/experts or replacement. Report changed-model comparisons separately.

Check-ins map to `interaction.checkpoints: required|quality|milestones`. Fewer check-ins never waive spending, scope or quality requirements. If human review is required, prepare reviewable samples and pause promotion, while continuing authorized independent work. An unavailable reviewer means UNPROVEN, not approval. A fully unattended run cannot promise human checkpoints.

Inspect supplied model files/hardware first; ask only unknown model, machine or task constraints. Translate ordinary answers into workload settings yourself.

## Record and proceed

Record choices, sources and assumptions in intake and `state.decisions`. `interaction.show_plan` defaults to false. Show/create plans only on request; that does not require approving every step.

Keep planning internal and state compact. Read [intake.md](intake.md) and [fidelity.md](fidelity.md). Broader policy needs a new pinned run; reconsider numerical-only rejects with linked history and fresh checks, never rewritten verdicts.
