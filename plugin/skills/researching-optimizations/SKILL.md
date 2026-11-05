---
name: researching-optimizations
description: Find current, applicable inference optimizations and turn them into testable experiments for a specific model, shape, hardware and runtime. Use early in optimizing-inference, before custom kernel work, when progress stalls, or after a bottleneck/runtime change. Covers implementations, compiler docs, papers, engineering talks and issues beyond text models.
license: Apache-2.0
compatibility: Needs web/repository search or supplied source material. No GPU or paid action is required for research. Verify installed tool versions before using examples.
metadata:
  version: "0.1.0"
---

# Research to an experiment

Called by [optimizing-inference](../optimizing-inference/SKILL.md) and [optimizing-kernels](../optimizing-kernels/SKILL.md). Return a small ranked experiment queue, not a literature review.

1. **Fingerprint the problem.** Model architecture/components and revision; modality and input/output shapes; batch/concurrency; device architecture and memory; runtime/compiler versions; quality and model-change permissions; current profile and target. Missing facts are unknown, not guessed.
2. **Search what can actually run.** Check installed engine features first, then official releases, code, tests, issues/PRs, compiler/vendor docs and papers. Search architecture/operator and device identifiers, not just the model name. An issue is a hypothesis until code or a reproducer confirms it.
3. **Look beyond papers.** Use the [source map](references/sources.md) as a starting point, and actively discover sources beyond it: GPU MODE talks/slides/code, hardware and runtime engineering blogs, kernel libraries and compiler repositories often expose actionable details before papers. Search recent releases, merged PRs, conference/workshop material and new engineering groups for the latest relevant work and largest credible gains. Use original sources; follow secondary summaries to their implementation.
4. **Check freshness and applicability.** Record URL, access date, release/commit, publication date when available, supported device, dtype, shape and assumptions. Check whether the installed compiler already implements the idea. Compare high reported gains against baseline strength, quality/precision, shapes and full-request scope; prioritize the biggest applicable opportunity with credible headroom. A newer article is not automatically better. Do not call old material “latest” without checking updates.
5. **Produce an experiment.** Each useful finding needs the table below. Start implementing the cheapest informative test once a credible candidate exists; do not finish a catalog first. Retain at most three immediate candidates, with other links in a backlog.
6. **Refresh on evidence.** Search again after a moved bottleneck, two uninformative attempts, a toolchain change or an unexplained fallback. Reuse prior findings and rejected hypotheses instead of repeating the same search.

| Finding | Required record |
|---|---|
| Source and evidence | Exact source/section/commit; what was read or inspected; reported result versus your inference. |
| Proposed change | File/operator/runtime path; implementation or flag to reuse; how to prove it executed. |
| Compatibility | Device generation, compiler/runtime, dtype, shapes, memory, license and model/quality permissions. |
| Expected headroom | Critical-path fraction and Amdahl bound, or bytes/FLOPs/launches saved; mark estimates inferred. Unknown profile means first measure attribution. |
| Cheapest useful test | Small representative shapes and quality screen, expected mechanism, falsifying outcome, complete-request confirmation. |
| Decision | Try, defer or reject with reason and linked experiment id. |

Read linked slides, code or transcripts when a video is unavailable. Record that limitation; never claim to have watched a talk from its title. A tutorial example or datacenter result is not proof of support or speed on consumer hardware. Downloading or executing unfamiliar code requires inspection within the existing authorization.

Write compact findings into the existing experiment ledger or a linked research note. Independent research can run alongside kernel writing; [parallel.md](../optimizing-inference/references/parallel.md) governs shared resources.
