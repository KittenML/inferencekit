# Parallel implementation and isolated measurement

Research, analysis, review and independent kernel writing may run in parallel. Each implementation worker gets a distinct branch or owned files, one hypothesis, the quality policy, a reference revision and an explicit resource scope. Writing code does not require owning a GPU.

## Shipped read-only agents

| Agent | When | Returns |
|---|---|---|
| [engine-scout](../../../agents/engine-scout.md), `inferencekit:engine-scout` | per candidate, early research | sourced support, formats, flags and known issues; launch still decides support |
| [quant-scout](../../../agents/quant-scout.md), `inferencekit:quant-scout` | allowed formats or memory pressure | checkpoint revisions, size/fit and compatible runtimes |
| [profile-analyst](../../../agents/profile-analyst.md), `inferencekit:profile-analyst` | per profile | critical-path bottleneck with evidence |
| [claims-skeptic](../../../agents/claims-skeptic.md), `inferencekit:claims-skeptic` | once at freeze | refutations grounded in artifacts |
| [log-reader](../../../agents/log-reader.md), `inferencekit:log-reader` | one question about a log/doc over ~200 lines | at most 40 lines with file:line |

Use `log-reader` for big reads and `profile-analyst` for traces. These roles stay read-only. For kernel development use a normal implementation-capable worker with [optimizing-kernels](../../optimizing-kernels/SKILL.md), not a scout with silently expanded permissions. If delegation is unavailable, perform the same tasks sequentially.

Give each worker model/revision, shapes, device, runtime, goal, permissions, evidence paths and the expected handoff. Do not assume it sees the conversation or state. Subagent output is input, never a verdict; verify sources and executed changes.

## Integration and device ownership

- One coordinator owns shared `state.json`, budget and experiment records. Workers return patches/commits, hypotheses, quick checks and dependencies.
- Integrate one candidate at a time, resolve conflicts and run the relevant screen. Measure its interaction with previously kept changes; separate isolated gains from combined gains.
- **Never run two timed measurements on the same GPU at once.** A measurement owner acquires a host/device lock before launch/warmup and holds it through captures, quality and timing. Use the existing runner/supervisor lock; if none exists, queue all device work through one owner. Autotuning, JIT probes and correctness kernels use the same owner; plain code writing does not. A lock filename without acquisition/enforcement is not isolation.
- Pause competing builds, downloads, analysis and other CPU/I/O/memory-heavy work on the benchmark host during measurement. Code writing and research on another machine can continue. On unified-memory hardware, host memory pressure also affects GPU results.
- Independent authorized devices may run concurrently with separate intake, ledger and lock. Check shared CPU, PCIe, network, memory and thermal contention before parallel screening; dedicate the host for final rounds where necessary.
- Final interleaved rounds use the same device class, one engine at a time and the same harness. Never splice rounds from different devices or independent calls into one comparison.
- Limit compiler workers by available RAM; watch swap and device memory. Serialize resource-heavy compilation when necessary. Never kill unrelated host processes.

## Client execution

Use parallel task calls when supported; otherwise work sequentially. Codex workers need the relevant agent/skill body explicitly if the client does not discover it. OpenCode's shipped agents remain read-only; use an authorized implementation-capable worker for code changes.

If the client offers workflow orchestration, use it only within available permissions and existing user authorization. Do not require a workflow approval ceremony or a plan merely to parallelize work. Review findings proportionately; reuse one final `claims-skeptic` review instead of repeating full reviews per experiment.
