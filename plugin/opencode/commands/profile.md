---
# Generated: edit plugin/commands/profile.md
description: Profile an inference step with inferencekit - Nsight Systems timeline and critical path, then Nsight Compute on the dominant kernel; the PyTorch profiler when counters or nsys are unavailable. Needs GPU approval.
---
Profile this workload: $ARGUMENTS

Follow the profiling skill. Read the `profiling` skill (load it with the skill tool, which also prints its base directory) and its references before capturing.

Steps:
1. Approval (`intake.yaml` `budget.approved: true`, or an explicit yes to the cost); analysing existing `.nsys-rep` / `.sqlite` / `.ncu-rep` files or torch traces needs none, so start with the person's own files for this workload.
2. The command runs bounded steady-state work with no credentials (weights downloaded, `HF_HUB_OFFLINE=1`), nothing else uses the GPU (`inferencekit doctor`), and the benchmark's ms/token (or TTFT) for the same workload is at hand.
3. Capture only through `inferencekit nsys capture -- <command>` (nsys under `env -i` with an allowlisted environment), never a raw `nsys profile`: `.nsys-rep` files embed the shell environment, secrets included.
4. Read the capture record: trace mode (host-side costs only from graph-level tracing, kernel attribution only from node-level), capture window, limitations.
5. `inferencekit nsys analyze <rep or sqlite from the record>`; a profiled step more than a few percent off the benchmark's ms/token measures something else.
6. If `kernel_sum_us / span_us > 1`, kernels overlap (programmatic dependent launch, streams): rank only by critical-path shares (`critical_pct_of_step`), never by summed kernel time.
7. Answer the six questions in order, each with a number, ending with the Amdahl bound of the change you would try.
8. If one family dominates and is far from speed of light: `inferencekit ncu capture` with an anchored `--kernel-regex` and a `--launch-skip` past warmup, then `inferencekit ncu summarize --peak-gbs <measured bandwidth evidence>`; label the bound (memory / compute / latency) as inferred.
9. Counters blocked, no nsys, or not NVIDIA: the skill's "Fallbacks" (nsys only with ncu NOT RUN; the engine's PyTorch profiler, read by hand or by the `profile-analyst` agent, never timed with it on; the person's own profiles; the platform's profiler).
10. Fill the skill's fixed tables, each number observed / derived / inferred with its evidence id or file, and name the single next step (e.g. one optimizing-kernels change, with its Amdahl bound).

A profile ranks where time went; it proves no speedup. Never attach or upload `.nsys-rep`, `.ncu-rep`, `.sqlite` or trace files without the user checking them first.

## OpenCode notes

- **Authorization:** reuse the approved spending/scope; ask only for new decisions. Plans are opt-in. Respect the client's current execution permissions.
- **Subagents** (engine-scout, quant-scout, profile-analyst, claims-skeptic, log-reader) are read-only. Independent kernel writing may use implementation-capable workers with isolated files. Serialize integration and device measurements.
- **Long steps** (`bench rounds`, `ceiling bw`, quality runs, captures) **use the CLI, detached with a log:** OpenCode ends an MCP call at its `timeout`; this server sends no progress.
