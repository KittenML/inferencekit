---
name: profiling
description: Finds where inference time goes and why a kernel is slow on NVIDIA, AMD ROCm, Apple Metal and CPUs, routing to vendor timelines then supported kernel analysis; nsys only or the PyTorch profiler when GPU counters are blocked. Use for "where does the time go", "why is decode/prefill slow", "GPU idle", "launch overhead", "which kernel", "is this kernel memory-bound", "nsys", "ncu", "torch profiler", "profile". Never proves a speedup.
license: Apache-2.0
compatibility: Needs the inferencekit CLI (Python 3.10+, uv or uvx) or its MCP server. NVIDIA capture uses Nsight; supported AMD setups use the rocprof wrapper; Apple capture uses vendor tools with manual evidence files. Existing profiles need no GPU. Written for Claude Code, Codex and OpenCode.
metadata:
  version: "0.1.0"
---

# Profiling inference: timeline first, then kernels

**nsys** records a timeline (CPU threads, CUDA API calls, kernels, copies, NVTX ranges): *where the time goes* and *whether CPU or GPU is on the critical path*. **ncu** replays one kernel reading hardware counters: *why it takes as long as it does*. Neither proves a speedup; only an interleaved A/B does ([benchmarking](../benchmarking/SKILL.md)). In [optimizing-kernels](../optimizing-kernels/SKILL.md)' change loop: one nsys capture per lever area (again when the bottleneck moves), a mechanism profile only for a claimed change.

**Use** when decode or prefill is below ~85% of its ceiling for unknown reasons ([ceiling bands](../inference-triage/SKILL.md#ceiling-bands)), for launch overhead, host syncs or idle gaps, for one dominant kernel family's limiter, or for the mechanism behind a measured speedup. **Not** for a speed number (never time with a profiler on), nor before you have a benchmark and a ceiling (a profile without them has no scale).

**GPU approval:** captures occupy the GPU; run them only with `budget.approved: true` in `intake.yaml` or an explicit yes to the stated cost. Analysing existing files needs none.

## Route before capturing

Read `doctor`'s `data.hardware.platform`/`data.profiling_route` and the relevant [hardware reference](../inference-triage/references/hardware.md). NVIDIA: nsys → ncu. AMD: rocprofv3 HIP/kernel/copy timeline → architecture-supported counters (Instinct support does not establish Radeon support). Apple: Instruments CPU/Metal → Xcode/MLX capture. CPU: platform profiler. nsys/ncu capture is NOT RUN elsewhere; profiling itself still applies.

**Retrieve current official docs online** for unknown architectures, runtime/tool support, flags or errors before choosing a path; record URL/date/version and check local help. Use AMD ROCm, Apple Developer/MLX and NVIDIA primary sources. Retain timeline/op attribution when counters fail, marking unsupported checks NOT RUN. All platforms follow critical path → Amdahl → change → correctness → unprofiled interleaved A/B, with the same [environment hygiene](references/capture-hygiene.md). Include text encoding, image VAE, audio codec/vocoder and buffering where applicable.

## Preflight checklist

- [ ] Steady-state work for a bounded number of steps (load and warmup rank the wrong things).
- [ ] Weights downloaded, no credentials needed (`HF_HUB_OFFLINE=1`): the capture must hold no tokens.
- [ ] You know whether the engine uses CUDA graphs and programmatic dependent launch (PDL): both change how the timeline reads.
- [ ] For ncu: counters readable ("Fallbacks"), measured bandwidth (`inferencekit ceiling bw`), memory headroom for replay.
- [ ] Asked whether the person already has profiles of this workload (no GPU, no approval needed).

## Hard failure gates

- **Never capture outside `env -i` with an allowlisted environment** (`inferencekit nsys capture` / `ncu capture` do it). Why: a `.nsys-rep` stores the full environment, tokens included.
- **Never rank by summed kernel duration.** Rank by critical-path time per step (`inferencekit nsys analyze` -> `data.steps.families.*.critical_pct_of_step`, GPU idle in `data.steps.idle_pct_of_step`). `kernel_sum_us / span_us > 1` means kernels overlap and duration shares are inflated.
- **Never take host-side costs (`cudaGraphLaunch` time) from a node-traced capture.** Why: node tracing inflated it about 25x in one measured case.
- **Never use an unanchored ncu regex** or a `--launch-skip` landing in warmup, autotuning, graph capture or prefill (the wrong launch).
- **Never quote timeline, trace or ncu durations as step time or speedup.** Why: tracing adds overhead; ncu serialises kernels, flushes caches and removes PDL overlap.

## NVIDIA workflow (copyable)

```
Profile:
- [ ] 1. Benchmark number and ceiling in hand (ms/token or TTFT, % of ceiling)
- [ ] 2. inferencekit nsys capture -- <steady command>          -> ev_cap (.nsys-rep, .sqlite)
- [ ] 3. Check the record: trace mode (node/graph), env allowlist, capture window, limitations
- [ ] 4. inferencekit nsys analyze <rep|sqlite>                 -> ev_tl
- [ ] 5. Profile check: per-step wall p50 vs the benchmark's ms/token (within a few %?)
- [ ] 6. kernel_sum / span: > 1 means overlap; use critical-path shares only
- [ ] 7. Answer the six questions, each with a number
- [ ] 8. One family dominates and is far from speed of light -> ncu on it (counters blocked: NOT RUN, see "Fallbacks"):
         inferencekit ncu capture --kernel-regex '^<name>$' --launch-skip <N_pre + 2k> --count 5 -- <cmd>  -> ev_ncap
         inferencekit ncu summarize <.ncu-rep> --peak-gbs <ev_bw>                                      -> ev_k
- [ ] 9. Bound label per kernel; speed-of-light gap; Amdahl bound of the proposed change
- [ ] 10. Fill the fixed tables; every number observed/derived/inferred
```

## Exact commands

```bash
inferencekit nsys capture -- python run_decode.py --model ./models/qwen3-8b-fp8 --gen-len 128
inferencekit nsys analyze <.nsys-rep or .sqlite path from the capture record's artifacts>
inferencekit ncu capture --kernel-regex '^fp8_block_gemv_kernel$' --launch-skip 400 --count 5 \
  -- python run_decode.py --model ./models/qwen3-8b-fp8 --gen-len 64
inferencekit ncu summarize <.ncu-rep from the capture record> --peak-gbs <ev_bw>
inferencekit evidence show <ev_id>
```

MCP tools: `capture_nsys`, `analyze_nsys`, `capture_ncu`, `summarize_ncu`, `capture_rocprof`, `analyze_rocprof`, `inspect_evidence` (in OpenCode, run captures through the CLI, detached). Options: `--help`. Hygiene and manual `env -i` recipes: [references/capture-hygiene.md](references/capture-hygiene.md).

## Fallbacks: counters blocked, no nsys, supplied profiles, other hardware

Other skills link here for these rules.
1. **Check the counters.** `doctor`'s `data.profiling_counters.status`, from the driver setting and the process's capabilities (not root: a container's root usually lacks CAP_SYS_ADMIN): `allowed`, `admin_only` (no ncu, unless its reason says this process has CAP_SYS_ADMIN: then probe) or `unknown`. `inferencekit doctor --probe-profilers` settles it with a tiny nsys and ncu capture, or the first small `ncu capture` does (`counters` in its `coverage.missing`: blocked). Containers and many cloud instances block them, A100 and H100 included: a host setting, not the GPU model.
2. **Blocked: nsys only** (it needs no counters): the whole timeline half (steps 1-7); every ncu row **NOT RUN** ("GPU counters blocked on this host", doctor evidence id), and the report says, in these words: "ncu was not possible on this host; profiling used nsys timelines only"; the bound label only from arithmetic (bytes a kernel must read ÷ its in-step time vs measured bandwidth), **inferred, no counters**.
3. **No nsys either, or you want PyTorch-op attribution: the PyTorch profiler** (no counters, no Nsight): vLLM `--profiler-config` with `/start_profile` and `/stop_profile`, SGLang `/start_profile`, `torch.profiler` in a custom engine. The CLI does not parse its Chrome/Perfetto traces: read kernel time per step and GPU idle gaps by hand or with `profile-analyst`. It slows the run: never time with it on ([references/torch-profiler.md](references/torch-profiler.md)).
4. **Ask for the person's own profiles** captured where they have permission (`profiles: <path>, ...` in `constraints`; non_interactive: a `QUESTIONS.md` request): `inferencekit nsys analyze <file.nsys-rep or .sqlite>`, `inferencekit ncu summarize <file.ncu-rep> --peak-gbs <ev_bw or GB/s>` (no GPU; without the ncu CLI, CSV exports: [references/ncu-capture-recipes.md](references/ncu-capture-recipes.md)). A profile from another GPU or engine version explains that setup only: say so by every number. Never copy, commit or upload such files (they may hold secrets); cite the analysis.
5. **Not NVIDIA:** nsys and ncu rows are **NOT RUN**. On AMD use `inferencekit rocprof capture` and `rocprof analyze` for kernel/HIP traces, or a supported vendor profiler; rocprof-compute support depends on the exact GPU ([amd-rocm.md](../inference-triage/references/amd-rocm.md)). Use perf or likwid on CPU, Instruments or MLX on Apple, under `env -i`. Vendor captures outside the CLI are cited by file; kernel totals may overlap and the bound label stays **inferred**.

## Reading the timeline

- **Node vs graph tracing.** `--cuda-graph-trace=graph` shows realistic host timings (host costs, CPU vs GPU critical path); `node` shows every kernel inside graphs (ranking, gaps) but inflates `cudaGraphLaunch`. `inferencekit nsys capture` traces nodes; for host costs add one graph-level capture ([references/capture-hygiene.md](references/capture-hygiene.md)).
- **Critical path, not summed kernel time.** Under PDL (and multiple streams) the next kernel starts early and waits on the GPU, so durations include waiting. When the per-step overlap `kernel_sum_us - busy_us` is about one family's summed duration, that family is mostly hidden: size changes with `critical_pct_of_step` ([lessons](../optimizing-kernels/references/lessons.md), "Summed time lies under PDL").
- **Profile check.** The profiled token period (step wall plus the gap to the next step) vs the benchmark's ms/token: a few percent of inflation is normal; more means the profile measures something else (often the tracer): fix the window or trace mode first.

## The six questions (top-down, each with a number)

1. **Does this stage matter?** Its share of per-token (or per-prefill) wall time. 2. **CPU or GPU on the critical path?** GPU idle gaps and blocking calls (`cudaStreamSynchronize`, D2H of the sampled token) mean the CPU is. 3. **Which stage?** Attention, MLP, lm_head or sampling, by critical-path family. 4. **A few big kernels or a long tail?** A long tail of microsecond kernels means hand-off bound. 5. **Speed-of-light gap?** Bytes a big kernel must read / measured bandwidth vs its in-step time. 6. **Amdahl bound?** `1 / ((1 - f) + f / s)` for critical-path fraction `f` made `s` times faster. Traps and SQL on the `.sqlite`: [references/six-questions.md](references/six-questions.md).

## ncu: labelling the bound

| Label | Evidence pattern (heuristic thresholds) | Lever |
|---|---|---|
| memory-bound | DRAM throughput >= ~70% of the **measured** peak; SM pipes well below theirs | read fewer bytes (precision, fusion that avoids re-reads) |
| compute-bound | a pipe (tensor, FMA, ALU) >= ~70% of its peak; DRAM well below | fewer instructions (better maths path, tensor cores, cheaper dequant) |
| latency-bound | DRAM and SM both below ~50%; few eligible warps; long-scoreboard / barrier stalls; grid below one wave or low occupancy | more bytes in flight: more blocks (split-K), wider loads, fewer dependent steps |

The label is **inferred**: a hypothesis, not a verdict. Counters suggest; in-place timing decides ([lessons](../optimizing-kernels/references/lessons.md)). Capture recipes: [references/ncu-capture-recipes.md](references/ncu-capture-recipes.md); metrics and the L2-resident and nominal-peak traps: [references/ncu-metrics.md](references/ncu-metrics.md).

**Evidence kinds:** `nsys_capture` -> `nsys_timeline`; `ncu_capture` -> `ncu_kernels`; `rocprof_capture` -> `rocprof_summary`. Profile files hold host names, paths and command lines: never upload one unless the user checked it.

## Fixed result tables

| Question | Answer | Number | Class | Evidence |
|---|---|---|---|---|
| Profile vs benchmark (step ms vs ms/token) | | % difference | derived | ev_tl, ev_bench |
| kernel_sum / span | | ratio | derived | ev_tl |
| 1-6 (as above) | | | observed / derived | ev_tl, ev_bw |

| Kernel (anchored) | Launches profiled | Grid, waves | ncu duration us (not e2e) | DRAM GB/s | % of measured peak | SM % | Occupancy | Top 2 stalls | Bound (inferred) | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|

## Decision and reporting rules

- Idle above ~10% of the step: fix the host side (CUDA graphs, async scheduling, GPU-side sampling) before kernels, which would only lengthen the gaps. Memory-bound at >= ~85% of measured peak: stop tuning that kernel. Latency-bound with waves below 1: add parallelism (split-K, smaller tiles) first.
- Amdahl bound below the benchmark's CI half-width: do not start (you cannot prove the win). Stalls from kernels under ~10 us or one launch are qualitative only.
- Timeline and counter values are **observed** under tracing or replay; say so. Shares, Amdahl bounds, % of measured peak and speed-of-light gaps are **derived** (cite ev_bw). "Launch-bound", "memory-bound" are **inferred** until an A/B supports the change they motivate.
- A question the capture cannot answer (host costs from a node-only trace; % of peak without measured bandwidth) is UNPROVEN, naming the capture that would answer it. Every ncu duration says "ncu replay time, not end-to-end".
