# CPU inference

Read [hardware.md](hardware.md) first. Use `doctor` for CPU ISA, physical/SMT cores, L3 groups, raw hybrid core types, NUMA, affinity, governor, perf policy and host load. Unknown facts stay unknown. Host load is a contention warning, not per-core utilization. ISA support and package presence do not prove BLAS/oneDNN dispatch: retain loaded-library and runtime dispatch logs, otherwise mark it UNPROVEN.

## Freeze a tuned baseline

Screen thread counts × placement for the declared workload and lead metric before fixing a CPU baseline. Start with physical cores, then test SMT; defaults are not a tuned baseline. Prepare one manifest whose engine argv uses `{threads}` and `{port}`:

```bash
inferencekit cpu screen-manifest baseline.json --out cpu-screen.json \
  --threads 1 --threads 2 --threads 4 --cpus 0-7
```

Choose CPUs from actual topology and reserve others for the client. The helper launches nothing: it generates no-binding, compact and spread variants inside the selected CPU pool, with `OMP_PLACES=cores`. Use `--thread-env OMP_NUM_THREADS` instead of `{threads}` when appropriate; `--include-smt` permits siblings. Run the generated manifest with `bench rounds`, verify effective settings, select the best screen configuration, then collect independent final comparisons. Pinning controls affinity, not NUMA memory policy.

Use manifest `hardware.backend: cpu` and `hardware.cpu.model`. CPU rounds use an advisory CPU lock, record requested runtime controls plus observed OS thread counts/affinity, and sample process-group RSS. Requested counts do not prove effective runtime pools. RSS may miss peaks and counts shared pages per process; it is not VRAM, swap or a leak test. CPU load/capacity mode is unsupported.

## Engine selection and launch

| Candidate | What to verify |
|---|---|
| llama.cpp | CPU build, GGUF artifact hash and conversion command; installed server help and startup log. |
| vLLM CPU | Pinned build/image, supported ISA/precision, KV cache and actual thread binding. |
| OpenVINO Model Server | IR export/quantization identity and endpoint compatibility: `bench serve` needs `/v1/chat/completions` or `/v1/completions`. |
| Other engines | Research current CPU support with the exact CPU and ISA before exclusion; a launch decides support. |

Ollama/LM Studio wrappers matter when they are the user's baseline. Do not start new work on archived IPEX projects; TensorRT-LLM is excluded on CPU hardware. Format changes are separate tracks unless explicitly permitted: a BF16 GGUF can match BF16 weights, while Q8_0 and an int8 IR export are not the same format.

Keep the client off the engine cores. Engines inherit runner affinity, so launch them with explicit `taskset`, `numactl` or verified runtime binding. Record CPU and NUMA memory placement. Start with one logical CPU per core and measure SMT, rather than assuming it helps. Explicit ports and clean environment are required; container launch wrappers must stop the container on exit.

For llama.cpp retain build SIMD flags, thread counts, context, KV types and slots. For vLLM retain dtype, KV space and binding logs; inspect thread placement when logs omit it. Existing GPU-specific log gates may flag missing CUDA/offload on CPU: identify those as not applicable, preserve every other failure. Do not dismiss actual correctness failures.

## Profiling

Match the timed model, inputs, threads, affinity and NUMA placement. `doctor` reads policy without testing counter access. Use user-space `perf record`/`perf stat` when permitted; paranoid ≤2 or CAP_PERFMON suggests access but container policy may still block it. Preserve errors and fall back to engine per-layer timers or `torch.profiler` with `ProfilerActivity.CPU`. Do not change sysctls/capabilities to obtain access.

Use downloaded offline weights and an allowlisted environment. Warm up, bound the capture, keep stack/shape collection off unless needed, and retain traces locally. Inspect allocation, barriers/spin, placement, library dispatch and page faults. Summed operation time can overlap across threads; it is not wall-clock critical-path time. Profiles explain mechanisms; turn instrumentation off for independent timed proof.

## Bandwidth and host controls

Measure read bandwidth across the sockets used by the engine, preferably with a supported read-only LIKWID or MLC kernel. STREAM is a fallback: arrays exceed 4× total LLC; Triad counts 24 bytes/iteration and is not read-only bandwidth, so qualify the ceiling. Datasheet bandwidth remains an estimate. Feed measured GB/s to `roofline`; unsupported GGUF formats need explicit tensor/KV byte accounting.

Record governor/boost and ask before changes. Keep other work off engine cores. Respect NUMA capacity and runtime constraints; container permission changes need their own authorization.

Detailed launch commands, profiler examples, bandwidth arithmetic and primary sources: [CPU benchmark guide](https://github.com/KittenML/inferencekit/blob/main/docs/cpu-benchmarks.md). Check installed help and current official engine documentation before using version-specific flags.
