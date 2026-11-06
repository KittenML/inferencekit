# x86 CPUs (no GPU)

Read [hardware reference](../plugin/skills/inference-triage/references/hardware.md) first (what the CLI does here, what is NOT RUN). Gathered from project docs (sources at the end, read 14 Oct 2026); the installed version's `--help` is the authority and its startup log the proof.

## Engines

| Engine | Notes |
|---|---|
| llama.cpp `llama-server` | The default build is CPU-only (`cmake -B build && cmake --build build --config Release -j`). GGUF weights: a converted artifact, so record its sha256 and the conversion command. |
| vLLM CPU | Image `vllm/vllm-openai-cpu:latest-x86_64` (pin a tag). AVX-512 recommended, AVX2 limited; AMD CPUs need Zen 4 or newer. FP32, FP16 and BF16. Building it needs `numpy<2`. |
| OpenVINO Model Server | Serves OpenVINO IR models (`--source_model`, `--model_repository_path`, `--rest_port`). An IR export, and especially an int4/int8 one, is a different artifact and often a different precision: a separate track unless `precision_allowed` names it. `bench serve` always calls `<base-url>/v1/chat/completions` (or `/v1/completions`): if the pinned version serves only another path, the harness cannot time it, so it is NOT RUN. |
| IPEX-LLM, intel-extension-for-pytorch | Archived on GitHub: do not start new work on them. |
| TensorRT-LLM | NVIDIA-only: excluded for the hardware, not NOT SUPPORTED (nothing was launched). |
| Any other engine (SGLang, LMDeploy, ...) | Not covered by this guide. Run `engine-scout` with the CPU model and ISA (AVX-512, AMX) in the query before excluding it: a launch decides. |

Wrappers around llama.cpp (Ollama, LM Studio) add a layer, not a candidate. Include one only as the baseline the person already runs.

## Freeze a tuned baseline

Before fixing a CPU baseline, screen thread counts × placement on the declared workload and lead metric. Start with physical cores, then screen SMT explicitly; default thread counts and a single placement are not a tuned baseline. `doctor` records visible ISA, core/SMT/L3/NUMA topology, raw hybrid core types when exposed, affinity, governor, perf policy and host load. Missing facts stay unknown; high host load is a contention warning, not per-core utilization.

Prepare one baseline manifest whose launch uses `{threads}` and `{port}` in its argv, then generate variants beside it:

```bash
inferencekit cpu screen-manifest baseline.json --out cpu-screen.json \
  --threads 1 --threads 2 --threads 4 --cpus 0-7
```

The helper launches nothing. It sets `OMP_PLACES=cores` and screens no worker binding, compact (`close`) and spread placement, all inside the selected CPU pool. Choose that pool from the actual topology and leave client cores outside it. Use `--thread-env OMP_NUM_THREADS` when the runtime uses that variable instead of an argv thread count; `--include-smt` allows siblings. Run the generated manifest with `bench rounds`, verify the resolved thread/affinity settings, and retain the best screen configuration for independent final comparisons. The helper does not set NUMA memory policy.

ISA support does not prove library dispatch. Record the loaded BLAS/oneDNN build and runtime dispatch logs where supported; oneMKL's selected path cannot be inferred from the CPU vendor or package presence. Keep dispatch **UNPROVEN** when the engine provides no such evidence.

## Launching an OpenAI endpoint

**Core placement decides CPU results.** Give each engine an explicit core and memory placement in its launch line, and keep the benchmark client off those cores:
- Pin `bench rounds` to spare cores: `taskset -c <spare cores> inferencekit bench rounds ...`.
- Engines inherit that affinity from the runner, so every launch line must set its own: `numactl --physcpubind=<engine cores> --membind=<node> <engine command>`, or the engine's own binding (below).
- Start with one logical CPU per physical core, then use the screen to test SMT siblings for this workload.

```bash
# llama.cpp on NUMA node 0, physical cores 0-29 (leave the rest for the client and the OS)
numactl --physcpubind=0-29 --membind=0 llama-server -m <model.gguf> -t 30 -tb 30 -c <ctx> -np 1 \
  --port 8003 --seed 0 --metrics
```
Pass `--port` explicitly: the default has changed between releases. With more than one NUMA node, see `--numa distribute|isolate|numactl` in the server's help, and drop the page cache before switching modes.

```bash
# vLLM CPU: tensor parallel = number of NUMA nodes; leave 1-2 cores per node for the server
docker run --rm --network host --security-opt seccomp=unconfined --cap-add SYS_NICE \
  -v <models dir>:/models -e VLLM_CPU_KVCACHE_SPACE=<GiB> -e VLLM_CPU_OMP_THREADS_BIND=<cores, e.g. 0-30> \
  vllm/vllm-openai-cpu:<pinned tag> --model /models/<dir> --served-model-name <name> --port 8001 \
  --dtype bfloat16 --seed 0 --no-enable-prefix-caching
```
Under `bench rounds`, launch the container through a wrapper that stops it on exit ([container cleanup](../plugin/skills/inference-triage/references/amd-rocm.md), "Containers under `bench rounds`"). A venv install needs `LD_PRELOAD` of tcmalloc and libiomp5, as the vLLM CPU docs describe.

- **Identity:** use `hardware.backend: cpu` and `hardware.cpu.model` in the run manifest. CPU rounds record a CPU identity, not a GPU name; direct media clients use `--device cpu --hardware-name` with the server CPU model.
- **Price:** `hardware.price_usd_h` is the whole host's price. CPU load/capacity mode remains unsupported; do not label CPU throughput per GPU.

## Proving settings in effect

- **llama.cpp:** `system_info` (the SIMD flags it was built with, e.g. AVX512, AMX), the thread counts, `n_ctx`, `type_k`/`type_v`, and the slot count.
- **Expected HARD GATE lines.** On a CPU host, `bench rounds` tags rows `[HARD GATE]` for GPU settings that cannot apply: llama.cpp's `no GPU backend in this llama.cpp build (system_info)` and `only 0/N layers offloaded to the GPU`, vLLM CPU's `CUDA graphs off (...)`, and `fallback attention backend (TORCH_SDPA)` if its log names that backend. Report each as "not applicable on x86 CPU" next to the row. Any other HARD GATE is still a FAIL ([hardware reference](../plugin/skills/inference-triage/references/hardware.md), note 3).
- **vLLM CPU:** the engine config line (dtype, KV cache space) and the thread-binding lines it prints for `VLLM_CPU_OMP_THREADS_BIND`. If the binding is not printed, check the threads' placement while it runs: `ps -L -o pid,lwp,psr,comm -p <pid>` (`psr` is the core each thread is on).
- **Placement:** record the `numactl`/`taskset` line and `numactl -H` in `runs/<run-id>/hw/`.

## Profilers and how to read them

Run them under `env -i` with an allowlist ([hardware reference](../plugin/skills/inference-triage/references/hardware.md), section 5). `doctor` reads `perf_event_paranoid` and capabilities without probing counters: a policy that allows user-space events (commonly paranoid ≤ 2) does not prove container access. Use `perf` when permitted; otherwise use [engine timers or CPU-only torch.profiler](#cpu-profiling-fallback). Do not change host policy as part of profiling.

| Question | Tool | Command |
|---|---|---|
| How close to memory bandwidth is decode? | likwid-perfctr, MEM group | `likwid-perfctr -c <engine cores> -g MEM -S 30s` while steady decode runs: memory bandwidth in MBytes/s for those cores |
| Memory-bound or compute-bound? | perf stat | `perf stat -e cycles,instructions,cache-misses -p <engine pid> -- sleep 30` |
| Where does the time go (functions)? | perf record | `perf record -F 99 -g -p <engine pid> -- sleep 30`, then `perf report --no-children` or `perf script report flamegraph` |
| DRAM-bound share, bandwidth histogram, remote NUMA traffic | VTune memory-access | `vtune -collect memory-access -r vt -- <cmd>` |

Reading them with the profiling skill's six questions:
- **Bandwidth used.** likwid's MEM bandwidth during decode ÷ the measured peak (below) is the CPU's "% of peak" (derived). Near the peak, only fewer bytes per token help, as on a GPU.
- **Compute-bound decode.** Low memory bandwidth together with high instructions per cycle (`instructions / cycles`, roughly above 2) means compute-bound: dequantising Q4/Q8 blocks, or narrow vector units (AVX2 rather than AVX-512/AMX). The bandwidth ceiling is then not the binding one: say so, inferred.
- **Threading problems.** Large shares in OpenMP spin or barrier functions (`__kmp_*`, `gomp_*`) in `perf report` mean too many threads, SMT siblings, or uneven work.
- **NUMA.** VTune's "NUMA: % of Remote Accesses" should be near zero when each rank and its memory sit on one node.
- **Page faults** in the first requests mean the weights are memory-mapped and not resident yet. llama.cpp's `--mlock` keeps them in RAM; the harness's warm-up requests are not timed.

## Bandwidth and the ceiling

Measure all the memory the engine uses: every socket the engine runs on, with one thread per physical core.
- **Read-only first** (decode is read-dominant):
  - `likwid-bench -a` lists the kernels; use the widest read-only one (`load_avx512`, `load_avx` or `load`), one workgroup per socket: `likwid-bench -t load_avx512 -w S0:4GB -w S1:4GB`.
  - Intel MLC (Intel CPUs, needs root and the `msr` module): `mlc --peak_injection_bandwidth`, then quote the "ALL Reads" line. `mlc --bandwidth_matrix` shows bandwidth between NUMA nodes.
- **STREAM** when those are unavailable:
  ```bash
  gcc -O3 -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE=<elements> stream.c -o stream
  OMP_NUM_THREADS=<physical cores> OMP_PROC_BIND=spread OMP_PLACES=cores ./stream
  ```
  Each array must be at least 4x the sum of all last-level caches. STREAM counts Triad as 24 bytes per iteration and has no read-only kernel, so quote Triad and label it "read-only bandwidth not measured; % of ceiling may be overstated".
- **Datasheet fallback:** channels × transfer rate × 8 bytes per socket, e.g. 8 channels of DDR5-4800 = 307.2 GB/s (derived, labelled "datasheet-based; measured ceiling NOT RUN").
- Then `inferencekit roofline --model <dir> --dtype <fmt> --gpu-bw <GB/s> --context <ctx>` per bucket. `--dtype` covers bf16, fp8 and q8_0 only. For other GGUF types (Q4_K and so on), derive bytes per token by hand from the tensor sizes that `gguf-dump` (llama.cpp's `gguf` Python package) lists: every weight tensor once, the token-embedding table as one row per token, plus the KV term. Show the arithmetic (derived).

## Pitfalls

- **Each vLLM rank must fit on one NUMA node,** or it is killed with `exitcode 9`. Set tensor parallel to the number of NUMA nodes.
- **In Docker, `get_mempolicy` errors** mean the container needs `--cap-add SYS_NICE`.
- **Frequency and power.** Record the governor and boost state (`cpupower frequency-info`). Ask before changing them. A long run on a hot or power-capped host drifts, and the balanced round order spreads that drift over every engine.
- **Nothing else on the engine's cores,** the benchmark client included (above).
- **Same precision:** BF16 in vLLM CPU and a BF16 GGUF compare; Q8_0 and an OpenVINO int8 export do not (rule 8).

## Sources (read 14 Oct 2026)

- llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md; build: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- vLLM CPU: https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html
- OpenVINO Model Server LLM quickstart: https://github.com/openvinotoolkit/model_server/blob/main/docs/llm/quickstart.md
- IPEX-LLM: https://github.com/intel/ipex-llm; intel-extension-for-pytorch: https://github.com/intel/intel-extension-for-pytorch
- perf: https://perfwiki.github.io/main/tutorial/; VTune memory access: https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-0/memory-access-analysis.html
- likwid: https://github.com/RRZE-HPC/likwid/wiki/Likwid-Bench; STREAM: https://www.cs.virginia.edu/stream/ref.html; Intel MLC: https://www.intel.com/content/www/us/en/support/articles/000055898.html

# CPU profiling fallback

Use the [CPU thread and placement screen](#freeze-a-tuned-baseline) before fixing the baseline. Keep the same model, inputs, thread count, CPU affinity and NUMA placement for the diagnostic run. `doctor` supplies OS facts and a route; it does not test profiler access or establish library dispatch.

When Linux permits process-level events, use a short steady-state `perf record` for function attribution and `perf stat` for supported counters. Request user-space events when policy excludes kernel activity. `perf_event_paranoid ≤ 2` or CAP_PERFMON suggests permission; container/security policy can still refuse it. Check installed help, preserve the actual error, and use the fallback below if unavailable. Do not change sysctls or capabilities to make a capture work. [Kernel permission rules](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html).

When perf is blocked or missing, use existing engine per-layer timers or `torch.profiler` with `activities=[ProfilerActivity.CPU]`. Warm up first, bound the capture to representative steady requests, and keep stack/shape recording off unless needed. Timers cover only instrumented regions; tensor operation totals can overlap across threads, so they are not wall-clock critical-path shares. [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html).

Capture with downloaded offline weights and an allowlisted environment, retaining only the CPU runtime settings needed to match the benchmark. Logs/traces may include paths, commands and input data; keep them local under the run directory. Inspect thread placement, barriers/spin time, memory allocation and the loaded library/kernel dispatch. Missing counter or dispatch evidence remains NOT RUN or UNPROVEN.

Profiles explain a mechanism. Turn instrumentation off and use independent interleaved client-timed measurements for any speedup claim.
