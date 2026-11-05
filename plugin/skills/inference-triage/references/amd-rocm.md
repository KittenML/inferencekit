# AMD ROCm: Instinct and consumer Radeon

Read [hardware routing](hardware.md). Native Linux and WSL are different support targets. Retrieve **current official docs online** for unfamiliar architectures, tools or flags; record URL/access date, exact GPU/gfx, OS/kernel/driver, ROCm and tool versions. Installed help and resolved startup logs establish behavior; a smoke request establishes model support. Recipes below were reviewed 27 Oct 2026; pin versions rather than treating them as a permanent support matrix.

## Hardware and engine selection

`doctor` reads DRM sysfs for product/VRAM facts; on WSL it can fall back to rocminfo GPU-agent Name/Marketing Name. A ROCm installation or CPU agent is not proof of an AMD GPU, and rocminfo heaps are not proof of dedicated VRAM. WSL may expose `/dev/dxg` rather than `/dev/kfd` or DRM. Save sanitized `rocminfo`, `amd-smi version`, `amd-smi static`, `amd-smi metric` and `amd-smi process`; missing facts remain unknown.

These are text-serving examples. Image, audio and video work uses the actual model's supported runtime and modality adapter, including decoding/preprocessing in the timing contract.

| Engine | Starting point | Verify on the actual host |
|---|---|---|
| vLLM | [Official GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), pinned `vllm/vllm-openai-rocm` tag/digest | GPU/gfx support, compatible ROCm/driver, architecture, dtype and quantization. Do not copy an Instinct configuration onto Radeon. |
| SGLang | [AMD instructions](https://docs.sglang.io/docs/hardware-platforms/amd_gpu), matching ROCm image | AITER is conditional: enable `SGLANG_USE_AITER=1` only for a supported configuration. The documented gfx115x image keeps it off (`SGLANG_USE_AITER=0`); verify the current target-specific instructions. |
| llama.cpp | [HIP build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip) from a pinned revision | Build for the genuine gfx target, use a supported GGUF conversion, record hashes and layer residency. |
| Other runtimes | Official release/code plus the research skill | A hardware exclusion needs a cited reason; unsupported model/format claims need evidence. Do not assume a platform is covered because HIP imports successfully. |

[AITER](https://github.com/ROCm/aiter) offers attention, matrix, normalization, quantization and other kernels. Library presence does not prove dispatch or support: record which implementation actually ran. Do not use unverified gfx overrides to disguise unsupported hardware.

## Build and launch recipes

Replace placeholders with recorded values. Inspect the pinned version's help first; launch settings are candidates for tuning, not proof of the strongest baseline.

```bash
# From the pinned llama.cpp source; IK_GFX is the observed rocminfo target.
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS="$IK_GFX" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel "$IK_BUILD_JOBS"

# In each engine's compatible environment/container, with local offline weights.
vllm serve <model-dir> --host 0.0.0.0 --port 8000
python3 -m sglang.launch_server --model-path <model-dir> --port 30000 --disable-radix-cache
llama-server -m <model.gguf> -ngl 999 -c <context> -np 1 --port 8003 --seed 0
```

The SGLang line is a cold-prefix-cache text example; verify that flag in the pinned release. A permitted cache-enabled workload has its own declared contract. Use `inferencekit bench rounds` for served-text comparisons; SGLang's `sglang.bench_serving` is not the inferencekit harness. Tune batching, memory and graphs for the declared goal before timing.

**Container cleanup.** A killed Docker client can leave the container using the GPU. Use a wrapper such as this for manifest launch commands; pass only the reviewed Docker arguments for the host (device nodes/masks, ports, memory, offline model mounts and pinned image). Native Linux commonly needs `/dev/kfd` and selected `/dev/dri` access; WSL and profiler permissions need separate verification. Do not copy blanket privileged settings.

```bash
#!/usr/bin/env bash
# usage: ik-docker <label> <docker-run arguments...>
set -eu
name="ik-${1:?engine label required}-$$"
shift
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker run --rm --name "$name" "$@" &
wait "$!"
```

A trap cannot handle SIGKILL or host failure. The coordinator must also check the named container and vendor process telemetry after every engine; remove only its own leftover container before continuing. `bench rounds`' memory-return check still uses NVIDIA telemetry, so `gpu_memory.checked: false` is not a clean AMD device. WSL telemetry limitations must be recorded.

For missing/incomplete identity, pass a verified override such as `--serve-args "--gpu-name MI300X --gpu-count 1"` on `bench rounds`, using the actual name. Cite the hardware card in REPORT.md; an override is not automatic discovery.

## Prove settings in effect

- **vLLM:** save the resolved config (dtype, quantization, attention, graph mode) and selected-backend log, such as `Using ROCM_AITER_FA backend.` or `ROCM_AITER_MLA` for a supported path. Names and defaults move between versions. Confirm graph capture/replay and the selected kernels; omit `--enforce-eager` for a valid graph baseline. Eager execution is a labelled diagnostic or supported fallback, not a silent replacement for a faster valid baseline.
- **SGLang:** save the `server_args=ServerArgs(...)` output, resolved AITER/backend choice and graph-capture lines. An environment variable alone is insufficient evidence.
- **llama.cpp:** `system_info` must establish HIP/ROCm; save `offloaded N/N layers`, buffer allocation, attention and KV-type lines. The standard full-residency protocol requires all layers on device. An authorized partial-offload task needs its own residency/latency/quality contract; do not override a failed gate.
- **Precision:** record weights, activations, cache, scales, excluded modules and conversions. “FP8” or a bit count alone does not identify the arithmetic.

## Profilers and how to read them

Run captures under `env -i` with the [capture hygiene](../../profiling/references/capture-hygiene.md) allowlist, offline weights and no tokens. Review PATH/HOME/TMPDIR, required runtime/library paths and device masks; avoid conflicting HIP/ROCR/CUDA masks. Use bounded warmed steps with ROCTx/task markers. Keep traces private.

| Question | Tool and starting command |
|---|---|
| GPU gaps, host launches, copies or synchronization? | `rocprofv3 --sys-trace --output-format pftrace -d prof -- <steady-command>`; inspect the Perfetto trace locally. |
| Which kernels dominate? | `rocprofv3 --kernel-trace --stats --output-format csv -d prof -- <steady-command>` |
| Why is a dominant kernel slow? | `rocprof-compute profile -n step -- <steady-command>`, then `rocprof-compute analyze -p <emitted-workload-path>` |
| CPU stacks alongside whole-application activity? | `rocprof-sys-sample --output-format proto -- <steady-command>`; inspect the emitted Perfetto trace. |

Check installed help for formats and filters. Some releases default to a rocpd database; use that release's conversion commands when another format is needed. For rocprof-compute, `-k <kernel>` limits collection; roofline-only options and available metrics are version/device specific. Counter collection can replay kernels: its timing is diagnostic, not end-to-end evidence.

Read the trace with the [six profiling questions](../../profiling/SKILL.md):

1. **Mark the task interval.** Token step, image denoising/encoder/VAE, audio codec/vocoder or video stage as appropriate; keep preprocessing and final output visible.
2. **Inspect launches and gaps.** HIP API activity distinguishes graph launches from many small submissions. Substantial idle time suggests a host, dependency or synchronization problem; trace the cause rather than declaring a universal 10% threshold. Confirm that graphs cover the intended region.
3. **Rank kernels on the critical path.** Kernel statistics identify candidates, but summed durations overstate elapsed time when streams overlap. Use interval unions/step spacing and check profiler overhead against the unprofiled workload.
4. **Explain the kernel.** Supported memory, matrix/VALU, occupancy/wave and stall metrics distinguish bandwidth, compute and latency hypotheses. Speed-of-light panels may use theoretical peaks; convert to the measured bandwidth basis before comparing with a read ceiling.

`rocprofv3-avail` enumerates installed-device metrics. Check the exact [rocprof-compute accelerator matrix](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/reference/compatible-accelerators.html); support includes selected Instinct and APU targets and must not be inferred for every Radeon. Unsupported/failed counters are NOT RUN with tool/version/error; retain timeline/op attribution. Verify any engine's built-in profiler on ROCm instead of assuming CUDA documentation applies.

RDNA counter collection can require a stable performance state. Inspect the installed documentation and current setting; record and restore authorized changes. Do not copy permissive sysfs chmod recipes. WSL tracing/counter support is a separate check.

For CLI/MCP capture, `inferencekit rocprof capture --gpu-name '<verified GPU>' --keep-env HF_HUB_OFFLINE -- python model.py` collects kernel and HIP API CSV traces under an allowlisted environment. Use `inferencekit rocprof analyze <kernel-trace.csv>` for dispatch counts and summed durations. MCP operations are `capture_rocprof` and `analyze_rocprof`. Check the installed `rocprofv3` version and exact GPU support; a missing trace or nonzero exit is incomplete. Raw traces can contain sensitive data. CSV capture covers these trace types; newer ROCm recommends rocpd for broader tracing. Summed kernel durations can overlap and do not establish end-to-end speedup.

## Bandwidth, memory and tuning

`inferencekit ceiling bw` is NOT RUN on ROCm even when PyTorch calls the GPU `cuda`: NVIDIA buffer/cache assumptions do not establish an AMD read ceiling. Use a pinned vendor read test or [BabelStream HIP](https://github.com/UoB-HPC/BabelStream). From its pinned source:

```bash
cmake -S . -B build -DMODEL=hip -DCMAKE_CXX_COMPILER=hipcc
cmake --build build --parallel "$IK_BUILD_JOBS"
./build/hip-stream --device "$IK_DEVICE_INDEX" --arraysize "$IK_ARRAY_ELEMENTS" --numtimes 20
```

Choose each array at least 4x the largest relevant cache, including Infinity Cache, while fitting **all three arrays**, reduction scratch and OS/runtime headroom in the approved memory budget. Default double precision is eight bytes per element; record `--float` if chosen instead. Verify the selected device and validation output. Report Dot (two input reads plus reduction work/output) with Copy/Triad, buffer bytes, sample statistics, version and command. It is an attained read-dominant measurement, not a promise that every inference kernel reaches it. [rocm_bandwidth_test](https://github.com/ROCm/rocm_bandwidth_test) measures links/transfers, not a VRAM-read ceiling.

Supply the observed GB/s via `inferencekit roofline --model <dir> --dtype <format> --gpu-bw <GB/s> --context <context>`, citing the raw measurement. A datasheet fallback must identify the exact part/source and label measured ceiling NOT RUN; do not invent a measured percentage.

- **FP8 formats:** FNUZ and OCP encodings/scales differ; bit reinterpretation is not conversion. Verify the exact architecture's [precision support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html) and engine conversion path, record resolved format and converted-artifact hashes, and follow the quality policy. Equal labels across engines/devices do not establish equal numerics.
- **Autotuning:** PyTorch TunableOp requires separate tuning and measurement passes; set `PYTORCH_TUNABLEOP_TUNING=0` during measurement where supported, retain the resolved cache/config and prove it is loaded. Rebuilding/tuning is outside timed rounds.
- **Target-specific tuning:** investigate attention/AWQ backends, speculative depth, GEMM strides/layouts, NUMA/kernel-argument handling and collective topology in the installed engine and [AMD workload guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html). Treat old knob values or stride rules as hypotheses; prove the dispatch and benefit on this device. Host/power changes require existing authorization and recorded settings.
- **Consumer memory:** discrete VRAM and an APU's shared RAM are different capacities. Use [affordability](../../optimizing-inference/references/affordability.md), record pressure/swap and reserve headroom; do not sum overlapping allocations. Serialize device work and cap compilation parallelism to host RAM.

Profiler references: [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html), [rocprof-compute usage](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/use.html), [rocprof-sys output](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/understanding-rocprof-sys-output.html). Search beyond these starting points when the installed stack or bottleneck changes.
