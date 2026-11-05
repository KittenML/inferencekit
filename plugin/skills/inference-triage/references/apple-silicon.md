# Apple Silicon: Metal, MLX and shared memory

Read [hardware routing](hardware.md). `doctor` records selected sysctl/system_profiler facts and labels capacity unified; it does not prove runtime support, contention or headroom. Verify the actual Mac/chip, RAM, OS, runtime and model architecture with official docs plus a pinned smoke request. Recipes were reviewed 27 Oct 2026; recheck installed help and support before use.

Save sanitized `sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu`, `system_profiler SPDisplaysDataType`, `pmset -g`, runtime/compiler versions, memory pressure and swap facts in the hardware card. CPU and GPU share memory; the capacity is not dedicated VRAM. Do not infer the bottleneck or bandwidth from the chip name alone.

## Runtimes and launch recipes

| Runtime | Starting point | Verify |
|---|---|---|
| MLX / mlx-lm for text | [MLX](https://github.com/ml-explore/mlx), [mlx-lm server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) | Supported architecture, resolved dtype/group size, batching, cache and attention paths. The current server docs warn against production deployment; record that limitation for deployment claims. |
| llama.cpp Metal | [Build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#metal-build) | Metal is enabled by default on macOS builds; startup logs must prove the active GPU backend and residency. GGUF conversion commands, revisions and hashes belong in the manifest. |
| Other text engines, including Ollama, vLLM or SGLang | Current official Apple/Metal implementation, release and supported-model docs | Discover the actual backend and launch a smoke request before including or excluding it. A wrapper can still be a relevant baseline; an experimental backend is not automatically production-ready. |
| Image, audio, video or other models | The model's MLX, Metal, MPS or native implementation | Operator fallbacks, preprocessing/decoder placement and supported shapes. Use the modality adapter, not text token timing. |

From a pinned llama.cpp source checkout, build with bounded host parallelism:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel "$IK_BUILD_JOBS"

# Text-serving starting points, with already-downloaded local weights.
mlx_lm.server --model <local-model-dir> --port 8080
llama-server -m <model.gguf> -ngl 999 -fa on -c <context> -np 1 --port 8003 --seed 0
```

Check flags in the installed release. Tune concurrency, cache sizing and attention for the same workload before comparison; these examples do not establish the best settings.

- **Exact text output length:** `bench serve` with `--engine-kind generic` sends no engine-specific exact-length field. If a pinned MLX server stops at end-of-sequence before the required length, record the failed gate; do not silently shorten the candidate's work or bypass the gate.
- **Prompt caching:** verify the text caching gate and resolved server behavior. A cache-enabled task needs a declared comparable contract. Cache reuse is not a free improvement in a cold-prefix-cache comparison.
- **Identity:** when discovery is incomplete, use a verified override, for example `--serve-args "--gpu-name 'M3 Ultra 80-core GPU' --gpu-count 1"` on `bench rounds`, replacing the name with the actual chip. Match intake and REPORT.md to the hardware card.
- **Quality reference:** `quality score` needs a reference with vLLM prompt-logprob support; an MLX/llama.cpp endpoint is not sufficient. A reference may run on another documented host. If unavailable, mark that check NOT RUN; token identity/GSM8K alone do not establish its PASS. Other modalities require their own quality checks.

## Prove settings and numerics

For MLX, retain startup arguments, loaded `config.json`, checkpoint/conversion hashes and resolved quantization modules/group size; inspect the actual execution path for CPU fallback. For llama.cpp, retain the Metal `system_info`, buffer allocation, `offloaded N/N layers`, flash-attention and KV-type lines. All layers must be resident for the standard full-residency comparison. An authorized partial-offload task needs its own declared checks, not a silently overridden gate.

MLX affine 4/8-bit and GGUF Q4_K/Q8_0 are different formats. Record weights, activations and cache precision separately; same bit count is not same numerics. Follow the chosen quality policy and label cross-precision comparisons. Pin conversion commands and resulting artifact hashes.

In the current [mlx-lm server instructions](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md), quantized KV cache disables batching; a large prefill step can also dominate memory. Verify the installed behavior of `--kv-bits` and `--prefill-step-size` before crediting a memory or throughput improvement. Do not reuse old concurrency defaults without checking help/logs.

## Profilers and interpretation

Every capture uses `env -i` with reviewed PATH/HOME/TMPDIR, required Metal variables and offline cache settings; weights must already be downloaded and tokens absent. Apply [capture hygiene](../../profiling/references/capture-hygiene.md) to subprocesses; keep traces private. nsys/ncu capture remain NVIDIA-only.

| Question | Tool and starting command |
|---|---|
| CPU encoding, GPU execution, copies, blocking or gaps? | Enumerate templates with `xcrun xctrace list templates`, then, if installed, `xcrun xctrace record --template 'Metal System Trace' --attach <pid> --time-limit 30s --output step.trace`. |
| Which kernels ran and what limits them? | [MLX Metal capture](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html), opened in Xcode; GPU counters/replay where supported. |
| GPU activity, power or thermal behavior? | Optional authorized telemetry: `sudo powermetrics --samplers gpu_power -i 1000 -n 30`; confirm sampler availability in installed help. |

Read the trace with the [six profiling questions](../../profiling/SKILL.md). Select the actual task interval, then correlate CPU encode/commit/wait with command buffers and GPU work. Gaps suggest host or dependency delays; confirm the cause instead of applying a universal idle-percentage threshold. Low GPU residency is a clue, not proof. Rank critical-path kernels, memory traffic and synchronization rather than summing overlapping durations.

Instruments can include system-wide work: filter to the inference process but account for WindowServer and other contenders when assessing interference. Compare traced step spacing with the unprofiled workload. Counter/replay durations explain mechanism and never replace unprofiled end-to-end timing; unsupported counters are NOT RUN.

For a bounded MLX capture, adapt this helper to the actual warmed step (it returns arrays or a supported nested container). Launch with `MTL_CAPTURE_ENABLED=1` in the clean environment. Choose a trace path that does not already exist.

```python
import mlx.core as mx

def capture_step(step, trace_path):
    # step must execute the same model, precision, shapes and state as the target.
    for _ in range(3):
        mx.eval(step())
    mx.synchronize()
    mx.metal.start_capture(trace_path)
    try:
        mx.eval(step())
        mx.synchronize()
    finally:
        mx.metal.stop_capture()
```

Inspect dispatch order, dependencies, threadgroup layout and available memory/occupancy counters in Xcode. The [MLX debugger guide](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html) describes a source build with `CMAKE_ARGS="-DMLX_METAL_DEBUG=ON"` for richer labels. Record the debug build; do not mix its timings with the release engine. A script explains the server only when model/settings/work match; explicitly label that distinction.

## Consumer memory and sustained performance

Use [affordability](../../optimizing-inference/references/affordability.md) for 16/24 GB limits and a baseline tuned to the same memory/latency/quality envelope. Record peak allocator usage with the installed [MLX memory API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html), resetting its peak before each case. It omits OS, other processes and some runtime allocations; do not add overlapping measures as if they were independent.

Capture `vm_stat`, `sysctl vm.swapusage` and memory-pressure observations before, during and after each case. Report swap deltas and pressure, not just allocated tensor bytes. Swapping invalidates an ordinary no-swap fit claim; a permitted offload/paging result must declare that condition and include its latency.

macOS GPU wired-memory limits vary by OS/runtime. Inspect before changing; do not automatically raise `iogpu.wired_limit_mb`. An authorized adjustment must reserve OS/runtime headroom, be recorded, and be restored where appropriate. API allocator limits are not evidence of physical free memory.

Record power source/mode (`pmset -g`) and thermal behavior. On laptops, a stable plugged-in configuration may be appropriate; make power-setting or competing-app changes only within authorization. Do not kill unrelated processes. Serialize measurements on the Mac and bound compilation concurrency by available RAM. Sustained throttling, shared CPU/GPU bandwidth and other activity affect the result; balanced rounds help expose drift but do not erase contention.

## Read bandwidth and lazy evaluation

`inferencekit ceiling bw` is NOT RUN on Apple. Use a measured read-dominant MLX/Metal workload; [MLX is lazy](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), so construct fresh work each sample, materialize it and synchronize at timing boundaries. Re-evaluating an already evaluated result can time no work.

This standalone example measures a reduction's attained input-read rate, including launch/reduction overhead. Run it only with exclusive device ownership and an approved memory budget. Choose `--array-mib` at least 4x the relevant system cache while allowing scratch, runtime and OS headroom; try multiple safe sizes and disclose unknown cache capacity. There is deliberately no automatic large allocation.

```python
import argparse
import json
import math
import statistics
import time
import mlx.core as mx

parser = argparse.ArgumentParser()
parser.add_argument("--array-mib", type=int, required=True)
parser.add_argument("--samples", type=int, default=20)
args = parser.parse_args()
if args.array_mib < 1 or args.samples < 3:
    parser.error("use a positive array size and at least three samples")

mx.set_default_device(mx.gpu)
mx.random.seed(0)
n = args.array_mib * 1024 * 1024 // 4
x = mx.random.uniform(shape=(n,), dtype=mx.float32)
mx.eval(x)
mx.synchronize()

def read_once():
    y = mx.sum(x)  # A new operation, not repeated evaluation of one cached result.
    mx.eval(y)
    mx.synchronize()
    return y

for _ in range(3):
    read_once()
times = []
for _ in range(args.samples):
    mx.synchronize()
    start = time.perf_counter()
    result = read_once()
    times.append(time.perf_counter() - start)
    if not math.isfinite(result.item()):
        raise RuntimeError("non-finite reduction result")
print(json.dumps({
    "input_bytes": n * 4, "seconds": times,
    "median_read_GB_s": n * 4 / statistics.median(times) / 1e9,
    "scope": "MLX reduction, synchronized, includes launch/reduction overhead",
}))
```

Save the script, output, exact command, MLX/macOS versions and memory/thermal observations under the run's hardware evidence. A reduction does arithmetic and small writes; this is not a guaranteed DRAM peak or a fixed percentage below one. Check the profile and size scaling for cache/dispatch effects. Report Copy/Triad or another native read probe alongside it when helpful.

Supply the observed GB/s to `inferencekit roofline --model <dir> --dtype <format> --gpu-bw <GB/s> --context <context>` and cite the raw file. A datasheet-only fallback needs the exact chip/configuration and primary source, with measured ceiling NOT RUN; do not copy an unverified M-series bandwidth table.

Additional primary references: [Apple GPU performance](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance), [CPU/Metal timeline analysis](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app), [MLX synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html). Search current runtime releases and other sources when the installed stack or bottleneck changes.
