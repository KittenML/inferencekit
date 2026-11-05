# Hardware routing and evidence

Read this before using AMD ROCm, Apple Metal or a CPU. `doctor` records selected observed OS facts and `data.profiling_route`; detection does not prove runtime support. Verify the exact device, OS, driver and engine against current official docs, then launch a pinned smoke request. Missing support or measurements stay NOT RUN/UNPROVEN.

| Platform | Identification | Timeline → kernel analysis | Bandwidth |
|---|---|---|---|
| NVIDIA | nvidia-smi; full doctor clock/process/counter checks | nsys → ncu when counters work | `ceiling bw` |
| AMD | DRM sysfs name/VRAM; rocminfo GPU agents as WSL fallback | rocprofv3 HIP/kernel/copy trace → supported architecture metrics | vendor read test or BabelStream |
| Apple | sysctl/system_profiler chip and shared capacity | Instruments CPU/Metal → Xcode Metal/MLX capture | read-dominant MLX/Metal test |
| CPU | basic architecture/core count; complete with lscpu/sysctl | perf/VTune/likwid or Instruments Time Profiler | STREAM-style read test |

Vendor details: [AMD](amd-rocm.md), [Apple](apple-silicon.md), [CPU](cpu.md), [Intel/TPU](other-accelerators.md). Unknown architecture, runtime, profiler, flag or error requires **online official-doc retrieval** before selection; record URL/date, installed version and the support matrix checked. Platform references are starting points, not durable support guarantees.

## CLI boundaries

- `doctor`: AMD/Apple selected facts, identity and profiling route; process contention, clocks, runtime and counter permissions are not automatically established there. AMD PCI-id fallback names are labelled unverified. Apple capacity is unified, never dedicated VRAM. WSL rocminfo heap sizes are not dedicated VRAM evidence. `/dev/dxg`/other accelerator interfaces without vendor facts route to unknown accelerator, never confidently CPU.
- `ceiling bw` and nsys/ncu **capture** remain NVIDIA-only, NOT RUN elsewhere. `doctor --probe-profilers` does not launch CUDA probes on positively detected AMD/Apple or unknown accelerator interfaces. Supplied NVIDIA profiles can be analysed anywhere with the needed analysis tool.
- `bench serve`, `bench rounds`, gates and token quality use HTTP and can test compatible text servers on any hardware; a first run still verifies the adapter. `roofline --gpu-bw <GB/s>`, intake, manifest, ledger and reports are hardware-neutral. `bench decode` off NVIDIA is untested. Text token timings are not image or audio latency; use the registered modality client and quality protocol ([affordability](../../optimizing-inference/references/affordability.md)).
- Evidence identity now uses discovered AMD/Apple names too. When missing/incomplete, pass a verified override (`--serve-args "--gpu-name MI300X --gpu-count 1"` on `bench rounds`, `--gpu-name` on `bench serve`) and cite the hardware card in REPORT.md. Do not fabricate a tool record for manual telemetry.
- `bench rounds` memory-return checking still uses nvidia-smi; elsewhere `gpu_memory.checked: false` requires a manual vendor process check after every engine. Check that killed container clients did not leave a running container.

## Hardware card and uncontended timing

Save sanitized vendor output in `runs/<run-id>/hw/`, with command and tool version. Remove hostnames, IPs, serials and UUIDs; cite these as **observed (vendor tool)** files. They are not automatically included in report/review.

| Platform | Complete the facts with |
|---|---|
| AMD | `amd-smi version`, `amd-smi static`, `amd-smi metric`, `amd-smi process`, `rocminfo`; WSL support checked separately |
| Apple | `sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu`, `system_profiler SPDisplaysDataType`, `pmset -g`, memory pressure/swap |
| CPU | `lscpu`, `numactl -H`, governor/boost, memory size; on macOS use sysctl |

Stop timing for another accelerator process, substantial CPU/I/O load, low-power mode, thermal throttling or swap pressure. Explicitly authorized independent devices can run in parallel with one owner/lock and separate state per device; no simultaneous measurements on one device or shared-host contention.

## Existing text-engine hard gates

A CPU host has no CUDA graphs or GPU layers. Only these exact existing gate reasons may be reported **not applicable on x86 CPU**, with the reason quoted next to the row:

- llama.cpp CPU build: `no GPU backend in this llama.cpp build (system_info)` and `only 0/N layers offloaded to the GPU`.
- CPU engine: `CUDA graphs off (...)`, with the actual reason from its log.
- vLLM CPU: `fallback attention backend (TORCH_SDPA)` when that is its documented CPU path.

Other HARD GATE failures remain FAIL; those reasons on NVIDIA/AMD/Metal do too. Partial GPU offload (`only K/N layers` with K>0) still fails the existing text comparison protocol. A task explicitly permitting offload needs its own declared residency/latency/quality contract and applicable checks; never silently override a gate or claim it passed. ROCm has HIP graphs and llama.cpp Metal/HIP builds count as GPU backends.

`quality score` needs a reference exposing vLLM prompt_logprobs. A llama.cpp/MLX server does not suffice; a higher-precision reference can be on another documented host. Otherwise that check is NOT RUN; token identity/GSM8K alone do not establish its PASS. Images/speech use modality checks instead of LLM proxies.

## Ceilings and profiles

Measure read-dominant bandwidth on the actual device with buffers >=4x its last-level/system cache, including AMD Infinity Cache and Apple system cache. Record command, version, buffers and output. Copy/triad includes writes and can flatter a decode roofline; label that limitation. Datasheet-only fallback means measured ceiling NOT RUN and percentage-of-ceiling UNPROVEN. Supply observed GB/s to `inferencekit roofline --model <dir> --dtype <fmt> --gpu-bw <GB/s> --context <ctx>` and cite the bandwidth file. Low throughput may instead be compute/dequantization bound: use the profile, not a bandwidth-only assumption.

Do not run `ceiling bw` on ROCm even if its PyTorch fallback starts: its NVIDIA-oriented buffer/cache handling is not verified for AMD Infinity Cache. A result from it is not an AMD ceiling.

Use [capture hygiene](../../profiling/references/capture-hygiene.md) on every vendor tool: offline weights, `env -i` allowlist, no secrets, private traces. Critical-path/interval-union attribution, supported counters and Amdahl explain mechanism; unprofiled interleaved A/B proves speed. Vendor telemetry is **observed (vendor tool)**; bound labels are **inferred**.
