# Follow the code that executes

Load after a profile identifies an expensive operation, before writing its replacement. Use source and supported tools on the installed runtime; do not require a GUI or privileged counters when headless evidence is available.

1. Pin model/runtime/compiler commits, device target, dtype, shape, strides and graph mode. Reproduce one representative call, including awkward sizes.
2. Follow the Python/module call into the compiled graph. Identify graph breaks, guards/recompilation, fallback operators, layout copies and precision conversions.
3. Follow dispatch to the loaded library or generated kernel. Record operator, symbol, source/cache artifact and compile options. Prove the selected specialization ran; a source edit or successful build is insufficient.
4. Compare generated IR/code with the hypothesis. Investigate register spills, redundant loads/conversions, missed vectorization and synchronization. Check whether fusion already occurred and whether it introduced pressure or repeated work.
5. Change one suspected cause behind a selectable path, check correctness under the run's policy, and measure the complete inference path. Dumping code, profiling and debug flags stay outside published timing.

## PyTorch and Triton example

Check flags against the installed [PyTorch compiler documentation](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) and [Triton developer README](https://github.com/triton-lang/triton). These commands assume a small local `repro.py` you have inspected; it must exercise the actual selected operation.

```bash
# Compiler inspection, not a timed run.
TORCH_LOGS="graph_breaks,recompiles,output_code" python repro.py
# A fresh per-experiment cache forces compilation without deleting shared caches.
ik_inspect_dir="$(mktemp -d)"
TRITON_CACHE_DIR="$ik_inspect_dir/cache" TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR="$ik_inspect_dir/dump" python repro.py
```

Inspect emitted TTIR, TTGIR, LLVM IR and final PTX/AMDGCN where that backend exposes them. For a narrower pass investigation, supported versions accept `MLIR_ENABLE_DUMP=<kernel-name>` and `MLIR_DUMP_PATH=<file>`; `LLVM_IR_ENABLE_DUMP=1` emits LLVM stages. Use a fresh experiment cache if compilation would otherwise be skipped. Preserve source/kernel hashes and a small relevant excerpt, not a whole compiler log in context.

| Backend | Inspect and change | Constraints |
|---|---|---|
| NVIDIA | PTX then actual SASS from the loaded cubin (`cuobjdump --dump-sass <binary>`); ptxas resource reports and supported Nsight attribution. Adjust tiling, layout, staging or fusion before hand-editing assembly. | PTX is virtual ISA; it does not by itself prove issued instructions. Capture target SM and registers/shared memory. |
| AMD | Generated AMDGCN/LLVM for the exact gfx target, supported ROCm disassembly and rocprofv3 attribution. Check VGPR spills, LDS traffic, vector loads and MFMA/WMMA selection. | CDNA and RDNA differ. Verify installed tool help and target support; never prescribe NVIDIA disassemblers or counters. |
| Apple | Loaded Metal pipeline, generated shader/source where available, MLX dispatch and Instruments/Metal capture. Inspect threadgroup memory, SIMD layout, lazy evaluation and CPU/GPU synchronization. | Use [Metal tools](https://developer.apple.com/metal/tools/) and [MLX custom kernels](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/custom_metal_kernels.rst). Public ISA/IR visibility may be limited; record the gap rather than promise PTX-like access. |
| CPU | Actual vectorized library/compiled loop, compiler optimization remarks, supported disassembly and a CPU profile. | Match ISA and thread/NUMA configuration; scalar fallback, oversubscription and memory traffic can dominate. |

IR overrides and assembly are escalation tools. Supported Triton versions offer `TRITON_KERNEL_OVERRIDE=1` with `TRITON_OVERRIDE_DIR`; first inspect that revision's loader/naming rules and copy the selected dumped artifact into a private experiment directory. Change one stage, verify a changed artifact hash and executed symbol, then test. Keep the original path and pin compiler/target; an override that is silently ignored proves nothing. Never leave override environment variables enabled in unrelated runs.

## Evidence and stopping

Record source call → graph operator → dispatched symbol → generated artifact, the suspected limiter, one change and its measured result. Check irregular shapes, boundaries, dtypes and synchronization hazards appropriate to the change. A faster isolated kernel can hurt neighbors or add packing/dispatch overhead; only end-to-end rounds establish the gain.

If inspection shows the compiler already did the proposed transformation, reject duplicate work and update the [research queue](../../researching-optimizations/SKILL.md). Missing counters do not justify invented occupancy/traffic measurements.
