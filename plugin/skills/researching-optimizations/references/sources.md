# Primary sources to revisit

These are starting points, not an exhaustive catalog or an approved-source boundary. This field changes quickly: actively search beyond this table for the latest relevant implementations and the largest credible improvements for the actual workload. Recheck releases and support against the installed stack; record access dates and chosen revisions.

Search the model architecture, expensive operators, input shapes, device/gfx/SM target and runtime/compiler versions. Check recent releases, merged PRs, issues with reproducers, original engineering blogs, conference/workshop material, papers with code and talks. Follow useful citations and implementation links to sources not listed here; search different terms if the first pass only repeats familiar projects. Community posts can reveal leads, but verify technical claims in primary sources.

Use a recent time window for discovery, then include older foundational techniques that still fit. Compare advertised gains with the authors' baseline, precision, quality, shapes and timing scope. Rank the largest **applicable end-to-end** opportunities by evidence and measured headroom; a large microkernel speedup or popularity alone is insufficient. Record why a promising result transfers, fails to transfer, or still needs a probe. Do not call a search exhaustive or a result “latest” when coverage was limited.

| Need | Start here | Inspect |
|---|---|---|
| Practical GPU engineering | [GPU MODE lectures](https://github.com/gpu-mode/lectures), [resource stream](https://github.com/gpu-mode/resource-stream) | Linked slides, notebooks, source and transcripts. Examples: Triton Internals (029), AMD Composable Kernel (025), Metal kernels (031), SASS/microarchitecture (037), Mirage megakernels (079), Gluon (104). |
| Compiler stages and low-level kernels | [Triton developer README](https://github.com/triton-lang/triton), [PyTorch compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html), [CUTLASS](https://github.com/NVIDIA/cutlass) | Installed-version dump flags, generated code, dispatch, shape constraints, tests and regressions. |
| Launch/synchronization research | [CUDA Graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html), [Hazy Research](https://hazyresearch.stanford.edu/blog), [Megakernels](https://github.com/HazyResearch/Megakernels) | Capture constraints, residency, synchronization and supported GPUs. H100/B200 examples do not establish AMD, Apple or consumer NVIDIA support. |
| AMD implementations | [ROCm blogs](https://rocm.blogs.amd.com/), [Composable Kernel](https://github.com/ROCm/composable_kernel), [FlyDSL docs](https://rocm.docs.amd.com/projects/FlyDSL/en/latest/) | Exact gfx target, RDNA versus CDNA, matrix instructions, LDS/register pressure, supported kernels and build versions. |
| Apple implementations | [MLX](https://github.com/ml-explore/mlx), [custom Metal kernels](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/custom_metal_kernels.rst), [Metal tools](https://developer.apple.com/metal/tools/) | Lazy evaluation, synchronization, unified-memory pressure, SIMD/threadgroup layout and native runtime integration. |
| Runtime and kernel integration | [PyTorch engineering blog](https://pytorch.org/blog/), [Hugging Face kernels](https://github.com/huggingface/kernels), the chosen engine's official repo/releases | Dispatch overhead, supported modalities/shapes, packing, quantization, graph compatibility and whether a feature is already enabled. |

Search task components: vision convolutions, diffusion denoisers/decoders, audio vocoders/streaming, video temporal state.

