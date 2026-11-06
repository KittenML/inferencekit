<div align="center">

<img src="https://raw.githubusercontent.com/KittenML/inferencekit/main/assets/inferencekit-kitten.png" width="200" height="200" alt="KittenML kitten holding a GPU">

# inferencekit

**Turn your coding agent into an inference engineer. Make your models faster across any hardware.**

Benchmark, profile and optimize text, image, audio and video models with your coding agent.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/KittenML/inferencekit/blob/main/LICENSE)
[![PyPI: inferencekit-cli](https://img.shields.io/pypi/v/inferencekit-cli.svg?label=inferencekit-cli)](https://pypi.org/project/inferencekit-cli/)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-plugin-d97757.svg)](#install)
[![Codex](https://img.shields.io/badge/Codex-plugin-111111.svg)](#install)
[![OpenCode](https://img.shields.io/badge/OpenCode-plugin-4a7bd0.svg)](#install)

[Install](#install) · [How it works](#how-it-works) · [Showcase](#showcase) · [Features](#features) · [Engines](https://github.com/kittenml/inferencekit-engines) · [Roadmap](#roadmap) · [Contributing](https://github.com/KittenML/inferencekit/blob/main/CONTRIBUTING.md) · [Discord](https://discord.com/invite/VJ86W4SURW)

</div>

inferencekit is a plugin for Claude Code, Codex and OpenCode, backed by reusable CLI and MCP tools. Give your agent a model, hardware and a goal. It finds, benchmarks and tunes compatible inference engines for your workload. You can stop with the best tested configuration or go further with custom optimizations, including GPU kernels. Model coverage depends on runtime and measurement adapter support; see [supported workloads and hardware](#supported-hardware-and-agents).

## How it works

inferencekit gives your agent the playbook and the tools of a senior inference engineer. On every run it works the problem the same way.

1. **Understand the task.** The agent first asks about your goal, deadline, budget and how much the results may change, reusing answers you already gave. It identifies the model, hardware and real inputs, then estimates memory, compute and launch costs per request. Text, image, audio and video use their own metrics and quality checks.
2. **Find and tune an engine.** It researches current implementations, then tunes and benchmarks supported engines (vLLM, SGLang, llama.cpp, TensorRT-LLM and others) for your workload on your hardware. The fastest tested configuration that meets your quality requirements can be the result, or the tuned baseline for further work.
3. **Find the bottleneck.** It selects the profiler for your hardware: Nsight on NVIDIA, ROCm tools on AMD, or Instruments and Metal on Apple. It measures the critical path and investigates memory, compute and launch costs, checking current vendor docs when support is uncertain.
4. **Investigate and implement, if requested.** Search current code, papers, compiler docs and engineering talks; inspect dispatch, IR and assembly when useful. Choose graph execution, fusion, matrix kernels, packing, state reuse or other techniques from measured bottlenecks. Independent implementations can run in parallel; one coordinator integrates changes, and measurements stay serialized per device. Separate authorized devices may run concurrently.
5. **Prove every step.** Every change gets a fast correctness check and a timed screen; milestones get the full proof: interleaved rounds against the tuned baseline, the anti-cheat gates and quality against a reference model. Changes that miss the agreed performance or quality requirements are reverted.
6. **Ship it.** You get the fastest proven configuration and a [`REPORT.md`](https://github.com/KittenML/inferencekit/blob/main/plugin/skills/reporting-perf-claims/references/wording-guide.md) with the result, quality checks, commands to install and run it, saved outputs, and details to reproduce the measurements.

## When it asks you

The agent works on its own and stops only when it needs you:

- **Missing details.** If you didn't say which model, hardware or workload, it asks before it starts.
- **Your preferences.** Choose exactly the same results, small differences with the same quality, or speed with an agreed quality tradeoff. Set your goal, spending, deadline and check-ins; it reuses answers you already gave.
- **Model changes.** Permission to simplify the model is separate from permission to use stronger compression. Plans are optional: ask to see one if you want it.
- **The budget cap.** When the run reaches the cap you set, it stops and asks.
- **Things only you can give.** For example, permission to read the GPU's profiler counters, or a Hugging Face token for a gated model.

Everything else runs without asking. For a hands-off run, approve a budget with a cap up front (`non_interactive: true` and `budget.approved: true` in the intake file); the agent then writes its questions and the choices it made to `QUESTIONS.md` instead of waiting for you.

<a id="results"></a>

## Showcase

Examples of coding agents using inferencekit to optimize inference. The workflow applies beyond these models wherever compatible runtimes and measurement adapters are available. Each result applies to its stated hardware, baseline and quality settings.

**Baseline selection includes competing engines:** Gemma's tuned vLLM baseline followed a vLLM/SGLang screen; LTX's ComfyUI baseline was fastest in the tested ComfyUI, Diffusers and official LTX pipeline screen.

### Gemma 4: faster decode on RTX 5090

Claude Opus 5.5 used inferencekit to build an exact sampler, batch-1 NVFP4 MoE kernels, a pruned output layer and lossless weight packing. **1.33× faster single-user decode than tuned vLLM**, with the same NVFP4 weights and activation scheme.

[How it was built, results and usage](https://github.com/KittenML/inferencekit-engines/tree/main/gemma-4-26b-a4b-nvfp4_rtx-5090/)

### LTX-2.5: faster video generation on RTX PRO 6000

Claude Opus 5.5 used inferencekit to compile transformer blocks, fuse kernels and reuse work between guidance passes. **1.25× faster than ComfyUI** for one 768 × 512, 121-frame clip at a time, with the same BF16 weights and 30 steps; quality checks passed, but outputs are not bit-identical.

[How it was built, results and usage](https://github.com/KittenML/inferencekit-engines/tree/main/ltx-2.5_rtx-pro-6000/)

### Drama3: speech optimization on Apple M4 (16 GB)

GPT-6.1 Sol used inferencekit to remove a redundant sampler sort, compile single-token GDN blocks and release the codec encoder after reference preparation. **1.112× faster single-user speech generation than the MLX-Audio baseline selected from the tested configurations**, on identical quantized weights. Output-preservation checks passed for all 144 paired outputs, matching audio, codes, chunks and completion. Neither engine met every-request realtime playback.

[How it was built, results and usage](https://github.com/KittenML/inferencekit-engines/tree/main/drama-3_apple-m4-16gb/profiles/main5-tower5/)

### MOSS-Image-SFT: an accepted speed/quality trade-off on AMD

GPT-6.1 Terra used inferencekit to wire online FP8 into selected transformer projections and tune the quantization scale and step count. **2.46× faster than the tested tuned vLLM-Omni baseline** on a 16 GB RX 9070 XT: FP8/24 steps versus BF16/50 steps, with quality secondary.

[How it was built, results and usage](https://github.com/KittenML/inferencekit-engines/tree/main/moss-image-sft_rx-9070-xt-fp8/)

### Qwen3-8B on RTX 5090: 1.22× faster single-user decode than vLLM with FlashInfer

**Fastest decode in our tested vLLM, SGLang and llama.cpp panel.** Claude Opus 5.5 used inferencekit to build **Qwen3-8B on RTX 5090** with custom FP8 matrix-vector kernels, a CUDA graph for the decode step and fused attention. Single-user decode was measured with in-process timers and no speculative decoding; llama.cpp Q8_0 is a separate format comparison. [Design, results and limits](https://github.com/KittenML/inferencekit-engines/tree/main/qwen3-8b-fp8_rtx-5090/).

![Bar chart of speedups against engine baselines on the same hardware, including tuned vLLM and MLX-Audio: Gemma 4 26B-A4B on RTX 5090, 1.33× faster than tuned vLLM; LTX-2.5 on RTX PRO 6000, 1.25× faster than ComfyUI; Qwen3-8B on RTX 5090, 1.22× faster than vLLM with FlashInfer; Drama3 on Apple M4, 1.11× faster than tuned MLX-Audio; MOSS-Image-SFT on RX 9070 XT, 2.46× with quality secondary (FP8, 24 vs 50 steps; paired 95% CI 2.453–2.465×, lower bound 2.453×)](https://raw.githubusercontent.com/KittenML/inferencekit/main/assets/hero-speedups.png)

**Start optimizing models without a $200k/year hire. Our published agent runs cost as little as $18**

<details>
<summary>Measurement details and agent time, tokens and cost</summary>

| Model / workload | Hardware | Speedup | Agent | Agent time | Tokens | Cost |
|---|---|---|---|---|---|---|
| [Gemma 4 26B-A4B (NVFP4)](https://github.com/kittenml/inferencekit-engines/tree/main/gemma-4-26b-a4b-nvfp4_rtx-5090) | RTX 5090 | **1.33×** vs tuned vLLM | Claude Opus 5.5 (high) | 19 h | 630M | $199 |
| [LTX-2.5 text-to-video](https://github.com/kittenml/inferencekit-engines/tree/main/ltx-2.5_rtx-pro-6000) | RTX PRO 6000 | **1.25×** vs ComfyUI | Claude Opus 5.5 (high) | 3.6 h | 47M | $18 |
| [Qwen3-8B (FP8)](https://github.com/kittenml/inferencekit-engines/tree/main/qwen3-8b-fp8_rtx-5090)* | RTX 5090 | **1.22×** vs vLLM with FlashInfer | Claude Opus 5.5 (xhigh) | about 10 h | about 240M | about $125 |
| [Drama3 speech](https://github.com/KittenML/inferencekit-engines/tree/main/drama-3_apple-m4-16gb/profiles/main5-tower5)† | Apple M4 (16 GB) | **1.112×** vs tuned MLX-Audio | GPT-6.1 Sol (high) | ~70 h elapsed | ~2.07B | ~$475 standard-rate equivalent |
| [MOSS-Image-SFT (FP8)](https://github.com/KittenML/inferencekit-engines/tree/main/moss-image-sft_rx-9070-xt-fp8)‡ | RX 9070 XT (16 GB) | **2.46×** vs tuned vLLM-Omni | GPT-6.1 Terra (high) | ~19.6 h elapsed | ~1.80B | ~$206 standard-rate equivalent |

<sub>* Against vLLM 0.31.0 with FlashInfer attention; neither side uses speculative decoding. † Drama3 used identical quantized weights and matched PCM audio, codes, chunks and completion in 144 fresh output pairs. Four paired timing rounds gave a 95% CI of 1.059×–1.156×. [Comparison details](https://github.com/KittenML/inferencekit-engines/tree/main/drama-3_apple-m4-16gb/profiles/main5-tower5).</sub>

<sub>‡ Lossy speed-first experiment: FP8/24 steps vs BF16/50 steps; paired 95% CI **2.453–2.465×**, conservative lower bound **2.453×**. [Outputs, timing and accounting](https://github.com/KittenML/inferencekit-engines/tree/main/moss-image-sft_rx-9070-xt-fp8/).</sub>

Time and tokens include setup, benchmarking, optimization, validation and reporting. Costs are API list-price estimates; most tokens are cached conversation reads. How each engine was built and measured: [inferencekit-engines](https://github.com/kittenml/inferencekit-engines).

</details>

<a id="speed-you-can-trust"></a>

## Performance and quality validation

Performance gains need matched workloads and quality checks. inferencekit checks for cached responses, incomplete outputs and unintended quality changes before accepting a speedup.

Without these checks, one agent thought its video engine was 2% faster, but it was rendering a completely different video. Another was sure it had a 1.12× speedup; measured again on a clean machine, its engine was 5% slower than where it started.

A speedup counts only when both engines pass every check that applies.

| A way to game a benchmark | What catches it |
|---|---|
| Answer repeated prompts from a cache | `caching`: a replayed prompt may not be over 10% faster; a prompt changed by one character may not get the old answer |
| Report more tokens than were made | `token_count`: the reported count must match the streamed tokens and the re-tokenized text |
| Stop early or cut answers short | `output_length`: with end-of-text ignored, every answer must be exactly the requested length |
| Drop rare tokens from the vocabulary | `rare_tokens`: random CJK text, emoji, long numbers and code names must come back exactly |
| Ignore sampling settings or fake probabilities | `hidden_probes`: temperature must change answers; probabilities must be valid |
| Use another chat template so prompts get shorter | Render check: every prompt must become the same number of tokens as on the baseline |
| Speculative decoding that changes answers | `spec_exact`: greedy answers must match the same engine with speculation off |
| Lower quality for speed | Quality vs a reference, GSM8K, per-step video check |
| Numbers that only hold on the agent's machine | Every run writes `manifest.json` with each engine's install and launch commands, so anyone can re-run it on a clean machine, on prompts sealed with `inferencekit prompts seal` that the agent never saw |

Each check ends PASS, FAIL, UNPROVEN (not enough evidence) or NOT RUN. A missing check is never a pass.

## Features

- **9 skills, 4 commands and 5 read-only subagents**, backed by [47 executable CLI/MCP operations](https://github.com/KittenML/inferencekit/tree/main/src/inferencekit/ops/).
- **Reusable run tools:** [ikstep](https://github.com/KittenML/inferencekit/blob/main/plugin/skills/optimizing-inference/references/remote-host.md) manages jobs on local or remote Linux hosts, readiness checks and concise status output.
- **Compact output:** CLI summaries and bounded MCP responses keep full evidence records on disk, with paging for large lists.
- **Speed-of-light ceilings:** measured memory bandwidth (`ceiling bw`) and exact bytes per token read from the checkpoint's safetensors headers (`roofline`), including NVFP4, FP8 and mixture-of-experts models.
- **Engine triage and tuning:** vLLM, SGLang, llama.cpp, TensorRT-LLM, ComfyUI, Diffusers and others. Only settings that the engine's own logs show in effect are counted.
- **Kernel work:** profiling routed by hardware, CUDA/HIP, Triton and Metal kernels, and torch.compile. The included [engine skeleton](https://github.com/KittenML/inferencekit/blob/main/plugin/skills/optimizing-kernels/skeleton/README.md) targets NVIDIA CUDA.
- **One timer for every engine:** interleaved rounds in a balanced order, medians with 95% confidence intervals.
- **Anti-cheat gates:** caching, token counts, output length, rare tokens, sampling probes, prompt rendering, speculative-decoding exactness.
- **Quality checks:** reference quality and GSM8K for language models; latent and final-clip checks for video. Images and speech use client timing and structural checks plus an external semantic quality protocol; see [media benchmarks](https://github.com/KittenML/inferencekit/blob/main/docs/media-benchmarks.md).
- **Budget control:** spending authorized up front, with hardware-hour, dollar and elapsed-time limits, or explicitly uncapped spending with a finite deadline. No repeated plan approval inside the agreed scope.
- **Reproducible output:** a ranked table labelled PASS, FAIL, UNPROVEN or NOT RUN, a launch command, a review page and the commands that reproduce every number.

### Tools behind each step

| Step | Tools |
|---|---|
| Size the problem | `intake`, `doctor`, `ceiling bw`, `roofline` |
| Start from the best engine | `bench rounds`, `gates run`, `quality` |
| Find the bottleneck | `doctor` routes profiling; `nsys`/`ncu` on NVIDIA, `rocprof` on supported AMD setups, external vendor tools on Apple |
| Pull the right levers, prove every step | `correctness`, `bench rounds`, `gates run`, `quality` |
| Ship it | `ledger`, `review`, `report` |

## Install

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer. Install the engine runtime for your hardware. NVIDIA capture commands require CUDA and Nsight; AMD and Apple profiling uses vendor tools with separate evidence files. See [supported hardware](#supported-hardware-and-agents).

### Claude Code

```bash
claude plugin marketplace add kittenml/inferencekit
claude plugin install inferencekit@inferencekit
```

### Codex

```bash
codex plugin marketplace add kittenml/inferencekit
codex plugin add inferencekit@inferencekit
```

### OpenCode

```bash
git clone --depth 1 https://github.com/kittenml/inferencekit ~/.local/share/inferencekit
python3 ~/.local/share/inferencekit/plugin/opencode/install.py
```

| Agent | Commands / skills | Check the install |
|---|---|---|
| Claude Code | `/inferencekit:optimize-inference`, `/inferencekit:bench`, `/inferencekit:profile`, `/inferencekit:perf-report` | `claude plugin list`; `/mcp` lists the inferencekit server |
| Codex | `$inferencekit:optimizing-inference`, `$inferencekit:benchmarking`, `$inferencekit:profiling`, `$inferencekit:reporting-perf-claims` | `codex plugin list`; `codex mcp list` |
| OpenCode | `/optimize-inference`, `/bench`, `/profile`, `/perf-report`, and each skill as `/<skill-name>` | `opencode mcp list` |

Updating, removing, per-project installs and OpenCode's options: [docs/install.md](https://github.com/KittenML/inferencekit/blob/main/docs/install.md).

## Try it

```text
/inferencekit:optimize-inference Qwen/Qwen3-8B-FP8 on 1x RTX 5090, fastest single-user decode, by 6h
```

To tune existing engines without custom kernels:

```text
/inferencekit:optimize-inference Tune vLLM and SGLang for my model and hardware using their built-in settings only; do not write custom kernels.
```

In OpenCode the command is `/optimize-inference`. It asks only for missing preferences and spending authorization; a plan is optional. For a repeatable setup, start from an intake file: `inferencekit intake template --out intake.yaml`, then `/inferencekit:optimize-inference intake.yaml`. For unattended runs, set `non_interactive: true` and `budget.approved: true` with a cap.

More things to ask for: [docs/examples.md](https://github.com/KittenML/inferencekit/blob/main/docs/examples.md).

## CLI and MCP server

```bash
uv tool install "inferencekit-cli==0.1.0"
inferencekit doctor                                         # Hardware, runtime and profiler availability
inferencekit gates run --base-url http://127.0.0.1:8000     # anti-cheat gates on any OpenAI-compatible server
uvx --from 'inferencekit-cli[mcp]==0.1.0' inferencekit mcp serve   # MCP server (stdio) for any MCP client
```

Every CLI command is also an MCP tool with the same arguments. Text benchmarks, gates and quality checks talk HTTP to OpenAI-compatible servers. Image and streamed speech measurements use the [media adapter contracts](https://github.com/KittenML/inferencekit/blob/main/docs/media-benchmarks.md).

## Supported hardware and agents

| | Status | Run end to end |
|---|---|---|
| NVIDIA GPUs (CUDA) | Full tool set | Yes: RTX 5090, RTX PRO 6000 |
| AMD ROCm GPUs | Hardware discovery, HTTP benchmarks and ROCm trace capture/analysis where supported; bandwidth measured externally | Yes: RX 9070 XT |
| Apple silicon | Hardware discovery and vendor profiling guidance; HTTP benchmarks; vendor bandwidth/captures collected externally | Yes: Apple M4 |
| x86 CPUs | Skills and HTTP benchmarks; platform profiling collected externally | No |
| ARM CPUs (CPU-only inference) | Platform support unverified | No |
| Language models | Benchmarks, gates, GSM8K, quality vs a reference | Yes |
| Text-to-video | Per-clip timing, per-step check, final-clip PSNR | Yes |
| Text-to-image | Client-timed decoded images and structural checks; external quality proof required | Yes: MOSS-Image-SFT on RX 9070 XT |
| Speech models | Client-timed streamed PCM, first-signal latency, RTF and underruns; external quality proof required | Yes: Drama3 TTS on Apple M4 |
| Claude Code | Supported | Yes |
| Codex | Supported | Yes: AMD and Mac case studies |
| OpenCode | Install path and CI install checks | No |

Profiles and logs can contain environment variables, host names and paths: scan them before you share them.

## Roadmap

- **More hardware:** H100, H200, B200, RTX 4090, Intel GPUs and Jetson, with the hardware tools (`doctor`, `ceiling bw`, profiling) for each.
- **Broader workload coverage:** more speech, image and video models, including speech recognition (STT).
- **Multi-user serving:** published results for many users at once (load sweeps), not just one user at a time.
- **Token and time efficiency:** further reduce agent token usage and turnaround time with reusable scripts, compact tool output and focused context.
- **OpenCode end to end.**

## Contributing

We welcome bug reports, engine settings and new skills. See [CONTRIBUTING.md](https://github.com/KittenML/inferencekit/blob/main/CONTRIBUTING.md). Develop with `uv sync --all-extras`, then `uv run --all-extras pytest -q` (no GPU needed). Report security problems privately: [SECURITY.md](https://github.com/KittenML/inferencekit/blob/main/SECURITY.md).

## License

Apache-2.0: [LICENSE](https://github.com/KittenML/inferencekit/blob/main/LICENSE) and [NOTICE](https://github.com/KittenML/inferencekit/blob/main/NOTICE). Copyright 2026 Stellon Labs. inferencekit is an independent project, not affiliated with NVIDIA, any engine project or any model publisher. Product and model names belong to their owners. Every result holds for the hardware, driver, engine version, model and workload it was measured on; change any of them and the numbers can change. The software comes with no warranty, as the Apache-2.0 licence states, and you are responsible for the cost of hardware use with it.
