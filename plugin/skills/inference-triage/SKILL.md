---
name: inference-triage
description: "Route inference performance questions for text, image, audio, video and other models to the cheapest useful evidence: latency, throughput, memory, hardware cost or quality. Use when unsure where to start; select tools for NVIDIA, AMD, Apple Silicon or CPU and preserve resource authorization."
license: Apache-2.0
compatibility: Uses the inferencekit CLI (Python 3.10+, uv or uvx) or its MCP server. NVIDIA GPUs are fully supported (profiling needs Nsight); AMD ROCm, x86 CPUs and Apple Silicon are guided by references/hardware.md. Written for Claude Code, Codex and OpenCode.
metadata:
  version: "0.1.0"
---

# Inference triage

Entry point. Turn a vague request into one measurable question, collect the cheapest evidence that answers it, then hand off to exactly one skill. The rules here apply to every other skill in this plugin.

## Which skill

| The request | Skill |
|---|---|
| Current techniques, implementation research, a stalled optimization | [researching-optimizations](../researching-optimizations/SKILL.md) |
| "Make this model fast", "which engine should I use", "fastest config for my model on my GPU", a target (tok/s, TTFT, users per GPU, $/1M tokens) | [optimizing-inference](../optimizing-inference/SKILL.md): the end-to-end flow |
| A speed number: tok/s, TTFT, per-token latency, capacity `N*`, engine vs engine, before vs after, % of ceiling | [benchmarking](../benchmarking/SKILL.md) |
| Where the time goes; why one kernel is slow | [profiling](../profiling/SKILL.md) |
| Make an engine or kernel faster (custom engine, engine patch, decode or prefill) | [optimizing-kernels](../optimizing-kernels/SKILL.md) |
| Is it still the same model; did quality regress | [proving-correctness](../proving-correctness/SKILL.md) |
| Fit consumer memory while meeting latency and quality | [optimizing-inference](../optimizing-inference/SKILL.md), [affordability](../optimizing-inference/references/affordability.md) |
| Should I quantize; is FP8 / 4-bit / a GGUF quant faster at acceptable quality; which quant; will it fit my GPU; a big model on a small GPU | [evaluating-quantization](../evaluating-quantization/SKILL.md) |
| Write up, review or audit a claim | [reporting-perf-claims](../reporting-perf-claims/SKILL.md) |
| Not sure | stay here: run the workflow below |

## When to use / not use

Use when the right skill or the question is unclear ("why is it slow", "is X tok/s good", "did my change help"), or a number is about to be quoted unchecked against a ceiling. Not when the request clearly matches a row above (go there), or to rank different models by quality (model evaluation, not equivalence).

## GPU approval (every skill)

GPU work (benchmarks, captures, `ceiling bw`, `--gpu-bw measure`, quality runs, renting) runs only with `budget.approved: true` in `intake.yaml` or an explicit yes after you stated the cost (GPU, $/h x hours, cap): it occupies shared hardware and spends money. Analysing existing evidence, roofline arithmetic with a numeric bandwidth, and reporting need no approval.

Explicit session authorization persists. An approved finite-deadline run may specify `budget.spend_cap: none`, `stage_timeboxes: false` and `report_reserve_h: 0`; these run-scoped overrides replace the respective defaults without repeated approval questions.

## Getting a GPU

- Your own GPU, or one rented by the hour from any cloud GPU marketplace; never name or recommend a provider in shared files.
- Rent only after an explicit yes to the GPU model, $/h x hours and a cost cap, or within a pre-approved budget (end to end: optimizing-inference's opening questions; existing authorization persists).
- **Need ncu counters? Prefer a VM or bare-metal offer.** Containers and many cloud instances (A100 and H100 included) block them; `inferencekit doctor` reports it (`data.profiling_counters.status`). If blocked: nsys only with ncu NOT RUN, then the PyTorch profiler (no counters needed) and the person's own profiles ([profiling](../profiling/SKILL.md), "Fallbacks").
- Record GPU model, count, driver, CUDA and price/h (`inferencekit doctor`, and the price in the intake's `hardware.price_usd_h`).
- Keep the provider's credentials out of commands, logs and profiles (capture only through inferencekit, under `env -i`).
- Copy `runs/` back, destroy the instance after use (not just stop it), and confirm none is left running.

## Not NVIDIA? (AMD ROCm, x86 CPU, Apple Silicon)

NVIDIA is fully supported, and the only hardware inferencekit has been run on so far. On other hardware, the skills guide the same flow, and the CLI does less: [references/hardware.md](references/hardware.md) has the table.
- **Hardware-neutral, not yet run off NVIDIA:** `bench rounds`, `bench serve`, `gates run` and quality (they talk HTTP to any OpenAI-compatible server), `roofline` with a numeric `--gpu-bw`, and the intake, manifest, ledger and report tools. Treat a first run there as a test.
- **NOT RUN off NVIDIA:** `ceiling bw` and nsys/ncu captures. `doctor` records selected AMD DRM/rocminfo and Apple sysctl/system_profiler facts with a vendor profiling route; unknown details remain unknown. On AMD, `rocprof capture` and `rocprof analyze` create trace evidence with rocprofv3; device/version support must be checked. Use current official docs for unfamiliar tools and a vendor or STREAM-style bandwidth test. Captures made outside the CLI are vendor evidence files.
- **Platform guides:** [AMD ROCm](references/amd-rocm.md), [x86 CPU](references/cpu.md), [Apple Silicon](references/apple-silicon.md), [Intel GPU and TPU](references/other-accelerators.md).

## Preflight checklist

- [ ] One sentence names the workload: model id + revision, precision (weights / activations / KV cache), GPU, input bucket(s), output length, and the goal (single-user latency or capacity under an SLO). Why: a number without its workload cannot be reproduced or compared.
- [ ] `inferencekit doctor` has run on the target machine this session (not NVIDIA: plus the hardware card from [references/hardware.md](references/hardware.md)). Why: driver, clocks, counter permissions and engine versions decide which captures are possible.
- [ ] You know whether anything else uses the GPU or host CPUs. Why: shared GPUs and busy "spare" CPUs silently slow timed runs.
- [ ] The goal is a falsifiable claim (e.g. "candidate decode tok/s >= 1.10x baseline, medium bucket" or "N* >= 32 at p95 TTFT <= 300 ms"). Why: a claim chosen after the data is a forking-paths result.

## Hard failure gates

Stop and report the gate instead of continuing when:
- `inferencekit doctor` shows no usable GPU, a mismatched driver/CUDA pair, or an unknown engine version. On AMD, CPU or Apple hardware, empty GPU facts are expected: continue with [references/hardware.md](references/hardware.md) instead.
- The two sides of a comparison differ in precision, model revision, GPU type or workload.
- A secret (API key, token) would end up in a capture, log or report.
- GPU-occupying work is about to run without recorded approval.

## Workflow (copyable)

```
Triage:
- [ ] 1. Restate the request as one question + one pre-registered claim
- [ ] 2. End-to-end "make it fast" request? -> optimizing-inference, done
- [ ] 3. inferencekit doctor -> ev_env
- [ ] 4. inferencekit roofline -> bytes/token and ceiling tok/s (no GPU needed with a numeric --gpu-bw)
- [ ] 5. Existing evidence first: inferencekit evidence list; profiles the person already has
- [ ] 6. % of ceiling = measured tok/s / ceiling tok/s (derived), if a measurement exists
- [ ] 7. Pick ONE skill from the symptom table; say why in one line
- [ ] 8. Hand off with: workload sentence, evidence ids, the claim, what is still NOT RUN
```

## Exact commands

| Step | CLI | MCP tool |
|---|---|---|
| Environment card | `inferencekit doctor` | `doctor` |
| Ceiling (arithmetic only) | `inferencekit roofline --model Qwen/Qwen3-8B-FP8 --dtype fp8 --gpu-bw <measured GB/s or ev_bw>` | `roofline` |
| Existing evidence | `inferencekit evidence list` / `inferencekit evidence show <id>` | `list_evidence` / `inspect_evidence` |
| Intake for the end-to-end flow | `inferencekit intake template` / `inferencekit intake validate intake.yaml` | `intake_template` / `intake_validate` |
| Report | `inferencekit report <ids...> --format md` | `report` |
| Analyse a profile the person supplied (no GPU) | `inferencekit nsys analyze <file>` / `inferencekit ncu summarize <file> --peak-gbs <ev or GB/s>` | `analyze_nsys` / `summarize_ncu` |

MCP clients prefix tool names (Claude Code: `mcp__plugin_inferencekit_inferencekit__roofline`); the arguments are the CLI's. **MCP server does not start** (no `uvx`, no network): the CLI at this plugin's version, `uv tool install "inferencekit-cli==0.1.0"`. **In OpenCode,** long steps (`bench rounds`, `gates run`, quality runs, `ceiling bw`, captures) go through the CLI, detached with a log: it ends an MCP call after the server's `timeout` (120 s), and the server sends no progress. Bandwidth: your GPU's measured `inferencekit ceiling bw` (off NVIDIA, the platform's micro-benchmark: [references/hardware.md](references/hardware.md)), or say the ceiling is datasheet-based and therefore unreachable.

## Symptom routing

| Symptom or question | Cheapest first evidence | Hand off to |
|---|---|---|
| "Make it fast", "which engine", a target to hit | intake | optimizing-inference |
| "How fast could this possibly go?" | `roofline` with a known bandwidth | benchmarking (ceilings) |
| "What tok/s / TTFT do we get?", engine vs engine | doctor + roofline | benchmarking |
| "How many users per GPU?", p95 latency rises under load, cost per token | the SLO and traffic shape | benchmarking (capacity, `N*`) |
| Below ~85% of the ceiling ([bands](#ceiling-bands)), cause unknown | one nsys capture of the steady step | profiling |
| "I already have a profile" (`.nsys-rep`, `.sqlite`, `.ncu-rep`) | `inferencekit nsys analyze <file>` / `inferencekit ncu summarize <file>` (no GPU, no approval) | profiling |
| "I have a PyTorch profiler trace" (`*.json.gz` from vLLM, SGLang or `torch.profiler`) | read by hand or with `profile-analyst`: the CLI does not parse it (no GPU, no approval) | profiling |
| GPU idle between kernels, high kernel count | nsys per-step busy/idle | profiling, then optimizing-kernels |
| One kernel or family dominates the step | nsys critical-path shares | profiling (ncu) |
| "Make this engine/kernel faster" with evidence in hand | ceiling + profile | optimizing-kernels |
| TTFT slow (prefill) | TTFT per bucket, first-seen vs repeated length | benchmarking, then optimizing-kernels (prefill) |
| Outputs changed; before any speedup claim | trajectory score vs the noise floor | proving-correctness |
| "Should I quantize?", "is 4-bit faster?", "which quant?", "will it fit?", near the ceiling and only fewer bytes can help | `roofline --method headers` per checkpoint (no GPU): bytes/token and fit | evaluating-quantization |
| "Did my change help?" / regression hunt | interleaved A/B (`bench rounds`; `compare` for `bench decode`) | benchmarking, then reporting-perf-claims |
| Writing up a result, PR, blog post | the evidence ids | reporting-perf-claims |

## Ceiling bands

One scheme for every skill in this plugin: % of a measured-bandwidth ceiling, for single-user decode. These are heuristics that pick the next measurement, not verdicts.
- **Above 100%:** impossible; the ceiling, the bytes per token or the token count is wrong.
- **~85% or more:** near the floor. Only reading fewer bytes per token helps (an exact byte cut, or a lower precision: evaluating-quantization).
- **~50% to ~85%:** look for launch gaps and under-filled kernels (profiling, then optimizing-kernels).
- **Below ~50%:** suspect eager mode, host syncs or a CPU-bound loop.
- **In every band the ceiling can move** (exact byte cuts, allowed speculation): estimate both at the start (optimizing-inference's headroom estimates) and work them alongside kernel fixes.

The table with the next evidence for each band: [references/symptom-map.md](references/symptom-map.md).

## Artifact/evidence contract

- Every inferencekit operation writes an evidence record (schema v2; v1 records still read) and returns its id. Why: the id is the citation; a number without one is hearsay.
- Read `identity`, `coverage.complete`, `coverage.missing` and `limitations` before quoting any metric. Why: a record can be valid and still not cover your claim.
- Never paste raw environment values; inferencekit stores none, and you must not add them.
- Global rules with their reasons: [references/evidence-rules.md](references/evidence-rules.md) (the numbered rules are also in the repository's `AGENTS.md`).

## Fixed result table

| Fact | Value | Class | Evidence id | Note |
|---|---|---|---|---|
| Workload and goal | e.g. Qwen3-8B FP8 W / BF16 KV, medium bucket, single-user | observed | ev_... | from `identity.workload` |
| GPU, driver, clocks | | observed | ev_... | doctor |
| Ceiling tok/s | | derived | ev_... | bytes/token and bandwidth source |
| Measured value | ... or NOT RUN | observed | ev_... | statistic, n |
| % of ceiling | | derived | (both ids) | formula shown |
| Next skill | | inferred | — | one-line reason |

## Decision and no-go rules

- Measure before optimising. Why: most "obvious" bottlenecks are not where the time goes.
- One skill at a time. Why: profiling during benchmarking contaminates both.
- No speedup claim without a quality verdict. Why: the fastest kernel is the one that skips work.
- Measured above the ceiling: stop; the bytes/token model, the bandwidth or the token count is wrong.
- Datasheet bandwidth is a label, not a ceiling.

## Reporting rules

- Label every number **observed** (a tool measured it; cite the evidence id), **derived** (arithmetic; show the formula) or **inferred** (reasoning; say what would confirm it).
- Claim verdicts are **PASS / FAIL / UNPROVEN / NOT RUN**; missing data is UNPROVEN or NOT RUN, never zero and never PASS. Comparison verdicts come from the tools (`bench rounds` ratio CIs for served engines, `inferencekit compare` for `bench decode`), never from eyeballed medians.
