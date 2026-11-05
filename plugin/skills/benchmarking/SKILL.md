---
name: benchmarking
description: Measure end-to-end inference latency, throughput, memory and cost against tuned baselines with balanced rounds and quality evidence. Use for engine/config comparisons and performance claims. Route text, image, audio and video to their own client timers and protocols; isolate device measurements.
license: Apache-2.0
compatibility: Needs the inferencekit CLI (Python 3.10+, uv or uvx) or its MCP server, each engine at a pinned version (any OpenAI-compatible server), and an NVIDIA GPU for the full tool set (elsewhere ceiling bw is NOT RUN; roofline with a numeric --gpu-bw needs no GPU). Written for Claude Code, Codex and OpenCode.
metadata:
  version: "0.1.1"
---

# Benchmarking inference speed

Choose the workload before the metric: complete-request latency, sustained throughput, peak memory or cost at the agreed quality. Use the image/audio route in [affordability](../optimizing-inference/references/affordability.md) and [video checks](references/video-checks.md) for those modalities. Token metrics, text load grids and GSM8K below apply to text; other tasks need a registered adapter, with unsupported checks NOT RUN.

Speed numbers that survive review: a timer outside the engine, one precision for every engine, settings proven in effect, warm engines, fresh-process rounds in a balanced order, confidence intervals, a speed-of-light ceiling and anti-cheat gates. [optimizing-inference](../optimizing-inference/SKILL.md) calls this skill.

**Use** for latency and tok/s per input bucket, capacity (`N*` within an SLO, goodput, $/1M tokens), comparisons of engines, configs or builds, and any reported number. A change-loop step needs only a screen; the full protocol is for milestones (fixed rounds: [references/statistics.md](references/statistics.md)) and anything published ([optimizing-kernels](../optimizing-kernels/SKILL.md), "Checks"). **Not** for *why* it is slow ([profiling](../profiling/SKILL.md)), nor while builds' outputs differ unchecked ([proving-correctness](../proving-correctness/SKILL.md) first).

**GPU approval:** benchmarks, `ceiling bw`, `--gpu-bw measure` and gates run only with `budget.approved: true` in `intake.yaml` or an explicit yes after you stated the cost (GPU, $/h x hours). Record who approved and when.

## Which timer

| | `bench rounds` / `bench serve` (MCP `bench_rounds` / `bench_serve`) | `bench decode` (MCP `bench_decode`) |
|---|---|---|
| Timer | the client, `time.perf_counter_ns`, outside the engine process | inside the engine's process (engine-timed) |
| Engines | any OpenAI-compatible server: `vllm serve`, SGLang, `llama-server`, `trtllm-serve`, a custom engine | in-process vLLM `LLM`, SGLang `Engine`, `llama-bench`, or `--engine cmd` |
| Modes | single-user (sequential) and open-loop load | batch 1 only |
| Published? | **yes: published numbers come only from here** | **no**: label every number "engine-timed" |

`bench rounds` launches each manifest engine fresh per round, warms it up, runs `bench serve` once, stops it, checks GPU memory came back and parses the resolved config, in a Williams order under one GPU lock. Why one client timer: an engine that times itself chooses its own boundaries.

## Preflight checklist

- [ ] Approval recorded; `inferencekit doctor` this session -> ev_env.
- [ ] Nothing else on the GPU or host CPUs while timing (they shift latency by tens of percent); the client on pinned cores or another host.
- [ ] Engines and model pinned in `runs/<id>/manifest.json`; `inferencekit manifest validate` passes.
- [ ] Same precision (weights, activations, KV, lm_head) for every engine, or a numerics change the task allows, labelled "numerics changed"; the same sampling and seed for every engine (`--extra-body`: the workload's, else `generation_config.json` defaults); each baseline at its best documented config for the goal.
- [ ] Prompt set verified (`inferencekit prompts verify --set <dev|path>`), identical for every engine.
- [ ] Claim written first: metric, bucket, threshold, decision rule, round count R.

## Hard failure gates

Any of these makes a record unusable for a speed claim; stop and report it:
- A setting not proven in effect from the resolved config or log, or CUDA graphs off where the engine supports them.
- Generated tokens per request differ from the requested `max_tokens`, or token counts disagree between methods.
- Repeated prompts inside a load level, or a prefix/response cache serving repeats.
- Precision, model revision, GPU type or prompt set differ between the engines compared.
- Fewer completed rounds than pre-registered, rounds not interleaved, or warmups counted. R is intake `workload.rounds` (default 5, never below 3; kernel-loop milestones excepted), sized with `n ~= (2.5c/h)^2` and passed as `--rounds R`, or an adaptive plan pre-registered as `--ci-target`, `--min-rounds` and a maximum `--rounds` ([references/statistics.md](references/statistics.md)).
- A profiler or debugger attached, or the GPU shared during timing.
- Any `inferencekit gates run` FAIL: each gate catches a known way to improve the number without improving the engine.

## Workflow (copyable)

```
Benchmark:
- [ ] 1. Approval recorded; claim and R pre-registered; doctor -> ev_env
- [ ] 2. Ceiling: ceiling bw -> ev_bw; roofline at each context you report -> ev_roof
- [ ] 3. Run manifest: per engine its launch line, env, port, ready_check; best documented config for the goal
- [ ] 4. Prove each knob took effect (resolved config / log line); copy it into manifest resolved_config
- [ ] 5. Single-user: bench rounds --mode single --rounds R  (fresh server per cell, Williams order, warmup excluded)
- [ ] 6. Capacity (if a goal): bench rounds --mode load (grid, SLO, price, one shared --ref-e2e-ms) -> N*
- [ ] 7. Gates, per engine, while its server is up: relaunch it from the manifest, gates run, stop it -> ev_gates_<label>
- [ ] 8. Verdicts per bucket and metric: bench rounds' <metric>_ratio_to_best with its paired CI95 (see Comparisons)
- [ ] 9. % of ceiling per engine (derived); fill the fixed tables; inferencekit report
```

## Exact commands

```bash
inferencekit doctor
inferencekit ceiling bw                                                    # -> ev_bw
inferencekit roofline --model <hf-id or config.json> --dtype fp8 --gpu-bw <ev_bw> --context <prompt + gen/2>
inferencekit prompts verify --set dev            # prints path + sha256; or --set <path.jsonl> --expected-sha256 <pre-registered>
inferencekit manifest validate runs/<id>/manifest.json
# single-user. Engines start in the calling directory (a relaunch: the run dir), so with relative launch paths
# cd runs/<id> first and pass --manifest manifest.json
inferencekit bench rounds --manifest runs/<id>/manifest.json --prompts <set.jsonl> --rounds <R> --mode single
# capacity: open loop, seeded; --concurrency repeats; the same --ref-e2e-ms gives every engine the same schedule
inferencekit bench rounds --manifest runs/<id>/manifest.json --prompts <dev-load set.jsonl> --rounds <R_load> --mode load \
  --serve-args "--concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 --concurrency 16 --concurrency 32 \
  --concurrency 64 --ref-e2e-ms <ms> --seed 0 --slo-ttft-p95-ms 500 --slo-tpot-p95-ms 50 --gpu-price-usd-h <price>"
# gates: per engine, its server up (launched from the manifest line), then stopped
inferencekit gates run --base-url http://127.0.0.1:8000 --engine <label> --run-dir runs/<id> \
  --reference-engine <baseline label>    # + --spec-off-url <same model, no speculation> if it speculates
inferencekit report <ids...> --format md
# engines without a server only (engine-timed; never published as such):
inferencekit bench decode --engine cmd --cmd '<argv>' --model <m> --prompt-len 1024 --gen-len 256 --rounds 1 --extend <ev_prev>
inferencekit compare <ev_baseline> <ev_candidate> --metric decode_tok_s   # same-kind round records
```

MCP tools: `doctor`, `measure_bandwidth`, `roofline`, `prompts_verify`, `validate_manifest`, `bench_rounds`, `bench_serve`, `run_gates`, `report`, `bench_decode`, `compare` (OpenCode: long steps through the CLI, detached). If `bench rounds` cannot launch an engine: a fresh server per call, `bench serve --rounds 1 --engine-kind <vllm|sglang|llamacpp|trtllm> --label <label> --extend <its previous ev>`, in the Williams order yourself (`generic` sends no exact-length field; the default `--rounds` passes share one process). Serving method, buckets, SLOs, `N*`, goodput and cost: [references/serving.md](references/serving.md).

## Prove every knob took effect

Record each setting's requested value, where it was set and the value the engine **resolved** (config or log line; `bench rounds` writes `resolved_config/<label>.r<k>.json`, RESOLVED / PARTIAL / UNPROVEN), copy it into the manifest, and read every warning: a silently dropped setting once turned an A/B into an A/A ([references/engine-configs.md](references/engine-configs.md)).

## Ceilings (speed of light)

At batch 1 each decode step reads all weights plus the KV cache, so time per token >= bytes per token / **measured** DRAM bandwidth. `inferencekit ceiling bw` measures it (quote the large-buffer **read** plateau, never a figure above the datasheet or a copy test); `inferencekit roofline` gives bytes per token. Numbers quoted from the early engine are engine-timed (rule 18). Batched, prefill and non-NVIDIA ceilings: [references/ceilings.md](references/ceilings.md); worked example: [references/worked-example-qwen3-8b.md](references/worked-example-qwen3-8b.md); hand-off cost: [references/launch-cost.md](references/launch-cost.md).

## Rounds, order and comparisons

- Each round starts every engine as a **fresh process**, warms it up and measures, in a **balanced (Williams) order** (period k for k engines when k is even, 2k when odd; pick R as a multiple). CIs are **bootstrap over rounds**, not requests.
- **Served engines:** verdicts come from `bench rounds`' `<metric>_ratio_to_best` = median(engine) / median(best), with a paired bootstrap CI95, read with `compare`'s rule (`improved | regressed | within_noise | inconclusive`, pre-registered effect e, default 0.02; [references/statistics.md](references/statistics.md), "Reading the result"). A claim about two engines, neither of them the best one, needs its own `bench rounds --engines <baseline>,<candidate>` run.
- `compare` accepts matching decode, single-user serve, clip, image or audio records. Load mode is refused. Media needs separate quality and lifecycle proof.

## Anti-cheat gates (`inferencekit gates run`, MCP `run_gates`)

Per engine, against its live server plus the run directory: caching, token counts, output length, unique prompts under load, speculative/MTP exactness (`--spec-off-url`), hidden logprob and temperature probes, rare tokens. A live gate run after the server is gone is NOT RUN, not a pass ([references/reward-hacks.md](references/reward-hacks.md)). Video: [references/video-checks.md](references/video-checks.md).

## Other hardware (AMD ROCm, x86 CPU, Apple Silicon)

The protocol is unchanged; the differences (hardware name via `--serve-args "--gpu-name MI300X --gpu-count 1"`, GPU-only gates on a CPU host, the memory check, a vendor-measured ceiling under `hw/` with `ceiling bw` NOT RUN even if it starts on ROCm) are in [hardware.md](../inference-triage/references/hardware.md).

## Fixed result tables

| Engine (label) @ version | Precision W / A / KV / lm_head | Resolved key settings (evidence) | Bucket | TTFT p50 / p95 ms | TPOT p50 / p95 ms | Decode tok/s median [CI95] | % of ceiling | Timer | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | client / engine-timed | ev_... |

| Engine (serving config) | SLO (TTFT p95, TPOT p95) | N* | Goodput req/s at N* | Output tok/s per GPU at N* | $/1M output tokens at N* | Knee | Gates | Evidence |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | PASS / FAIL | ev_... |

| Pre-registered claim | Ratio [CI95] (source: bench rounds / compare) | Comparison verdict | Claim verdict | Evidence |
|---|---|---|---|---|
| | | | PASS / FAIL / UNPROVEN / NOT RUN | |

## Decision and reporting rules

- No comparison across precisions; if engines share none, compare at BF16 and report quantised runs against their own ceilings. The one exception is a format trade-off ([evaluating-quantization](../evaluating-quantization/SKILL.md), "The cross-precision rule"): speed alone across precisions is never reported.
- A speedup needs a ratio whose paired CI95 over interleaved rounds excludes 1. A CI wider than the effect means more rounds, never a conclusion (UNPROVEN, not a tie).
- Above 100% of the ceiling: stop and find the error (token count, dtype, L2-resident weights). After an exact byte cut, recompute bytes per token for the new storage format first. `N*` comes from open-loop runs with each engine's best serving config; closed loops hide overload.
- Latencies, tok/s and `N*` are **observed** (cite the record), `compare` ratios too; `ratio_to_best` with its CI, % of ceiling, goodput and cost are **derived** (show the formula); any *why* is **inferred** until a profile shows it.
- `bench decode` numbers carry "engine-timed" and never sit in a published table next to client-timed ones. Publish losing engines, refuted configurations and NOT RUN buckets too ([reporting-perf-claims](../reporting-perf-claims/SKILL.md)).
