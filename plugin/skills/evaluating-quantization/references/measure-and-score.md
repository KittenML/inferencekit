# Measuring and scoring formats

The same harness for every row: one `bench rounds` call for speed, `gates run` per row, and the quality commands of [proving-correctness](../../proving-correctness/SKILL.md) against one full-precision reference. This file covers what is specific to comparing formats.

## 1. Rows and labels

A row is one (engine at a pinned version, checkpoint at a pinned sha, KV dtype) configuration. Label it so the format is readable, e.g. `vllm-bf16` (the reference), `vllm-fp8`, `vllm-awq`, `vllm-awq-kvfp8`, `llamacpp-q4km`.

| Row kind | Isolates | Label in the report |
|---|---|---|
| Same engine, reference checkpoint | nothing: the reference | `reference` |
| Same engine, quantized checkpoint | the format | `cross-precision: format trade-off, not an engine comparison` |
| Same engine and weights, FP8 KV cache | the KV dtype | the same, plus "KV dtype only" |
| Another engine, another format | nothing: a deployment option | `cross-precision and cross-engine: deployment option` |
| A weight-only fallback path (e.g. W8A16 on a GPU without FP8 tensor cores) | the fallback | its own row, with the path named |

## 2. One manifest, several checkpoints

`bench rounds` interleaves only the engines of one manifest, and the paired ratio CIs exist only between rows of one call, so every format goes into one `runs/<id>/manifest.json`. Its `model` block is the reference checkpoint; each quantized row downloads its own checkpoint, pinned, in its `install` commands, and records it in `resolved_config`:

```json
{"name": "vllm", "label": "vllm-awq", "version": "x.y.z",
 "install": ["uv pip install vllm==x.y.z huggingface_hub",
             "hf download Qwen/Qwen3-8B-AWQ --revision <40-hex sha> --local-dir models/qwen3-8b-awq"],
 "launch": ["vllm", "serve", "models/qwen3-8b-awq", "--port", "8002", "--served-model-name", "qwen3-8b-awq",
            "--seed", "0", "--no-enable-prefix-caching"],
 "port": 8002, "ready_check": "/v1/models",
 "resolved_config": {"checkpoint": "Qwen/Qwen3-8B-AWQ@<sha>", "quantization": "<from the startup log>",
                     "gemm_kernel": "<from the startup log>", "kv_cache_dtype": "<from the startup log>",
                     "source": "startup log lines ..."}}
```

- GGUF rows download one file (and every shard): `hf download <repo> <file.gguf> --revision <sha> --local-dir models/<dir>`.
- Ports distinct, labels unique; `inferencekit manifest validate` must pass. A re-run replays the `install` commands, so each checkpoint is pinned there and nowhere else.
- The bench records do not carry each row's precision: the report takes it from `resolved_config` and the startup log, row by row.

## 3. Speed

- Single-user goal: `inferencekit bench rounds --manifest runs/<id>/manifest.json --prompts <dev set path> --rounds <R> --mode single`.
- Capacity goal: `--mode load` with the intake's concurrency grid and SLO (benchmarking's exact command), so every row gets `N*`, goodput and $/1M at `N*`. Tune each row's serving config for capacity separately (more KV room is the usual reason a quantized row wins).
- Rounds: the pre-registered R (default 5, never below 3), rounded to the Williams period of the row count. Estimate the run time first (rows x rounds x (launch + warm-up + prompt set)) and drop rows, never rounds, to fit the deadline.
- Read each row's `<metric>_ratio_to_best` and CI95. For a row's speed vs the reference: when the reference is not the best row, the reference row's ratio is relative to the best too; divide the two medians for the point value and run a two-row `bench rounds --engines <reference>,<row>` if the claim needs that pair's own CI (benchmarking, "Rounds, order and comparisons").
- Gates per row while its server is up: `inferencekit gates run --base-url http://127.0.0.1:<port> --engine <row label> --run-dir runs/<id> --reference-engine <reference label>`.
- **% of ceiling per row uses that row's own bytes per token**, at the KV dtype it resolved: it shows how well that format's kernels use the bandwidth.

## 4. Quality margin tiers (pre-registered)

`quality score` margins are **increases over the A/A floor**. Agree them with the user before measuring and write them into the plan (or the intake's `constraints`, e.g. `quality margins: fp8 kl=0.01 top1=0.02 gsm8k=1.0; int4 kl=0.03 top1=0.05 gsm8k=2.0`; tier keys as in the first column below).

| Tier (key in `constraints`) | `--margin-kl` (nats) | `--margin-agreement` (top-1, absolute) | `--margin-nll` (nats) | GSM8K max drop (points) |
|---|---|---|---|---|
| FP8 weights, FP8 KV (`fp8`) | 0.01 | 0.02 | 0.02 | 1.0 |
| 4-bit: NVFP4, MXFP4, INT4, Q4_K_M, IQ4_XS (`int4`) | 0.03 | 0.05 | 0.05 | 2.0 |
| 5-8 bit GGUF: Q5_K, Q6_K, Q8_0 (`gguf5-8`) | 0.01 | 0.02 | 0.02 | 1.0 |
| Below 4 bits: Q3_K, IQ3, IQ2, Q2_K (`sub4`) | 0.03 (the 4-bit default; no looser one) | 0.05 | 0.05 | 2.0 |

The conservative defaults when nobody set margins (`non_interactive`; log the choice in `QUESTIONS.md`), from proving-correctness's starting points (absolute KL: FP8 around 0.005-0.015, 4-bit 0.02-0.04): calibrate them to your measured floor, never widen them after seeing data. **Below 4 bits there is no looser default** (any default loose enough to pass that loss would pass anything): such rows use the 4-bit margins unless a `sub4` margin was agreed first, and a failing row is reported as a loss with its measured KL and GSM8K delta, never relabelled as passing.

## 5. Quality commands

```bash
# A/A floor: the reference's own greedy runs (restart your reference server between them)
inferencekit quality trajectory --base-url <reference url> --model <ref name> --prompts <dev set path> --label reference --run-dir runs/<id>
inferencekit quality trajectory --base-url <reference url> --model <ref name> --prompts <dev set path> --label reference-r2 --run-dir runs/<id>
# each row, on its exact speed configuration (relaunched from its manifest line)
inferencekit quality trajectory --base-url http://127.0.0.1:<port> --model <served name> --prompts <dev set path> \
  --label <row label> --run-dir runs/<id> --engine-kind vllm      # sglang | llamacpp | openai to match the engine
# one score per margin tier
inferencekit quality score --reference-url <reference url> --reference-model <ref name> \
  --trajectories runs/<id>/quality/vllm-awq --trajectories runs/<id>/quality/llamacpp-q4km \
  --baseline runs/<id>/quality/reference --baseline-repeat runs/<id>/quality/reference-r2 \
  --margin-kl 0.03 --margin-agreement 0.05 --margin-nll 0.05 --out runs/<id>/quality/score-4bit.json
# task accuracy: the reference first, then every row paired with it
inferencekit quality gsm8k --base-url <reference url> --model <ref name> --run-dir runs/<id> --label reference
inferencekit quality gsm8k --base-url http://127.0.0.1:<port> --model <served name> --run-dir runs/<id> --label <row label> \
  --paired-with runs/<id>/quality/gsm8k/reference.jsonl
# FP8 KV cache rows, or a long-context claim
inferencekit quality niah --base-url http://127.0.0.1:<port> --model <served name> --run-dir runs/<id> --label <row label> \
  --lengths 8192 --lengths 32768
```

- Use the same `--chat-template-kwargs` in every call when the model has a thinking switch.
- **GSM8K verdict (derived):** the record's `data.paired` gives the paired counts b = `only_this_correct`, c = `only_other_correct` and n = `n_paired`. Delta = (b - c) / n; its 95% CI = delta +/- 1.96 x sqrt(b + c - (b - c)^2 / n) / n. PASS when the CI's lower end is above minus the margin, FAIL when its upper end is below it, UNPROVEN otherwise. Quote the McNemar p next to it.
- A row's quality verdict combines its checks with evidence rule 4's precedence (FAIL > UNDERPOWERED > INCOMPLETE > NO-DATA > PASS; [evidence-rules.md](../../inference-triage/references/evidence-rules.md)): one FAIL fails the row, and a check that did not run is never a pass.
- The reference's own GSM8K accuracy goes in the table, so a reader sees the starting point.

## 6. The Pareto table and the recommendation

| Row (engine @ version, format) | Label | Decode tok/s median [CI95] (bucket) | Ratio to reference [CI95] | % of own ceiling | Mean KL vs floor (margin) | Top-1 vs floor | GSM8K acc [CI95], McNemar p | Quality verdict | Gates | On the frontier? | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vLLM x.y.z BF16 | reference | | 1.00 | | floor | floor | | — | PASS | | ev_... |
| vLLM x.y.z AWQ | cross-precision: format trade-off, not an engine comparison | | | | | | | PASS / FAIL / UNPROVEN | | | |

Capacity goals add `N*`, goodput and $/1M at `N*` per row.

1. Put every row in the table: measured, failed, did not fit, NOT RUN (with the reason).
2. Speed axis: the headline metric for the goal (decode tok/s or TTFT per bucket for single user, `N*` for capacity), with CI. Quality axes: mean KL increase over the floor with CI, and the GSM8K delta with its derived CI.
3. A row is **dominated** when another row is faster (paired ratio CI entirely above 1) and its quality is at least as good (verdict PASS, KL CI upper end not above this row's). Rows not dominated are on the frontier. Rows whose CIs overlap on both axes are ties: say so, never break them by point estimates.
4. The recommendation is the frontier row with the highest conservative speed whose quality is PASS and which fits the intake's context and concurrency. If the user's accepted loss is tighter than the tier, apply the tighter one.
5. If no quantized row passes quality, the recommendation is the reference precision; if the best passing row's speed CI includes the reference, the format has no proven speed benefit (state any memory benefit separately: fit, context, `N*`).

## 7. Wording

- Good (illustrative numbers): "On an RTX 5090 with vLLM x.y.z, AWQ INT4 decodes the medium bucket at least 2.1x the BF16 row (ratio CI [2.1, 2.3]; cross-precision: format trade-off, not an engine comparison), with mean KL +0.012 nats over the A/A floor (4-bit margin 0.03: PASS) and GSM8K -0.8 points (CI [-2.0, +0.4], McNemar p = 0.41: PASS at a 2-point margin)."
- Refuse: "AWQ is 2.2x faster than BF16" (no quality, no CI, no label); "vLLM is 3x faster than llama.cpp" when the rows also differ in format; "4-bit is lossless" (the test can only show a loss smaller than its margin).
- Every predicted ratio stays labelled **derived (ceiling)** and never appears in the measured column.
