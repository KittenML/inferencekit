# Harness commands for stages 4–5, in order

Run each command on the machine that runs the engines, **from the run directory** (`.` below), where a re-run starts engines too, so relative launch paths resolve the same: for `access: ssh` on the GPU host after `$IK` ([remote-host.md](remote-host.md)), which `cd`s to `~/ik/runs/<run-id>` and sets the evidence root; for `access: local`, `cd runs/<run-id>` first. A flag the CLI refuses means these notes are out of date: `inferencekit <command> --help` is the authority.

- **One host-side command per step, one blocking wait per job, `$STEP stop --all` before each sweep, gates session and quality run, and JSON or nested quotes in a deployed job script** ([remote-host.md](remote-host.md), "One command per step" and "GPU hygiene"). Never poll with `sleep`, `tail` or `cat` of a log. For `access: local`, the same with `--local`, from the run directory.
- **Output:** a terse view per evidence record (summary line, one line per check that did not PASS, `evidence: <path>`: `evidence/<id>/evidence.json`, plus `evidence/<id>.json` with `--run-dir .`). Read single fields with `inferencekit evidence show <id> --path <dotted path>`, never a whole record. `INFERENCEKIT_OUTPUT=json` prints the full JSON (`$STEP bench` sets it and saves `logs/<name>.json`).
- **Placeholders:** `<run-id>` is the run id. `<dev>` is the `path` that `prompts verify --set dev` printed on the host. `<max_tokens>` is `workload.max_tokens`. `<buckets>` is one `--buckets <b>` per intake bucket (e.g. `--buckets very_short --buckets short --buckets medium --buckets long`). `<port>` and `<served-name>` come from the engine's manifest entry. `<template>` is the chat-template argument for that command (table below).
- **Rounds (`<R>`), one rule for stages 4 and 5:** R is `workload.rounds` rounded **up** to a multiple of the balance period for the number of engines in that call (period: 2 engines 2, 3 engines 6, 4 engines 4, 5 engines 10). For example, 5 rounds with 2 engines becomes 6. The **floor** is the smallest multiple of the period that is at least 3 (2 engines 4, 3 engines 6, 4 engines 4, 5 engines 10); never run below it. `rounds/summary.md` must say `balanced: yes`. Or a pre-registered adaptive plan: `inferencekit bench rounds --ci-target <h> --min-rounds <n> --rounds <R_max>` stops only at whole Williams periods; a run that hits R_max is never extended by another call.
- **Stage 7 milestones** run as `$STEP job light|full` ([state.md](state.md), "Best so far") at fixed rounds (2, then 3): the floor and `balanced: yes` apply only to stages 4 and 5, and "order not balanced" is expected. An inconclusive result stays UNPROVEN, with no second call.
- **Timed sampling:** every timed call uses the workload's sampling: the run manifest's `workload.extra_body` for `bench rounds` (no flag needed), `--extra-body '<that JSON>'` for a `bench serve` screen.

## 0. Once per run
Locally, check the draft manifest, then copy it to the host ([remote-host.md](remote-host.md), "Getting the manifest to the host"):
```bash
inferencekit manifest validate runs/<run-id>/manifest.json --no-strict   # draft; fix every error before bench rounds
```
On the host, from the run directory:
```bash
inferencekit prompts verify --set dev        # verdict PASS; note "path" (= <dev>) and "sha256" (manifest harness.dev_prompt_set_sha256)
inferencekit prompts verify --set dev-load   # only when load mode runs; its "path" is <dev-load>
```

## Chat template: one setting, every call
Hybrid-thinking models (e.g. `enable_thinking`) think by default. Pass one setting, the same JSON, to every harness call for every engine (thinking **off** unless the intake says otherwise; logged in `QUESTIONS.md` and `state.decisions`).

| Command | How to pass it |
|---|---|
| `bench serve`, `bench rounds`, `gates run`, `quality trajectory`, `quality gsm8k`, `quality niah`, `quality sampling` | `--chat-template-kwargs '{"enable_thinking": false}'` (`bench rounds`: or the manifest's `workload.chat_template_kwargs`) |
| `gates spec-exact` | `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'` |

- **Not inside `--serve-args`:** `bench rounds` refuses a `--serve-args` `--extra-body` that differs from `workload.extra_body` ("give one value"), so a chat template sent that way breaks a sampled workload.
- **Record it:** write the setting in `REPORT.md` and in the manifest's `workload.chat_template_kwargs` (a re-run uses it).
- **An engine that rejects the field** (HTTP 400) is NOT SUPPORTED under the common protocol. Record its error.

## 1. Stage 4 screening (one config, server already up)
```bash
inferencekit bench serve --base-url http://127.0.0.1:<port> --model <served-name> --prompts <dev> \
  --engine-kind <vllm|sglang|llamacpp|trtllm|generic> --buckets short --buckets long --rounds 1 --max-tokens <max_tokens> \
  --label <label>-screen --run-dir . [--chat-template-kwargs '<JSON>'] [--extra-body '<workload.extra_body JSON>']
```
- **`--engine-kind` is required in practice.** The default `generic` sends no exact-length field, so outputs stop at EOS and the lengths no longer match the other engines.
- **Screening only ranks configs** (two buckets, short and long: [tuning.md](tuning.md), step 3). It never decides keep or drop. Read TTFT and TPOT as well as the lead metric: a config ahead on one of them goes to `bench rounds` too (tuning.md, step 4).
- **As one step:** `$STEP bench <label>-screen -- <the bench serve flags above>`. It prints one verdict line with the evidence id and summary, and keeps the full JSON in `logs/<label>-screen.json`.

## 2. Full protocol: `bench rounds` (stage 4 keep/drop and stage 5 ranking)
Every config under test is a manifest engine entry (its own label, port and launch line). Copy the manifest to the host after every change ([remote-host.md](remote-host.md)). Pass the same workload a re-run passes: `<buckets>` and `--max-tokens` inside `--serve-args`, the chat template with `--chat-template-kwargs`; the timed sampling comes from the manifest's `workload.extra_body`.
```bash
inferencekit bench rounds --manifest manifest.json --prompts <dev> --rounds <R> --mode single --engines <labels> \
  --serve-args "<buckets> --max-tokens <max_tokens>" --chat-template-kwargs '{"enable_thinking": false}'
```
- **What it does:** launches each engine fresh, once per round, in a balanced order, and runs `bench serve` against it.
- **As steps:** write the command above into `runs/<run-id>/jobs/rounds-<n>.sh` locally; then `$STEP stop --all`, `$STEP deploy runs/<run-id>/jobs/rounds-<n>.sh jobs`, `$STEP start rounds-<n> -- 'bash jobs/rounds-<n>.sh'` and one `$STEP wait rounds-<n>` (again on `RUNNING`) until `FINISHED`. `$STEP compare` then prints one line per engine from `rounds/summary.json`.
- **Output:** `rounds/summary.md`, and on stdout a short summary (best engine, each engine's ratio to the best with its 95% CI) plus `evidence: <path>`; `INFERENCEKIT_OUTPUT=json` prints the full record.
- **Which engines:** always pass `--engines`. Without it every manifest entry runs, capacity entries included. Stage 4: the configs under test. Stage 5 single mode: the single-user entries (every entry except the capacity ones). Load mode: the capacity entries (labels like `vllm-capacity`).
- **Fit the stage box before you start (stages 4 and 5):**
  1. Estimate: engines × R × (startup time from stage 3 + warm-up + one pass over the buckets you pass). Compare it with what `inferencekit intake budget runs/<run-id>` leaves in the stage box.
  2. Prefer one call over several pairs: `--engines <recipe>,<c1>,<c2>,<c3>` at the 4-engine floor is 16 launches; three pairwise calls at the 2-engine floor are 24.
  3. If it still does not fit, drop candidates first: keep the `baseline-recipe` plus the best one or two by screening. Report the others' screening numbers as UNPROVEN ("not run under the full protocol").
  4. Stage 5 never runs fewer than R. Stage 4 may run at the floor instead of R if you write that in `state.decisions` before the call. Never go below the floor, and never merge rounds from separate `bench rounds` calls into one comparison.
- **Reading the result:** the rule `compare` applies (`stats.decide`; the table is in the benchmarking skill's [statistics reference](../../benchmarking/references/statistics.md), "Reading the result"), on the pre-registered lead metric in every target bucket, with the effect `e` written in `state.decisions` before the call (default 0.02). `e` is a detection threshold for claims, not a stopping point. For each engine that is *not* best, the best engine's advantage A is 1 / ratio for tok/s (CI [1 / CI high, 1 / CI low]) and the ratio as printed for latencies. `improved` for the best engine is `regressed` for that row. `inconclusive`: more rounds in a new call if the box allows, otherwise UNPROVEN.
- **Keep or drop (stage 4):**
  - **Keep** a config only when it is the best engine and the incumbent's row (the best so far; first the recipe) reads `improved` for it in at least one target bucket, and it is `regressed` in none.
  - **`within_noise`:** drop it and stop exploring that knob. **`regressed`:** drop it.
  - **`inconclusive`:** do not keep it; record it as UNPROVEN. That alone is not a reason to stop exploring the knob.
  - With more than two engines, only the best engine has a verdict against each other row. A second candidate that is not best is not kept.
- **`compare`:** use `bench rounds`' own paired results for its aggregate. `inferencekit compare` accepts matching decode, single-user serve, clip, image or audio records; load-mode and round-summary records are not supported.
- **Capacity (load mode):** use the `dev-load` set (buckets `short` and/or `medium` only) and give every engine the same arrival schedule. `--concurrency` repeats once per level, for every level of the grid (extended until the SLO fails). Size each level for at least 100 requests ([capacity.md](capacity.md), section 1):
  ```bash
  inferencekit bench rounds --manifest manifest.json --prompts <dev-load> --rounds <R> --mode load \
    --engines <capacity labels> --serve-args "<buckets> --max-tokens <max_tokens> --concurrency 1 --concurrency 2 \
    --concurrency 4 --concurrency 8 --ref-e2e-ms <e2e_ref> --duration-s <s> --max-requests-per-level <m> \
    --slo-ttft-p95-ms <ttft> --slo-tpot-p95-ms <tpot> --gpu-count <GPUs per server> \
    --gpu-price-usd-h <price_usd_h / count>" <template>
  ```
  - `<e2e_ref>`: the baseline's unloaded end-to-end latency in ms, the same for every engine. `<s>` = 125 × e2e_ref (s) ÷ the lowest level, rounded up (a mean of 125 leaves a margin over 100; check the seeded count first, [capacity.md](capacity.md) section 1); `<m>` = the larger of 100 and 10 × the top level.
  - `--gpu-price-usd-h` is per GPU: `hardware.price_usd_h` ÷ `hardware.count`. `--gpu-count` is the number of GPUs the server uses ([parallelism.md](parallelism.md)).
  - Add `--sweep-all` only to see the curve past the first failing level, when the box allows.

## 3. Gates, per engine, while its server is up
`bench rounds` stops every server. To gate an engine, relaunch it from its manifest line (same argv, env and port): `$STEP start <label> --ready-url http://127.0.0.1:<port><ready_check> -- '<its env as VAR=value> <launch line>'`, then `$STEP wait <label>` until `READY`. Then:
```bash
inferencekit gates run --base-url http://127.0.0.1:<port> --model <served-name> --engine <label> \
  --run-dir . --reference-engine <baseline-recipe label> [--spec-off-url <same model, no speculation>] \
  [--chat-template-kwargs '<JSON>']
```
- **Keep the server up** for the quality trajectory below, then stop it: `$STEP stop <label>`.
- **Verdicts:** a gate FAIL voids that engine's speed numbers. `data.verdict` UNPROVEN or NOT RUN is not a pass.

## 4. Quality
The reference is the intake's `quality.reference.base_url`, a vLLM server (`--base-url` / `--reference-url` take it with or without `/v1`; `<reference root>` drops the `/v1`; its model name is the `id` from `<reference root>/v1/models`).
- **Engine trajectories** run on the GPU host (engines listen on 127.0.0.1 there); **reference trajectories and `quality score`** wherever the reference answers (`curl -s <reference root>/v1/models` on the host first). Unreachable from the host: run them locally after syncing `quality/<label>` and `evidence/` back, with `--run-dir runs/<run-id>` and the `path` that a local `inferencekit prompts verify --set dev` prints (same `sha256` as the host's).
- **The A/A repeat (`reference-r2`, when `quality.noise_floor` is `measure`)**: the same command with `--label reference-r2`, restarting a reference server you started yourself in between (a remote endpoint: twice back to back, and `REPORT.md` says the floor has no restart).
```bash
# the reference's own greedy run (the A/A floor), then the repeat
inferencekit quality trajectory --base-url <reference url> --model <reference id> --prompts <dev> --label reference \
  --run-dir . --engine-kind vllm --max-tokens <max_tokens> [--chat-template-kwargs '<JSON>']
inferencekit quality trajectory <same arguments> --label reference-r2
# each engine, its exact speed configuration up (same session as its gates)
inferencekit quality trajectory --base-url http://127.0.0.1:<port> --model <served-name> --prompts <dev> --label <label> \
  --run-dir . --engine-kind <vllm|sglang|llamacpp|openai> --max-tokens <max_tokens> [--chat-template-kwargs '<JSON>']
# score every engine against the floor; --trajectories repeats once per engine
inferencekit quality score --reference-url <reference url> --reference-model <reference id> \
  --trajectories quality/<label-a> --trajectories quality/<label-b> \
  --baseline quality/reference --baseline-repeat quality/reference-r2 \
  --margin-agreement <m> --margin-kl <m> --margin-nll <m>
```
- **Margins:** write them into `state.decisions` before you score. The defaults are loose placeholders. Same-precision starting points are in the proving-correctness skill's [quality reference](../../proving-correctness/references/quality.md), section 4.
- **`quality.reference: none`:** use the `baseline-recipe` engine as the reference, if it is vLLM (scoring needs vLLM's `prompt_logprobs`). Otherwise, the proving-correctness [no-reference route](../../proving-correctness/SKILL.md#no-higher-precision-reference); without it, UNPROVEN, with that reason.
- **Task accuracy (all 1,319 at stage 5, for a task-named baseline and for final or scored commits; `--limit 100` in light checks):** `inferencekit quality gsm8k --base-url ... --model ... --run-dir . --label <label> --concurrency 0`. Add `--paired-with <baseline results jsonl>` for candidates.

## 5. After each stage
Sync the run directory back ([remote-host.md](remote-host.md)). Locally, with `INFERENCEKIT_RUNS_DIR` set, check `inferencekit evidence show <id>` for each id you will cite.

At stage 8, run these locally:
- `inferencekit review build runs/<run-id>`;
- `inferencekit ledger trailers <ev_rounds> <ev_quality_score> [<ev_gates>] --label <label>`, once per milestone commit (other kept commits carry trailers from their correctness record, plus their screen's record when the screen was `bench decode`);
- `inferencekit history --repo <repo.path>`.
