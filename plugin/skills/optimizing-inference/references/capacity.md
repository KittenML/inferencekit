# Capacity: goodput at the SLO, the load grid, the frontier and the replica plan

For runs whose `derived.modes` includes `load` (`goals.lead: capacity` or an `n_star` target). The method (open loop, N\*, the capacity table) is in the benchmarking skill's [serving reference](../../benchmarking/references/serving.md); this page adds what experienced engineers do on top.

## Terms

- **SLO:** the latency every user must get, here p95 TTFT and p95 TPOT, plus a failure budget (`goals.slo`).
- **Goodput:** requests per second that met the SLO. Requests that finished late do not count.
- **N\*:** the highest concurrency level at which the SLO holds at that level **and at every lower one**. A pass at 32 after a fail at 16 is noise, not capacity.
- **In flight:** requests sent and not yet finished. The record's `mean_inflight` is the time average over the level (Little's law).
- **tok/s/user** (derived) = output tok/s ÷ mean in-flight requests. It is the speed one user sees.
- **tok/s/GPU** = output tok/s ÷ the GPUs behind the server (the record's `output_tok_s_per_gpu`; `--gpu-count`).
- **ISL_m, OSL_m:** the prompt and output lengths the load test actually sent (section 7 says how to get them). They are not the real workload's ISL and OSL ([workload.md](workload.md)), and every number on this page belongs to them.

## 1. At least 100 requests per level

- **Requests per level:** load mode offers concurrency N as a Poisson rate of N ÷ e2e_ref (the unloaded end-to-end latency), so a level sends N × duration_s ÷ e2e_ref requests on average (derived), a Poisson draw around that mean.
- **The seed fixes the count** (`--seed`, default 0, plus the level's index in the sorted grid; `bench rounds` reuses it): every round and engine gets the **same** count, so a short level stays short.
- **Every level needs at least 100 requests:** the p95 of 20 is the second-worst one, a guess (the tool only warns below 20). Quote p99 only at levels with at least 1,000.
- **Settings, passed inside `--serve-args`:**
  - `--ref-e2e-ms <e2e_ref>`: the baseline's unloaded end-to-end latency, the same value for every engine, so every engine gets the identical schedule;
  - `--duration-s` = **125** × e2e_ref (s) ÷ the lowest level in the grid, rounded up. The margin matters: a mean of exactly 100 falls short of 100 about half the time; a mean of 125 falls short about 1% of the time (Poisson, derived);
  - `--max-requests-per-level` = the larger of 100 and 10 × the top level. This keeps the top levels short: each still runs for about 10 × e2e_ref, long enough for the in-flight count to settle.
- **Check the lowest level's count before the run** (schedule only: no GPU or server, under a second), locally with inferencekit's Python (`head -1 "$(command -v inferencekit)"`; the same harness version as the host's):
  ```bash
  <that python> -c 'from inferencekit.serve.loadgen import poisson_schedule as ps, level_seed as ls
  print(len(ps(<lowest level> / <e2e_ref s>, <duration_s>, ls(<seed>, 0), <max-requests-per-level>)))'
  ```
  Below 100: add 10% to `--duration-s` and check again (only the lowest level needs this).
- **Example:** e2e_ref = 2.5 s and grid 1–64 give `--duration-s 313 --max-requests-per-level 640`.
  - Levels 1, 2 and 4 run 313 s each: about 125, 250 and 500 requests on average. With `--seed 0` the counts are 103, 256 and 493 (`poisson_schedule`, computed). With `--duration-s 250` (a mean of 100), level 1 would get 87 and decide nothing.
  - Levels 8 to 64 stop at 640 requests, after about 190, 105, 50 and 25 s.
  - That is about 22 min per engine per round, plus drain (derived). The `dev-load` pool is smaller than that; reused prompts get a unique prefix tag, and the record counts them.
- **Fit the stage box** with fewer load rounds or fewer capacity entries, and write the choice in `state.decisions` before the call. Never go below 100 requests per level.
- **Check** `n_requests` per level in the record (`data.levels[].summary`). A level below 100 cannot decide N\*: that N\* is UNPROVEN.
- **A re-run of the manifest** replays the grid with `bench serve`'s default 30 s window, because the manifest has no field for the window. Its low levels have fewer requests: say so next to any reported N\*.

## 2. Extend the grid until the SLO fails

- `bench serve` stops a sweep at the first level that misses the SLO (`--sweep-all` runs the rest).
- If the capacity record says "SLO holds at every level tried" (and its limitation says "the SLO held at the top of the grid"), N\* is only a **lower bound**. Double the top level (64 → 128 → 256, ...) and measure again. Stop at the first failing level, or when the KV cache cannot hold the level: KV cache tokens from the startup log ÷ (ISL_m + OSL_m) tokens per sequence is the most sequences that fit (derived).
- **Find the failing level cheaply first** with a one-off `bench serve --mode load` against a running server (screening), then run `bench rounds --mode load` once with the final grid, the same grid for every capacity entry.
- **Pre-registration:** `workload.concurrency` is the floor; write the extension rule in the plan (or `state.decisions`) before the run and the final grid in `manifest.json` `workload.concurrency` (a re-run replays it). Higher levels never lower a target.

## 3. Is the engine's batch cap setting N\*?

Open loop offers N as a rate, not N workers: a saturated server holds more than N in flight, a short window fewer. Compare the engine's cap with the **measured** in-flight count, not with N.

Read three numbers at N\* and at the first failing level:
1. **The resolved cap** from the startup log or `resolved_config`: vLLM `max_num_seqs` (per data-parallel rank), SGLang `max_running_requests`, llama.cpp `n_seq_max` (`-np`), TRT-LLM `max_batch_size`.
2. **`mean_inflight`** from the capacity record (`data.curve[]`).
3. **Running and waiting requests** during the level:
   - vLLM `/metrics`: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc` [vLLM metrics];
   - SGLang decode log lines: `#running-req`, `#queue-req`, `token usage` [SGLang tuning guide].

What they mean:
- **Cap-bound:** requests are waiting while running sits at the cap. The cap sets N\*, not the GPU. Raise the cap (and the KV memory and graph capture sizes to match) in the capacity config, prove it in effect, and measure again.
- **KV-bound:** running is below the cap, requests are waiting, and KV cache usage is near 100%. Give the KV cache more memory (`--gpu-memory-utilization`, `--mem-fraction-static`), lower the maximum context length to what the workload needs, or use an fp8 KV cache if `kv_cache_dtype_allowed` permits it (a precision track).
- **Under-offered:** `mean_inflight` < 0.9 × N\*. The capacity record flags this. Read the capacity as about `mean_inflight` streams, or rerun with a longer `--duration-s`.

Record the three numbers for each capacity entry in `experiments.jsonl` and in `REPORT.md`.

## 4. Diagnose under load: bandwidth use per level

- **MBU (model bandwidth utilisation) at a level** (derived) = (W + n_run × KV_seq) ÷ TPOT_p50 ÷ BW, where:
  - W = weight bytes read per decode step: roofline's weight term (for MoE, use the experts the batch activates; all experts is the upper bound);
  - n_run = running sequences (from the engine's metrics; `mean_inflight` is an upper bound, because it includes queued requests);
  - KV_seq = KV bytes per token × L, with L the mean context ≈ ISL_m + OSL_m ÷ 2;
  - TPOT_p50 = seconds per decode step;
  - BW = the measured `ceiling bw`.
- **B_crit** (inferred) is the batch size at which the weight read stops dominating: B_crit ≈ peak dense FLOP/s × bytes per weight ÷ (2 × BW).
  - For an H100 with bf16 weights and compute: 989e12 × 2 ÷ (2 × 3.35e12) ≈ 295.
  - 8-bit weights with bf16 compute roughly halve it.
  - Below B_crit, the batch shares one weight read, but each sequence still reads its own KV cache. At long context, step time grows with N well before B_crit.
- **Reading it:**
  - MBU near 1 at N\*: bandwidth-bound. The levers are fewer bytes (precision or KV dtype, only within what the intake allows), not the scheduler.
  - Low MBU with GPU idle gaps between kernels in nsys: overhead (CUDA graphs that do not cover the batch sizes reached, scheduler CPU time).
  - Low MBU, no gaps, and N at or above B_crit: compute-bound (large batches or prefill).
- Do not use the "double the batch and runtime rises only ~10%, so it is overhead-bound" test. Memory-bound decode passes it too, because the batch shares one weight read.

## 5. The Pareto frontier across configs

One config's levels give one curve. A frontier needs every capacity config.
- **Points:** every level of every capacity config measured with `bench rounds --mode load`, taken from each round's capacity record (`data.curve`, median over rounds per level). **Every point comes from the `dev-load` set**, so the frontier compares one workload.
- **The N = 1 end** is each config's level-1 load point (the grid starts at 1). Do not add the single-user rows: they come from the `dev` set (every intake bucket, `long` included) and a separate config, so they differ in prompt length, not only in config. Show them, if at all, in a separate table labelled "single-user, `dev` set, not on the frontier".
- **Axes:** x = tok/s/user, y = tok/s/GPU. vLLM's `vllm bench sweep plot_pareto` uses the same axes, with x = throughput ÷ concurrency [vLLM sweeps].
- **Drop** every point that misses the SLO, and every point above its config's N\*.
- **The frontier** is the points no other point beats on both axes. **The operating point** is the frontier point with the highest goodput.
- **`--sweep-all`** adds the levels past the first failure. They show the knee, but they are never frontier points. Use it only when the box allows.
- **In `REPORT.md`:** a table of config, level, tok/s/user, tok/s/GPU, goodput and SLO held, plus a plot if you can (`review build` does not draw it yet); screening points only labelled "screening". y is per GPU so different GPU counts compare ([parallelism.md](parallelism.md)).

## 6. The ranked table when the lead is capacity

| Rank | Engine / label | Precision | GPUs | N\* (streams) [CI] | Goodput at N\* (req/s) | Output tok/s/GPU at N\* | $/1M output tok at N\* | p95 TTFT / TPOT at N\* (ms) | Single-user decode tok/s [CI] | Single-user TTFT (ms) | Quality vs floor | Gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

- **Order** by N\*, then goodput. The values come from `bench rounds --mode load` (medians over rounds, with a CI over rounds when there are at least 3; with one round there is no CI, so say so).
- **The single-user columns** come from the same engine's single-user entry (its own config, on the `dev` set with every intake bucket). Label them as such: they are a different prompt set from the load columns, so never divide one by the other.
- **Separate tracks** (another precision, another GPU count) go in their own blocks, as in [stop-point.md](stop-point.md).

## 7. Replica and cost plan (stage 8)

**Inputs:** the peak RPS (`goals.traffic.arrival_rate_rps`), plus the operating point's goodput at N\* per replica, its GPUs per replica, and its price per GPU-hour (`hardware.price_usd_h` ÷ `hardware.count`). A replica is one server endpoint: one GPU, or one tensor-parallel group.

**The lengths are the measured ones, not the real workload's.** The goodput was measured on `dev-load` prompts, so every token conversion uses what those requests carried:
- **ISL_m** = the mean prompt length of the `dev-load` buckets you passed: nominally ~35 tokens for `short` and ~300 for `medium` (a word-count proxy). Prefer the engine's own count when it has one: vLLM `vllm:prompt_tokens_total` ÷ the requests served [vLLM metrics]. Say which you used.
- **OSL_m** = `workload.max_tokens`: every request generated exactly that many tokens.

```
replicas        = ceil(peak RPS / goodput per replica at N*)                   (derived)
GPUs            = replicas x GPUs per replica
$ per hour      = GPUs x price per GPU-hour;   $ per month = $ per hour x 730
$ per request   = $ per hour per replica / (goodput per replica at N* x 3600)
$ per 1M tokens = $ per request / (ISL_m + OSL_m) x 1e6                         (input and output together)
split, output:input price ratio r:  input $ per 1M = $ per request / (ISL_m + r x OSL_m) x 1e6;  output = r x input
```

- **Show every formula with its inputs.** Cite the evidence id of the goodput figure.
- **Valid only for the measured length mix.** The replica count assumes real requests cost what `dev-load` requests cost. When the real ISL or OSL (the `workload:` constraint) differs from ISL_m or OSL_m, add a caveat to `REPORT.md` (inferred) and a row in the gap table ([workload.md](workload.md)):
  - real prompts longer than ISL_m (e.g. a real p50 of 900 tokens against `medium` at ~300): each request costs more prefill and holds more KV cache, so goodput per replica is overstated and **the replica count is too small**;
  - real outputs shorter than OSL_m (e.g. `max_tokens` set to the OSL p90): each request is cheaper than measured, so the count is conservative;
  - never mix the two: do not divide a cost measured at ISL_m + OSL_m by the real ISL + OSL.
- **The ratio r** is an assumption: name it. The NVIDIA cost blog splits by the ratio of list prices, e.g. $1 in to $3 out (r = 3) [NVIDIA cost].
- **Harness cost:** the harness's `$/1M output tok at N*` charges every GPU-hour to output tokens. Show it next to the split.
- **Bursts and failover:** the plan sizes replicas for the peak RPS with Poisson arrivals. Say whether it includes headroom for bursts or one spare replica for failover (an operations choice, inferred).
- **No `arrival_rate_rps`:** the replica plan is NOT RUN. Still give $ per request and $ per 1M tokens.
- **A single-user lead** gets the same plan with N\* = 1 when a peak RPS is given: one replica serves 1 ÷ e2e requests per second.

## Sources (read 2026-10-14)

- [vLLM metrics] https://docs.vllm.ai/en/latest/design/metrics.html
- [SGLang tuning guide] https://docs.sglang.io/advanced_features/hyperparameter_tuning.html
- [vLLM sweeps] https://docs.vllm.ai/en/latest/benchmarking/sweeps/
- [NVIDIA cost] https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost
- B_crit and per-sequence KV reads: https://jax-ml.github.io/scaling-book/inference/
