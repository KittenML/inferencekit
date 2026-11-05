# Serving benchmarks: single-user and capacity with `bench serve`

The flow optimises two goals, and the intake says which one leads (`goals.lead`: `single_user` or `capacity`). Both are measured by the same client, `inferencekit bench serve`, against each engine's OpenAI-compatible server, driven round by round by `inferencekit bench rounds` (fresh server per cell, Williams order, GPU lock, resolved-config capture).

## Terms

- **TTFT (time to first token):** from just before opening the connection to the first token-bearing streamed event; connection time is included and also recorded separately.
- **TPOT (time per output token):** `(t_last - t_first) / (tokens after the first event)`; the gaps between token events are recorded too (inter-token latency). Report p50 and p95.
- **Decode tok/s:** the inverse of TPOT: tokens after the first token-bearing event, divided by the time between the first and last token-bearing events. Why: the first event carries prefill; mixing it in makes long prompts look like slow decode.
- **End-to-end (text client):** from just before opening the connection to the completion marker, or stream end when no completion marker is sent.
- **SLO (service-level objective):** the latency a user must get, e.g. p95 TTFT <= 500 ms and p95 TPOT <= 50 ms, plus zero failed requests.
- **Open loop:** requests arrive on a fixed, seeded schedule whether or not earlier ones finished, like real users. **Closed loop:** each client sends its next request only when the previous one finished.
- **Goodput:** requests per second that met the SLO (not all requests completed per second).
- **`N*` (capacity):** the highest number of concurrent streams per GPU at which every SLO holds at that level **and at every lower level**: p95 TTFT <= target, p95 TPOT <= target (for streaming audio: zero playback underruns), 0 failed requests, and quality inside the noise floor. Why "every lower level": a lucky pass at 32 after a fail at 16 is noise, not capacity.

## Input buckets

| Bucket | LLM prompt |
|---|---|
| very_short | 1-5 words |
| short | one sentence, ~20-50 tokens |
| medium | a paragraph, ~300 tokens |
| long | ~2,000-4,000 tokens of context |

Report every metric per bucket. A single "average prompt" hides that TTFT at 4,000 tokens is a prefill (compute) problem and at 5 words a host-overhead problem. The dev prompt sets ship with inferencekit: `dev` (16 prompts per bucket, single-user and quality) and `dev-load` (short and medium prompts, enough for unique prompts at every load level); `inferencekit prompts show --set dev` gives counts and lengths. A sealed set with different items (`inferencekit prompts seal`) is only for re-running the final numbers on prompts the engine never saw. Check the set with `inferencekit prompts verify --set dev` (or `--set <path.jsonl> --expected-sha256 <pre-registered>`) before each session; it prints the file path to pass as `--prompts`.

## Single-user mode (Goal 1)

- One request at a time, sequentially over the prompt set; each request asks for its `max_tokens` (`--exact-length` asks for exactly that many where the engine supports it; the output-length gate checks the rest).
- R rounds (the pre-registered count; statistics.md), run by `inferencekit bench rounds --manifest runs/<id>/manifest.json --prompts <set.jsonl> --rounds R --mode single`. In each round every engine is started fresh from its manifest launch line, checked ready (`ready_check`, default `/v1/models`), warmed up, measured with one `bench serve --rounds 1` call (chained across rounds with `--extend`), and stopped; the engine order across rounds is a Williams design.
- Do not run `bench serve` on its own with its default `--rounds` (5): those passes share one server process, so the CI "over rounds" would be a CI over one process. Manual fallback only: a fresh server per call, `bench serve --rounds 1 --engine-kind <kind> --label <label> --extend <previous ev of that engine>`, in a balanced order you write down.
- Per bucket: TTFT, TPOT, decode tok/s and end-to-end, median with a bootstrap CI over rounds.
- Token counts come from streamed token ids where the server returns them, else `usage.completion_tokens`, else by re-tokenising with the model's tokenizer; the record states which. Why: counting methods disagree when an engine merges or splits tokens in the stream.

## Load mode (Goal 2: capacity)

- **Command:** `inferencekit bench rounds --manifest runs/<id>/manifest.json --prompts <dev-load set.jsonl> --rounds <R_load> --mode load --serve-args "--concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 --concurrency 16 --concurrency 32 --concurrency 64 --ref-e2e-ms <ms> --seed 0 --slo-ttft-p95-ms <ms> --slo-tpot-p95-ms <ms> --gpu-price-usd-h <price>"`. `--concurrency` is repeated once per level (or `--rates` for request rates); `--duration-s` sets each level's arrival window. A load sweep is long, so the pre-registration may set fewer load rounds than single-user rounds; with one round, `N*` has no CI over rounds: say so.
- **Arrivals:** seeded Poisson; level N is offered as the rate N / e2e_ref, with the same `--ref-e2e-ms` for every engine (e.g. the baseline's unloaded end-to-end latency), so all get the identical schedule. At least 100 requests per level (size `--duration-s` for a mean of about 125 at the lowest level, check `n_requests`, cap the top levels with `--max-requests-per-level`; p99 only from levels with 1,000 or more); record `duration_s`.
- **Extend the grid** while the SLO holds at its top level (N* is then only a lower bound), and **check the batch cap against the measured in-flight count** (`mean_inflight`, running/waiting from `/metrics`), not N. Sizing, the cap check and the frontier in detail: the optimizing-inference skill's [capacity.md](../../optimizing-inference/references/capacity.md).
- **Unique prompts:** no prompt repeats inside a level. Why: repeats hit prefix caches and fake capacity.
- **Per level:** p50/p95/p99 TTFT and TPOT, failures, goodput, output tok/s per GPU.
- **Outputs:** `N*`; goodput and throughput at `N*`; the latency-vs-concurrency curve with CIs and the knee (where p95 starts rising steeply) marked; cost. Across several serving configs, a Pareto frontier: output tok/s per GPU against tok/s per user (output tok/s ÷ mean in-flight), keeping only levels at or below each config's `N*`, all from the same load prompt set (never mix in single-user rows measured on another set).
- **Cost (derived):** `$ per 1M output tokens at N* = gpu_price_usd_h / (output tok/s per GPU at N* x 3,600) x 1e6`, with `--gpu-price-usd-h` from the intake. Show the formula and the price source.
- **Gates:** after the sweep, relaunch each engine from its manifest line and run `inferencekit gates run --base-url <its url> --engine <label> --run-dir runs/<id>` while it is up; `unique_prompts` reads the load schedule from the run directory.
- **Tuning:** every engine gets its own capacity tuning (batching limits, KV-cache size, scheduling knobs), separately from its single-user tuning. `N*` uses each engine's best serving configuration. Why: the best single-user config is rarely the best serving config, and comparing one engine's tuned serving config against another's single-user config is a strawman.
- **Closed loop:** report only as a stress test, labelled. Why: under overload a closed loop slows its own arrivals, so latency looks bounded while real users would queue.

## Input and output modes

The latency a user feels depends on how input arrives (complete or streaming, and at what rate and chunk size) and how output is consumed (complete or streamed). The intake records both, but the shipped text and audio clients send complete JSON inputs; streaming-input measurements need a dedicated adapter. If an engine does not support a mode natively, say NOT SUPPORTED; an emulation (e.g. client-side chunking) is allowed only when labelled as such.

Latency definitions per modality:
- **LLM:** time to first token from just before opening the connection; time per output token.
- **TTS:** time to first audio from the first text chunk; end-of-input latency (last text chunk to last audio); gaps between audio chunks and playback underruns.
- **STT:** first-partial latency; finalisation latency after end of speech; partial stability; real-time factor.

TTS and STT buckets: very_short 1-5 words (text) or ~1-2 s (audio); short ~10-20 words or ~5-10 s; medium ~60-120 words or ~30-60 s; long ~300-500 words or ~5-10 min.

## What the record holds

- `kind=serve_benchmark` (single and load) and `kind=capacity` (the `N*` summary), evidence schema v2, with workload fields `mode`, `bucket`, `concurrency`, `rate_rps`, `duration_s`, `prompt_set {name, sha256}`, `slo`, `endpoint_kind`.
- Every request's text and token ids under `runs/<run-id>/outputs/<engine>/<bucket>/<id>.json`. Why: the gates and the quality checks read the exact outputs that were timed.

## Fixed capacity table

| Engine (serving config) | SLO | Level | Arrival rate req/s | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | Failures | Goodput req/s | Output tok/s per GPU | SLO held |
|---|---|---|---|---|---|---|---|---|---|

Then one line per engine: `N*` (observed), goodput and tok/s at `N*` (observed), $/1M at `N*` (derived, formula shown), knee (inferred from the curve).
