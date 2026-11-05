# Reward hacks and the gates that catch them

A reward hack is anything that makes the number better without making the engine better for the workload you claim. Agents and humans both drift into them under pressure, and an agent told to "make it faster" will find the cheap ones first. Each line: the hack, why it inflates the number, and how to detect it.

## Gates: `inferencekit gates run` (MCP `run_gates`)

Run once per engine, against its **live** server and the run directory together:

```bash
inferencekit gates run --base-url http://127.0.0.1:8000 --engine <label> --run-dir runs/<id> \
  --reference-engine <baseline label> [--reference-url <reference endpoint>] [--spec-off-url <same model, no speculation>]
```

- `--base-url` drives the live probes (caching, token count, output length with `ignore_eos`, hidden probes, rare tokens, and spec exactness with `--spec-off-url`). They need the server up, so run them before stopping the engine: after `bench rounds`, relaunch each engine from its manifest line, gate it, stop it. A live gate run without a server is NOT RUN, never a pass.
- `--run-dir` + `--engine` add the static checks on the timed outputs (`outputs/<label>/...`): token counts per request, per-prompt lengths against `--reference-engine`, and unique prompts per load level from the schedule files.
- `--reference-url` excuses rare-token misses the reference shares; `--spec-off-url` (or `inferencekit gates spec-exact --a <spec url> --b <no-spec url>`) runs the exactness gate.

Each gate ends PASS / FAIL / UNPROVEN / NOT RUN in a `kind=gates` record; any FAIL makes that engine's speed numbers unusable for a claim. Exact checks and thresholds: `inferencekit gates run --help`.

| Gate | What it catches | How it works (in outline) |
|---|---|---|
| Caching probe (`caching`, live) | prefix, radix or response caching that serves repeated work (TTFT near zero, or a whole response replayed) | replays prompts mixed in with other probes: a repeat may not be more than 10% faster (TTFT or decode) than its first sighting, and a prompt changed by one character must not get the old answer back |
| Token-count consistency (`token_count`, live + run dir) | tok/s inflated by counting prompt tokens, or a stream whose chunks do not match the claimed token count | `usage.completion_tokens`, streamed token ids (or logprob entries) and a re-tokenisation of the text (with `--tokenizer`) must agree (a logprob stream short of usage at max_tokens is UNPROVEN unless the re-tokenised text covers usage; copied rare/echo strings and text after EOS are not held to the re-tokenised count) |
| Output-length gate (`output_length`, live + run dir) | early EOS or stop strings that shorten generation (fewer steps, less KV read) | with `ignore_eos` the output is exactly `max_tokens` long; in the run dir, per-prompt lengths stay within 10% of the reference engine's |
| Unique prompts under load (`unique_prompts`, run dir) | repeated prompts inside a load level that hit caches and fake capacity | the load schedule never repeats a prompt id or prompt hash within a level |
| Speculative-decoding / MTP exactness (`spec_exact`, live) | a lossy draft (speculative decoding, multi-token prediction) presented as the same model | greedy outputs with speculation on vs off: early divergences may not exceed the A/A floor of the no-speculation engine run twice; otherwise it is an approximation and needs a quality verdict |
| Hidden logprob and T > 0 probes (`hidden_probes`, live) | an engine that ignores sampling settings (always greedy, or cached text), or returns fabricated logprobs | T = 1 requests (top_p 1, top_k off) must not equal the greedy output on every prompt (8 prompts by default, `--sampling-prompts`; 3/4 or more equal is UNPROVEN, all equal is FAIL); top-k logprobs must be finite, normalised and sorted, and the greedy token must be their argmax. Probes are shuffled in with ordinary traffic so an engine cannot tell them apart by position |
| Rare-token pack (`rare_tokens`, live) | a lossy pruned or truncated vocabulary in the lm_head, a logit mask, or a tokenizer shortcut, that speeds the head up | CJK, emoji, code identifiers, long numbers and uncommon Unicode must be reproduced verbatim (misses the reference shares are not counted; a slip in the random ASCII text around a string is recorded, not counted) |

Special-casing of the published prompts is caught separately: re-run the final numbers from `manifest.json` on a prompt set the engine never saw (`inferencekit prompts seal`), and score quality per engine against a reference (proving-correctness).

## The full checklist

| # | Hack | Why it inflates | How to detect |
|---|---|---|---|
| 1 | Lower precision in one engine (weights, activations, KV cache, or lm_head) | Fewer bytes per token in a memory-bound workload | Compare `identity.model.dtype/quant` and the logged KV dtype and GEMM kernel for every engine |
| 2 | Speculative decoding or MTP in one engine | Several tokens per weight read; speed depends on draft acceptance, which depends on the prompt | Resolved flags; exactness gate; tok/s varies strongly by prompt |
| 3 | Prefix/radix cache or a response cache with repeated prompts | Prefill skipped (TTFT) or the whole response served from cache | Caching probe; unique-prompts gate; TTFT near zero after the first sighting |
| 4 | Early stop (EOS, stop strings) | Fewer tokens, shorter context, less KV read | Output-length gate |
| 5 | Counting prompt tokens as output tokens, or miscounting the stream | Prefill tokens are processed in parallel and are cheap per token | Token-count consistency gate |
| 6 | Shorter context in one engine (different template, llama-bench depth 0) | Less KV cache to read per step | Prompt token count after templating; identical prompt set sha256 |
| 7 | Strawman baseline (defaults, eager mode, old version, single-user config in a capacity test) | Measures configuration, not engineering | Baseline's resolved config vs its documented config for that goal |
| 8 | Best-of-N instead of median | Selects the luckiest round | Statistic field must be median, CI over rounds |
| 9 | Non-interleaved rounds, or one engine kept warm while others restart | Drift and warm state land on one engine | Round timestamps; fresh process per round (`bench rounds` plan and cell log); `compare` interleave check for `bench decode` |
| 10 | Engine-timed numbers next to client-timed ones | Excludes work (HTTP, queueing, detokenisation) from one engine | Timer field; only `bench serve` numbers are published |
| 11 | Locked clocks, raised power limit, or exclusive GPU for one engine only | Hardware configuration, not software | `doctor` records before each engine's runs |
| 12 | Closed-loop load reported as capacity | The client slows its arrivals when the server slows | Load records must be open-loop with a seeded schedule |
| 13 | A setting that was silently ignored | The "candidate" is the baseline; a no-difference result looks like a finding | Resolved config per knob (engine-configs.md) |
| 14 | Profiler attached to one engine | Overhead on that engine | Process tree during the run |
| 15 | Workload, bucket or metric chosen after seeing results | Forking paths | Pre-registered claim timestamp precedes the run |
| 16 | Different model revision, or a pruned/truncated model or a lossy pruned vocabulary | Fewer layers, parameters or vocab rows | `identity.model.revision`; rare-token pack; quality score against the reference |
| 17 | Weights resident in L2 (tiny models, repeated microbenchmarks) | Reads served from cache, above DRAM bandwidth | % of ceiling (for the format read) above 100% |
| 18 | Engine tuned on the published prompts | Special-cased inputs | Re-run the final numbers on a sealed prompt set (`inferencekit prompts seal`); score quality on the same prompts |
| 19 | Correctness not checked | A broken kernel can skip work | proving-correctness evidence absent |

**Sampled exactness** (speculation, a pruned head or custom sampler above temperature 0): `spec_exact` checks greedy only and `inferencekit quality sampling` first tokens only, which no draft touches; speculation also needs its accept/resample check ([speculative.md](../../optimizing-inference/references/speculative.md)). Missing: UNPROVEN.

**Allowed: an exact bound-pruned lm_head**, with all three: a proof the chosen token never changes (bounds rigorous through the final logit transforms, e.g. a softcap; at T > 0, pruned rows proven outside the top-k / top-p set), a `rare_tokens` PASS, and the full head whenever logprobs are requested or the bounds cannot decide. Missing any one: row 16.

## Disclosure rule

If a line cannot be fixed (for example an engine cannot disable a cache), keep the run but put the item in `limitations` and in the report's findings. A disclosed caveat is honest; a hidden one is a reward hack.
