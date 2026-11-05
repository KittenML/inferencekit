# Quality: noise floor, trajectory scoring, pass rules

## 1. The A/A noise floor

The noise floor is how much two correct computations of the same model disagree when run through **exactly** the pipeline you will use for the candidates (same prompts, template, lengths, scoring code). Measure it before any candidate.

In `inferencekit quality score` the floor is the reference's own greedy trajectory (`quality trajectory` against the reference server), passed as `--baseline`, scored exactly like an engine. It measures how much the reference's decode path and its teacher-forced one-pass scoring already disagree. A second greedy run of the reference (`--baseline-repeat`, e.g. after a relaunch) adds run-to-run variation and is the better floor when you have it.

Other second computations, for checks outside `quality score`:
- the reference with a different valid kernel path (early engine runs: transformers with `attn_implementation="eager"` vs `"sdpa"`);
- the baseline engine against itself in a fresh process with a different valid backend.

>>>>

## 2. Trajectory scoring (`quality trajectory` + `quality score`)

1. **Trajectory.** `inferencekit quality trajectory --base-url <engine> --model <name> --prompts <set.jsonl> --label <engine> --run-dir runs/<id> --top-logprobs 20 --engine-kind vllm` (`--engine-kind` matches the server: `vllm`, `sglang`, `llamacpp`, else `openai`; with a hybrid-thinking model add the same `--chat-template-kwargs '{"enable_thinking": false}'` to every run): for each prompt, the engine generates its own greedy continuation through its serving endpoint and returns token ids and the top-k logprobs at every generated position (the engine's logprob cap is detected; the k actually returned is recorded). Files: `runs/<id>/quality/<label>/<prompt-id>.json` plus `index.json`; one `kind=quality_trajectory` record per engine.
2. **Scoring.** `inferencekit quality score --trajectories runs/<id>/quality/<label-a> --trajectories runs/<id>/quality/<label-b> --reference-url <vLLM reference> --reference-model <name> --baseline runs/<id>/quality/reference [--baseline-repeat ...] --margin-agreement <m> --margin-kl <m> --margin-nll <m> [--margin-tv <m>]`. The reference must be a vLLM server: it is sent prompt + that exact continuation as token ids (`/v1/completions`, `max_tokens=1`, `prompt_logprobs`), with prompt ids from its `/tokenize`. Per position:
   - **top-1 agreement:** the engine's token is the reference's argmax;
   - **KL(reference || engine):** on the top-k supports, an approximation, because the engine returned only its top-k logprobs; state it;
   - **reference NLL of the engine's token** and its mean over positions (perplexity);
   - **first divergence** from the reference's own greedy run (`--baseline`).

   Total variation (TV, the share of probability mass that moved, 0 to 1) is computed on the same top-k lists as KL and reported for every engine; it is judged only when `--margin-tv` is given (there is no default: a KL margin of 0.02 nats allows a TV of up to 0.1). An engine that returns no logprobs gets a text-only trajectory (the trajectory record lists `logprobs` as missing): its text is re-tokenized on the reference and scored for top-1 agreement and NLL against the noise floor's text re-tokenized the same way; KL and TV are NOT RUN, and the score record lists them per engine in `data.not_run`.
3. **Aggregate** per engine, per position bucket (0-15, 16-63, 64-255, 256+) and per input bucket. The verdict per engine comes from a paired bootstrap CI over prompts of "engine minus floor" against the margins (positions within one prompt are correlated, so prompts are resampled).

The default margins (`--margin-agreement 0.02`, `--margin-kl 0.02`, `--margin-nll 0.05`) are placeholders, far looser than a same-precision margin; always pass the pre-registered values. Agreement margins are absolute fractions (0.005 = 0.5 points); KL and NLL margins are nats above the floor.

Why the engine generates and the reference scores: the engine's numbers come from its decode path (the kernels users run), and the reference does one forward per prompt. Free-running texts diverge after the first flip, so later positions answer different questions.

## 3. The decode-path vs one-pass caveat

- **Decode path:** one token per forward with a KV cache, the code path that generates tokens. Trajectory logprobs come from here.
- **One pass ("score mode"):** prompt + continuation in a single prefill forward. This is what prompt-logprob endpoints (`prompt_logprobs`, `return_logprob`) compute. It exercises prefill kernels (large-M GEMMs, causal prefill attention), not decode kernels.
- The reference scores in one pass, which differs from its own decode path (early engine runs: 57 of 6,656 positions, 0.86%). So the floor uses the candidates' scoring path, and decode-path numbers are never compared with score-mode numbers. Trajectory scoring needs no teacher-forcing API from the engine.

## 4. Pass rule

Pre-register, per check: the metric, the comparison (candidate vs noise floor, or candidate vs baseline engine), the margin, and the decision rule. Then:

- **PASS:** the whole CI95 of "engine minus floor" (oriented so positive is worse) is at or below the margin, and there is no large-margin disagreement above the control's maximum.
- **FAIL:** the whole CI95 is above the margin, or a hard gate fired. Off NVIDIA, the reasons that [hardware.md](../../inference-triage/references/hardware.md), note 3, lists as expected on that platform (a CPU has no CUDA graphs or GPU backend) are reported as "not applicable on <platform>" instead; any other hard gate is still a FAIL.
- **UNPROVEN:** the CI straddles the margin (underpowered), or a check is incomplete. No `--baseline`, edited trajectory files, unscored prompts or a chat-template mismatch never give PASS.
- **NOT RUN:** never attempted.

Starting points from the design brief, to be calibrated by your own floor (absolute values; the `quality score` margins are increases over the floor, so derive them from these and your measured floor): same precision, KL around 0.002 or less and top-1 agreement around 98% or more; FP8, KL 0.005-0.015; 4-bit, KL 0.02-0.04. Task scores: no significant drop, and above -1 point for a change claimed lossless. Measure the floor; do not adopt these blindly.

## 5. Under load

Batch-dependent kernels can change numerics when other requests run. When `N*` is claimed, quality must hold at that level: run the trajectory at the `N*` concurrency too (or check that the load run's greedy outputs match the single-user ones), and report it per level.

## 6. Sampler check (`quality sampling`)

Greedy scoring never exercises temperature, top-k, top-p or seeding. `inferencekit quality sampling --base-url <engine> --model <name> --run-dir runs/<id> --extra-body '<the default sampling settings as JSON>' [--prompts <sealed set>]` checks one server. Per prompt (32 very_short/short ones by default) it asks twice for the top-20 log-probabilities of the first token, builds the distribution the settings imply (temperature, then top-k, then top-p; min-p must keep the same tokens before or after top-p), and draws the first token with 256 seeds (`max_tokens 1`). Per prompt: TV between the draws and that distribution, minus the TV that sampling noise alone gives (its exact expectation). Verdict on the mean over prompts with a bootstrap 95% CI: FAIL when the lower bound is above the margin (`--margin`, 0.02), PASS when the upper bound is below it, else UNPROVEN. Also FAIL when more than 0.5% of draws (95% lower bound) are tokens the settings remove. Fewer than 8 informative prompts (not dropped, top token under 98%): UNPROVEN; raise `--n-prompts`.

What it proves: the server's sampler draws from the distribution its own log-probabilities give. It does not compare logits with the baseline. Two correct engines differ there (batch shape, GEMM kernel, bf16 rounding on near-ties), and the teacher-forced check above already judges that against its noise floor. Together, the two checks mean the draws follow the baseline's distribution within noise. Check the baseline the same way, one engine on the GPU at a time. When both are up, `--baseline-url` checks both: each gets its own verdict, and the TV between their implied distributions is reported as information only.

How fine it is: the margin bounds the mean excess, not the sampler's own error, and the excess grows with about the square of a small error. On synthetic data at the defaults, samplers off by a TV of 0.05 or more (top_p 0.85 for 0.95, temperature 0.2 too high, top-p off by one token) FAILed in 15 to 20 of 20 runs; off by 0.025 to 0.03 (top_p 0.9 for 0.95, or temperature 1.0 off by 0.1), 1 to 11 of 20 runs PASSed and most of the rest were UNPROVEN. It catches real sampler bugs, not small drifts.

- Run with `--concurrency 1` (the default). A batched engine's logits change with the batch, so draws taken next to other requests need not follow the log-probabilities of a request run alone.
- The log-probabilities must be raw (before temperature), as vLLM returns them. A server that returns them after temperature FAILs, and the reason says so.
- Send every sampling field. `top_k`, `top_p`, `min_p` or `repetition_penalty` not sent are read as off, but vLLM fills them from `generation_config.json`. Settings the check cannot compute (`repetition_penalty` other than 1, `typical_p`, `logit_bias`, guided output) give UNPROVEN, never PASS or FAIL.
- A prompt is dropped, and counted in `data.prompts_used.dropped`, when its two logprob requests list different tokens, when the kept tokens might reach past the top 20, or when a cut falls within the rounding tolerance. A prompt whose top token holds 98% or more counts only for the removed-token check.
- top_k 20 or more, or off: a prompt counts only when the top-p cut falls inside the listed 20 whatever the unlisted tokens hold (any number of them may tie with the 20th, and ties are kept). At temperature 1 with top_k 20 that dropped a third to a half of the prompts on synthetic data; if too few remain, raise `--n-prompts` on a larger `--prompts` set.
- SGLang takes top-p over the whole distribution, then the first top_k tokens; Hugging Face and vLLM take it over the top-k tokens. A prompt counts only when both keep the same tokens (`top_p_order`), so a correct sampler in either order passes.
- Whether the server honours the seed is reported; it never blocks a PASS.
- For a thinking model, pass `--chat-template-kwargs '{"enable_thinking": false}'`.

## 7. Needle in a haystack (optional)

When the claim covers long contexts (or the change touches attention or the KV cache: split-K merges, FP8 KV, cache indexing), run a needle-in-a-haystack retrieval at the claimed lengths and the production length (`inferencekit quality niah --base-url <engine> --model <name> --run-dir runs/<id> --label <engine> --lengths 2048 --lengths 4096`; repeat `--lengths` once per length, Wilson CIs per cell); pass at least ~95% of the reference's retrieval rate at every length. Skipped: NOT RUN, with the reason.

## 8. Reading the numbers (patterns)

| Pattern | Likely meaning |
|---|---|
| Agreement inside the floor, disagreements at near-ties | correct, different arithmetic |
| Disagreement at token 0-2 on most prompts | wrong weights, template, RoPE configuration or tokenisation |
| Fine for short prompts, bad for long ones | attention or KV-cache bug at longer context |
| Fine alone, bad under load | state leaking between requests, or batch-dependent kernels |
| Agreement fine, KL or log-prob drift rising with position | accumulating error (KV precision, residual precision) |
| One engine's agreement lower than others at the same precision | a different arithmetic path; early engine runs: vLLM's default FP8 path re-quantised block scales to powers of two at load (inferred from its code); disabling DeepGEMM raised its score-mode agreement by most of a percentage point |
