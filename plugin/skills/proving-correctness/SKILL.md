---
name: proving-correctness
description: Verify correctness and agreed quality after engine, kernel, precision or model changes. Use before a performance claim and for quick exploration screens. Select exact, task or perceptual checks for text, image, audio, video and other supported tasks; keep missing evidence UNPROVEN.
license: Apache-2.0
compatibility: Needs inferencekit (Python 3.10+, uv or uvx) and the workload's reference/quality adapter. Text examples use OpenAI-compatible endpoints; media uses native pipelines. Disassembly tools and byte-identity proofs depend on the backend.
metadata:
  version: "0.1.0"
---

# Proving correctness and quality

A speedup only counts if the engine still computes the same model. Engines almost never agree bit for bit, so "same" is defined per kind of change, against a noise floor of correct-but-different arithmetic. [optimizing-inference](../optimizing-inference/SKILL.md) runs the full workflow per candidate engine and at milestones; inside the change loop each change gets only the fast check below.

## When to use / not use

Use when:
- before claiming any speedup ([benchmarking](../benchmarking/SKILL.md), [optimizing-kernels](../optimizing-kernels/SKILL.md));
- a kernel, fusion, precision, engine, attention backend or serving config changes (inside a change loop: the fast check only);
- `N*` is being claimed (quality must hold at that load level too).

Do not use when:
- you want to rank different models by quality (evaluation, with far larger sets);
- the change intentionally makes a new model (new weights, a 4-bit recipe you have not agreed to treat as the same model): that is a quality study with the user's sign-off.

**GPU approval:** reference and candidate runs occupy GPUs; run only with `budget.approved: true` in `intake.yaml` or an explicit yes to a stated cost.

## Choose the contract and modality first

Read the run's [quality choice](../optimizing-inference/references/starting-questions.md). Exact requires equality on its declared deterministic/seeded outputs; legacy lossless distribution agreement alone is insufficient. Balanced/speed tracks use their registered task/perceptual budget, so rounding differences are not an automatic rejection. Model changes need separate permission and separate reporting.

Use modality-appropriate structural integrity and task quality: text accuracy/distributions, image semantics/readability, speech content/completion/speaker, video temporal consistency. Other tasks need a registered adapter; do not substitute GSM8K or token identity. Missing listening/viewing remains UNPROVEN. The text procedures below apply only where their inputs and claims match.

For TTS, register the acceptance policy before capture with `quality audio-policy`, then run `quality audio` on the final configuration records and saved PCM. It checks parity and imports hash-bound ASR/MOS/listening observations under the registered margins; [speech quality protocol](https://github.com/KittenML/inferencekit/blob/main/docs/audio-quality.md). It does not run the scorer models or replace required listening with a MOS predictor.

## Which rule applies (decide before measuring)

| Kind of change | Examples | Fidelity | Pass rule |
|---|---|---|---|
| Pure refactor | moving code, launch restructuring, a switch that is off, prefill tiles that keep the K order | lossless | byte identity (ladder below) |
| Equivalence | CUDA graphs, fusion, the same precision in another engine or attention kernel | lossless | no regression beyond the A/A noise floor |
| Exact byte cut | lossless weight re-encoding, an exact bound-pruned lm_head | lossless | its proof recipe ([byte-cuts.md](../optimizing-kernels/references/byte-cuts.md): weights decode bit-equal; pruned token = full token every step; logprob requests take the full head), `rare_tokens`, then equivalence |
| Exact speculation | MTP or EAGLE heads, a same-tokenizer draft model, n-gram lookup, with standard verification | lossless | the `spec_exact` gate at temperature 0; under sampling, the accept/resample check and `quality sampling` ([sampled exactness](#sampled-exactness)); then equivalence |
| Declared numerics change | 16-bit activations where the checkpoint quantises them; a KV dtype the task lists | lossless only if the task allows it; declared in the engine's `numerics`, labelled "numerics changed" | quality against the reference no worse than the baseline's, within the noise floor |
| Numeric | an FP8 KV cache the checkpoint does not declare, lower-precision attention maths | lossy | no regression beyond the noise floor plus a pre-registered margin |
| Approximation | 4-bit weights, lossy speculative decoding, weight or expert pruning, a truncated vocabulary | lossy | a pre-registered margin agreed with the user; otherwise it is a different model |

`fidelity: lossless` (the intake default) allows only the lossless rows. `fidelity: lossy` adds the others, each judged against the intake's `quality.budget` and reported on its own track, as speed at a stated quality cost. A task's own precision rule overrides this plugin's default. Full rules: [fidelity.md](../optimizing-inference/references/fidelity.md).

For images, TTS and STT, an explicitly authorized numerical change uses the registered modality checks rather than LLM token/logprob gates. Relative degradation budgets belong in `quality.budget.relative`, with a matching modality and `quality.protocol` / `quality.protocol_sha256` binding. A valid intake leaves external quality proof UNPROVEN. Different pixels, PCM, generated codes or rounding are diagnostic observations, not automatic quality failures for that numerical-change track. Pure refactors still use the declared exactness checks. Nonfinite values, invalid shapes, unsafe accesses, broken own-output/native integrity and omitted cases remain failures. A quality budget alone does not authorize architecture pruning or another model.

With explicit `quality.acceptance: perceptual_and_task_noninferiority`, image/TTS promotion requires the bound task checks and fresh blinded perceptual evidence. Numerical similarity (including image LPIPS/SSIM) is diagnostic, not an independent rejection gate; do not add dummy bounds. TTS ASR/VAD/speaker proxies do not replace listening. Missing perceptual evidence remains UNPROVEN. Retain old numerical failures under their original rules and reassess a baseline's eligibility prospectively under the new acceptance policy.

## Inside the change loop: one fast check

For an image or speech numerical change, use one short fixed public modality correctness/quality screen under the agreed protocol. Full registered quality and balanced timing remain milestone/final promotion requirements; they are not required for every intermediate kernel. Retain failed screens and all cases, and label any historical output reuse exploratory. No numerical candidate is promoted from exact-output diagnostics alone.

For LLM candidate changes, against the previous build, greedy, on a handful of fixed prompts: a pure refactor or exact change keeps the greedy tokens identical; a numerics change (any change to the arithmetic) keeps next-token and logprob agreement, flips only at near-ties (`inferencekit correctness --reference <prev.jsonl> --candidate <new.jsonl>`); a fail drops the change. Inputs: one `{"tokens": [...], "logprobs": [...]}` line per prompt from each build's server (a `bench decode` id compares tokens only). For a numerics change, `--min-prefix-agreement` below 1 (written down first) and `--logprob-tol <t>`; at each first divergence, the top-2 gap from the previous build's top-20 logprobs must be below the A/A control's largest flip margin, or the change drops ([margin test](references/token-agreement.md#first-divergence-and-the-margin-test)). The A/A floor, trajectory score, GSM8K, sampling and byte-identity ladder below are the full proof: milestones and published claims only ([state.md](../optimizing-inference/references/state.md)).

## Preflight checklist

- [ ] The prompt set fixed and verified (`inferencekit prompts verify --set <dev|path>`), the same token ids for every engine. Why: changing prompts changes the result.
- [ ] Greedy decoding, every sampler setting explicit, fixed length, same chat template. Why: sampling turns every comparison into noise. Speed is timed at the workload's sampling; a sampled claim (sampler, speculation, a pruned head above temperature 0) also needs [sampled exactness](#sampled-exactness).
- [ ] A higher-precision reference endpoint (intake `quality.reference`): a vLLM server running the model in BF16 for an FP8 engine (`quality score` needs vLLM's `prompt_logprobs`). Why: distance to a better reference is the quality signal; two candidates agreeing with each other proves little. None: [below](#no-higher-precision-reference).
- [ ] Quality runs on the **exact speed configuration** (the same resolved config that was timed), never a debug or batch-invariant mode. Why: those modes run different, slower kernels.
- [ ] An A/A control planned, run through the identical pipeline: the reference's own greedy trajectory (`--baseline`), ideally twice (`--baseline-repeat`, e.g. after a relaunch). Why: it sets the noise floor; no tolerance may be invented.
- [ ] Margins pre-registered and passed explicitly (`--margin-agreement`, `--margin-kl`, `--margin-nll`). Why: the tool's defaults (0.02 / 0.02 nats / 0.05 nats) are loose placeholders, ten times looser than a same-precision KL margin.

## Hard failure gates

- NaN or Inf anywhere in logits or logprobs. Always a bug.
- A disagreement where the reference was confident (large top-1 vs candidate-token margin, above the control's largest). Why: near-ties flip under rounding; large-margin flips mean wrong maths.
- Teacher-forced agreement or trajectory score below the noise floor by more than the pre-registered margin, with the CI excluding the margin.
- Any rung of the byte-identity ladder failing for a change declared a pure refactor.
- An A/A control that is not identical when the ladder requires identity. Why: then the harness is non-deterministic and no A/B difference can be interpreted.

## Workflow (copyable)

```
Quality:
- [ ] 1. Classify the change (table above); pre-register the rule and margin
- [ ] 2. Reference (vLLM, BF16) up: quality trajectory --label reference, then --label reference-r2
         (restart your own reference server in between; a remote endpoint cannot be restarted:
         run back to back and say so) -> the A/A floor's trajectories
- [ ] 3. Per engine, its exact speed config up: quality trajectory --label <engine> -> ev_traj_<engine>
- [ ] 4. quality score --trajectories <dir> (once per engine) --baseline <reference dir> [--baseline-repeat]
         with the pre-registered margins -> ev_qs (verdict per engine vs the floor, paired CI over prompts)
- [ ] 5. Every disagreement: position + reference margin; any above the control's max -> investigate / FAIL
- [ ] 6. quality gsm8k --concurrency 0 on every engine: all 1,319 for a claim, the final candidate
         or a scored commit; --limit 100 at the light milestone checks (only screens);
         candidates with --paired-with <baseline run> (McNemar) -> ev_gsm
- [ ] 6b. Sampling in the claim? quality sampling on each engine, the baseline too (draws vs own logprobs) -> ev_samp;
         plus the accept/resample check (speculation) or per-step support check (pruned head)
- [ ] 7. Long context in the claim? quality niah at the lengths claimed (optional otherwise)
- [ ] 8. Pure refactor? Byte-identity ladder instead, control first
- [ ] 9. Fill the fixed table; verdict per check
```

## Exact commands

```bash
# A/A floor: the reference's own greedy run (vLLM serving the model in BF16)
inferencekit quality trajectory --base-url http://127.0.0.1:8001 --model <ref-name> --prompts <set.jsonl> \
  --label reference --run-dir runs/<id> --top-logprobs 20
# each engine, at its exact speed configuration
inferencekit quality trajectory --base-url http://127.0.0.1:8000 --model <served-name> --prompts <set.jsonl> \
  --label vllm-tuned --run-dir runs/<id> --top-logprobs 20 --engine-kind vllm      # -> runs/<id>/quality/vllm-tuned
inferencekit quality score --trajectories runs/<id>/quality/vllm-tuned --reference-url http://127.0.0.1:8001 \
  --reference-model <ref-name> --baseline runs/<id>/quality/reference [--baseline-repeat runs/<id>/quality/reference-r2] \
  --margin-agreement <pre-registered> --margin-kl <pre-registered> --margin-nll <pre-registered>   # -> ev_qs
inferencekit quality gsm8k --base-url http://127.0.0.1:8000 --model <served-name> --run-dir runs/<id> --label baseline \
  --concurrency 0                                                                  # 0: sized from the server
inferencekit quality gsm8k --base-url http://127.0.0.1:8000 --model <served-name> --run-dir runs/<id> --label vllm-tuned \
  --concurrency 0 --paired-with runs/<id>/quality/gsm8k/baseline.jsonl             # McNemar vs the baseline
inferencekit quality sampling --help                                               # sampled exactness (below)
inferencekit correctness --reference ref_hf.jsonl --candidate <ev_bench>          # token agreement: the per-change fast check vs the previous build (or the older HF-reference method)
inferencekit report <ids...> --format md
```

MCP tools: `quality_trajectory`, `quality_score`, `quality_gsm8k`, `quality_niah`, `check_correctness`, `report` (in OpenCode, which ends an MCP call after 120 s, run quality through the CLI, detached with a log). Flag spellings: `inferencekit quality <op> --help`. Scoring, the noise floor and pass rules in detail: [references/quality.md](references/quality.md).

## Trajectory scoring (summary)

Each engine generates its **own** greedy continuation through its serving (decode) path, the code users run, with the top-20 logprobs per step. The reference (vLLM) is teacher-forced on that sequence and scores every position: top-1 match, KL on the top-k supports (an approximation), the reference's log-probability of the engine's token, and the first divergence from the reference's own greedy run. The A/A floor is the reference's own trajectory scored the same way, so it already holds the gap between decode path and one-pass scoring. Prompt-logprob ("score mode") endpoints exercise only prefill kernels.

## Sampled exactness

Lossless means the same sampling distribution, not only the same greedy token. The teacher-forced check holds the engine's log-probabilities to the reference's; `inferencekit quality sampling` shows the sampler draws from the distribution they give under the request's temperature, top-k and top-p (seeded first-token draws against the top-20 log-probabilities; mean TV beyond sampling noise, bootstrap CI over prompts; [references/quality.md](references/quality.md) section 6). PASS when the CI's upper end is inside the pre-registered margin, FAIL when its lower end is above it, UNPROVEN otherwise or when not run.

It checks only the first generated token (max_tokens 1, produced by prefill), so a change that acts only in decode steps is not exercised. Speculation also needs a unit check of the accept/resample step against the target distribution (synthetic p and q, many draws, TV or chi-square inside a pre-registered bound). A pruned head needs the per-step support check in [byte-cuts.md](../optimizing-kernels/references/byte-cuts.md), or the first token must also go through the pruned head. Without one of these the sampled claim is UNPROVEN.

## Why bitwise identity across engines is not expected

Floating-point addition is not associative: tiling, split-K, reduction order, FMA contraction, online softmax and shape-dependent heuristics each give a slightly different, equally valid result, and after one flip greedy outputs diverge. With the early engine almost no prompt matched the reference exactly for any engine (the reference's own second attention path included) while per-step agreement stayed inside the floor. Exact-match rates are a poor quality signal.

## Byte-identity ladder (pure refactors only)

Climb in order; stop at the first failing rung ([references/byte-identity-ladder.md](references/byte-identity-ladder.md)):
1. Kernel outputs byte-equal (memcmp) across a shape sweep, outputs pre-filled with NaN.
2. Identical SASS for kernels the refactor should not touch (`cuobjdump -sass`).
3. End-to-end logits (and KV cache) byte-equal on the fixed prompts, every path (eager, capture, replay).
4. Under load: an off/off (A/A) control first, then old vs new.

## Task eval

`inferencekit quality gsm8k`: the same problems, format and answer extraction for every engine, greedy, accuracy with a Wilson CI; `--paired-with <the baseline's results JSONL or evidence id>` adds McNemar's exact test. A claim uses all 1,319 items (the default); a `--limit` subset only screens (the first 100 detect only about 10-point effects). `--concurrency 0` sizes concurrency from the server. The first run on a host downloads the pinned test split (419 KB) into `~/.cache/inferencekit/gsm8k/`: unset `HF_HUB_OFFLINE` for it, or place the file with the `curl` command the error prints ([references/gsm8k-parity.md](references/gsm8k-parity.md); the older token-agreement method: [references/token-agreement.md](references/token-agreement.md)).

## No higher-precision reference

With `quality.reference: none`, compare with the baseline engine run twice, restarted between (the A/A floor); label results "vs baseline, no reference".
- **GSM8K, all 1,319:** candidate `--paired-with` baseline; the A/A pair gives the noise-only discordant count.
- **Held-out NLL** per token on a fixed text in no prompt set, from each engine's logprobs (outside the CLI; record the text's sha256 and code path). Pass: gap to baseline within the A/A gap plus a pre-registered margin.
- A vLLM baseline can also be the `quality score` reference: it bounds drift from the incumbent, not quality.

## Artifact/evidence contract

- `quality trajectory` -> `runs/<id>/quality/<label>/<prompt-id>.json` + `index.json` and a `kind=quality_trajectory` record (observed); `quality score` -> `runs/<id>/quality/score.json` and a derived record: per engine, per position bucket and input bucket, with a PASS / FAIL / UNPROVEN verdict against the floor; `quality gsm8k` -> `kind=quality_task` (per-problem outcomes, not only accuracy); `correctness` -> `kind=correctness`.
- Keep the prompt-set sha256, template and decoding settings in each record. Why: a quality record without its inputs proves nothing.

## Fixed result table

| Check | Engine (config) | Value [CI95] | Noise floor (A/A) [CI95] | Margin (pre-registered) | Verdict | Evidence |
|---|---|---|---|---|---|---|
| Top-1 agreement vs reference (decode path, tie-inclusive) | | % | % | pp | | |
| Mean / p99 KL (top-20 approx.) | | | | | | |
| Disagreements above the control's max margin | | count | must be 0 | | PASS / FAIL | |
| GSM8K (1,319), McNemar vs baseline | | x / n, p | | | | |
| Sampled distribution (mean TV vs baseline; speculation: + accept/resample check), if sampling is claimed | | | | | | |
| NIAH at lengths claimed (optional) | | | | | NOT RUN if skipped | |
| Byte ladder rung reached (refactors) | | 1-4 | control identical? | | | |

## Decision and no-go rules

- FAIL blocks any speed claim for that engine or change. Why: performance of a wrong engine is meaningless.
- Tolerances come from the A/A control, never from "looks small". Why: invented tolerances drift toward whatever the candidate produced.
- Divergences explained by near-ties only: PASS with the explanation. Why: expected behaviour of correct, different arithmetic.
- No reference: the [no-reference route](#no-higher-precision-reference); without it, UNPROVEN, not PASS. A test without enough items to detect the pre-registered effect: UNPROVEN (underpowered), not PASS.
- Speed claims stay within one precision even when quality is compared across precisions against the same reference. The one exception is a format trade-off ([evaluating-quantization](../evaluating-quantization/SKILL.md)): a speed ratio with its CI95 shown only next to this skill's quality verdict for that row, labelled "cross-precision: format trade-off, not an engine comparison".

## Reporting rules

- Agreement, KL and task scores are **observed**; noise floors from controls are **observed**; "divergence is due to rounding" is **inferred**, supported by the margin at divergence.
- State which path was measured: decode path (trajectory) or one prefill pass (score mode). The two are not comparable with each other.
- Verdicts PASS / FAIL / UNPROVEN / NOT RUN per check, combined with the precedence in [reporting-perf-claims](../reporting-perf-claims/SKILL.md).
