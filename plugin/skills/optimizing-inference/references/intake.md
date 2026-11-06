# intake.yaml

The intake is written **before** anything is measured. Its targets are the pre-registered claims the final report is judged against.

Schema `intake.v1` ships inside inferencekit (`intake validate` applies it). Start from `inferencekit intake template > intake.yaml` (annotated, every field) or the examples at the end of this page.

## Checking it

```bash
inferencekit intake validate intake.yaml --allow-placeholders   # structure only; placeholders become warnings
inferencekit intake validate intake.yaml                        # a real run: placeholders are errors
inferencekit intake validate intake.yaml --run-dir runs/<run-id> # start or resume the run directory
```

- **Placeholders** are `<angle brackets>`, `CHANGEME`, `TODO` and `__UPPER_SNAKE__` markers (e.g. `__HOST__`, `__REFERENCE_BASE_URL__`). Fill every one before a real run.
- **`--run-dir` writes only redacted copies** (`intake.yaml` / `intake.json`, `intake.normalized.json`): every `hardware.access` field except `kind`, and the host in any URL, become `<redacted>`; your home directory becomes `~`. `intake_sha256` hashes the **original** (named by `state.json` `intake_source`), so pre-registration stays checkable.
- **Resume** by validating the original again with the same `--run-dir` (the redacted copy is refused).
- **A changed intake is a new run:** a different recorded `intake_sha256` is refused (targets are pre-registered); start a new run id, and never lower a target mid-run.
- **Output:** `{ok, intake, warnings, placeholders, derived, intake_sha256[, run]}`: `intake` normalized with every default; `derived` the plan (stop rule with `after_target` and its source, modes, binding hour cap, stage time boxes, precision, fidelity, quality budget, speculation and route, each with value and source, and the questions to log when non_interactive). Errors exit 1, all listed at once. Unknown keys are dropped with a warning, except inside `quality.budget.relative`, where unknown keys are errors: a misspelt relative bound must never disappear.

## Fields

| Field | Default | Meaning and rules |
|---|---|---|
| `run_id` | `<model>-<gpu>-<UTC date-time>` | Directory under `runs/`. |
| `model.id` | required | HF id or local path. |
| `model.revision` | resolved at stage 1 | Commit sha. Quote it in YAML. A branch name is a warning, and stage 1 resolves it to a 40-hex sha. |
| `model.precision` | — | The precision the checkpoint is stored in (nvfp4, fp8, bf16, q8_0…). |
| `model.precision_allowed` | `[precision]` | Weight/activation precisions an engine may run. Each extra entry (with `fidelity: lossy`) is a separate track, never ranked against the others. |
| `model.kv_cache_dtype_allowed` | `[checkpoint_default]` | KV cache dtypes. `checkpoint_default` = what the checkpoint config specifies, else the activation dtype. Add `fp8` only if a lossy KV cache is acceptable. |
| `model.tokenizer` | `model.id` | |
| `modality` | required | `llm`, `image`, `tts`, `stt` or `video`; other tasks require an adapter, not a misleading modality label. |
| `hardware.gpu`, `count` | required, 1 | Free text. Off NVIDIA, name the accelerator (`MI300X`, `M3 Ultra 80-core GPU`), or the CPU for a CPU-only host (`CPU 2x Xeon 6980P`, `count: 1`). What changes there: [hardware.md](../../inference-triage/references/hardware.md). |
| `hardware.price_usd_h` | — | Hourly price of the **whole host**. Needed for $ caps and $/1M. |
| `hardware.access` | `{kind: local}` | `{kind: ssh, host, user, port: 22, key_path}`: inferencekit and every engine run on that host. These values stay in the original intake file only. The run directory gets redacted copies (`<redacted>`); never copy the original into `runs/`. |
| `goals.lead` | `single_user` | `capacity` = N* streams per GPU inside the SLO. Derive it from the workload: more than one request in flight at peak means `capacity` ([workload.md](workload.md)). |
| `goals.targets` | none | Any of the targets below; with the target block, `buckets` (default: every intake bucket). |
| `goals.slo` | `max_fail: 0` | `ttft_p95_ms` is **required** for capacity or an `n_star` target; `tpot_p95_ms` is recommended. Always ask for both. A p90 SLO goes into these p95 fields (stricter). |
| `goals.traffic` | — | `arrival_rate_rps` (the **peak**: stage 8 sizes replicas for it), `burstiness`, `session_len`. |
| `goals.after_target` | `keep_climbing` when `non_interactive` or a budget is fixed; else `stop` | At the target, `keep_climbing` continues while headroom and budget remain ([stop-point.md](stop-point.md)); set it for benchmark and comparison runs too. |
| `io_modes` | input `complete`, output `streaming` | Streaming input warns: engines without incremental input are NOT SUPPORTED for it; any emulation is labelled. |
| `buckets` | all four; TTS declared text buckets when supplied | LLM: `very_short` 1–5 words, `short` ~20–50 tokens, `medium` ~300 tokens, `long` 2k–4k tokens. Choose them from the input-length p50 and p90 ([workload.md](workload.md)). TTS character bounds live in `workload.tts.text_length_buckets`. |
| `workload` | `rounds` 5; LLM only: `max_tokens` 256, `prompt_set` `dev` | LLM `max_tokens` comes from the output-length p50 (p90 for capacity when p90 > 2 × p50); omit it for media. `concurrency` defaults to `[1,2,4,8,16,32,64]` in load mode; it is a floor, extended until the SLO fails ([capacity.md](capacity.md)). |
| `workload.client`, `workload.tts` | unset | Timing client (`bench_audio` for TTS, `external` for another adapter); TTS voice, sample rate, character buckets and lead metric. `io_modes.output` selects streaming or complete utterance; see below. |
| `budget` | `approved: false` | `gpu_hours` (host hours × count), `usd`, `wall_clock_h` (the deadline; see "Time budget" below). Reuse explicit session/intake authorization; no separate plan approval ([starting-questions.md](starting-questions.md)). |
| `quality.reference` | `none` | `{kind: endpoint, base_url}` for a higher-precision reference server. With `none`, the reference is the baseline engine at the same precision. |
| `quality.noise_floor` | `measure` | `measure` = an A/A run of the reference; or a fixed agreement floor in [0, 1]. |
| `fidelity` | `lossless` | `lossless` or `lossy`: what may change and how it is proven ([fidelity.md](fidelity.md)). |
| `quality.budget` | — | Required with `fidelity: lossy`, agreed before measuring. Existing absolute keys retain their meaning: LLM `gsm8k_drop_pts_max`, `top1_agreement_min`, `kl_max`, `nll_increase_max`, `tv_max`; image/video `psnr_db_min`, `ssim_min`, `lpips_max`; speech `wer_max`, `cer_max`, `speaker_similarity_min`. Optional `relative` declares external modality degradation limits below. |
| `quality.protocol_sha256` | — | SHA256 of the external protocol's bytes, 64 lowercase hex characters; required with `quality.budget.relative`, along with nonempty `quality.protocol`. Validation checks the declaration, not the protocol's evidence. |
| `quality.acceptance` | — | Optional explicit `perceptual_and_task_noninferiority` for image/TTS: require hash-bound blinded perceptual and task evidence. Numerical similarity is diagnostic, with no dummy similarity limits. |
| `quality.diagnostic_only` | — | Typed modality-specific diagnostic comparisons under explicit perceptual/task acceptance; cannot quietly disable an existing gate. |
| `repo` | none (ledger NOT RUN) | `path`, `branch` (default `perf/<model>-<gpu>`), `push: ask\|never`. |
| `constraints` | `[]` | Free text, one quoted string per entry (unquoted, `a: b` is a YAML mapping and validate refuses it). Honour it literally and quote it in the report. Conventions the flow reads: `workload: ...` records the workload numbers that have no field (lengths, concurrency, prefix share, the SLO as given, the deadline as a clock time; [workload.md](workload.md)), `profiles: <path>, <path>` lists existing `.nsys-rep`, `.sqlite`, `.ncu-rep` or torch trace files to analyse without a GPU ("Profiles you already have", below), `plan: skip` is accepted as legacy text (plans are now opt-in; [plan.md](plan.md)), `after_target: stop\|keep_climbing` (as `goals.after_target`), `fidelity: lossless\|lossy` (as the field), `speculation: off\|auto` (default `auto`; [speculative.md](speculative.md)) and `route: build\|patch` (absent: from the goal; [stop-point.md](stop-point.md), "Route"). Several rules can share one entry, separated by `;` (`speculation: off; route: patch`). |
| `non_interactive` | `false` | `true` means never block. See non-interactive.md. |
| `harness` | — | `wheel`, `wheel_sha256`, `requirements`: how to install inferencekit on an ssh host. Absent on a host that came with inferencekit installed: validate warns, and remote-host.md says how to check the version and find the sha256. |
| `candidates` | `include: [], exclude: [], max_engines: 5` | |

### Targets
| Target | Better | Unit | Modality | Measured by |
|---|---|---|---|---|
| `decode_tok_s` | higher | tok/s | llm | `bench serve` single-user: tokens after the first ÷ time after the first |
| `ttft_ms` | lower | ms | llm | `bench serve` single-user |
| `decode_ceiling_fraction` | higher | fraction | llm | decode tok/s ÷ ceiling tok/s at that bucket's context (derived; ceilings.md) |
| `n_star` | higher | streams/GPU | not video | `bench serve` load mode, against `goals.slo` |
| `usd_per_1m` | lower | $/1M units | not video | `price_usd_h` ÷ (units/s at the lead operating point × 3600) × 1e6 |
| `ttfa_ms` | lower | ms | tts | `bench audio` or a matching modality client (UNPROVEN without one) |
| `rtf` | lower | × real time | tts, stt | `bench audio` for TTS; an external matching client for STT |
| `clip_ms`, `clips_per_hour` | lower, higher | ms, clips/h | video | `bench clip` |

A target that does not apply to the modality is reported NOT RUN.

## Given a written brief instead of an intake

Write an intake from the brief before `--run-dir`. The brief's rules (numerics, sampling, speculation, server contract, files to leave) override this plugin's defaults; record them in `state.decisions` and the report.
- **Map it:** model, revision and precision; GPU and access; goal and target; its baseline as the `baseline-recipe` row, exactly as given; `fidelity` from its precision rule; `speculation: off` if forbidden; timed sampling from its defaults; `non_interactive: true` unless someone answers questions.
- **Budget:** the brief's limits. With none, the given host is the approval: `budget.approved: true` and a `gpu_hours` planning horizon (logged in `QUESTIONS.md`); at `over_budget`, refresh the report and the files the brief asks for, then continue while the brief allows.
- **Never stop at its target unless it says so.** Write its files exactly as it says (e.g. a progress line per measured result, screened or verified, with the checks run so far); keep its launch file starting the commit you would hand over, and move a hand-over tag only to a checked or verified result ([state.md](state.md), "Best so far").

## Profiles you already have (`constraints`)

List existing profiles as one `constraints` entry (the schema has no field), e.g. `profiles: profiles/decode.nsys-rep, profiles/gemv.ncu-rep`, with the GPU, engine version and workload each came from (a profile explains its own GPU only).
- **Use them when this run cannot capture its own** (ncu counters are often blocked in containers and on cloud GPUs), or to skip a capture you have.
- **Analysis needs no GPU and no approval:** `inferencekit nsys analyze <file>`, `inferencekit ncu summarize <file> --peak-gbs <ev or GB/s>` (a `.nsys-rep` needs the nsys CLI, an `.ncu-rep` the ncu CLI; a `.sqlite` neither).
- **Relative paths only** (the report quotes `constraints`; absolute paths leak user names). The files stay where they are: never copied into `runs/`, committed or uploaded (they can hold environment values and host details).

## Time budget (`budget.wall_clock_h`)

- **It is your deadline,** in hours from the run start. `intake validate --run-dir` sets `budget.run_started`, and interactive runs do that only after your go, so review time does not count.
- **It is one of the three hour caps** (below). When it is the smallest, it binds, and every stage's time box is a share of it. `intake budget` counts it as wall clock, whether the GPU is busy or idle.
- **The agent time-boxes every stage to it.** When `intake budget` says `stage_over`, the agent finishes the current measurement, marks the rest NOT RUN, and moves on. The 10% report reserve means a report exists at the deadline.
- **A deadline given as a clock time** ("by 9am", "by 18:00 UTC") becomes `wall_clock_h` = hours from `--run-dir` to then, rounded down to 0.25 h, with the time zone pinned. Set it before `--run-dir`: a changed intake is a new run. The conversion, and why a shorter value caused by the approval wait is not a new question: [workload.md](workload.md), "The deadline".

## Rules validate enforces

**Errors:** `non_interactive: true` without `budget.approved: true` or without a spend cap (`gpu_hours`, or `usd` plus `price_usd_h`); `usd` without `price_usd_h` when non_interactive; `lead: capacity` or an `n_star` target without `slo.ttft_p95_ms`; target buckets outside `buckets`; placeholders in a real run (`--run-dir` always refuses them); `--run-dir` on a run started from a different intake; schema violations (types, enums, ranges such as `0 < decode_ceiling_fraction ≤ 1`).

**Warnings (report them):** a revision that is not a sha; an unapproved budget (every GPU step then asks first); no reference; no repo; streaming input; TTS/STT measurement coverage; more than one allowed precision; `decode_ceiling_fraction` with a precision other than bf16/fp8/q8_0 (its "dense models only" is out of date: [ceilings.md](ceilings.md)).

## Derived execution settings

- **Hour cap** = min(`gpu_hours / count`, `usd / price_usd_h`, `wall_clock_h`). The smallest one is `binding`.
- **Time boxes** are shares of the hour cap:

  | Stage | Share |
  |---|---|
  | intake | 2% |
  | hardware | 5% |
  | candidates | 18% |
  | tuning | 22% |
  | benchmark | 18% |
  | stop point | 2% |
  | custom | 23% |
  | report | 10% (reserved) |

  A stage that finishes early donates its time to the next one. The report reserve is never spent on anything else.
- **Stop rule:**
  - `target`: stop when met, continue when missed; with `goals.after_target: keep_climbing` (the default when unattended or with a fixed budget), past it too while budget remains;
  - `ask`: legacy interactive intake without a target (with a cap, offer continued work); explicit `interaction.checkpoints: required` instead follows the authorized stop rule without another continuation question;
  - `budget`: no target, unattended or explicit continued-work/check-in policy; continue while budget remains;
  - `stop_point`: no target, non_interactive, `after_target: stop` written; stop after the stage 6 ranking.
- **Weak target (your check, not validate's):** within `e` (default 2%) of the stated baseline, or met by the untouched baseline, it ends the run at a trivial win. Warn in `state.decisions` and the report (non_interactive: also `QUESTIONS.md`); before the go, suggest a `decode_ceiling_fraction` or larger target.

## Example (unattended LLM run, abbreviated)

```yaml
model: {id: <hf-org>/<checkpoint-nvfp4>, revision: "<40-hex sha>", precision: nvfp4}
modality: llm
hardware:
  gpu: RTX 5090
  price_usd_h: 0.60
  access: {kind: ssh, host: <gpu-host>, port: 22, user: <user>, key_path: ~/.ssh/<key>}
goals:
  lead: single_user
  targets: {decode_ceiling_fraction: 0.70, buckets: [short, medium]}
  slo: {ttft_p95_ms: 500, tpot_p95_ms: 50}
budget: {gpu_hours: 8, usd: 6, wall_clock_h: 8, approved: true}
repo: {path: <ledger repo>, push: never}
constraints: ["no changes to precision"]   # add existing profiles here, e.g. "profiles: profiles/decode.nsys-rep, profiles/gemv.ncu-rep"
non_interactive: true
harness: {wheel: <path to the inferencekit wheel>}
```

## Example (capacity lead, from a described workload)

"About 30 people use it at peak, 4 requests/s, prompts ~900 tokens (p90 3k) including a shared ~550-token system prompt, answers ~200 tokens (p90 600), p90 TTFT under 800 ms and TPOT under 40 ms, results by 9am London time." That becomes ([workload.md](workload.md)):

```yaml
goals:
  lead: capacity                                  # more than one request in flight at peak
  slo: {ttft_p95_ms: 800, tpot_p95_ms: 40, max_fail: 0}   # p90 given; judged at p95 (stricter)
  traffic: {arrival_rate_rps: 4, burstiness: poisson, session_len: "1 turn"}
buckets: [medium, long]                           # nearest ISL p50 (~300) and p90 (2k-4k)
workload:
  max_tokens: 600                                 # OSL p90 > 2 x p50, capacity lead
  concurrency: [1, 2, 4, 8, 16, 32, 64]           # a floor: extended until the SLO fails
budget: {gpu_hours: 10, usd: 12, wall_clock_h: 9.5, approved: true}   # hours from --run-dir to 09:00
constraints:
  - "workload: ISL p50 900 / p90 3000 tok; OSL p50 200 / p90 600 tok; peak 4 rps; concurrency peak 30; prefix share ~60% (a 550-token system prompt inside the 900-token p50; ISL counts the whole prompt); SLO as given: p90 TTFT <= 800 ms, p90 TPOT <= 40 ms; deadline 09:00 Europe/London; source: owner estimate"
```


## Consumer-device and run-policy fields

`modality: image` adds `image_ms`/`images_per_hour`; `tts` adds `ttfa_p95_ms` (first playable audio). Both need a real modality adapter, not a token endpoint. [Affordability](affordability.md) defines memory/latency/quality acceptance: `goals.affordability` requires `memory_gb_max`/`memory_kind`, optional smaller stretch cap, host cap and `offload_allowed`; `lead: affordability` additionally requires a latency target and `quality.protocol`. Image quality budgets use PSNR/SSIM/LPIPS, speech WER/CER/speaker similarity; protocols also cover reference/A/A, datasets/seeds, margins/CIs and human review.

For TTS, omit `max_tokens` and declare the speech workload:

```yaml
modality: tts
io_modes: {input: complete, output: streaming}
workload:
  client: bench_audio
  tts:
    voice: speaker-a
    sample_rate: 24000
    lead_metric: first_audio_ms       # or rtf / complete_audio_ms
    text_length_buckets:
      - {name: short, min_chars: 1, max_chars: 80}
      - {name: long, min_chars: 81, max_chars: 800}
```

These pins describe the intended run; pass them to the adapter and verify effective settings. `bench_audio` removes the missing-client warning, but timing does not prove quality or fresh-process interleaving. [Audio contracts and metric definitions](https://github.com/KittenML/inferencekit/blob/main/docs/media-benchmarks.md) apply; STT needs `client: external`.

### Relative modality budgets

Use `fidelity: lossy` and a typed `quality.budget.relative` when the agreed criterion is degradation relative to a baseline or A/A floor. Do not encode an LPIPS excess as `lpips_max`, or a WER increase as `wer_max`: those existing keys are absolute bounds. Absolute and relative branches may coexist; every declared bound still needs proof.

```yaml
fidelity: lossy
quality:
  protocol: protocols/image-relative.json
  protocol_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  budget:
    relative:
      modality: image
      lpips_excess_max: 0.02
      ssim_loss_max: 0.01
      ocr_cer_increase_max: 0.02
      semantic_score_drop_max: 0.05
      new_severe_regressions_max: 0
```

The example hash is illustrative; bind the actual protocol bytes before use. `relative.modality` must match the intake and be `image`, `tts` or `stt`. Only these keys apply:

| Modality | Allowed relative limits |
|---|---|
| image | `lpips_excess_max`, `ssim_loss_max`, `ocr_cer_increase_max`, `semantic_score_drop_max`, `new_severe_regressions_max` |
| tts | `wer_increase_max`, `cer_increase_max`, `completion_failure_increase_max`, `speaker_similarity_drop_max`, `perceptual_score_drop_max`, `new_severe_regressions_max` |
| stt | `wer_increase_max`, `cer_increase_max`, `completion_failure_increase_max`, `new_severe_regressions_max` |

Every increase/drop is oriented so positive is worse, in the protocol's declared units. Values must be finite and nonnegative; severe regressions are an integer count. Completion and normalized semantic degradation are at most 1; SSIM loss and cosine-similarity drops at most 2. WER/CER can exceed 1. The protocol must fix reference/A/A adjustment, cohorts, missingness, units and CI decisions; no defaults or generic metric names are inferred. Unknown keys, null bounds, another modality's metrics and missing protocol bindings are errors.

The derived `external_quality_proof` says evidence is required and the quality verdict is `UNPROVEN`. Valid intake syntax is not quality PASS. Built-in absolute scorers do not evaluate this branch, and no frozen scorer/client is changed. Numerical changes may produce different pixels, PCM or codes within the agreed quality budget; exact output comparisons remain diagnostic for them. Own output integrity, valid shapes, finite values and the task's model/architecture scope still apply. Preserve historical absolute failures separately.

### Perceptual and task acceptance

When explicitly authorized, set `quality.acceptance: perceptual_and_task_noninferiority` for image or TTS. The same protocol path/SHA binding is required. Image relative budgets must declare OCR CER increase, normalized semantic score drop and new severe regressions; TTS budgets must declare WER/CER, completion, speaker, perceptual score drop and severe regressions. These require actual task and fresh blinded perceptual proof, with the protocol's units, A/A controls, cohorts and CIs. ASR, VAD or speaker embeddings alone never establish listening quality.

Under this mode, image LPIPS/SSIM/PSNR acceptance thresholds are rejected, rather than invented or silently ignored. Omit them from the budget and, if measured, declare them in `quality.diagnostic_only`. Missing numerical similarity limits do not produce unbounded-gate warnings. Image diagnostics allow `lpips`, `ssim`, `psnr`, `cross_engine_byte_identity`, `cross_engine_tensor_equality`, `rounding_drift`, `eager_compiled_drift` and `repeated_run_variation`. TTS allows the drift/equality entries plus `cross_engine_pcm_equality` and `cross_engine_generated_code_equality`, excluding image LPIPS/SSIM. These describe cross-engine diagnostics; each artifact's own integrity and native input/codec contract remain required.

The derived plan marks blinded perceptual review required and its verdict `UNPROVEN`. Missing viewing/listening blocks a perceptual-quality claim; intake validation and proxy scores supply no substitute. Original protocols/FAILs stay unchanged. Reassess baseline eligibility under the newly bound task/perceptual rule; an old numerical-similarity-only failure cannot alone exclude a baseline from a new perceptual comparison. Historical reanalysis is exploratory; promotion needs fresh exact-configuration confirmation. STT retains the typed relative task-budget route without inventing an audio-output listening requirement.

Run-scoped explicit authorization can replace default spending/time policies:

```yaml
budget:
  approved: true
  spend_cap: none
  deadline: "2026-10-26T12:00:00+04:00"
  stage_timeboxes: false
  report_reserve_h: 0
```

Uncapped spend requires approval plus a timezone-qualified deadline or `wall_clock_h`, conflicts with numeric `usd`/`gpu_hours`, and remains null, never a huge sentinel. `intake budget` enforces the deadline. False stage timeboxes disables stage cutoffs; zero reserve disables early report/freeze. Validate/package continuously through that deadline. Other runs retain defaults.

## Simple preferences

[Starting questions](starting-questions.md) translate ordinary answers into optional fields:
- `quality.preference: exact|balanced|speed`: resolves fidelity to lossless for exact and lossy otherwise. Conflicting explicit fidelity is an error. This records intent, not proof; exact needs equality checks, and lossy needs its quality budget and allowed formats.
- `model.architecture_changes: forbid|allow|ask` (default forbid): separate permission for model simplification/replacement; preserve all quality requirements.
- `interaction.checkpoints: required|quality|milestones` and `interaction.show_plan: false`: when to pause and whether a plan was requested. Human quality/milestone checkpoints conflict with unattended mode.
Existing intakes without these fields retain their fidelity and stop-rule behavior. Preferences never approve spending or create quality evidence.
