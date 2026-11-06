---
name: optimizing-inference
description: Optimize a model for faster responses, more requests, less memory or cheaper hardware. Ask simple quality and budget questions, research current implementations, tune the strongest supported baseline, investigate compiler/kernel bottlenecks and prove end-to-end results. Covers text, image, audio and video with modality-specific checks; accepts an intake or written brief.
license: Apache-2.0
compatibility: Needs inferencekit (Python 3.10+, uv or uvx), the target runtime and git. NVIDIA, AMD, Apple Silicon and CPU tools differ; unsupported measurements stay NOT RUN. Written for Claude Code, Codex and OpenCode.
metadata:
  version: "0.1.0"
---

# Optimize inference

Deliver reproducible code and evidence against the strongest tuned, supported baseline under the agreed quality policy. Record time, tokens/API cost and device cost. Unsupported checks are NOT RUN; missing proof is UNPROVEN.

## Rules throughout the run

- Reuse the unmodified inferencekit harness and datasets. A missing modality adapter is a recorded limitation; do not disguise a custom timer as harness evidence.
- Fix input distribution, output work, sampling/seeds, component precisions and quality protocol before comparing. Treat baseline and candidate symmetrically. Numerical changes follow [fidelity](references/fidelity.md); cross-precision results disclose both formats and quality.
- Launch pinned implementations to establish support. Verify each knob in the resolved runtime config; an ignored flag is a finding.
- Keep every attempt, including failures, in the [state and ledger](references/state.md). Label numbers observed, derived or inferred and claims PASS / FAIL / UNPROVEN / NOT RUN.
- Check `inferencekit intake budget runs/<run-id>` before long steps. Preserve existing spending authorization; never expand it silently. Keep secrets, access details and provider names out of artifacts.
- For text, keep prefix caching off in the cold-request track; never use `--no-cache-salt`. Reuse needs a separate realistic workload. [Speculation](references/speculative.md) needs its own exactness proof when called lossless.
- Reuse existing engine features, compiler output, harness and resource supervisor before adding code. Run the smallest experiment that can disprove a hypothesis. Avoid unused helpers, headers, configuration and abstractions. Quick screens guide exploration; complete proof belongs to finalists.

## 0. A few questions, then work

Use [starting questions](references/starting-questions.md). Reuse answers in the conversation and intake; ask only missing decisions in short batches. Do not require technical metrics, YAML or a plan from the person.

Quality has three choices: **Exactly the same**, **Same quality, small differences**, **Prioritize speed**. Separately establish permission to change the model itself, the main objective, spending, deadline and desired check-ins. A speed preference does not authorize architecture changes or unchecked quality loss.

No plan or plan approval is required by default. Show or create one only when requested; [plan.md](references/plan.md) covers that optional path and legacy runs. Start once required choices and resource authorization are established. Record their source in state; do not ask again within that scope. For unattended runs use [non-interactive.md](references/non-interactive.md).

## Start or resume

Existing run: first call `inferencekit intake resume runs/<run-id>`; inspect files for reported issues or required details ([state.md](references/state.md)).

Fresh run:

```bash
inferencekit intake template > intake.yaml   # only when no intake was supplied
inferencekit intake validate intake.yaml --allow-placeholders
inferencekit intake validate intake.yaml --run-dir runs/<run-id>
export INFERENCEKIT_RUNS_DIR="$PWD/runs/<run-id>/evidence"
inferencekit intake budget runs/<run-id>
```

Use the derived run id; `--run-dir` pins the intake and starts the clock. SSH runtime and client share the [device host](references/remote-host.md).

## 1. Define the actual task

Validate [intake](references/intake.md); pin revision, component precisions, processing, dataset, shapes and output work. Derive workload details from examples/logs ([workload.md](references/workload.md)); record assumptions.

Choose response latency, sustained throughput, peak memory or affordable hardware under a latency/quality limit ([affordability.md](references/affordability.md)). Token metrics and GSM8K belong to text workloads. Speech needs content, completion and speaker checks; images need semantic/visual checks; video needs temporal checks. Other architectures need a registered task-specific adapter and protocol; unsupported CLI modalities remain UNPROVEN rather than being relabelled as text.

Use [researching-optimizations](../researching-optimizations/SKILL.md) now, from the pinned model, shapes, hardware and runtime. It returns a short experiment queue, including existing engine paths to try before custom code.

## 2. Hardware, resource limits and headroom

Run `inferencekit doctor`. Check competing work, memory, swap, thermals and profiler permissions before occupying the device. Stop only the run's own jobs; do not kill unrelated services to obtain a clean measurement.

On NVIDIA, measured `ceiling bw` and `roofline` inform supported text models. Elsewhere use the [hardware route](../inference-triage/references/hardware.md) and vendor measurements; a PyTorch CUDA namespace does not establish NVIDIA-tool support. For non-token workloads derive bytes/FLOPs per request or step with assumptions. Unsupported ceilings stay NOT RUN.

Estimate critical-path share, attainable efficiency, exact byte savings and permitted algorithmic savings ([ceilings.md](references/ceilings.md), [stop-point.md](references/stop-point.md)). Recompute after changing precision or work. An inferred ceiling orders experiments; it cannot establish a speedup.

## 3. Establish the strongest baseline

Use [candidates.md](references/candidates.md) and the research queue. Launch compatible engines in isolated environments; record versions, commands, resolved configs and smoke results. Text engines, image pipelines, speech runtimes and video pipelines have different candidates. NVIDIA graphs are not a generic launch flag for AMD or Apple.

[Tune](references/tuning.md) supported engines for the goal. A task-fixed baseline is labelled fixed, never automatically best in the field. Reconsider numerical-only baseline rejects under a newly authorized lossy policy.

For text, a fixed baseline with route build may skip the survey and use the [engine skeleton](../optimizing-kernels/skeleton/README.md). Other models use their native pipeline and [custom-engine contract](references/candidates.md#custom-engine-contract), not the text skeleton.

## 4. Screen and measure

Use [harness-commands.md](references/harness-commands.md) for text, [affordability.md](references/affordability.md) for image/audio and [video.md](references/video.md) for video. Screen representative cases before full rounds. Text screens include `short` and `long` lengths. Promote secondary wins the user values without hiding the lead metric.

Final comparisons use balanced fresh-process rounds, excluded warmup, registered stopping and confidence intervals. Include complete output work and transfers/synchronization/decoding. [Capacity](references/capacity.md) needs its own tuning and measured concurrency.

## 5. Quality on the exact speed configuration

Load [proving-correctness](../proving-correctness/SKILL.md) and the selected policy. Exact-output claims need equality on the declared deterministic/seeded contract; a lossless label alone does not prove it. Lossy tracks need agreed task/perceptual bounds, not accidental bitwise rejection. Preserve structural validity, finite outputs and completion in every track.

For text, run applicable gates, reference trajectory/score, full task evaluation and sampling checks. Image/audio/video use registered semantic, structural and perceptual checks instead of token gates or GSM8K. Missing human listening/viewing remains UNPROVEN. Approval frequency changes when to ask, never whether quality has been established.

## 6. Choose the next experiment

Rank experiments by headroom, cost and confidence. Refresh research after two stalled attempts or changed bottleneck/toolchain. [optimizing-kernels](../optimizing-kernels/SKILL.md) routes compiler inspection and conditional techniques.

At a met target follow `goals.after_target`. Otherwise continue within the approved scope and stop rule. With no target, honor the agreed check-in preference; do not repeatedly request approval for an already authorized search.

Default stage time boxes and reporting reserve come from intake. Explicit `budget.stage_timeboxes: false` or `report_reserve_h: 0` override them. A finite deadline still binds. "Unreachable" and "no headroom" are verdicts, never stops under a stated deadline; pursue another useful lever while resources remain. Unavailable devices, evidence or permission are recorded honestly.

## 7. Implement and investigate

Use [compiler inspection](../optimizing-kernels/references/compiler-inspection.md): source → graph → dispatch → executed kernel. Check fusion/fallbacks/conversions, form one hypothesis, change one cause, screen quality and measure the complete inference path. Inspect IR/assembly when useful.

Run independent implementation experiments in parallel under [parallel.md](references/parallel.md). Integrate one candidate at a time; serialize device measurements and avoid host contention. Reuse previous numeric-only rejects when policy broadens, with new evaluation and preserved history ([fidelity.md](references/fidelity.md)).

Keep a commit and screen evidence per retained change. Milestone/final checks follow [state.md](references/state.md): text uses existing light/full jobs; other modalities use registered checks. Final proof applies to the commit you hand over. No full evaluation after every small edit; no published claim from a screen alone.

At freeze, one `claims-skeptic` reviews draft, diff, intake and evidence while final proof runs. Answer refutations with evidence. With zero reserve, keep validation and packaging current through the deadline.

### 8. Final report (always runs, from the report reserve, not before)

Use the shared [report format and final reply](../reporting-perf-claims/references/wording-guide.md): Result, Run it, Files and outputs, then Details. The plain-English summary requirements are in [stop-point.md](references/stop-point.md), "Summary". Separate end-to-end from kernel gains, exact from cross-precision tracks, and screened from verified results. Keep material limits visible in Result; retain unsuccessful attempts and time/cost accounting in Details.

Keep the [manifest](references/manifest.md), reproduce commands, ledger and report consistent with the final commit. Run `inferencekit review build runs/<run-id>` and `inferencekit manifest validate runs/<run-id>/manifest.json --strict`. No freeze (no stage 7): one `claims-skeptic` on the draft before the host is released. Push only within explicit authorization. Record final spend and release resources as authorized.

After the final report, check the result for an [optional engine contribution](references/engine-contribution.md). Prepare a qualifying contribution locally, then open its PR when the user authorizes publication; reuse an explicit opt-in already recorded for this run. A declined contribution stays local. Missing proof or publication access does not restart optimization or hold rented hardware open.

## Parallel work

Shipped read-only agents: `engine-scout`, `quant-scout`, `profile-analyst`, `claims-skeptic`, `log-reader`. Implementation workers may write kernels concurrently in isolated files/branches. One coordinator integrates; one owner per device serializes measurements. See [parallel.md](references/parallel.md).
