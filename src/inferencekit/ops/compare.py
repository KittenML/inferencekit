"""compare: is the candidate faster than the baseline, by more than noise and more than it matters?

Method:
  1. Take one metric per round from each benchmark record (default decode_tok_s).
  2. Pair rounds by round index (baseline round i with candidate round i).
  3. Check that the arms were interleaved in time: for every pair, the two rounds
     are neighbours in the merged timeline (A B A B... or A B B A...). Without
     interleaving, slow drift (temperature, clocks, other jobs) is confounded with
     the change; the comparison still runs but says so in ``limitations``.
  4. Ratio of medians (candidate / baseline) with a paired bootstrap 95% CI
     (10,000 resamples, seed 0).
  5. Verdict against a practical threshold (default 2%): see stats.decide.

Inputs are ``benchmark`` records (bench decode) or single-user ``serve_benchmark`` records (bench serve,
bench rounds cells; their ``data.rounds`` has the same row shape). Both must be the same kind and the
same workload (prompt set sha256, mode, bucket, generation length, prefix-cache state, endpoint); a
load-mode record is per level, so it is refused (compare capacity records' N* instead).

For serve_benchmark records "the same workload" also means the same requests: the prompt-set sha256 is
the whole file's, so the selected subset (--buckets, --limit-per-bucket) and the per-request settings
(max_tokens, chat_template_kwargs, API, exact length, extra body) must match too. The records'
``workload.request_set`` hash says so when both have it; for records without it, the stored bench serve
arguments are compared instead (JSON arguments by value). A serve record with failed requests makes the result inconclusive:
the round medians drop failed requests, so an engine that fails the long prompts would look faster.

Clips (``clip_benchmark`` records from bench clip): metric ``clip_ms`` (lower is better) or ``clips_per_hour``
(higher is better). A round's clip_ms is the mean over its clips, so on a set that mixes tasks (t2v and i2v) a
change to either shows; ``clip_ms_<task>`` and ``clips_per_hour_<task>`` (e.g. clip_ms_i2v) compare one task's
clips only. The same clip set, requested shape (task, size, frames, fps, steps, guidance), request set and
seed policy are required, and each pair of rounds must have used the same seed. A failed clip, or a record whose
claim_validity is false (the engine reports fewer steps or another shape than asked, cached stages, or
byte-identical clips for different requests), makes the result inconclusive. Engines reached through different
endpoint kinds (clip_http, comfyui) can be compared: both are timed on the client until the clip is in hand. A
reused encoding of a fixed negative prompt is allowed for every engine; when only one side reused it, a limitation
says so.
"""

from __future__ import annotations

import math
import re

from inferencekit import evidence as ev
from inferencekit import stats
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve.clip_client import TASKS as CLIP_TASKS
from inferencekit.serve.round_evidence import MAX_ROUNDS, MAX_ROUND_INDEX, finite, memory_of, round_coverage

# bench clip's rounds: every clip, then one task's clips (bench_clip.round_key).
CLIP_METRICS = ("clip_ms", "clips_per_hour") + tuple(f"{m}_{t}" for t in CLIP_TASKS
                                                     for m in ("clip_ms", "clips_per_hour"))
MEDIA_METRICS = {"image_benchmark": ("image_ms", "images_per_hour"),
                 "audio_benchmark": ("audio_ms", "first_audio_p95_ms", "rtf")}
HIGHER_IS_BETTER = {"decode_tok_s": True, "prefill_tok_s": True, "ttft_ms": False, "tpot_ms": False,
                    "total_ms": False, **{m: m.startswith("clips_per_hour") for m in CLIP_METRICS},
                    "image_ms": False, "images_per_hour": True, "audio_ms": False,
                    "first_audio_p95_ms": False, "rtf": False}
BENCH_KINDS = ("benchmark", "serve_benchmark", "clip_benchmark", *MEDIA_METRICS)
UNITS = {**{m: "clips/h" for m in CLIP_METRICS if m.startswith("clips_per_hour")},
         "images_per_hour": "images/h", "rtf": "x"}
# Workload fields that must agree: bench decode's shape, then bench serve's (absent fields are None on both).
SAME_WORKLOAD = ("prompt_len", "gen_len", "batch", "mode", "bucket", "prefix_cache", "endpoint_kind")
# bench serve arguments that change which requests are sent or what they ask for (serve_benchmark only).
SAME_SERVE_ARGS = ("buckets", "limit_per_bucket", "max_tokens", "api", "exact_length", "cache_salt", "extra_body",
                   "chat_template_kwargs")
SERVE_METRICS = ("decode_tok_s", "ttft_ms", "tpot_ms", "total_ms")  # what bench serve's rounds carry
# Clip workload fields that must agree (clip_benchmark only), beside SAME_WORKLOAD.
SAME_CLIP_WORKLOAD = ("task", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "audio")


def _serve_arg(key: str, value):
    """A stored bench serve argument as compared here: JSON objects by value, lists in order-free form."""
    if key in ("extra_body", "chat_template_kwargs"):
        from inferencekit.ops.bench_serve import normalized_json
        return normalized_json(value)
    if key == "buckets":
        return sorted(value) if value else None
    if key == "limit_per_bucket":
        return value or None
    return value


def serve_workload_reasons(a: dict, b: dict) -> list[str]:
    """Why two serve_benchmark records did not send the same requests (empty when they did)."""
    out = []
    ra, rb = ((r["identity"].get("workload") or {}).get("request_set") or {} for r in (a, b))
    if ra.get("sha256") and rb.get("sha256"):
        # The hash covers what was actually sent (the selection, max_tokens, the merged template settings
        # however they were given: flag, --extra-body, intake pin), so the arguments are not compared.
        if ra["sha256"] != rb["sha256"]:
            out.append(f"workloads differ (request_set {ra['sha256'][:12]} vs {rb['sha256'][:12]}: different "
                       "prompt subset or per-request settings)")
        return out
    args_a, args_b = (((r.get("data") or {}).get("request") or {}).get("args") or {} for r in (a, b))
    for key in SAME_SERVE_ARGS:
        va, vb = _serve_arg(key, args_a.get(key)), _serve_arg(key, args_b.get(key))
        if va != vb:
            out.append(f"workloads differ (bench serve --{key.replace('_', '-')}: {va} vs {vb})")
    return out


def clip_workload_reasons(a: dict, b: dict) -> list[str]:
    """Why two clip_benchmark records did not ask for the same clips (empty when they did)."""
    out = []
    wa, wb = (r["identity"].get("workload") or {} for r in (a, b))
    for key in SAME_CLIP_WORKLOAD:
        if wa.get(key) != wb.get(key):
            out.append(f"workloads differ ({key}: {wa.get(key)} vs {wb.get(key)})")
    for key, what in (("clip_set", "clip sets"), ("request_set", "request sets")):
        sa, sb = ((w.get(key) or {}).get("sha256") for w in (wa, wb))
        if sa != sb:
            out.append(f"different {what} ({str(sa)[:12]} vs {str(sb)[:12]})")
    pa, pb = ({k: (w.get("seed_policy") or {}).get(k) for k in ("name", "base")} for w in (wa, wb))
    if pa != pb:
        out.append(f"different seed policies ({pa} vs {pb})")
    return out


def clip_seed_reasons(base: dict[int, dict], cand: dict[int, dict], pairs: list[int]) -> list[str]:
    """Paired clip rounds must have used the same seed (the same requests, minutes apart)."""
    bad = [i for i in pairs if base[i].get("seed") != cand[i].get("seed")]
    if not bad:
        return []
    i = bad[0]
    return [f"{len(bad)} paired round(s) used different seeds (round {i}: {base[i].get('seed')} vs "
            f"{cand[i].get('seed')})"]


def media_workload_reasons(a: dict, b: dict, image_tradeoff: bool = False) -> list[str]:
    """Media speed claims require complete request and checkpoint identity.

Quantization/offload studies may still report raw measurements and quality, but
different numerical formats cannot become a same-precision engine speed claim.
"""
    out = []
    for label, rec in (("baseline", a), ("candidate", b)):
        workload = rec["identity"].get("workload") or {}
        if not (workload.get("request_set") or {}).get("sha256"):
            out.append(f"{label} missing request_set hash")
        if not rec["identity"]["gpu"].get("name") and not (rec["identity"].get("cpu") or {}).get("model"):
            out.append(f"{label} missing hardware identity")
        if not rec["identity"]["model"].get("id"):
            out.append(f"{label} missing model identity")
        for key in ("revision", "dtype", "quant"):
            if not rec["identity"]["model"].get(key):
                out.append(f"{label} missing model {key}; declare unquantized weights explicitly as quant=none")
        if not rec["identity"]["engine"].get("version"):
            out.append(f"{label} missing pinned engine version")
        validity = rec.get("data", {}).get("claim_validity") or {}
        if validity.get("valid") is not True:
            out.append(f"{label} structural validity not proven")
    wa, wb = (r["identity"].get("workload") or {} for r in (a, b))
    if image_tradeoff:
        ha, hb = ((w.get("image_task_set") or {}).get("sha256") for w in (wa, wb))
        if not ha or not hb or ha != hb:
            out.append("image task identity missing or differs (prompts, dimensions, seeds or other settings)")
    elif (wa.get("request_set") or {}).get("sha256") != (wb.get("request_set") or {}).get("sha256"):
        out.append("workloads differ (request_set: fixtures, sampling, precision or generation settings)")
    for key in (("id", "revision") if image_tradeoff else ("id", "revision", "dtype", "quant")):
        if a["identity"]["model"].get(key) != b["identity"]["model"].get(key):
            out.append(f"model identity differs ({key}); format trade-offs require separate quality reporting")
    return out


def round_values(record: dict, metric: str) -> dict[int, dict]:
    """{round index: round} for rounds that have the metric."""
    out = {}
    rows = record.get("data", {}).get("rounds") or []
    for r in rows[:MAX_ROUNDS] if isinstance(rows, list) else []:
        if isinstance(r, dict) and type(r.get("index")) is int and 0 <= r["index"] <= MAX_ROUND_INDEX and finite(r.get(metric)):
            out[r["index"]] = r
    return out


def cpu_hardware_reasons(a: dict, b: dict) -> list[str]:
    ca, cb = (r["identity"].get("cpu") for r in (a, b))
    if ca is None and cb is None:
        return []
    if ca is None or cb is None:
        return ["hardware backends differ or CPU identity is missing from one arm"]
    reasons = []
    for label, rec, cpu in (("baseline", a, ca), ("candidate", b, cb)):
        if rec["identity"]["gpu"].get("name"):
            reasons.append(f"{label} mixes CPU and GPU hardware identities")
        if not cpu.get("model"):
            reasons.append(f"{label} missing CPU model identity")
    # Affinity and threads are tuning knobs; topology is the host's identity.
    for key in ("model", "architecture", "logical_cores", "physical_cores", "topology_sha256"):
        if ca.get(key) is not None and cb.get(key) is not None and ca[key] != cb[key]:
            reasons.append(f"different CPU hardware ({key})")
    return reasons


def lifecycle_verdict(record: dict, coverage: dict) -> str:
    data = record.get("data") or {}
    proof = data.get("lifecycle") or {}
    contexts = data.get("runtime_rounds") or []
    expected = coverage.get("expected_indices")
    if not (data.get("scheduler_final") is True and proof.get("source") == "bench rounds"
            and proof.get("fresh_process_per_cell") is True and proof.get("valid") is True
            and proof.get("verdict") == "PASS" and coverage.get("complete") is True
            and expected and len(contexts) == len(expected)):
        return "UNPROVEN"
    indices = [r.get("index") for r in contexts if isinstance(r, dict)]
    if (len(indices) != len(contexts) or any(type(i) is not int for i in indices)
            or len(set(indices)) != len(indices) or set(indices) != set(expected)):
        return "UNPROVEN"
    if not all(r.get("status") == "ok" and (r.get("ready") or {}).get("status") == "ready"
               and (r.get("stop") or {}).get("group_gone") is True for r in contexts):
        return "UNPROVEN"
    return "PASS"


def arm_environment(record: dict) -> dict | None:
    """Selected scheduler-recorded environment facts; no arbitrary engine flags."""
    env = (record.get("identity", {}).get("engine", {}).get("flags") or {}).get("server_env")
    if not isinstance(env, dict):
        return None
    return {key: env[key] for key in ("controls", "ld_preload", "omitted", "varied_between_rounds") if key in env}


def environment_differences(environments: dict) -> list[str]:
    a, b = environments["baseline"], environments["candidate"]
    if a is None or b is None:
        return ["availability"] if a != b else []
    ca, cb = a.get("controls") or {}, b.get("controls") or {}
    if not isinstance(ca, dict) or not isinstance(cb, dict):
        return ["unverified controls"]
    names = [k for k in sorted(set(ca) | set(cb)) if isinstance(k, str)
             and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", k) and ca.get(k) != cb.get(k)]
    if a.get("ld_preload") != b.get("ld_preload"):
        names.append("LD_PRELOAD")
    if a.get("varied_between_rounds") != b.get("varied_between_rounds"):
        names.append("round stability")
    if a.get("omitted") != b.get("omitted"):
        names.append("omitted controls")
    return names


def interleaved(base: dict[int, dict], cand: dict[int, dict], pairs: list[int]) -> bool | None:
    """True if each pair's rounds are adjacent in time; None if timestamps are missing."""
    events = []
    for arm, rounds in (("b", base), ("c", cand)):
        for i in pairs:
            t = rounds[i].get("t_start")
            if not finite(t):
                return None
            events.append((t, arm, i))
    events.sort()
    position = {(arm, i): k for k, (_, arm, i) in enumerate(events)}
    return all(abs(position[("b", i)] - position[("c", i)]) == 1 for i in pairs)


@operation("compare", ("compare",), "Interleaved-rounds check, ratio of medians with a paired bootstrap 95% CI, "
           "and a verdict: improved / regressed / within_noise / inconclusive.",
           [Arg("baseline", "string", "Baseline benchmark or single-user serve_benchmark evidence id.", required=True,
                positional=True),
            Arg("candidate", "string", "Candidate benchmark or single-user serve_benchmark evidence id.",
                required=True, positional=True),
            Arg("metric", "string", "Per-round metric to compare.", default="decode_tok_s",
                choices=tuple(HIGHER_IS_BETTER)),
            Arg("threshold", "number", "Practical threshold as a fraction (0.02 = 2%).", default=0.02),
            Arg("expected_rounds", "integer", "Intended total rounds per arm, indexed from zero; checks missing tail rounds when no plan was recorded."),
            Arg("image_tradeoff", "boolean", "Compare image timing with permitted step/precision differences; "
                "quality and lifecycle require separate proof. Requires image task identities.", default=False),
            Arg("resamples", "integer", "Bootstrap resamples.", default=stats.DEFAULT_RESAMPLES),
            Arg("seed", "integer", "Bootstrap seed.", default=0)])
def compare(call, baseline, candidate, metric, threshold, resamples, seed, image_tradeoff=False, expected_rounds=None):
    if expected_rounds is not None and not 1 <= expected_rounds <= MAX_ROUNDS:
        raise OpError(f"expected_rounds must be positive and <= {MAX_ROUNDS}")
    store = ev.Store()
    a, b = store.load(baseline), store.load(candidate)
    higher = HIGHER_IS_BETTER[metric]
    ra, rb = round_values(a, metric), round_values(b, metric)
    limitations, missing = [], []
    reasons_inconclusive = []
    arm_coverage = {label: round_coverage(rec, expected_rounds) for label, rec in (("baseline", a), ("candidate", b))}
    memory = {label: memory_of(rec) for label, rec in (("baseline", a), ("candidate", b))}
    environments = {label: arm_environment(rec) for label, rec in (("baseline", a), ("candidate", b))}

    for rec, label in ((a, "baseline"), (b, "candidate")):
        coverage = arm_coverage[label]
        for key, what in (("invalid_indices", "invalid or oversized round indices"),
                          ("invalid_plan", "invalid round plan"), ("oversized_plan", "oversized round plan"),
                          ("oversized_rows", "too many recorded rounds")):
            if coverage[key]:
                reasons_inconclusive.append(f"{label} {what}")
        plan = rec.get("data", {}).get("round_plan") or {}
        recorded_expected = plan.get("expected_indices") if isinstance(plan, dict) else None
        if expected_rounds is not None and recorded_expected is not None and recorded_expected != list(range(expected_rounds)):
            reasons_inconclusive.append(f"{label} --expected-rounds conflicts with its recorded round plan")
        raw_rows = rec.get("data", {}).get("rounds") or []
        rows = raw_rows[:MAX_ROUNDS] if isinstance(raw_rows, list) else []
        if any(not isinstance(r, dict) or type(r.get("index")) is not int or r["index"] < 0 for r in rows):
            reasons_inconclusive.append(f"{label} has invalid round indices")
            coverage["complete"] = False
        for key, what in (("missing_indices", "missing requested rounds"), ("unexpected_indices", "rounds outside the requested plan"),
                          ("gaps", "gaps in observed rounds"),
                          ("duplicate_indices", "duplicate round indices")):
            if coverage[key]:
                reasons_inconclusive.append(f"{label} {what}: {coverage[key]}")
                coverage["complete"] = False
        metric_indices = set((ra if label == "baseline" else rb))
        metric_missing = sorted(set(coverage["actual_indices"]) - metric_indices)
        coverage["metric_indices"] = sorted(metric_indices)
        coverage["missing_metric_indices"] = metric_missing
        if metric_missing:
            reasons_inconclusive.append(f"{label} missing finite {metric} in rounds: {metric_missing}")
            coverage["complete"] = False
        if coverage["expected_indices"] is None:
            limitations.append(f"{label} planned round count unknown; missing tail rounds cannot be checked "
                               "without --expected-rounds or a recorded round plan")
        if not memory[label]:
            limitations.append(f"{label} peak RSS/VRAM was not recorded; memory use is unknown")
        for key, measurement in memory[label].items():
            if measurement.get("complete") is not True:
                qualifier = "partial" if measurement.get("complete") is False else "unknown"
                limitations.append(f"{label} {key} coverage is {qualifier}; the recorded value does not establish "
                                   "the peak over every requested round")
            if measurement.get("verdict") not in (None, "PASS"):
                limitations.append(f"{label} {key} source memory verdict: {measurement['verdict']}")
            limitations.extend(f"{label} {key}: {note}" for note in measurement.get("limitations", [])
                               if isinstance(note, str))
            if measurement.get("method") == "sampled_process_group_rss":
                limitations.append(f"{label} peak RSS is sampled process-group RSS: a lower bound on the true peak, "
                                   "with shared pages counted per process")
        server_env = (rec.get("identity", {}).get("engine", {}).get("flags") or {}).get("server_env")
        envs = [server_env] if isinstance(server_env, dict) else []
        if isinstance(server_env, dict) and "varied_between_rounds" in server_env:
            reasons_inconclusive.append(f"{label} server tuning environment varied or was unavailable between rounds")
            envs = server_env["varied_between_rounds"] if isinstance(server_env["varied_between_rounds"], list) else []
        if any(isinstance(env, dict) and (env.get("omitted") or any(
                value is None for value in (env.get("controls") or {}).values())) for env in envs):
            limitations.append(f"{label} requested CPU controls were omitted or invalid; the effective tuning "
                               "configuration is unverified")
        if any(isinstance(env, dict) and any(isinstance(preload, dict) and preload.get("status") != "hashed"
                for preload in env.get("ld_preload", [])) for env in envs):
            limitations.append(f"{label} preload library identity was not fully verified")
        problem = ev.integrity_problem(rec)
        if problem:  # an edited input must not turn into a fresh, verifying comparison
            reasons_inconclusive.append(f"{label} {problem}")
            missing.append("input integrity")
        if rec.get("kind") not in BENCH_KINDS:
            reasons_inconclusive.append(f"{label} is a '{rec.get('kind')}' record, not a benchmark")
        elif rec.get("kind") == "serve_benchmark" and (rec.get("identity", {}).get("workload") or {}).get(
                "mode") == "load":
            reasons_inconclusive.append(f"{label} is a load-mode serve_benchmark (one level, no rounds); compare "
                                        "capacity records' N* instead")
        failed = sum(int(r.get("n_failed") or 0) for r in rows if isinstance(r, dict))
        if failed and rec.get("kind") in ("serve_benchmark", "clip_benchmark", *MEDIA_METRICS):
            # Round medians drop failed requests: an arm that fails the slow prompts would look faster.
            reasons_inconclusive.append(f"{label} had {failed} failed request(s); its round medians exclude them, "
                                        "so the arms did not time the same requests")
        elif failed:
            limitations.append(f"{label} had {failed} failed request(s); its round medians exclude them")
        validity = rec.get("data", {}).get("claim_validity") or {}
        if validity.get("valid") is False and rec.get("kind") in ("clip_benchmark", *MEDIA_METRICS):
            # Fewer steps, another shape, cached stages or repeated bytes: the clip was not the one asked for.
            reasons_inconclusive.append(f"{label} claim invalid: {'; '.join(validity.get('violations', [])[:3])}")
        elif validity.get("valid") is False:
            limitations.append(f"{label} violates hard gates: {'; '.join(validity.get('violations', []))}")
        if not rec.get("coverage", {}).get("complete", True):
            limitations.append(f"{label} coverage incomplete: {rec['coverage'].get('missing')}")
    clips = a.get("kind") == b.get("kind") == "clip_benchmark"
    media = a.get("kind") == b.get("kind") and a.get("kind") in MEDIA_METRICS
    if image_tradeoff and not (a.get("kind") == b.get("kind") == "image_benchmark"):
        raise OpError("image_tradeoff is only available for two image benchmark records")
    if a.get("kind") != b.get("kind") and a.get("kind") in BENCH_KINDS and b.get("kind") in BENCH_KINDS:
        how = ("bench clip times whole clips, the others time tokens" if "clip_benchmark" in (a.get("kind"), b.get(
            "kind")) else "bench decode times inside the engine process, bench serve on the client")
        reasons_inconclusive.append(f"different timers ({a.get('kind')} vs {b.get('kind')}): {how}")
    wa, wb = a["identity"]["workload"], b["identity"]["workload"]
    for key in SAME_WORKLOAD:
        if clips and key == "endpoint_kind":
            continue  # clip_http and comfyui share the client timer (until the clip is in hand); noted below
        if wa.get(key) != wb.get(key):
            reasons_inconclusive.append(f"workloads differ ({key}: {wa.get(key)} vs {wb.get(key)})")
    if a.get("kind") == b.get("kind") == "serve_benchmark":
        reasons_inconclusive += serve_workload_reasons(a, b)
    if media:
        reasons_inconclusive += media_workload_reasons(a, b, image_tradeoff=image_tradeoff)
        if metric not in MEDIA_METRICS[a["kind"]]:
            reasons_inconclusive.append(f"{metric} is not measured by {a['kind']}")
        limitations.append("This compares client-timed performance only; image/speech quality and repeated-run "
                           "reliability require their own evidence. Fresh-process lifecycle is checked separately.")
        if image_tradeoff:
            limitations.append("Image format/work tradeoff: steps and numerical formats may differ. "
                               "Timing is not a quality or same-work engine comparison.")
    if clips:
        reasons_inconclusive += clip_workload_reasons(a, b)
        if metric in ("clip_ms", "clips_per_hour") and wa.get("task") is None and wb.get("task") is None:
            limitations.append(f"the clip set mixes tasks: each round's {metric} covers every clip (clip_ms is their "
                               f"mean), so a change to one task is diluted by the others; --metric {metric}_t2v or "
                               f"{metric}_i2v compares one task")
        if wa.get("endpoint_kind") != wb.get("endpoint_kind"):
            limitations.append(f"engines reached through different endpoint kinds ({wa.get('endpoint_kind')} vs "
                               f"{wb.get('endpoint_kind')}): the same client timer, different transport")
        reuse = {label: int(((rec.get("data") or {}).get("claim_validity") or {}).get("cache_allowed_clips") or 0)
                 for rec, label in ((a, "baseline"), (b, "candidate"))}
        if (reuse["baseline"] > 0) != (reuse["candidate"] > 0):
            who = "baseline" if reuse["baseline"] else "candidate"
            limitations.append(f"only the {who} reused its fixed negative prompt's encoding ({reuse[who]} clip(s)); "
                               "the other encoded it per clip. Allowed for every engine, so part of the difference "
                               "may be that reuse")
    sa, sb = ((w.get("prompt_set") or {}).get("sha256") for w in (wa, wb))
    if sa != sb and not image_tradeoff:
        reasons_inconclusive.append(f"different prompt sets ({str(sa)[:12]} vs {str(sb)[:12]})")
    ga, gb = a["identity"]["gpu"].get("name"), b["identity"]["gpu"].get("name")
    if ga and gb and ga != gb:
        reasons_inconclusive.append(f"different GPUs ({ga} vs {gb})")
    reasons_inconclusive += cpu_hardware_reasons(a, b)
    if a["identity"].get("cpu") is not None and b["identity"].get("cpu") is not None:
        if any(not (r["identity"]["cpu"] or {}).get("topology_sha256") for r in (a, b)):
            limitations.append("CPU topology identity is incomplete; equal model labels alone do not verify the same topology")

    unavailable = [label for rec, rounds, label in ((a, ra, "baseline"), (b, rb, "candidate"))
                   if not rounds and rec.get("data", {}).get("rounds")]
    if unavailable:
        if any(r.get("kind") == "clip_benchmark" for r in (a, b)) and metric not in CLIP_METRICS:
            why = f"{metric} is not measured by bench clip (its rounds carry {', '.join(CLIP_METRICS)})"
        elif any(r.get("kind") == "serve_benchmark" for r in (a, b)) and metric not in SERVE_METRICS:
            why = f"{metric} is not measured by bench serve (its rounds carry {', '.join(SERVE_METRICS)})"
        else:
            why = f"{metric} is absent from every round of the {' and '.join(unavailable)} record"
        reasons_inconclusive.append(why + "; more rounds will not help, pick another --metric")
    pairs = sorted(set(ra) & set(rb))
    if set(ra) != set(rb):
        reasons_inconclusive.append(f"unmatched metric rounds (baseline only {sorted(set(ra) - set(rb))}; "
                                    f"candidate only {sorted(set(rb) - set(ra))})")
    if clips or media:
        reasons_inconclusive += clip_seed_reasons(ra, rb, pairs)
    paired = len(pairs) >= 2
    inter = interleaved(ra, rb, pairs) if pairs else None
    if media:
        if len(pairs) < 3 or set(ra) != set(rb):
            reasons_inconclusive.append("media comparisons need at least three complete matching round pairs")
        if inter is not True:
            reasons_inconclusive.append("media comparison requires verified interleaving")
        windows = []
        for label, rows in (("baseline", ra), ("candidate", rb)):
            for index, row in rows.items():
                begin, end = row.get("t_start"), row.get("t_end")
                if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                           for t in (begin, end)) or end <= begin:
                    reasons_inconclusive.append(f"{label} round {index} has no valid timing window")
                else:
                    windows.append((begin, end))
                if not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool):
                    reasons_inconclusive.append(f"{label} round {index} has no recorded integer seed")
        windows.sort()
        if any(right[0] < left[1] for left, right in zip(windows, windows[1:])):
            reasons_inconclusive.append("media rounds overlapped in time")
    if inter is None:
        limitations.append("rounds carry no timestamps: interleaving cannot be verified")
    elif inter is False:
        limitations.append("rounds were not interleaved (A B A B ...): drift between runs is confounded with the "
                           "change; rerun alternating rounds (bench rounds, or bench decode / bench serve --extend)")
    # Pair only when interleaving is verified: round i of each arm must have run back to back.
    if paired and inter is True:
        res = stats.paired_ratio_ci([ra[i][metric] for i in pairs], [rb[i][metric] for i in pairs], resamples, seed)
        method = "paired bootstrap over round index"
        n = res["n_pairs"]
    else:
        res = stats.unpaired_ratio_ci([r[metric] for r in ra.values()], [r[metric] for r in rb.values()],
                                      resamples, seed)
        method = "unpaired bootstrap (rounds could not be paired by interleaved index)"
        n = min(res["n_baseline"], res["n_candidate"])
        if pairs:
            limitations.append("used an unpaired bootstrap: interleaving not verified, so rounds are not paired")
    lo, hi = res["ci95"]
    decision = stats.decide(res["ratio"], lo, hi, threshold, higher)
    if reasons_inconclusive:
        decision = {**decision, "verdict": "inconclusive", "reason": "; ".join(reasons_inconclusive)}
        missing.append("comparable inputs")
    if n < 2:
        missing.append("rounds (need >= 2 per arm)")
    # More rounds help only when the CI is too wide; not when the inputs cannot be compared.
    needed = None
    if decision["verdict"] == "inconclusive" and not reasons_inconclusive and decision.get("powered") is False:
        needed = stats.rounds_for_power(n, decision.get("half_width"), threshold)
    if needed:
        limitations.append(f"about {needed} interleaved rounds per arm would likely make the CI narrow enough "
                           f"(+/-{stats.POWER_RATIO * threshold:.1%}); estimate scaled from this run's CI")
    ratio = res["ratio"]
    change = f"{(ratio - 1) * 100:+.1f}%" if ratio else "n/a"
    ci_text = f" (95% CI {lo:.3f}-{hi:.3f})" if lo is not None and hi is not None else ""
    summary = (f"{metric}: candidate/baseline = {ratio:.3f}{ci_text}, {change}, n={n}: {decision['verdict']}"
               if ratio else f"{metric}: no ratio ({decision['reason']})")
    if image_tradeoff:
        summary = "image format/work tradeoff timing; quality requires separate evidence: " + summary
    memory_summary = []
    for key, name in (("peak_rss_mib", "peak RSS"), ("peak_vram_mib", "peak VRAM")):
        if any(key in row for row in memory.values()):
            values = []
            for label in ("baseline", "candidate"):
                measurement = memory[label].get(key)
                if measurement is None:
                    values.append(f"{label} unknown")
                    continue
                qualifiers = []
                if measurement.get("method") == "sampled_process_group_rss":
                    qualifiers.append("sampled")
                if measurement.get("complete") is not True:
                    qualifiers.append("partial" if measurement.get("complete") is False else "coverage unknown")
                suffix = " (" + ", ".join(qualifiers) + ")" if qualifiers else ""
                values.append(f"{label} {measurement['value']:.1f} MiB{suffix}")
            memory_summary.append(name + " " + ", ".join(values))
    if memory_summary:
        summary += "; " + "; ".join(memory_summary)
    env_differences = environment_differences(environments)
    if env_differences:
        note = ", ".join(env_differences[:4]) + (", ..." if len(env_differences) > 4 else "")
        summary += "; server env differs: " + note
        limitations.append("Requested server environment differs between arms; controls and preload identities "
                           "are retained in data.arm_environment. Requested settings do not prove runtime dispatch.")
    metrics = [ev.metric(f"{metric}_ratio", ratio, "x", "ratio_of_medians", n=n, ci95=[lo, hi])]
    for rec, rounds, label in ((a, ra, "baseline"), (b, rb, "candidate")):
        vals = [r[metric] for r in rounds.values()]
        s = stats.summarize(vals)
        metrics.append(ev.metric(metric, s.get("median"), UNITS.get(metric) or ("tok/s" if higher else "ms"),
                                 "median", n=len(vals), subject=label))
        for key, measurement in memory[label].items():
            metrics.append(ev.metric(key, measurement["value"], "MiB", "max", subject=label))
    lifecycle = {label: lifecycle_verdict(rec, arm_coverage[label]) for label, rec in (("baseline", a), ("candidate", b))}
    record = ev.build(call, kind="comparison", claim_class="derived", summary=summary, identity_=b["identity"],
                      metrics=metrics, missing=missing, limitations=limitations,
                      data={"verdict": decision["verdict"], "decision": decision, "metric": metric,
                            "comparison_scope": "image_format_work_tradeoff" if image_tradeoff else "matched_workload",
                            "higher_is_better": higher, "threshold": threshold, "method": method,
                            "interleaved": inter, "pairs": len(pairs), "rounds_for_power": needed,
                            "resamples": resamples, "seed": seed,
                            "baseline": a["evidence_id"], "candidate": b["evidence_id"],
                            "baseline_identity": a["identity"],
                            "arm_round_coverage": arm_coverage, "expected_rounds": expected_rounds,
                            "memory": memory, "arm_environment": environments,
                            **({"source_kind": a["kind"], "required_proofs": {
                                "media quality": "UNPROVEN", "fresh-process lifecycle": "PASS" if all(v == "PASS" for v in lifecycle.values()) else "UNPROVEN",
                                "repeated-run reliability": "UNPROVEN"},
                                "arm_lifecycle_verdicts": lifecycle,
                                "arm_timing_verdicts": {label: rec.get("data", {}).get("timing_verdict", "NOT RUN")
                                    for label, rec in (("baseline", a), ("candidate", b))}}
                               if media else {})})
    return store.save(record)
