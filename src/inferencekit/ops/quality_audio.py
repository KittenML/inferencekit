"""Registered speech quality against the exact PCM saved by bench audio."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from inferencekit import evidence as ev, stats
from inferencekit.canonical import canonical_json
from inferencekit.ops.bench_audio import load_fixtures
from inferencekit.quality import audio as qa, metrics as qm
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve.round_evidence import round_coverage


def _write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


@operation("quality_audio_policy", ("quality", "audio-policy"),
           "Register a speech quality policy and fixture identity before benchmark capture; no model execution.",
           [Arg("policy", "string", "JSON acceptance, margins, scorer pins and bootstrap plan.", required=True),
            Arg("fixtures", "string", "The speech fixture JSONL to be benchmarked.", required=True),
            Arg("run_dir", "string", "Also copy the registered policy evidence to this run.")])
def quality_audio_policy(call, policy, fixtures, run_dir):
    doc, _ = qa.read_json(policy)
    protocol = qa.validate_policy(doc)
    fixture_rows, fixture_sha = load_fixtures(fixtures)
    for fixture in fixture_rows:
        if not qa.words(fixture.text, protocol["normalization"]):
            raise OpError("audio policy fixture text has no words after normalization")
    data = {"format": qa.POLICY_FORMAT, "source": "quality audio-policy", "policy": protocol,
            "policy_sha256": qa.sha(canonical_json(protocol).encode()), "fixture_sha256": fixture_sha,
            "fixtures": [f.identity() for f in fixture_rows]}
    store = ev.Store()
    with store.staging_dir() as staging:
        relative = "quality/audio/policies/" + qa.sha(canonical_json(data).encode()) + ".json"
        path = _write(staging, relative, data)
        if run_dir:
            _write(Path(run_dir), relative, data)
        record = ev.build(call, kind="quality_audio", claim_class="derived", summary="Speech quality policy registered; measurements NOT RUN",
                          artifacts=[ev.artifact(path, staging, "audio_quality_policy")],
                          data={**data, "verdict": "NOT RUN"})
        return store.save(record, staging)


def _timestamp(value):
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError, TypeError):
        return None


def _source_compatibility(a, b, policy):
    """Engine/precision changes cannot change the spoken task, conditioning or seeds."""
    left, right = a["data"], b["data"]
    if left["fixture_set"] != right["fixture_set"]:
        raise OpError("audio quality requires the same complete fixture set")
    if {r["index"]: r["seed"] for r in left["rounds"]} != {r["index"]: r["seed"] for r in right["rounds"]}:
        raise OpError("audio quality requires identical round indices and seeds")
    for key in ("id", "revision"):
        if a["identity"]["model"].get(key) != b["identity"]["model"].get(key):
            raise OpError("audio quality requires the same model and checkpoint revision")
    if not policy["allow_numerics_change"] and any(a["identity"]["model"].get(k) != b["identity"]["model"].get(k)
                                                   for k in ("dtype", "quant")):
        raise OpError("audio numerical changes must be explicitly allowed in the registered policy")
    def settings(data):
        config = data["audio_config"]
        ignored = {"engine_version"} | ({"dtype", "quant"} if policy["allow_numerics_change"] else set())
        return {k: config[k] for k in ("sample_rate", "channels", "seed_base")} | {
            "settings": {k: v for k, v in config["settings"].items() if k not in ignored}}
    if settings(left) != settings(right):
        raise OpError("audio quality requires matching generation and audio settings")


def _metric_check(name, policy, observations, keys):
    """Pair on fixtures, retaining all rounds within each resampled fixture."""
    spec = policy["metrics"][name]
    totals = {"baseline": {}, "candidate": {}}
    missing, outliers, retry_missing = [], [], []
    retry_totals = {"baseline": {}, "candidate": {}}
    for fid, round_index in keys:
        pair = {arm: observations.get((arm, fid, round_index), {}) for arm in totals}
        if any(name not in v for v in pair.values()):
            missing.append(f"{fid}/r{round_index}")
            continue
        if name == "listening":
            left, right = pair["baseline"][name], pair["candidate"][name]
            if set(left) != set(right) or len(left) < spec["min_raters"]:
                missing.append(f"{fid}/r{round_index}: paired listener panel")
                continue
        outlier = name == "wer" and any(v[name]["wer"] > spec["outlier_wer"] for v in pair.values())
        if outlier:
            outliers.append(f"{fid}/r{round_index}")
            if any("wer_retry" not in v for v in pair.values()):
                retry_missing.append(f"{fid}/r{round_index}")
        for arm, values in pair.items():
            value = values[name]
            amount, count = (value["errors"], value["words"]) if name == "wer" else (
                sum(value.values()) / len(value), 1) if name == "listening" else (value, 1)
            old = totals[arm].get(fid, (0, 0))
            totals[arm][fid] = (old[0] + amount, old[1] + count)
            if name == "wer":
                retry = values.get("wer_retry") if outlier else value
                if retry is not None:
                    old = retry_totals[arm].get(fid, (0, 0))
                    retry_totals[arm][fid] = (old[0] + retry["errors"], old[1] + retry["words"])
    ids = sorted(set(totals["baseline"]) & set(totals["candidate"]))
    def score(values):
        baseline = [values["baseline"][i] for i in ids]
        candidate = [values["candidate"][i] for i in ids]
        result = qm.paired_delta_ci(candidate, baseline, policy["resamples"], policy["seed"])
        lo, hi = result["ci95"]
        if name != "wer":
            lo, hi = (-hi if hi is not None else None), (-lo if lo is not None else None)
        verdict, reason = qm.margin_verdict(lo, hi, spec["margin"])
        return {**result, "baseline": qm.pooled(baseline), "candidate": qm.pooled(candidate),
                "worse_ci95": [lo, hi], "verdict": verdict, "reason": reason}
    result = score(totals)
    result.update(margin=spec["margin"], missing=missing, outliers=outliers,
                  method="paired fixture-cluster bootstrap; all rounds retained per fixture")
    if missing or len(ids) < policy["min_fixtures"]:
        # A sufficiently complete observed regression remains a regression. Otherwise no omitted cases may PASS.
        if result["verdict"] != "FAIL":
            result.update(verdict="UNPROVEN", reason="missing observations or fewer than the registered fixture count")
    if outliers:
        result["primary_verdict"] = result["verdict"]
        if retry_missing:
            result.update(verdict="UNPROVEN", reason="ASR outliers require both arms' registered corroborating decode",
                          retry_missing=retry_missing)
        else:
            retry = score(retry_totals)
            result["retry"] = retry
            if retry["verdict"] != result["verdict"]:
                result.update(verdict="UNPROVEN", reason="ASR primary and corroborating decode disagree; review both transcripts")
    return result


@operation("quality_audio", ("quality", "audio"),
           "Score saved speech PCM with exact parity and registered, hash-bound ASR/MOS/listening observations.",
           [Arg("baseline", "string", "Baseline bench audio evidence id.", required=True),
            Arg("candidate", "string", "Candidate bench audio evidence id.", required=True),
            Arg("fixtures", "string", "Original speech fixture JSONL.", required=True),
            Arg("protocol", "string", "Evidence id from quality audio-policy.", required=True),
            Arg("scores", "string", "Optional external observation JSON; never executes or downloads models."),
            Arg("run_dir", "string", "Also copy evidence and quality/audio_score.json to this run.")])
def quality_audio(call, baseline, candidate, fixtures, protocol, scores, run_dir):
    store = ev.Store()
    registered, policy_root = qa.load_record(store, protocol, "quality_audio")
    if registered["data"].get("format") != qa.POLICY_FORMAT:
        raise OpError("protocol must be a quality audio-policy record")
    qa.artifact_bytes(policy_root, qa.one_artifact(registered, "audio_quality_policy"))
    policy = qa.validate_policy(registered["data"]["policy"])
    fixture_rows, fixture_sha = load_fixtures(fixtures)
    if fixture_sha != registered["data"]["fixture_sha256"]:
        raise OpError("audio fixture set differs from the registered policy")
    a, arows = qa.load_benchmark(store, baseline)
    b, brows = qa.load_benchmark(store, candidate)
    _source_compatibility(a, b, policy)
    if a["data"]["fixture_set"]["sha256"] != fixture_sha:
        raise OpError("audio source fixtures differ from the policy and supplied texts")
    if set(arows) != set(brows):
        raise OpError("audio source request pairs are incomplete")
    keys = sorted(arows)
    registered_at = _timestamp(registered["provenance"]["ended"])
    starts = [r.get("t_start") for rec in (a, b) for r in rec["data"]["rounds"]]
    preregistered = bool(registered_at is not None and starts and all(
        qa.finite(t) and registered_at <= t for t in starts))
    observations, imported = qa.load_scores(scores, registered["evidence_id"], a["evidence_id"], b["evidence_id"],
                                            policy, {"baseline": arows, "candidate": brows},
                                            {f.id: f.text for f in fixture_rows})
    missing, pairs = [], []
    source_coverage = {arm: round_coverage(rec) for arm, rec in (("baseline", a), ("candidate", b))}
    for arm, rec in (("baseline", a), ("candidate", b)):
        coverage = source_coverage[arm]
        if (coverage["complete"] is False or coverage["gaps"] or
                rec["data"].get("scheduler_final") and coverage["complete"] is not True):
            missing.append(f"complete {arm} source round coverage")
    round_verdict = "UNPROVEN" if missing else "PASS"
    structural = "PASS"
    for key in keys:
        left, right = arows[key], brows[key]
        pair = {"id": key[0], "round": key[1], "seed": left["seed"],
                "baseline_pcm_sha256": left["pcm_sha256"], "candidate_pcm_sha256": right["pcm_sha256"]}
        if not left["ok"] or not right["ok"]:
            structural = "FAIL"
            pair["error"] = "source request failed structural checks"
        elif left["pcm"] is None or right["pcm"] is None:
            missing.append(f"saved PCM {key[0]}/r{key[1]}")
        else:
            pair.update(qa.parity(left["pcm"], right["pcm"], left["channels"]))
        pairs.append(pair)
    if any(rec["data"].get("structural_verdict") == "FAIL" for rec in (a, b)):
        structural = "FAIL"
    exact = "FAIL" if structural == "FAIL" or any(p.get("bit_exact") is False for p in pairs) else (
        "UNPROVEN" if missing else "PASS")
    checks = {"structural": {"verdict": structural},
              "parity": {"verdict": exact, "required": policy["acceptance"] == "exact"},
              "round_coverage": {"verdict": round_verdict, "arms": source_coverage},
              "registration": {"verdict": "PASS" if preregistered else "UNPROVEN"}}
    if not preregistered:
        missing.append("policy registered before every measured round")
    rounds = len(a["data"]["rounds"])
    enough = len(fixture_rows) >= policy["min_fixtures"] and rounds >= policy["min_rounds"]
    if not enough:
        missing.append("registered minimum fixture/round count")
    compared = [p for p in pairs if "bit_exact" in p]
    snrs = [p["snr_db"] for p in compared if p["snr_db"] is not None]
    maxima = [p["maxabs"] for p in compared if p["maxabs"] is not None]
    metric_rows = [ev.metric("bit_exact_pct", 100 * sum(p["bit_exact"] for p in compared) / len(compared)
                             if compared else None, "%", "fraction", n=len(compared), subject=b["data"]["label"]),
                   ev.metric("same_sample_count_pct", 100 * sum(p["same_sample_count"] for p in compared) / len(compared)
                             if compared else None, "%", "fraction", n=len(compared)),
                   ev.metric("waveform_snr_db", min(snrs) if snrs else None, "dB", "min_finite", n=len(snrs)),
                   ev.metric("waveform_maxabs", max(maxima) if maxima else None, "full_scale", "max", n=len(maxima))]
    for name in policy["metrics"]:
        result = _metric_check(name, policy, observations, keys)
        checks[name] = result
        if result["missing"]:
            missing.append(f"complete {name} observations")
        if result["verdict"] == "UNPROVEN":
            missing.append(f"conclusive {name} evidence")
        metric_rows += [ev.metric(f"audio_{name}", result[arm], "fraction" if name == "wer" else "score", "pooled",
                                  n=result["n_prompts"], subject=arm) for arm in ("baseline", "candidate")]
        metric_rows.append(ev.metric(f"audio_{name}_delta", result["delta"], "fraction" if name == "wer" else "score",
                                    "candidate_minus_baseline", n=result["n_prompts"], ci95=result["ci95"],
                                    subject=b["data"]["label"]))
    required = [structural, checks["registration"]["verdict"], "PASS" if enough and not missing else "UNPROVEN"]
    if policy["acceptance"] == "exact":
        required.append(exact)
    required.extend(checks[name]["verdict"] for name in policy["metrics"])
    verdict = stats.combine_claims(required)
    scope = {"exact": "bit-exact preservation of the named baseline outputs",
             "task_noninferiority": "registered speech task/proxy noninferiority (predicted MOS, not human listening)",
             "perceptual_and_task_noninferiority": "registered speech task and blinded-listening noninferiority"}[policy["acceptance"]]
    limitations = ["Quality is relative to the named baseline; baseline intelligibility and suitability are not established.",
                   "Imported scorer outputs and listening blinding are producer attestations; hashes bind inputs and configuration, not methodological honesty.",
                   "SNR and maxabs use aligned raw PCM only; unequal lengths are not truncated or padded. Numerical parity is diagnostic outside exact acceptance.",
                   "Paired CIs resample fixtures, retaining all rounds; listening inference is conditional on the supplied listener panel.",
                   "Quality does not prove fresh-process lifecycle, timing, memory fit or sustained reliability."]
    if policy["acceptance"] == "task_noninferiority":
        limitations.append("WER, predicted MOS and speaker scores are proxies; human perceptual quality remains UNPROVEN.")
    data = {"format": qa.RESULT_FORMAT, "source": "quality audio", "label": b["data"]["label"],
            "baseline": a["evidence_id"], "candidate": b["evidence_id"], "protocol": registered["evidence_id"],
            "policy": policy, "fixture_sha256": fixture_sha, "preregistered": preregistered,
            "acceptance": policy["acceptance"], "scope": scope, "checks": checks, "pairs": pairs,
            "verdict": verdict, "reason": scope, "scorer_provenance": policy["scorers"]}
    with store.staging_dir() as staging:
        artifact_prefix = f"quality/audio/{b['data']['label']}/" + qa.sha(canonical_json(data).encode())
        relative = f"{artifact_prefix}/audio_score.json"
        path = _write(staging, relative, data)
        artifacts = [ev.artifact(path, staging, "audio_quality_result")]
        if imported is not None:
            imported_relative = f"{artifact_prefix}/imported_scores.json"
            imported_path = staging / imported_relative
            imported_path.write_bytes(imported)
            artifacts.append(ev.artifact(imported_path, staging, "audio_quality_observations"))
        if run_dir:
            _write(Path(run_dir), relative, data)
            if imported is not None:
                (Path(run_dir) / imported_relative).write_bytes(imported)
            _write(Path(run_dir), "quality/audio_score.json", data)
        record = ev.build(call, kind="quality_audio", claim_class="derived", summary=f"Speech quality {verdict}: {scope}",
                          identity_=b["identity"], metrics=metric_rows, artifacts=artifacts, missing=missing,
                          limitations=limitations, data=data)
        return store.save(record, staging)


def report_verdict(record):
    """Require the retained observations, not just a claimed verdict in data."""
    if record.get("data", {}).get("format") != qa.RESULT_FORMAT:
        return "NOT RUN", "speech quality protocol; measurements not run"
    try:
        check = ev.verify(record)
        if not check["id_ok"] or not check["schema_ok"]:
            raise OpError("invalid quality evidence")
        root = ev.Store().run_dir(record["evidence_id"])
        if not root.is_dir():
            copied_run = record["data"].get("request", {}).get("args", {}).get("run_dir")
            if copied_run:
                root = Path(copied_run).expanduser()
        for item in record["artifacts"]:
            qa.artifact_bytes(root, item)
        qa.one_artifact(record, "audio_quality_result")
        if record["data"]["policy"]["metrics"]:
            qa.one_artifact(record, "audio_quality_observations")
    except (OpError, KeyError, TypeError):
        return "UNPROVEN", "speech quality evidence or retained observations are missing/changed"
    verdict = record["data"].get("verdict")
    if verdict == "PASS" and (not record["coverage"]["complete"] or not record["data"].get("preregistered")):
        return "UNPROVEN", "speech quality coverage or preregistration is incomplete"
    return (verdict if verdict in ("PASS", "FAIL", "UNPROVEN", "NOT RUN") else "UNPROVEN"), record["data"]["scope"]
