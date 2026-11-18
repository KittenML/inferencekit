"""Validate an open-loop speech capture and derive its capacity under the registered SLOs."""

from collections import defaultdict
import json
from pathlib import Path
import re

from inferencekit import evidence as ev, stats
from inferencekit.canonical import canonical_json, file_sha256, sha256_hex
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve.audio_client import playback_metrics
from inferencekit.serve.metrics import pct
from inferencekit.serve.round_evidence import finite

KIND = "audio_capacity"
PLAN_KEYS = {"levels", "rounds", "waves", "period_ms", "sample_rate", "mode",
             "fixture_ids", "chunks_per_request", "fixture_sha256"}
ROW_KEYS = {"round", "concurrency", "wave", "stream", "id", "complete_ms", "chunks"}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}")
SHA = re.compile(r"[0-9a-f]{64}")

ARGS = (
    Arg("timings", "string", "JSONL timing rows produced by the on-device client.", required=True, sensitive=True),
    Arg("plan", "string", "Preregistered load plan JSON.", required=True, sensitive=True),
    Arg("fixtures", "string", "Optional fixture file whose SHA-256 must match the plan.", sensitive=True),
    Arg("label", "string", "Runtime configuration label.", required=True),
    Arg("model", "string", "Pinned speech model identifier.", required=True),
    Arg("model_revision", "string", "Checkpoint revision.", required=True),
    Arg("engine_version", "string", "Runtime build or revision.", required=True),
    Arg("hardware_name", "string", "Measured device model, SoC and OS.", required=True),
    Arg("model_dtype", "string", "Weights and compute precision."),
    Arg("slo_first_audio_p95_ms", "number", "Simulated first nonzero playback onset p95 per stream.", default=400.0),
    Arg("slo_rtf_p95", "number", "Complete-response real-time factor p95 per stream.", default=1.0),
)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def read_plan(path):
    try:
        plan = json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError):
        raise OpError("audio capacity plan is not readable JSON") from None
    if not isinstance(plan, dict) or set(plan) - PLAN_KEYS or PLAN_KEYS - {"fixture_sha256"} - set(plan):
        raise OpError("invalid audio capacity plan fields")
    levels = plan["levels"]
    if (not isinstance(levels, list) or not levels or len(levels) > 256
            or any(type(level) is not int for level in levels) or levels != list(range(1, len(levels) + 1))):
        raise OpError("audio capacity levels must be consecutive integers starting at 1")
    if (not _integer(plan["waves"], 1, 10000) or not _integer(plan["rounds"], 1, 1000)
            or not _integer(plan["sample_rate"], 8000, 192000) or not _integer(plan["period_ms"], 1, 3600000)
            or not _integer(plan["chunks_per_request"], 1, 1000) or plan["mode"] not in ("stream", "full")
            or (plan["mode"] == "full" and plan["chunks_per_request"] != 1)
            or len(levels) * (len(levels) + 1) // 2 * plan["waves"] * plan["rounds"] > 1000000):
        raise OpError("invalid audio capacity plan bounds or delivery mode")
    ids = plan["fixture_ids"]
    if (not isinstance(ids, list) or len(ids) != plan["waves"] * levels[-1]
            or any(not isinstance(x, str) or not ID.fullmatch(x) for x in ids) or len(set(ids)) != len(ids) or ev.redact(ids) != ids):
        raise OpError("one distinct fixture ID is required per wave and stream at the largest level")
    fixture_sha = plan.get("fixture_sha256")
    if fixture_sha is not None and (not isinstance(fixture_sha, str) or not SHA.fullmatch(fixture_sha)):
        raise OpError("fixture_sha256 must be a lowercase SHA-256 digest")
    if ev.redact(plan) != plan:
        raise OpError("audio capacity plan fields must be safe to retain")
    return plan


def request_metrics(row, plan):
    complete, chunks = row["complete_ms"], row["chunks"]
    if not isinstance(chunks, list):
        raise ValueError
    if (complete is not None and (not finite(complete) or complete <= 0)
            or len(chunks) > plan["chunks_per_request"]):
        raise ValueError
    latest = complete if complete is not None else float("inf")
    rate = plan["sample_rate"]
    previous, end, onset, frames = 0.0, 0.0, None, 0
    playback = []
    for value in chunks:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError
        arrival, count, silent = value
        if (not finite(arrival) or not previous <= arrival <= latest
                or not _integer(count, 1, rate * 3600) or not _integer(silent, 0, count)):
            raise ValueError
        start = max(end, arrival)
        if onset is None and silent < count:
            onset = start + silent * 1000 / rate
        end = start + count * 1000 / rate
        frames += count
        previous = arrival
        playback.append((round(arrival * 1e6), count))
    if complete is None or len(chunks) != plan["chunks_per_request"] or onset is None:
        return None
    seconds = frames / rate
    return {"first_pcm_ms": chunks[0][0], "first_audio_ms": onset, "rtf": complete / 1000 / seconds,
            "audio_s": seconds, "complete_ms": complete, **playback_metrics(playback, rate)}


def read_timings(path, plan):
    cells = defaultdict(dict)
    try:
        with Path(path).open(encoding="utf8") as handle:
            for index, line in enumerate(handle, 1):
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != ROW_KEYS:
                    raise ValueError
                level, round_index, wave, stream = (row[key] for key in ("concurrency", "round", "wave", "stream"))
                if (not _integer(level, 1, len(plan["levels"])) or not _integer(round_index, 0, plan["rounds"] - 1)
                        or not _integer(wave, 0, plan["waves"] - 1) or not _integer(stream, 0, level - 1)
                        or row["id"] != plan["fixture_ids"][wave * level + stream]):
                    raise ValueError
                cell = cells[level, round_index]
                if (wave, stream) in cell:
                    raise ValueError
                cell[wave, stream] = (row, request_metrics(row, plan))
    except (OSError, ValueError, OverflowError):
        raise OpError(f"invalid or unreadable audio timing row {locals().get('index', 0)}") from None
    return cells


def summarize_cell(cell, level, plan, first_audio_ms, rtf):
    count = plan["waves"] * level
    valid = [(wave, stream, stats) for (wave, stream), (_, stats) in sorted(cell.items()) if stats is not None]
    per_stream = [[] for _ in range(level)]
    for _, stream, stats in valid:
        per_stream[stream].append(stats)
    streams = [{"stream": i, "requests": len(rows),
                "first_pcm_p95_ms": pct([r["first_pcm_ms"] for r in rows], 95),
                "first_audio_p95_ms": pct([r["first_audio_ms"] for r in rows], 95),
                "rtf_p95": pct([r["rtf"] for r in rows], 95),
                "underruns": sum(r["underruns"] for r in rows)} for i, rows in enumerate(per_stream)]
    und = sum(r["underruns"] for r in streams)
    period = plan["period_ms"]
    overruns = sum(r["complete_ms"] > period for _, _, r in valid)
    failed = len(cell) - len(valid)
    slo_missed = any(r["requests"] and (r["first_audio_p95_ms"] > first_audio_ms or r["rtf_p95"] > rtf)
                     for r in streams)
    verdict = "UNPROVEN" if len(cell) < count else ("FAIL" if failed or und or overruns or slo_missed else "PASS")
    finish = max((wave * period + r["complete_ms"] for wave, _, r in valid), default=0)
    observation_ms = max(plan["waves"] * period, finish)
    return {"requests": len(cell), "expected_requests": count, "failed_requests": failed, "underruns": und,
            "wave_overruns": overruns, "streams": streams, "verdict": verdict,
            "audio_s_per_s": 1000 * sum(r["audio_s"] for _, _, r in valid) / observation_ms if valid else None}


@operation("import_audio_capacity", ("import", "audio-capacity"),
           "Derive speech capacity and audio throughput from a client-timed open-loop capture.", ARGS)
def import_audio_capacity(call, timings, plan, fixtures, label, model, model_revision, engine_version, hardware_name,
                          model_dtype, slo_first_audio_p95_ms, slo_rtf_p95):
    slo = {"first_audio_p95_ms": slo_first_audio_p95_ms, "rtf_p95": slo_rtf_p95, "underruns": 0}
    if any(not finite(v) or v <= 0 for v in (slo_first_audio_p95_ms, slo_rtf_p95)):
        raise OpError("audio capacity SLOs must be finite and positive")
    for value in (label, model, model_revision, engine_version, hardware_name, model_dtype):
        if value is not None and (not value.strip() or ev.redact(value) != value):
            raise OpError("audio capacity requires nonempty, public identity labels")
    plan_doc = read_plan(plan)
    if fixtures:
        try:
            fixture_digest, _ = file_sha256(Path(fixtures))
        except OSError:
            raise OpError("audio capacity fixture file is not readable") from None
        if fixture_digest != plan_doc.get("fixture_sha256"):
            raise OpError("audio capacity fixture file differs from the registered plan")
    cells = read_timings(timings, plan_doc)
    levels, highest, metrics = [], None, []
    for level in plan_doc["levels"]:
        rounds = [summarize_cell(cells[level, index], level, plan_doc, slo_first_audio_p95_ms, slo_rtf_p95)
                  for index in range(plan_doc["rounds"])]
        statuses = {row["verdict"] for row in rounds}
        verdict = "UNPROVEN" if "UNPROVEN" in statuses else ("FAIL" if "FAIL" in statuses else "PASS")
        if verdict == "PASS" and (level == 1 or highest == level - 1):
            highest = level
        levels.append({"concurrency": level, "verdict": verdict, "rounds": rounds})
        throughput = [row["audio_s_per_s"] for row in rounds if row["verdict"] == "PASS"]
        if verdict == "PASS":
            ci = stats.bootstrap_ci(throughput, ps=(50,))[50] if len(throughput) >= 3 else None
            metrics.append(ev.metric("audio_s_per_s", pct(throughput, 50), "audio s/s", "median",
                                     n=len(throughput), ci95=ci, subject=f"concurrency {level}"))
    boundary = next((row["verdict"] for row in levels if row["concurrency"] == (highest or 0) + 1), None)
    confirmed = plan_doc["rounds"] >= 3 and boundary == "FAIL" and bool(fixtures)
    verdict = ("PASS" if highest else "FAIL") if confirmed else "UNPROVEN"
    missing = ["audio quality reference proof", "independent timer and runtime configuration verification"]
    if plan_doc["rounds"] < 3:
        missing.append("at least three fresh-process rounds per level")
    if not fixtures:
        missing.append("fixture file verified against its registered digest")
    if boundary != "FAIL":
        missing.append("complete failing level immediately above the passing prefix")
    if any(row["verdict"] == "UNPROVEN" for row in levels):
        missing.append("complete timing rows for all planned cells")
    metrics.append(ev.metric("highest_passing_tested", highest, "streams", "max", n=len(levels)))
    metrics.append(ev.metric("n_star", (highest or 0) if confirmed else None, "streams", "capacity", n=len(levels)))
    store = ev.Store()
    with store.staging_dir() as staging:
        timing_path = staging / "timings.jsonl"
        timing_path.write_text("".join(json.dumps(row) + "\n" for _, cell in sorted(cells.items())
                                       for _, (row, _) in sorted(cell.items())), encoding="utf8")
        plan_path = staging / "plan.json"
        plan_path.write_text(json.dumps(plan_doc, indent=2) + "\n", encoding="utf8")
        record = ev.build(
            call, kind=KIND, claim_class="derived",
            summary=f"{label}: highest passing tested speech concurrency {highest}; capacity {verdict}",
            identity_=ev.identity(device={"class": "mobile", "name": hardware_name},
                                  engine={"name": label, "version": engine_version},
                                  model={"id": model, "revision": model_revision, "dtype": model_dtype},
                                  workload={"mode": "load", "concurrency": plan_doc["levels"],
                                            "request_set": {"sha256": sha256_hex(canonical_json(plan_doc)), "n": len(plan_doc["fixture_ids"])},
                                            "task": "tts", "endpoint_kind": "on_device_audio"}),
            metrics=metrics, missing=missing,
            artifacts=[ev.artifact(timing_path, staging, "requests"), ev.artifact(plan_path, staging, "load_plan")],
            limitations=["Native on-device client times and process isolation are supplied by the capture producer.",
                         "Playback uses a PCM buffer simulation with immediate start and no extra prebuffer.",
                         "Burst arrivals represent this registered workload only.",
                         "Audio throughput is decoded seconds divided by the full scheduled window plus any drain past it."],
            data={"source": "import audio-capacity", "hardware_name": hardware_name, "timer": "imported on-device client",
                  "delivery_mode": plan_doc["mode"], "levels": levels, "slo": slo, "verdict": verdict,
                  "highest_passing_tested": highest, "n_star": (highest or 0) if confirmed else None,
                  "quality_verdict": "UNPROVEN"},
        )
        return store.save(record, staging)
