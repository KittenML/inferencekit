"""Attach hash-checked STT timings, transcripts and policy verdicts to the review."""
from __future__ import annotations

import json

from inferencekit import evidence as ev
from inferencekit.quality.audio import artifact_bytes, one_artifact
from inferencekit.registry import OpError
from inferencekit.review.rundir import Cell, Item, artifact_lines, bucket_key, natural_key


def attach_benchmark(run, info, rec, source):
    for row in artifact_lines(run, rec, source, "transcripts", "STT transcripts") or []:
        if not isinstance(row.get("id"), str) or type(row.get("round")) is not int:
            continue
        info.stt_rows[f"r{row['round']}/{row['id']}"] = row
    for row in rec["data"].get("rounds", []):
        bucket = f"r{row['index']}"
        info.per_bucket.setdefault(bucket, {}).update({key: row[key] for key in
            ("stt_ms", "first_partial_p95_ms", "rtf", "audio_s_per_s") if row.get(key) is not None})
    for metric in rec.get("metrics", []):
        if metric["name"] in ("stt_ms", "first_partial_p95_ms", "rtf", "audio_s_per_s") and metric["value"] is not None:
            info.per_bucket.setdefault("all", {})[metric["name"]] = metric["value"]
    info.metric_classes.update({"rtf": "derived", "audio_s_per_s": "derived"})


def attach_quality(run, rec, problem):
    data = rec["data"]
    for role, label in data.get("arms", {}).items():
        info = run.engine(label)
        if info is not None:
            for metric, check in data.get("checks", {}).items():
                info.ev_quality.append({"name": metric.upper(), "unit": "fraction" if metric != "exact" else "",
                                        "value": None if problem else check.get(role),
                                        "verdict": "UNPROVEN" if problem else check.get("verdict", "UNPROVEN"),
                                        "evidence": rec["evidence_id"]})
            yield info
    if problem:
        return
    root = run.root / "artifacts" / rec["evidence_id"]
    if not root.is_dir():
        root = ev.Store().run_dir(rec["evidence_id"])
    try:
        raw = artifact_bytes(root, one_artifact(rec, "stt_quality"))
        rows = json.loads(raw)["transcripts"]
    except (OpError, ValueError, KeyError, TypeError):
        run.warnings.append(f"{rec['evidence_id']}: STT reference artifact missing or changed")
        return
    items = {run.key(item): item for item in run.items}
    for row in rows:
        if not isinstance(row.get("id"), str) or type(row.get("round")) is not int:
            continue
        bucket = f"r{row['round']}"
        key = f"{bucket}/{row['id']}"
        item = items.get(key)
        if item is None:
            item = items[key] = Item(id=row["id"], bucket=bucket)
            run.items.append(item)
            for info in run.engines:
                run.cells.setdefault((info.label, key), Cell(engine=info.label))
        item.ground_truth = row["ground_truth"]


def merge_transcripts(run):
    if not any(info.stt_rows for info in run.engines):
        return False
    items = {run.key(item): item for item in run.items}
    for info in run.engines:
        for key, row in info.stt_rows.items():
            if key not in items:
                bucket, fid = key.split("/", 1)
                items[key] = Item(id=fid, bucket=bucket)
            cell = run.cells.setdefault((info.label, key), Cell(engine=info.label))
            cell.text = row.get("transcript")
            complete = row.get("complete_ms")
            cell.metrics.update(first_partial_ms=row.get("first_partial_ms"), final_ms=complete,
                                rtf=complete * row["sample_rate"] / (1000 * row["frames"])
                                if complete is not None and row.get("frames") else None)
            cell.status = "ok" if row.get("ok") else "failed"
            cell.reason = "; ".join(row.get("errors", [])) if cell.status == "failed" else None
    run.items = sorted(items.values(), key=lambda item: (bucket_key(item.bucket), natural_key(item.id)))
    for item in run.items:
        for info in run.engines:
            run.cells.setdefault((info.label, run.key(item)), Cell(engine=info.label))
    return True
