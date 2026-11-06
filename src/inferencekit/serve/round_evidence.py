"""Attach scheduler-observed execution context without changing client timers or source records."""
from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.canonical import file_sha256
from inferencekit.registry import OpError

MAX_ROUNDS = 100_000
MAX_ROUND_INDEX = 100_000


def resource_mode(manifest: dict, requested: str = "auto") -> str:
    if requested not in ("auto", "cpu", "gpu"):
        raise OpError("device must be auto, cpu or gpu")
    hardware = manifest.get("hardware") or {}
    declared = hardware.get("backend")
    if declared == "cpu" and hardware.get("gpu") or declared in ("cuda", "rocm", "metal") and hardware.get("cpu"):
        raise OpError("manifest hardware has conflicting CPU and accelerator declarations")
    if requested == "auto":
        return "cpu" if declared == "cpu" else "gpu"
    if declared == "cpu" and requested == "gpu":
        raise OpError("--device gpu conflicts with the manifest's CPU hardware")
    if requested == "cpu" and declared in ("cuda", "rocm", "metal"):
        raise OpError("--device cpu conflicts with the manifest's accelerator hardware")
    return requested


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def memory_of(record: dict) -> dict:
    """Known memory measurements, with their measurement method retained for comparison."""
    data = record.get("data") or {}
    memory = data.get("memory") or {}
    result = {}
    for key in ("peak_rss_mib", "peak_vram_mib"):
        value = memory.get(key)
        if finite(value) and value >= 0:
            result[key] = {"value": value, "unit": "MiB", "method": memory.get("method") or memory.get("source"),
                           "complete": memory.get("complete"), "verdict": memory.get("verdict"),
                           "limitations": memory.get("limitations", [])}
    for metric in record.get("metrics") or []:
        key, value = metric.get("name"), metric.get("value")
        if key in ("peak_rss_mib", "peak_vram_mib") and key not in result and finite(value) and value >= 0:
            result[key] = {"value": value, "unit": "MiB", "method": memory.get("method") or "recorded metric",
                           "complete": memory.get("complete"), "verdict": memory.get("verdict"),
                           "limitations": memory.get("limitations", [])}
    return result


def round_coverage(record: dict, expected_rounds: int | None = None) -> dict:
    rows = record.get("data", {}).get("rounds") or []
    oversized_rows = isinstance(rows, list) and len(rows) > MAX_ROUNDS
    invalid_indices = not isinstance(rows, list)
    rows = rows[:MAX_ROUNDS] if isinstance(rows, list) else []
    indices = [r.get("index") for r in rows if isinstance(r, dict)]
    invalid_indices = invalid_indices or len(indices) != len(rows) or any(
        type(i) is not int or not 0 <= i <= MAX_ROUND_INDEX for i in indices)
    counts = Counter(i for i in indices if type(i) is int and 0 <= i <= MAX_ROUND_INDEX)
    actual = sorted(counts)
    plan = record.get("data", {}).get("round_plan") or {}
    requested = plan.get("expected_indices") if isinstance(plan, dict) else None
    invalid_plan = not isinstance(plan, dict)
    oversized_plan = isinstance(requested, list) and (len(requested) > MAX_ROUNDS or any(
        type(i) is int and i > MAX_ROUND_INDEX for i in requested[:MAX_ROUNDS]))
    if requested is not None and (not isinstance(requested, list) or oversized_plan or any(
            type(i) is not int or not 0 <= i <= MAX_ROUND_INDEX for i in requested)):
        invalid_plan = True
        requested = None
    if requested is not None and (not requested or len(set(requested)) != len(requested)):
        invalid_plan = True
        requested = None
    if expected_rounds is not None:
        if type(expected_rounds) is not int or not 1 <= expected_rounds <= MAX_ROUNDS:
            invalid_plan = True
            oversized_plan = type(expected_rounds) is int and expected_rounds > MAX_ROUNDS
            requested = None
        else:
            requested = list(range(expected_rounds))
    # Gaps are visible even when a manually chained run never recorded its intended size.
    gaps = sorted(set(range(min(actual), max(actual) + 1)) - set(actual)) if actual else []
    missing = sorted(set(requested or []) - set(actual))
    unexpected = sorted(set(actual) - set(requested)) if requested is not None else []
    duplicates = [i for i in actual if counts[i] > 1]
    failed = [r["index"] for r in rows if isinstance(r, dict) and r.get("n_failed")
              and type(r.get("index")) is int and 0 <= r["index"] <= MAX_ROUND_INDEX]
    invalid = invalid_indices or invalid_plan or oversized_rows
    return {"actual_indices": actual, "expected_indices": requested, "missing_indices": missing,
            "unexpected_indices": unexpected, "gaps": gaps, "duplicate_indices": duplicates, "failed_indices": failed,
            "invalid_indices": invalid_indices, "invalid_plan": invalid_plan, "oversized_plan": oversized_plan,
            "oversized_rows": oversized_rows,
            "complete": False if invalid else not (missing or unexpected or gaps or duplicates or failed)
            if requested is not None else None}


def finalize_arm_records(last_ev: dict, cells: list[dict], plan: dict, *, cpu: dict | None = None) -> dict:
    """Save immutable enriched copies after every cell has stopped; never alter the raw client record.

    These records are final scheduler results, not inputs for a later manual --extend.
    Their source_evidence remains available for inspecting the original client capture.
    """
    store, final = ev.Store(), {}
    expected = list(range(plan["rounds"]))
    for label, reference in last_ev.items():
        source = store.load(reference)
        if ev.integrity_problem(source):
            raise OpError(f"{label}: client evidence integrity failed")
        record = copy.deepcopy(source)
        data = record["data"]
        mine = [c for c in cells if c["label"] == label]
        contexts = [{"index": c["round"] - 1, "status": c.get("status"),
                     "position": c.get("position"), "ready": c.get("ready"), "stop": c.get("stop"),
                     "resources": c.get("resources"), "server_env": c.get("server_env"),
                     "t_start": c.get("t_start"), "t_end": c.get("t_end")} for c in mine]
        envs = [c.get("server_env") for c in mine if c.get("server_env") is not None]
        stable_env = len(envs) == len(mine) and bool(envs) and all(e == envs[0] for e in envs)
        data["source_evidence"] = source["evidence_id"]
        data["scheduler_final"] = True
        data["round_plan"] = {"expected_indices": expected, "requested_rounds": plan["rounds"],
                              "design": plan["design"]["design"], "balanced": plan["design"]["balanced"]}
        data["runtime_rounds"] = contexts
        cell_indices = [c["index"] for c in contexts]
        source_rows = data.get("rounds") or []
        windows = {c["index"]: (c.get("t_start"), c.get("t_end")) for c in contexts}
        timing_bound = len(source_rows) == len(expected) and all(
            type(row.get("index")) is int and row["index"] in windows and
            all(finite(v) for v in (*windows[row["index"]], row.get("t_start"), row.get("t_end"))) and
            windows[row["index"]][0] <= row["t_start"] < row["t_end"] <= windows[row["index"]][1]
            for row in source_rows)
        valid = (sorted(cell_indices) == expected and timing_bound and stable_env and
                 all(c.get("status") == "ok" and (c.get("stop") or {}).get("group_gone") is True
                     and (c.get("ready") or {}).get("status") == "ready" for c in mine))
        data["lifecycle"] = {"source": "bench rounds", "fresh_process_per_cell": True,
                             "valid": bool(valid), "verdict": "PASS" if valid else "UNPROVEN",
                             "scope": "server launch/readiness/process-group cleanup, not sustained reliability"}
        flags = record["identity"]["engine"].setdefault("flags", {})
        flags["server_env"] = envs[0] if stable_env else {"varied_between_rounds": envs}
        if cpu is not None:
            record["identity"]["cpu"] = cpu
            record["identity"]["gpu"] = ev.identity()["gpu"]
        by_index = {c["index"]: c for c in contexts}
        peaks = []
        for row in data.get("rounds") or []:
            context = by_index.get(row["index"], {})
            resource = context.get("resources") or {}
            peak = resource.get("peak_rss_mib")
            if finite(peak) and peak >= 0:
                row["peak_rss_mib"] = peak
                peaks.append(peak)
        if peaks:
            complete_memory = len(peaks) == len(expected) and all(
                (c.get("resources") or {}).get("status") == "observed" for c in mine)
            data["memory"] = {"peak_rss_mib": max(peaks), "method": "sampled_process_group_rss",
                               "verdict": "PASS" if complete_memory else "UNPROVEN",
                               "complete": complete_memory, "rounds_measured": len(peaks),
                               "limitations": ["Sampled RSS is a lower bound on the true peak and includes shared pages per process.",
                                                "This is host RAM, not VRAM, swap, or sustained leak testing."]}
            record["metrics"].append(ev.metric("peak_rss_mib", max(peaks), "MiB", "max", n=len(peaks)))
        record["limitations"] = [s for s in record.get("limitations", [])
                                  if not s.startswith("Rounds share one server process:")
                                  and not (peaks and s.startswith("Memory, swap and sustained"))]
        if peaks:
            record["limitations"].append("Host RSS was sampled; swap, device memory and sustained memory growth were not measured.")
            if data["memory"]["complete"]:
                record["coverage"]["missing"] = [x for x in record["coverage"].get("missing", [])
                                                  if x != "external memory telemetry"]
        record["limitations"].append("Scheduler execution context is recorded separately from unchanged client timing; CPU locks are advisory.")
        if not stable_env:
            record["limitations"].append("Server tuning environment was unavailable or changed between rounds.")
        coverage = round_coverage(record)
        data["round_coverage"] = coverage
        problems = []
        if coverage["missing_indices"]:
            problems.append(f"missing requested rounds: {coverage['missing_indices']}")
        if not valid:
            problems.append("complete fresh-process lifecycle")
        record["coverage"]["missing"] = sorted(set(record["coverage"].get("missing", []) + problems))
        record["coverage"]["complete"] = not record["coverage"]["missing"]
        old_root = store.run_dir(source["evidence_id"]).resolve()
        with store.staging_dir() as staging:
            for artifact in source.get("artifacts", []):
                relative = Path(artifact["path"])
                path = (old_root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(old_root):
                    raise OpError("client artifact escapes its evidence directory")
                if not path.is_file() or file_sha256(path)[0] != artifact["sha256"]:
                    raise OpError("client artifact missing or changed before scheduler finalization")
                dest = staging / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())
            proof = staging / "scheduler-context.json"
            proof.write_text(json.dumps(ev.redact({"plan": data["round_plan"], "cells": contexts}), indent=2) + "\n")
            record["artifacts"].append(ev.artifact(proof, staging, "scheduler_context"))
            saved = store.save(record, staging)
            final[label] = saved["evidence_id"]
    return final
