"""Read-only run resumption: pinned intake, state, budget and cited evidence in one call."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
import re

from inferencekit import evidence as ev
from inferencekit.canonical import canonical_json
from inferencekit.ops import intake
from inferencekit.ops.store import page
from inferencekit.registry import Arg, OpError, operation


_BUDGET_BASIS = ("Calculated from mutable checkpoint run_started, meter and stage timing; history is not "
                 "authenticated. Remaining allowance and the calculated action are not verified authorization.")


def _object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise OpError(f"cannot read {path}: {exc}") from None
    if not isinstance(value, dict):
        raise OpError(f"{path}: expected a JSON object")
    return value


def _citations(value, location=""):
    """Find every evidence list, including candidates and best_history; do not infer proof from it."""
    if isinstance(value, dict):
        for key, child in value.items():
            at = f"{location}.{key}" if location else key
            if key == "evidence":
                if not isinstance(child, list):
                    raise OpError(f"state.json: {at} must be a list of full evidence ids")
                for i, ref in enumerate(child):
                    if not isinstance(ref, str) or not re.fullmatch(r"ev_[0-9a-f]{64}", ref):
                        raise OpError(f"state.json: {at}.{i} must be a full evidence id")
                    yield ref, at
            else:
                yield from _citations(child, at)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _citations(child, f"{location}.{i}")


def _changed(actual, expected, location):
    """Report edited policy paths, never their potentially sensitive values."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(actual.keys() | expected.keys()):
            if key not in actual or key not in expected:
                yield f"{location}.{key}"
            else:
                yield from _changed(actual[key], expected[key], f"{location}.{key}")
    elif canonical_json(actual) != canonical_json(expected):
        yield location


def _saved_intake_policy(validated, saved):
    """Reproduce saved home redaction without weakening the exact policy comparison."""
    raw = intake.redact(validated, home="")
    prefixes = set()

    def find_prefixes(original, stored):
        if isinstance(original, dict) and isinstance(stored, dict):
            for key in original.keys() & stored.keys():
                find_prefixes(original[key], stored[key])
        elif isinstance(original, list) and isinstance(stored, list):
            for left, right in zip(original, stored):
                find_prefixes(left, right)
        elif (isinstance(original, str) and isinstance(stored, str) and stored.startswith("~/")
              and original.startswith("/") and original.endswith(stored[1:])):
            prefix = original[:-len(stored[1:])]
            if prefix:
                prefixes.add(prefix)

    find_prefixes(raw, saved)
    if len(prefixes) == 1:
        return intake.redact(validated, home=prefixes.pop())
    # No prior home substitution, or conflicting prefixes: the full comparison below
    # accepts unchanged absolute paths and rejects incompatible saved redactions.
    return raw


def _check_policy(state, normalized, validated, original):
    # run_id is assigned once when absent from the original intake; validating again must not invent a new
    # timestamp or rename the run when its directory moves. All explicitly pinned intake fields stay fixed.
    if not original.get("run_id"):
        if not isinstance(state.get("run_id"), str) or not state["run_id"]:
            raise OpError("state.json: run_id must be nonempty text")
        validated["intake"]["run_id"] = state["run_id"]
        validated["derived"] = intake.derive(validated["intake"])
    expected = {"intake": _saved_intake_policy(validated["intake"], normalized.get("intake")),
                "derived": validated["derived"],
                "ok": True, "placeholders": []}
    # Older normalizers added LLM defaults to media intakes. Accept only those exact
    # implied values; explicit source fields and every other saved policy pin stay binding.
    saved_intake = normalized.get("intake")
    saved_workload = saved_intake.get("workload") if isinstance(saved_intake, dict) else None
    if validated["intake"]["modality"] in ("tts", "stt", "image", "video") and isinstance(saved_workload, dict):
        source_workload = original.get("workload") or {}
        for key, default in (("max_tokens", 256), ("prompt_set", "dev")):
            if (key not in source_workload and key not in expected["intake"]["workload"]
                    and saved_workload.get(key) == default):
                expected["intake"]["workload"][key] = default
    changed = []
    for key, value in expected.items():
        changed.extend(_changed(normalized.get(key), value, f"intake.normalized.json.{key}"))
    initial = intake.initial_state(validated, validated["intake_sha256"])
    budget = state.get("budget")
    if not isinstance(budget, dict):
        raise OpError("state.json: budget must be an object")
    # Meter entries, stage progress and reassigned stage time boxes are mutable run history. Spending caps
    # and policy overrides are not: changing those requires a new pinned intake, not a state-only edit.
    for key, value in initial["budget"].items():
        if key not in ("meter", "run_started"):
            changed.extend(_changed(budget.get(key), value, f"state.json.budget.{key}"))
    for key in ("run_id", "workload_pin"):
        changed.extend(_changed(state.get(key), initial.get(key), f"state.json.{key}"))
    model = state.get("model")
    if not isinstance(model, dict):
        raise OpError("state.json: model must be an object")
    for key in ("id", "revision"):
        changed.extend(_changed(model.get(key), initial["model"].get(key), f"state.json.model.{key}"))
    revision = initial["model"].get("revision")
    resolved = model.get("revision_resolved")
    # A branch/tag may legitimately acquire a resolved commit during stage 1. An immutable commit pin
    # cannot resolve to another commit, even though revision_resolved is otherwise dynamic checkpoint data.
    if (isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision) and resolved is not None
            and (not isinstance(resolved, str) or resolved.lower() != revision.lower())):
        changed.append("state.json.model.revision_resolved")
    if changed:
        raise OpError("pinned run policy changed: " + ", ".join(changed) +
                      ". Restore the saved policy or start a new run from a changed intake; nothing was written.")


def _evidence(root: Path, ref: str, locations: list[str]) -> dict:
    # Both normal Store output and the run-directory copy made by benchmark/quality operations are valid
    # sources. Do not consult the global store: a resume must work from this run's synced artifacts alone.
    path = root / ref / "evidence.json"
    if not path.exists():
        path = root / f"{ref}.json"
    row = {"evidence_id": ref, "cited_by": locations}
    if not path.is_file():
        return {**row, "status": "missing", "reason": "no record in this run's evidence directory"}
    row["path"] = str(path)
    try:
        rec = _object(path)
        integrity = ev.verify(rec)
    except (OpError, TypeError, ValueError, OverflowError) as exc:
        return {**row, "status": "invalid", "reason": str(exc)}
    integrity["cited_id_ok"] = rec.get("evidence_id") == ref
    row["integrity"] = integrity
    if not all(integrity[key] for key in ("id_ok", "schema_ok", "cited_id_ok")):
        return {**row, "status": "invalid", "reason": "evidence id, content hash or schema check failed"}
    return {**row, "status": "available", "kind": rec["kind"], "summary": rec["summary"],
            "coverage": rec["coverage"], "limitations": rec["limitations"]}


def _audit_evidence(state, root):
    refs: dict[str, list[str]] = {}
    for ref, location in _citations(state):
        locations = refs.setdefault(ref, [])
        if location not in locations:
            locations.append(location)
    rows = [_evidence(root / "evidence", ref, locations) for ref, locations in refs.items()]
    rows.sort(key=lambda row: row["status"] == "available")
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("available", "missing", "invalid")}
    return rows, {"checked": len(rows), **counts, "order": "missing/invalid first"}


def _blocked(reason, root, source, state, normalized, validated, query, offset, limit):
    """Keep exit 1 / MCP is_error while avoiding a second manual diagnostic workflow.

    The original pin failed: no saved policy, budget or next action is authorized by this response. Evidence
    content checks are independent of that failed policy check. Return counts and bounded issue pages only.
    """
    def brief(value, length=240):
        if isinstance(value, str):
            return value if len(value) <= length else {"text": value[:length], "truncated": True}
        if isinstance(value, float) and not math.isfinite(value):
            return "see source (non-finite value)"
        return value if value is None or isinstance(value, (bool, int, float)) else "see source (non-scalar)"

    def snapshot(data):
        if not isinstance(data, dict):
            return {"error": "malformed saved intake; see source"}
        budget = data.get("budget") if isinstance(data.get("budget"), dict) else {}
        return {"budget": {key: brief(budget[key]) for key in
                           ("approved", "wall_clock_h", "gpu_hours", "usd", "spend_cap", "deadline") if key in budget},
                "fidelity": brief(data.get("fidelity"))}

    result = {
        "status": "blocked", "reason": brief(reason, 2000),
        "action": "Do not resume. Restore the pinned files or start a new run for policy changes.",
        "trust": {"saved_policy_and_budget": "UNPROVEN: not validated against the original intake",
                  "authorization": "not evaluated; saved approval fields are diagnostic only",
                  "evidence": "content hash and schema checks only; not performance or quality proof"},
        "sources": {"original_intake": str(source), "state": str(root / "state.json"),
                    "normalized_intake": str(root / "intake.normalized.json")},
        "intake_hashes": {"supplied": (validated or {}).get("intake_sha256"),
                          "state": brief(state.get("intake_sha256")),
                          "normalized": brief(normalized.get("intake_sha256"))},
        "saved_state": {"current_stage": state["current_stage"],
                        "status": state["stages"][state["current_stage"]]["status"],
                        "next_action": brief(state["next_action"], 1600)},
        "saved_policy_snapshot_unvalidated": snapshot(normalized.get("intake")),
    }
    if validated is not None:
        result["supplied_intake_snapshot_not_pinned"] = snapshot(validated["intake"])
    try:
        timestamp = intake._parse_time(query["now"]) if query.get("now") else dt.datetime.now(dt.timezone.utc)
        saved_budget = intake.budget_status(state, timestamp)
        saved_budget["basis"] = _BUDGET_BASIS
        # A numeric-looking NaN is not a meaningful saved budget, even for diagnostics.
        json.dumps(saved_budget, allow_nan=False)
        result["saved_budget_calculation_unvalidated"] = saved_budget
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["saved_budget_calculation_unvalidated"] = {"error": brief(str(exc))}
    try:
        rows, counts = _audit_evidence(state, root)
        problems = [{key: row[key] for key in ("evidence_id", "status", "reason") if key in row}
                    for row in rows if row["status"] != "available"]
        for row in problems:
            row["reason"] = brief(row.get("reason"), 500)
        def notes(field):
            values = sorted({str(note) for row in rows for note in
                             ((row.get("coverage") or {}).get("missing", []) if field == "missing"
                              else row.get("limitations", []))})
            return {"total": len(values), "items": [brief(value) for value in values[:10]],
                    "truncated": len(values) > 10}
        result["evidence_summary"] = {**counts,
            "incomplete_records": sum((row.get("coverage") or {}).get("complete") is False for row in rows),
            "coverage_missing": notes("missing"), "limitations": notes("limitations")}
        result["evidence_issue_page"] = page(problems, offset, min(limit, 10))
        if result["evidence_issue_page"]["next_offset"] is not None:
            result["next_call"] = {"tool": "intake_resume", "arguments": {
                **query, "offset": result["evidence_issue_page"]["next_offset"], "limit": min(limit, 10)}}
    except (OpError, TypeError, ValueError) as exc:
        result["evidence_summary"] = {"error": brief(str(exc), 800), "complete_check": False}
    result = ev.redact(result)
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(text) > 24000:  # Unusually long paths or malformed metadata: retain the hard rejection and pointers.
        result = {key: result[key] for key in ("status", "reason", "action", "trust", "sources")}
        result["diagnostics_omitted"] = "Inspect the named files; diagnostic output exceeded 24000 characters."
        text = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    raise OpError(text)


@operation("intake_resume", ("intake", "resume"),
           "Resume context in one read-only call: validate the pinned original intake, show state and budget, "
           "and check every state-cited evidence record's hash and schema. Checkpoint budget history is mutable; "
           "does not establish authorization or proof verdicts.",
           [Arg("run_dir", "string", "Run directory containing state.json and intake.normalized.json.",
                required=True, positional=True),
            Arg("file", "string", "Original intake path; overrides state.intake_source if its relative path "
                "no longer resolves. The pinned hash must still match."),
            Arg("now", "string", "Evaluate the budget at this ISO-8601 UTC time instead of the current time."),
            Arg("offset", "integer", "First evidence item to return; missing/invalid records sort first.", default=0),
            Arg("limit", "integer", "Evidence page size. Every cited record is checked on every call.", default=20)])
def intake_resume(call, run_dir, file, now, offset, limit):
    if offset < 0 or limit < 1:
        raise OpError("intake resume: offset must be nonnegative and limit must be positive")
    query = {"run_dir": run_dir, **({"file": file} if file is not None else {}),
             **({"now": now} if now is not None else {})}
    root = Path(run_dir).expanduser().resolve()
    state = _object(root / "state.json")
    normalized = _object(root / "intake.normalized.json")
    if state.get("state_version") != 1:
        raise OpError(f"{root / 'state.json'}: unsupported state_version {state.get('state_version')!r}")
    stages = state.get("stages")
    statuses = {"pending", "in_progress", "done", "skipped", "over_box"}
    if (not isinstance(stages, dict) or not stages or
            any(not isinstance(s, dict) or s.get("status") not in statuses for s in stages.values())):
        raise OpError(f"{root / 'state.json'}: malformed stages or stage status")
    if (not isinstance(state.get("current_stage"), str) or state["current_stage"] not in stages
            or not isinstance(state.get("next_action"), str)):
        raise OpError(f"{root / 'state.json'}: current_stage must name a stage and next_action must be text")
    source = file or state.get("intake_source")
    if not isinstance(source, str) or not source:
        raise OpError("state.json has no intake_source; pass --file <original intake>")
    source = Path(source).expanduser()
    if not source.is_file():
        raise OpError(f"original intake not found: {source}; pass --file <original intake> (the pinned hash "
                      "must still match)")
    # The existing validator rejects redacted copies and invalid policy. With no run_dir it writes nothing.
    try:
        validated = intake.intake_validate(call, str(source), run_dir=None, allow_placeholders=False)
    except OpError as exc:
        _blocked(str(exc), root, source, state, normalized, None, query, offset, limit)
    if validated["placeholders"]:
        raise OpError("cannot resume with placeholder values: " + ", ".join(validated["placeholders"]))
    sha = validated["intake_sha256"]
    changed = [name for name, data in (("state.json", state), ("intake.normalized.json", normalized))
               if data.get("intake_sha256") != sha]
    if changed:
        _blocked(f"the intake changed since this run started: {', '.join(changed)} does not match the "
                 "original intake. Targets are pre-registered; start a new run id. Nothing was written.",
                 root, source, state, normalized, validated, query, offset, limit)
    original, _ = intake.load_file(source)
    try:
        _check_policy(state, normalized, validated, original)
    except OpError as exc:
        _blocked(str(exc), root, source, state, normalized, validated, query, offset, limit)
    except (TypeError, ValueError) as exc:
        _blocked(f"malformed pinned run policy ({exc})", root, source, state, normalized,
                 validated, query, offset, limit)
    try:
        timestamp = intake._parse_time(now) if now else dt.datetime.now(dt.timezone.utc)
        budget = intake.budget_status(state, timestamp)
        budget["basis"] = _BUDGET_BASIS
        json.dumps(budget, allow_nan=False)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise OpError(f"{root / 'state.json'}: malformed budget/stage fields ({exc})") from None
    evidence, counts = _audit_evidence(state, root)
    issues = counts["missing"] + counts["invalid"]
    pinned = intake.redact(validated["intake"])
    result = {
        "run_dir": str(root), "run_id": state.get("run_id"),
        "status": "needs_attention" if issues or budget["verdict"] != "ok" else "context_ready",
        "scope": "resume context and evidence integrity only; not a benchmark or quality proof",
        "trust": {"checkpoint_history_authenticated": False, "remaining_authorization_verified": False,
                  "authorization": "not evaluated; context_ready and budget.action do not authorize work"},
        "current_stage": state["current_stage"], "next_action": ev.redact(state["next_action"]),
        "stages": ev.redact({key: {field: value for field, value in stage.items() if field in ("status", "notes")}
                             for key, stage in stages.items()}),
        "best": ev.redact(state.get("best")), "budget": budget,
        "policy": ev.redact({"source": "validated intake only; session authorization is not evaluated",
                             "budget_approved": pinned["budget"]["approved"], "fidelity": pinned["fidelity"],
                             "quality": pinned["quality"],
                             "architecture_changes": validated["derived"]["architecture_changes"]}),
        "intake": {"source": str(source.resolve()), "sha256": sha, "warnings": ev.redact(validated["warnings"])},
        "evidence_root": str(root / "evidence"),
        "evidence_summary": counts,
        **page(ev.redact(evidence), offset, limit),
        "pagination": {"tool": "intake_resume", "args": query},
        "state_path": str(root / "state.json"),
    }
    if result["next_offset"] is not None:
        result["next_call"] = {"tool": "intake_resume",
                               "arguments": {**query, "offset": result["next_offset"], "limit": limit}}
    return result
