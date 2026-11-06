"""evidence list / evidence show: query the store. Long lists are paginated (offset / limit)."""

from __future__ import annotations

from collections import Counter

from inferencekit import evidence as ev
from inferencekit.registry import Arg, OpError, operation

_FAILURES = {"FAIL", "failed", "error", "regressed", "port_busy", "launch_failed", "startup_timeout",
             "exited_early", "warmup_failed", "serve_failed", "serve_no_metrics", "stop_failed", "timeout"}
_UNPROVEN = {"UNPROVEN", "NOT RUN", "inconclusive"}
_SUCCESSES = {"PASS", "ok", "success", "improved", "within_noise"}
_COUNTED_VALUES = _FAILURES | _UNPROVEN | _SUCCESSES | {"RESOLVED", "SELF_REPORTED"}


def join_path(path: str, key: str | int) -> str:
    """Keep familiar dotted paths where unambiguous; use JSON Pointer for dotted or empty keys."""
    key = str(key)
    if not path:
        return "/" + key.replace("~", "~0").replace("/", "~1") if not key or "." in key else key
    if path.startswith("/"):
        return path + "/" + key.replace("~", "~0").replace("/", "~1")
    if not key or "." in key:
        return "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in [*path.split("."), key])
    return f"{path}.{key}"


def get_path(record: dict, path: str):
    """Follow a dotted path, or a JSON Pointer such as '/data/verdicts/engine.v1'."""
    node = record
    parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")] if path.startswith("/") \
        else [p for p in path.split(".") if p]
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.lstrip("-").isdigit() and -len(node) <= int(part) < len(node):
            node = node[int(part)]
        else:
            raise OpError(f"path '{path}' not found (stopped at '{part}')")
    return node


def page(items: list, offset: int, limit: int) -> dict:
    offset = max(0, offset)
    chunk = items[offset:offset + max(1, limit)]
    nxt = offset + len(chunk)
    return {"total": len(items), "offset": offset, "items": chunk, "next_offset": nxt if nxt < len(items) else None}


def _findings(node, path="data", verdict_map=False):
    """Stored decisions only: never infer a verdict from coverage or the absence of failures."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = join_path(path, key)
            is_verdict = key == "verdict" or key.endswith("_verdict") or verdict_map
            if (is_verdict and isinstance(value, str)) or \
                    (key == "status" and isinstance(value, str)) or \
                    (key == "valid" and isinstance(value, bool)):
                finding = {"path": here, "field": "verdict" if is_verdict else key, "value": value}
                # Keep decision qualifiers exact; detailed metrics and raw samples stay at the source path.
                finding.update({k: node[k] for k in ("reason", "applicable", "error", "detail", "violations", "warnings",
                                                     "limitations") if k in node})
                yield finding
            yield from _findings(value, here, verdict_map or key == "verdicts" or key.endswith("_verdicts")
                                 or key == "required_proofs")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _findings(value, join_path(path, i))


def _finding_order(finding):
    value = finding["value"]
    # Surface every explicit failure before successful checks, regardless of its position in a large list.
    if value is False or value in _FAILURES:
        return 0
    if value in _UNPROVEN:
        return 1
    if value is True or value in _SUCCESSES:
        return 3
    return 2


def summary_view(rec: dict, integrity: dict, ref: str, offset: int, limit: int) -> dict:
    """An opt-in inspection view, not a new evidence record or a new overall verdict.

    All decisions are scanned before pagination. Counts describe stored fields (including duplicate
    aggregate and individual verdicts), not independent checks or an inferred pass/fail decision.
    """
    data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
    findings = sorted(_findings(data), key=_finding_order)
    # Free-form status/verdict strings must not become unbounded JSON keys. Unknown values remain exact
    # in paged findings, and their combined count is explicit instead of being silently dropped.
    counts = Counter()
    for finding in findings:
        value = finding["value"]
        counted = value if isinstance(value, bool) or value in _COUNTED_VALUES else "<other>"
        counts[f"{finding['field']}:{counted}"] += 1
    index = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            index[key] = {"path": join_path("data", key), "type": "object" if isinstance(value, dict) else "array",
                          "count": len(value)}
        else:
            index[key] = value
    result = {"view": "summary", "source": ref, "integrity": integrity,
              **{k: rec.get(k) for k in ("evidence_id", "kind", "claim_class", "summary", "identity", "provenance",
                                        "metrics", "coverage", "limitations")},
              "artifact_count": len(rec["artifacts"]) if isinstance(rec.get("artifacts"), list) else None,
              "data_index": index,
              "finding_counts": dict(sorted(counts.items())), **page(findings, offset, limit),
              "note": "Stored decision fields, failures first; counts include aggregate and individual fields. "
                      "Unrecognized values are counted under <other> and retained in paged items. "
                      "No overall verdict is inferred. Inspect source paths for details or view='full' for the record."}
    accounting = data.get("request_accounting")
    if rec.get("kind") == "serve_benchmark" and isinstance(accounting, dict):
        keys = ("timed_requests", "successful_requests", "failed_requests", "selected_token_counts",
                "unknown_token_counts", "potentially_estimated_counts", "selected_output_tokens_total", "selected_output_tokens_min", "selected_output_tokens_max",
                "successful_count_mismatches")
        result["request_accounting"] = {k: accounting[k] for k in keys
                                        if k in accounting and (accounting[k] is None or type(accounting[k]) is int)}
    if result["next_offset"] is not None:
        result["more"] = (f"inspect_evidence(id={ref!r}, view='summary', offset={result['next_offset']}, "
                          f"limit={max(1, limit)})")
    return result


@operation("list_evidence", ("evidence", "list"), "List stored evidence records, newest first.",
           [Arg("kind", "string", "Only this kind.", choices=ev.KINDS),
            Arg("offset", "integer", "Skip this many records.", default=0),
            Arg("limit", "integer", "Page size.", default=20)])
def list_evidence(call, kind, offset, limit):
    rows = []
    for rec in ev.Store().records():
        if kind and rec.get("kind") != kind:
            continue
        rows.append({"evidence_id": rec.get("evidence_id"), "kind": rec.get("kind"),
                     "claim_class": rec.get("claim_class"), "started": rec.get("provenance", {}).get("started"),
                     "summary": rec.get("summary"), "complete": rec.get("coverage", {}).get("complete")})
    rows.sort(key=lambda r: r["started"] or "", reverse=True)
    return {"root": str(ev.Store().root), **page(rows, offset, limit)}


@operation("inspect_evidence", ("evidence", "show"), "Inspect evidence with an integrity check. Use view=summary "
           "for metrics, limitations and stored decisions; path=data.top_kernels for detail; default is full.",
           [Arg("id", "string", "Evidence id, a unique prefix (6+ hex chars), or a path to evidence.json.",
                required=True, positional=True),
            Arg("path", "string", "Dotted path (data.kernels), or JSON Pointer (/data/verdicts/engine.v1) for "
                "keys containing dots. Use paths returned by summary."),
            Arg("offset", "integer", "For list values or summary findings: first item.", default=0),
            Arg("limit", "integer", "For list values or summary findings: page size.", default=50),
            Arg("view", "string", "Full record (default), or summary with all decision counts and paged findings. "
                "Summary cannot be combined with path.", default="full", choices=("full", "summary"))])
def inspect_evidence(call, id, path, offset, limit, view="full"):  # noqa: A002
    if view == "summary" and path:
        raise OpError("view=summary cannot be combined with path; omit view to inspect the exact source path")
    rec = ev.Store().load(id)
    integrity = ev.verify(rec, include_error_count=True) if view == "summary" else ev.verify(rec)
    if view == "summary":
        return summary_view(rec, integrity, id, offset, limit)
    if not path:
        return {"integrity": integrity, "record": rec}
    value = get_path(rec, path)
    if isinstance(value, list):
        return {"evidence_id": rec.get("evidence_id"), "path": path, "integrity": integrity,
                **page(value, offset, limit)}
    return {"evidence_id": rec.get("evidence_id"), "path": path, "integrity": integrity, "value": value}
