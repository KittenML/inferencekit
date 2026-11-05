"""evidence list / evidence show: query the store. Long lists are paginated (offset / limit)."""

from __future__ import annotations

from inferencekit import evidence as ev
from inferencekit.registry import Arg, OpError, operation


def get_path(record: dict, path: str):
    """Follow a dotted path such as 'data.top_kernels' or 'metrics.0'."""
    node = record
    for part in [p for p in path.split(".") if p]:
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


@operation("inspect_evidence", ("evidence", "show"), "Show one evidence record (or a page of a list inside it, "
           "e.g. path=data.top_kernels), with an integrity check.",
           [Arg("id", "string", "Evidence id, a unique prefix (6+ hex chars), or a path to evidence.json.",
                required=True, positional=True),
            Arg("path", "string", "Dotted path inside the record, e.g. data.kernels or metrics."),
            Arg("offset", "integer", "For list values: first item.", default=0),
            Arg("limit", "integer", "For list values: page size.", default=50)])
def inspect_evidence(call, id, path, offset, limit):  # noqa: A002
    rec = ev.Store().load(id)
    integrity = ev.verify(rec)
    if not path:
        return {"integrity": integrity, "record": rec}
    value = get_path(rec, path)
    if isinstance(value, list):
        return {"evidence_id": rec.get("evidence_id"), "path": path, "integrity": integrity,
                **page(value, offset, limit)}
    return {"evidence_id": rec.get("evidence_id"), "path": path, "integrity": integrity, "value": value}
