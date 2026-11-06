"""``inferencekit mcp serve``: every registered operation as an MCP tool, over stdio.

Built on the low-level server of the official ``mcp`` Python SDK (v2), because the
tool list and input schemas already exist in the registry; no per-tool code here.

Tool results are trimmed to stay well under ~10k tokens: big lists inside a record
are cut to their first items and replaced by a pointer that says how to page through
the rest with the ``inspect_evidence`` tool (``path``, ``offset``, ``limit``). A result
that is already a page (``list_evidence``, or ``inspect_evidence`` with a list path)
is shortened to fewer items instead, with ``next_offset`` moved back to the first item
that was cut, so following it skips nothing. The stored record is never modified; only
the view sent to the model is shortened.
"""

from __future__ import annotations

import copy
import json
from functools import partial

from inferencekit import __version__, registry
from inferencekit.registry import OpError

# ~4 characters per token for JSON: 32k characters is about 8k tokens.
MAX_CHARS = 32_000
LIST_PREVIEW = 10
TOOLS_PAGE = 50

INSTRUCTIONS = (
    "inferencekit measures inference performance and writes evidence records. Start with doctor; "
    "use roofline (with gpu_bw from measure_bandwidth) for the ceiling. For an optimisation run: "
    "intake_template / intake_validate (start the run dir), validate_manifest (the run manifest), "
    "bench_rounds (every manifest engine, interleaved rounds) or bench_serve (one OpenAI-compatible server: "
    "single-user TTFT/TPOT/decode tok/s, or load mode for N* and $/1M tokens) for published speed numbers; "
    "bench_decode only for engines without a server. run_gates for the anti-cheat checks; quality_trajectory "
    "+ quality_score (agreement with a reference), quality_gsm8k / quality_niah for task accuracy; "
    "prompts_show / prompts_verify for the prompt sets. analyze_nsys / summarize_ncu to explain speed; "
    "bench_image / bench_audio for client-timed media; compare with image_tradeoff=true only for a "
    "labelled image step/precision tradeoff, with separate quality proof. AMD: capture_rocprof / "
    "analyze_rocprof for supported ROCm kernel traces. "
    "compare for A/B verdicts; report to write it up; "
    "build_review for the run's review page; ledger_trailers / history for Perf-* commit trailers. "
    "Pass chat_template_kwargs (e.g. {\"enable_thinking\": false}) identically to bench_serve, bench_rounds, "
    "run_gates, quality_trajectory, quality_gsm8k and quality_niah (an intake pin in the run dir is applied and "
    "enforced when run_dir is given). Start evidence inspection with inspect_evidence(id, view='summary') "
    "for integrity, metrics, limitations and stored decisions (failures first); follow path for details. "
    "Long lists in results are truncated: page them with "
    "inspect_evidence(id, path, offset, limit)."
)


def _size(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False, default=str))


def _lists(node, path=()):
    """Yield (path, parent, key, list) for every list inside node, largest first by caller."""
    if isinstance(node, dict):
        for k, v in node.items():
            p = (*path, k)
            if isinstance(v, list):
                yield p, node, k, v
            yield from _lists(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _lists(v, (*path, i))


def _trim_strings(node, limit: int) -> None:
    """Cut every string longer than ``limit`` characters in place, saying how much was cut."""
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else []
    for k, v in list(items):
        if isinstance(v, str) and len(v) > limit:
            node[k] = v[:limit] + f"\n... [truncated {len(v) - limit} characters]"
        elif isinstance(v, (dict, list)):
            _trim_strings(v, limit)


def _is_page(view: dict) -> bool:
    """A paged result from store.page(): {..., total, offset, items, next_offset}."""
    return isinstance(view.get("items"), list) and "offset" in view and "next_offset" in view


def _fit_page(view: dict, max_chars: int) -> dict:
    """Keep the first k items of a page that fit, and point next_offset / the hint at item offset+k."""
    items = view["items"]
    k = len(items)
    # Continuation hints also consume the response budget. Recheck the complete page
    # after each reduction so generic trimming cannot hide rows behind next_offset.
    while k > 1 and _size(view) > max_chars:
        k = max(1, k * 3 // 4)
        nxt = view["offset"] + k
        view["items"], view["next_offset"] = items[:k], nxt
        pagination = view.get("pagination")
        if isinstance(pagination, dict) and isinstance(pagination.get("tool"), str) and \
                isinstance(pagination.get("args"), dict):
            view["next_call"] = {"tool": pagination["tool"],
                                 "arguments": {**pagination["args"], "offset": nxt, "limit": k}}
            view["more"] = f"page shortened to {k} items; continue with next_call (original arguments preserved)"
        elif view.get("view") == "summary":
            view["more"] = (f"inspect_evidence(id={view['source']!r}, view='summary', "
                            f"offset={nxt}, limit={k})")
        elif "path" in view:  # inspect_evidence: repeat the same path, from the first item not shown
            view["more"] = (f"page shortened to {k} items: inspect_evidence(id='{view.get('evidence_id')}', "
                            f"path='{view['path']}', offset={nxt}, limit={k})")
        else:  # list_evidence
            view["more"] = f"page shortened to {k} items: call again with offset={nxt}, limit={k}"
    return view


def _fit_resume_page(view: dict, max_chars: int) -> dict:
    """Keep resume checks visible when checkpoint prose, rather than page length, is oversized."""
    from inferencekit.ops.store import join_path

    # The first shortening adds continuation metadata. Account for those bytes before accepting the page;
    # generic list previews must never replace items while next_offset still skips the unshown rows.
    while len(view["items"]) > 1 and _size(view) > max_chars:
        view = _fit_page(view, max_chars)
    if _size(view) <= max_chars:
        return view

    state_path = view.get("state_path")
    normalized_path = str(state_path).rsplit("/", 1)[0] + "/intake.normalized.json" if state_path else None
    omitted = {}

    def brief(value, source, path, allowance=800):
        if _size(value) <= allowance:
            return value
        pointer = {"file": source, "path": path}
        omitted[path] = pointer
        result = {"truncated": True, "source": pointer}
        if isinstance(value, str):
            result.update(text=value[:allowance // 2], omitted_characters=len(value) - allowance // 2)
        elif isinstance(value, (dict, list)):
            result["total"] = len(value)
        return result

    # Keep policy/cap/authorization facts. Free-form checkpoint prose and full evidence qualifiers remain
    # accessible by their exact file paths; an omitted qualifier is never a successful proof.
    for key in ("next_action", "best"):
        if key in view:
            view[key] = brief(view[key], state_path, key)
    if isinstance(view.get("stages"), dict):
        for name, stage in view["stages"].items():
            if isinstance(stage, dict) and "notes" in stage:
                stage["notes"] = brief(stage["notes"], state_path, join_path(join_path("stages", name), "notes"), 400)
        view["stages"] = brief(view["stages"], state_path, "stages", 3000)
    if isinstance(view.get("policy"), dict):
        for key in ("quality", "architecture_changes"):
            if key in view["policy"]:
                source_key = "intake.quality" if key == "quality" else "derived.architecture_changes"
                view["policy"][key] = brief(view["policy"][key], normalized_path, source_key)
    if isinstance(view.get("intake"), dict) and "warnings" in view["intake"]:
        view["intake"]["warnings"] = brief(view["intake"]["warnings"], normalized_path, "warnings")
    for row in view["items"]:
        source = row.get("path")
        for key in ("summary", "limitations"):
            if key in row:
                row[key] = brief(row[key], source, key)
        if "cited_by" in row:
            row["cited_by"] = brief(row["cited_by"], state_path, "")
        if isinstance(row.get("coverage"), dict) and "missing" in row["coverage"]:
            row["coverage"]["missing"] = brief(row["coverage"]["missing"], source, "coverage.missing")
        if "reason" in row:
            row["reason"] = brief(row["reason"], source, "")
        if isinstance(row.get("integrity"), dict) and "schema_errors" in row["integrity"]:
            row["integrity"]["schema_errors"] = brief(row["integrity"]["schema_errors"], source,
                                                       "")
    view["truncated"] = True
    view["detail_sources"] = list(omitted.values())
    view["note"] = ("Resume details shortened; read the named files/paths for omitted facts. "
                    "Omissions establish neither proof nor authorization. next_call advances past every shown item.")
    # Sources can be long paths. Avoid repeating them in this optional index: each shortened field still
    # carries its source and the original operation arguments remain in pagination/next_call.
    if _size(view) > max_chars:
        view.pop("detail_sources", None)
    if _size(view) > max_chars:
        # Unusually large identifiers or source arguments cannot safely be truncated into executable
        # calls. Stop MCP pagination explicitly; the CLI returns the original, untrimmed operation result.
        optional = {"budget": view.get("budget"), "policy": view.get("policy"),
                    "next_action": view.get("next_action"), "state_path": state_path,
                    "intake": view.get("intake"), "evidence_root": view.get("evidence_root"),
                    "cli_arguments": view["pagination"]["args"], "items": view["items"]}
        stopped = {"truncated": True, "status": "needs_attention", "resume_context_status": view.get("status"),
                   "trust": view.get("trust"), "evidence_summary": view.get("evidence_summary"),
                   "error": "Resume metadata exceeds the response limit; no MCP continuation is provided.",
                   "recovery": "Use inferencekit intake resume with the original run_dir/file/now arguments in "
                               "the CLI for untrimmed output. Read state.json, intake.normalized.json and evidence/ "
                               "under the original run_dir. Do not repeat this MCP page; omitted facts establish "
                               "neither proof nor authorization.",
                   "omitted_fields": list(optional)}
        if _size(stopped) > max_chars:
            raise OpError("MCP response limit is too small to retain resume status, trust and evidence counts; "
                          "use inferencekit intake resume with the original arguments in the CLI")
        for key, value in optional.items():
            candidate = {**stopped, key: value,
                         "omitted_fields": [name for name in stopped["omitted_fields"] if name != key]}
            if _size(candidate) <= max_chars:
                stopped = candidate
        return stopped
    return view


def fit(result: dict, max_chars: int = MAX_CHARS) -> dict:
    """Shrink a result for a model's context window, with paging hints for what was cut."""
    if _size(result) <= max_chars:
        return result
    view = copy.deepcopy(result)
    if _is_page(view):
        view = _fit_page(view, max_chars)
        if _size(view) <= max_chars:
            return view
        if isinstance(view.get("pagination"), dict) and view["pagination"].get("tool") == "intake_resume":
            return _fit_resume_page(view, max_chars)
    record = view.get("record", view)  # inspect_evidence wraps the record
    eid = record.get("evidence_id") if isinstance(record, dict) else None
    candidates = sorted(_lists(view), key=lambda t: -_size(t[3]))
    for parts, parent, key, items in candidates:
        if _size(view) <= max_chars:
            break
        if len(items) <= LIST_PREVIEW:
            continue
        from inferencekit.ops.store import join_path

        parts = parts[1:] if "record" in view and parts[0] == "record" else parts
        inner = ""
        if view.get("view") == "summary" and parts[0] == "items" and len(parts) > 1:
            # Summary findings are projections: their qualifier paths live beside the source decision.
            source_path = view["items"][parts[1]]["path"]
            inner = source_path.rsplit("/" if source_path.startswith("/") else ".", 1)[0]
            parts = parts[2:]
        elif _is_page(view) and "path" in view and parts[0] == "items" and len(parts) > 1:
            # A list inside one item of a page: "items.0.x" is item offset+0 of the paged path.
            inner = join_path(view["path"], view["offset"] + parts[1])
            parts = parts[2:]
        elif "path" in view and "value" in view and parts[0] == "value":
            # inspect_evidence(path=...) wraps a non-list value; the wrapper is not a source key.
            inner, parts = view["path"], parts[1:]
        for part in parts:
            inner = join_path(inner, part)
        parent[key] = {"truncated": True, "total": len(items), "first": items[:LIST_PREVIEW],
                       "more": f"inspect_evidence(id={view.get('source', eid)!r}, path={inner!r}, "
                               f"offset={LIST_PREVIEW}, limit=50)"
                       if eid else "call again with offset/limit"}
    if _size(view) > max_chars:  # long text, e.g. a report's markdown (the result also gives its file path)
        _trim_strings(view, max_chars // 2)
    if _size(view) > max_chars and isinstance(record, dict) and "data" in record:
        record["data"] = {"truncated": True,
                          "more": f"inspect_evidence(id='{eid}', path='data.<key>') for one part at a time",
                          "keys": sorted(result.get("record", result).get("data", {}).keys())}
    if _size(view) > max_chars:  # last resort: preserve integrity and decision counts for summary inspections
        if result.get("view") == "summary":
            integrity = result["integrity"]
            # Validation errors can contain arbitrarily long invalid values. Keep the verdicts and exact count,
            # but do not restore the unbounded messages after trimming the view.
            compact_integrity = {"id_ok": integrity["id_ok"], "schema_ok": integrity["schema_ok"],
                                 "schema_error_count": integrity.get("schema_error_count", 0),
                                 "schema_errors_omitted": bool(integrity.get("schema_errors"))}
            def bounded_label(value):
                if not isinstance(value, str):
                    return None
                if len(value) > 512:
                    return value[:512] + "... [truncated; use the original input]"
                return value
            limitations = result.get("limitations")
            coverage = result.get("coverage")
            missing = coverage.get("missing") if isinstance(coverage, dict) else None
            view = {"view": "summary", "truncated": True, "evidence_id": bounded_label(eid),
                    "source": bounded_label(result["source"]), "integrity": compact_integrity,
                    "finding_counts": result["finding_counts"],
                    "total": result["total"],
                    "limitation_count": len(limitations) if isinstance(limitations, list) else None,
                    "missing_count": len(missing) if isinstance(missing, list) else None,
                    "more": "Summary exceeds response budget. Inspect source paths (metrics, coverage, limitations, "
                            "data) separately; omitted details cannot establish PASS. For schema errors, run "
                            "inferencekit evidence show with the original id or path; use inferencekit.evidence.validate "
                            "on the stored record for every validation error."}
        else:
            view = {"truncated": True, "evidence_id": eid, "summary": record.get("summary") if isinstance(record, dict)
                    else None, "more": "use inspect_evidence with a path"}
    return view


def build_server():
    try:
        import anyio
        import mcp.types as types
        from mcp.server.lowlevel import Server
    except ImportError as e:  # pragma: no cover - mcp is a declared dependency
        raise SystemExit(f"inferencekit mcp serve needs the MCP SDK (pip install 'mcp>=2.3,<3'): {e}") from None

    ops = registry.load_all()
    tools = [types.Tool(name=op.name, description=f"{op.summary} (CLI: inferencekit {' '.join(op.cli)})",
                        input_schema=op.input_schema()) for op in ops.values()]

    async def list_tools(ctx, params):
        start = int(params.cursor) if params is not None and params.cursor else 0
        page = tools[start:start + TOOLS_PAGE]
        nxt = start + TOOLS_PAGE
        return types.ListToolsResult(tools=page, next_cursor=str(nxt) if nxt < len(tools) else None)

    async def call_tool(ctx, params):
        try:
            # Operations block on subprocesses for minutes; keep the event loop free.
            result = await anyio.to_thread.run_sync(partial(registry.call, params.name, params.arguments or {}))
        except OpError as e:
            return types.CallToolResult(content=[types.TextContent(text=f"error: {e}")], is_error=True)
        view = fit(result)
        return types.CallToolResult(content=[types.TextContent(text=json.dumps(view, ensure_ascii=False))])

    server = Server("inferencekit", version=__version__, instructions=INSTRUCTIONS, on_list_tools=list_tools,
                    on_call_tool=call_tool)
    return server, anyio


def serve() -> None:
    server, anyio = build_server()
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(main)
