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
    "enforced when run_dir is given). Long lists in results are truncated: page them with "
    "inspect_evidence(id, path, offset, limit)."
)


def _size(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False, default=str))


def _lists(node, path=""):
    """Yield (path, parent, key, list) for every list inside node, largest first by caller."""
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, list):
                yield p, node, k, v
            yield from _lists(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _lists(v, f"{path}.{i}")


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
    while k > 1 and _size({**view, "items": items[:k]}) > max_chars:
        k = max(1, k * 3 // 4)
    if k == len(items):
        return view
    nxt = view["offset"] + k
    view["items"], view["next_offset"] = items[:k], nxt
    if "path" in view:  # inspect_evidence: repeat the same path, from the first item not shown
        view["more"] = (f"page shortened to {k} items: inspect_evidence(id='{view.get('evidence_id')}', "
                        f"path='{view['path']}', offset={nxt}, limit={k})")
    else:  # list_evidence
        view["more"] = f"page shortened to {k} items: call again with offset={nxt}, limit={k}"
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
    record = view.get("record", view)  # inspect_evidence wraps the record
    eid = record.get("evidence_id") if isinstance(record, dict) else None
    prefix = "record." if "record" in view else ""
    candidates = sorted(_lists(view), key=lambda t: -_size(t[3]))
    for path, parent, key, items in candidates:
        if _size(view) <= max_chars:
            break
        if len(items) <= LIST_PREVIEW:
            continue
        inner = path[len(prefix):] if path.startswith(prefix) else path
        if _is_page(view) and "path" in view and inner.startswith("items."):
            # A list inside one item of a page: "items.0.x" is item offset+0 of the paged path.
            i, _, rest = inner[len("items."):].partition(".")
            inner = f"{view['path']}.{view['offset'] + int(i)}" + (f".{rest}" if rest else "")
        parent[key] = {"truncated": True, "total": len(items), "first": items[:LIST_PREVIEW],
                       "more": f"inspect_evidence(id='{eid}', path='{inner}', offset={LIST_PREVIEW}, limit=50)"
                       if eid else "call again with offset/limit"}
    if _size(view) > max_chars:  # long text, e.g. a report's markdown (the result also gives its file path)
        _trim_strings(view, max_chars // 2)
    if _size(view) > max_chars and isinstance(record, dict) and "data" in record:
        record["data"] = {"truncated": True,
                          "more": f"inspect_evidence(id='{eid}', path='data.<key>') for one part at a time",
                          "keys": sorted(result.get("record", result).get("data", {}).keys())}
    if _size(view) > max_chars:  # last resort: a summary that always fits
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
