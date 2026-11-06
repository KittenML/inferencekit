"""The ``inferencekit`` command: argparse built from the operation registry.

Each operation's ``cli`` path becomes a (sub)command: ("nsys", "analyze") ->
``inferencekit nsys analyze``. Arguments map one to one: ``prompt_len`` -> ``--prompt-len``,
positional arguments stay positional, arrays repeat (``--keep-env A --keep-env B``).
Commands that run another program take it after ``--``:

    inferencekit nsys capture -- python serve_bench.py --steps 64

Output on stdout. An evidence record is printed tersely by default: its summary line (for
``bench rounds``, then each engine's ratio to the best one with its 95% CI), one line per gate or
claim that did not PASS (verdict, name, reason cut to 160 characters), a count of limitations and
missing items, and ``evidence: <path>`` (the full record, already on disk). The
full record as JSON comes with ``--json`` (before or after the command) or
``INFERENCEKIT_OUTPUT=json``; programs that parse stdout set one of them (``bench rounds`` does
for its ``bench serve`` children). Results that are not records (listings, validation results)
print as JSON either way, and Markdown results (``report --format md``, ``intake template``,
``ledger trailers``) as their text. Errors go to stderr with exit code 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from inferencekit import __version__, registry
from inferencekit.evidence import default_root, set_default_root
from inferencekit.registry import OpError

_PY_TYPES = {"string": str, "integer": int, "number": float}
OUTPUT_ENV = "INFERENCEKIT_OUTPUT"
OUTPUT_MODES = ("terse", "json")
JSON_HELP = "Print the full JSON result instead of the terse summary (same as INFERENCEKIT_OUTPUT=json)."
CLAIM_VERDICTS = ("FAIL", "UNPROVEN", "NOT RUN")  # the claim verdicts that are not PASS
REASON_CHARS = 160


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _help(text: str) -> str:
    # argparse %-formats help strings ("%(default)s"), so a literal "%" must be doubled.
    return text.replace("%", "%%")


def _add_arguments(parser: argparse.ArgumentParser, op: registry.Operation) -> None:
    for a in op.args:
        if a.name == "argv" and op.passthrough:
            continue  # filled from the words after "--"
        help_text = _help(a.help + (f" (default: {a.default})" if a.default not in (None, [], "") else ""))
        if a.positional:
            nargs = "+" if a.type == "array" else None
            parser.add_argument(a.name, nargs=nargs, type=_PY_TYPES.get(a.items if a.type == "array" else a.type),
                                choices=a.choices, help=help_text)
        elif a.type == "boolean":
            parser.add_argument(_flag(a.name), dest=a.name, action=argparse.BooleanOptionalAction, default=None,
                                help=help_text)
        elif a.type == "array":
            parser.add_argument(_flag(a.name), dest=a.name, action="append", type=_PY_TYPES.get(a.items, str),
                                choices=a.choices, default=None, help=help_text + " (repeatable)")
        else:
            parser.add_argument(_flag(a.name), dest=a.name, type=_PY_TYPES[a.type], choices=a.choices, default=None,
                                help=help_text)
    if not any(a.name == "json" for a in op.args):
        # Also accepted after the command (`inferencekit gates run ... --json`). SUPPRESS: a subcommand that
        # was not given --json must not reset a --json given before it.
        parser.add_argument("--json", dest="json_output", action="store_true", default=argparse.SUPPRESS,
                            help=JSON_HELP)
    parser.set_defaults(_op=op.name)


def build_parser() -> argparse.ArgumentParser:
    ops = registry.load_all()
    parser = argparse.ArgumentParser(prog="inferencekit", description="Measure-first LLM inference optimisation. "
                                     "Every command writes a content-addressed evidence record under runs/.")
    parser.add_argument("--version", action="version", version=f"inferencekit {__version__}")
    parser.add_argument("--runs-dir", help="Evidence store directory (default: $INFERENCEKIT_RUNS_DIR, else the "
                        "deprecated $INFERKIT_RUNS_DIR, else ./runs). Records made with --run-dir are also copied to "
                        "<run-dir>/evidence/.")
    parser.add_argument("--json", dest="json_output", action="store_true", default=False, help=JSON_HELP)
    top = parser.add_subparsers(dest="command", metavar="<command>")
    groups: dict[str, argparse._SubParsersAction] = {}
    for op in ops.values():
        if len(op.cli) == 1:
            sub = top.add_parser(op.cli[0], help=_help(op.summary), description=op.summary)
        else:
            head, leaf = op.cli
            if head not in groups:
                grp = top.add_parser(head, help=f"{head} commands")
                groups[head] = grp.add_subparsers(dest="subcommand", metavar="<subcommand>")
            sub = groups[head].add_parser(leaf, help=_help(op.summary), description=op.summary)
        _add_arguments(sub, op)
    mcp = top.add_parser("mcp", help="Model Context Protocol server")
    mcp_sub = mcp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    mcp_sub.add_parser("serve", help="Serve every operation as an MCP tool over stdio").set_defaults(_op="mcp:serve")
    return parser


# --------------------------------------------------------------------------- output

def output_mode(flag: bool) -> str:
    """``json`` with --json or INFERENCEKIT_OUTPUT=json, else ``terse`` (an unknown value is reported)."""
    if flag:
        return "json"
    value = (os.environ.get(OUTPUT_ENV) or "terse").strip().lower()
    if value not in OUTPUT_MODES:
        print(f"inferencekit: {OUTPUT_ENV}={value!r} is not one of {list(OUTPUT_MODES)}; using terse",
              file=sys.stderr)
        return "terse"
    return value


def is_record(result) -> bool:
    """An evidence record (it has an id, a kind and a summary), as opposed to a listing or a check result."""
    return isinstance(result, dict) and result.get("view") != "summary" and \
        all(isinstance(result.get(k), str) and result.get(k) for k in ("evidence_id", "kind", "summary"))


def _one_line(text, limit: int | None = None) -> str:
    out = " ".join(str(text).split())
    return out if limit is None or len(out) <= limit else out[: limit - 3] + "..."


def problem_lines(record: dict) -> list[str]:
    """One line per gate or claim of a record that did not PASS: ``<VERDICT> <name>: <reason>``.

    Read from ``data.gates`` (gates run, gates spec-exact), else ``data.verdicts`` (one verdict per
    engine: quality score), else the record's own ``data.verdict`` / ``data.reason`` (correctness), plus
    every ``HARD GATE`` limitation (bench decode, bench rounds). A gate that did not run and was not
    expected (``applicable`` false) is left out: it is not a finding.
    """
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    lines = []
    gates = data.get("gates")
    items = gates.items() if isinstance(gates, dict) else enumerate(gates) if isinstance(gates, list) else ()
    for key, g in items:
        if not isinstance(g, dict) or g.get("verdict") not in CLAIM_VERDICTS:
            continue
        if g["verdict"] == "NOT RUN" and not g.get("applicable"):
            continue
        reason = f": {_one_line(g['reason'], REASON_CHARS)}" if g.get("reason") else ""
        lines.append(f"{g['verdict']} {g.get('gate') or key}{reason}")
    verdicts = data.get("verdicts")
    if not lines and isinstance(verdicts, dict):
        lines += [f"{v} {name}" for name, v in verdicts.items() if v in CLAIM_VERDICTS]
    if not lines and data.get("verdict") in CLAIM_VERDICTS[:2]:
        name = (data.get("request") or {}).get("op") or record["kind"]
        reason = f": {_one_line(data['reason'], REASON_CHARS)}" if isinstance(data.get("reason"), str) else ""
        lines.append(f"{data['verdict']} {name}{reason}")
    for lim in record.get("limitations") or []:
        if isinstance(lim, str) and lim.startswith("HARD GATE"):
            lines.append("FAIL hard gate: " + _one_line(lim.split(":", 1)[1], REASON_CHARS))
    return lines


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def ratio_line(record: dict) -> str | None:
    """bench rounds: every other engine's ratio to the best engine, with its 95% CI, for the metric and bucket
    the summary names (the first row whose engine is the best), e.g.
    ``ratio to best decode_tok_s [all] (a): b 0.52 (95% CI 0.5-0.54)``. None for other records."""
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    if not isinstance(data.get("rows"), list) or not isinstance(data.get("plan"), dict):
        return None
    rows = [r for r in data["rows"] if isinstance(r, dict)]
    head = next((r for r in rows if r.get("engine") == r.get("best")), None)
    if head is None:
        return None
    parts = []
    for r in rows:
        if (r.get("bucket"), r.get("metric")) != (head.get("bucket"), head.get("metric")) or \
                r.get("engine") == head.get("engine") or not _num(r.get("ratio_to_best")):
            continue
        ci = r.get("ratio_ci95") if isinstance(r.get("ratio_ci95"), list) else []
        lo, hi = (ci + [None, None])[:2]
        parts.append(f"{r['engine']} {r['ratio_to_best']:.3g}" + (f" (95% CI {lo:.3g}-{hi:.3g})"
                                                                  if _num(lo) and _num(hi) else ""))
    if not parts:
        return None
    more = f", +{len(parts) - 6} more" if len(parts) > 6 else ""
    return (f"ratio to best {head.get('metric')} [{head.get('bucket')}] ({head.get('engine')}): "
            + ", ".join(parts[:6]) + more)


def evidence_paths(record: dict) -> tuple[Path | None, Path | None]:
    """Where the full record is: the evidence store copy, and the run-directory copy when there is one."""
    eid = record["evidence_id"]
    stored = (default_root() / eid / "evidence.json").resolve()
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    args = (data.get("request") or {}).get("args") or {}
    run_dir = args.get("run_dir") or data.get("run_dir")  # bench rounds defaults it to the manifest's directory
    copy = None
    if isinstance(run_dir, str) and run_dir:
        copy = (Path(run_dir).expanduser() / "evidence" / f"{eid}.json").resolve()
    return (stored if stored.is_file() else None), (copy if copy is not None and copy.is_file() else None)


def terse(record: dict) -> str:
    """The terse view of an evidence record (see the module docstring)."""
    lines = [_one_line(record["summary"])]
    ratios = ratio_line(record)
    if ratios:
        lines.append(ratios)
    lines += problem_lines(record)
    n_lim = len(record.get("limitations") or [])
    n_missing = len((record.get("coverage") or {}).get("missing") or [])
    if n_lim or n_missing:
        lines.append(f"{n_lim} limitation(s), {n_missing} missing item(s): listed in the evidence file")
    stored, copy = evidence_paths(record)
    if stored is None and copy is None:
        lines.append(f"evidence: {record['evidence_id']} (no evidence file found under {default_root()})")
    else:
        lines.append(f"evidence: {stored or copy}")
        if stored is not None and copy is not None:
            lines.append(f"run-dir copy: {copy}")
    return "\n".join(lines)


def print_result(result, mode: str) -> None:
    if isinstance(result, dict) and result.get("format") == "md" and "markdown" in result:
        print(result["markdown"])
        print(f"(written to {result['path']})", file=sys.stderr)
    elif mode == "terse" and is_record(result):
        print(terse(result))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Split at the first "--": everything after it is the command to profile, untouched by argparse.
    passthrough = None
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.runs_dir:
        set_default_root(ns.runs_dir)
    op_name = getattr(ns, "_op", None)
    if op_name is None:
        parser.print_help()
        return 2
    if op_name == "mcp:serve":
        from inferencekit.mcp_server import serve

        serve()
        return 0
    op = registry.get(op_name)
    args = {a.name: getattr(ns, a.name) for a in op.args if hasattr(ns, a.name)}
    if op.passthrough:
        if not passthrough:
            parser.error(f"inferencekit {' '.join(op.cli)} needs the command to run after '--'")
        args["argv"] = passthrough
    elif passthrough:
        parser.error("'--' is only valid for commands that run another program")
    mode = output_mode(getattr(ns, "json_output", False))
    try:
        result = registry.call(op_name, args)
    except OpError as e:
        print(f"inferencekit: error: {e}", file=sys.stderr)
        return 1
    print_result(result, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
