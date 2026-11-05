"""review build: one self-contained HTML page comparing every engine's outputs, per input bucket.

Reads ``runs/<run-id>/{inputs,outputs,metrics}`` (see ``inferencekit.review.rundir``) and
writes ``runs/<run-id>/review/index.html`` plus ``review/data.json``, a machine-readable
copy of what the page shows with the sha256 of every file it was built from.

The page is a view, not a measurement: speed and quality numbers come from the run's
metrics files (marked observed) and the page adds only derived comparisons (first
divergence from the reference, word error alignment), marked derived. A missing output
is shown as NOT RUN, never as a blank cell.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from inferencekit import __version__
from inferencekit import evidence as ev
from inferencekit import prompts as _sets
from inferencekit.canonical import file_sha256
from inferencekit.registry import Arg, OpError, operation, utc_now
from inferencekit.review.page import render
from inferencekit.review.rundir import load_run


def store_records_for(store: ev.Store, root: Path) -> list[dict]:
    """Evidence in the store made for this run dir (its request's ``run_dir`` argument resolves to ``root``).

    bench serve and gates save their records in the evidence store, not in <run-dir>/evidence/,
    so without this the page would show their verdicts as NOT RUN.
    """
    want = root.resolve()
    out = []
    for rec in store.records():
        args = ((rec.get("data") or {}).get("request") or {}).get("args") or {}
        rd = args.get("run_dir") if isinstance(args, dict) else None
        if isinstance(rd, str) and rd:
            try:
                if Path(rd).expanduser().resolve() == want:  # the evidence layer writes home as ~
                    out.append(rec)
            except (OSError, RuntimeError):
                continue
    return out


def _write(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a reader never sees half a page."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".review-", suffix=".tmp", mode="w",
                                         encoding="utf-8", delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(text)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@operation("build_review", ("review", "build"), "Build the run's review page (runs/<id>/review/index.html): inputs "
           "by bucket x engines, first divergence from the reference marked, word errors for speech, summary "
           "table and filters; one offline HTML file.",
           [Arg("run_dir", "string", "Run directory (runs/<run-id>) holding inputs/, outputs/ and metrics/.",
                required=True, positional=True),
            Arg("reference", "string", "Label of the outputs/<label> column the others are compared against "
                "(default: 'reference' when that directory exists)."),
            Arg("prompts", "string", "Prompt-set JSONL ({id, bucket, messages} per line) for the input column "
                "when the run has no inputs/ directory."),
            Arg("evidence", "array", "More evidence ids (gates, quality, capacity records) to show per engine; "
                "records in <run-dir>/evidence/ and store records made with --run-dir <run-dir> are read "
                "automatically.")])
def build_review(call, run_dir, reference, prompts, evidence):
    root = Path(run_dir)
    if not root.is_dir():
        raise OpError(f"run directory not found: {run_dir}")
    if not (root / "review").resolve().is_relative_to(root.resolve()):
        raise OpError("review directory escapes the run directory")
    if not (root / "outputs").is_dir() and not (root / "inputs").is_dir() and not (root / "metrics").is_dir():
        raise OpError(f"{run_dir} has no inputs/, outputs/ or metrics/: is it a run directory (runs/<run-id>)?")
    if prompts and not _sets.existing(prompts).is_file():
        raise OpError(f"prompt set not found: {prompts}")
    store = ev.Store()
    extra = [store.load(ref) for ref in evidence or []]
    found = store_records_for(store, root)
    run = load_run(root, reference, prompts, extra + found)
    from inferencekit.review.media import prepare_pcm

    playback_copies = prepare_pcm(run)
    if not run.evidence_attached:
        run.warnings.insert(0, "no evidence attached: <run-dir>/evidence/ is missing or empty, no --evidence was "
                               "passed and no stored record names this run dir; quality, gates and N* come only "
                               "from the metrics files (pass --evidence <id> for gates, quality and capacity records)")
    generated = utc_now()
    page, data = render(run, generated, __version__)
    out = root / "review"
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    for rel in run.sources:
        try:
            sha, size = file_sha256(root / rel)
        except OSError:
            continue
        sources.append({"path": rel, "sha256": sha, "bytes": size})
    data.update({"generated": generated, "tool_version": __version__, "request": call.request(),
                 "sources": sources, "playback_copies": playback_copies})
    _write(out / "index.html", page)
    _write(out / "data.json", json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    cells = len(run.items) * len(run.engines)
    not_run = sum(1 for c in run.cells.values() if c.status == "not_run")
    failed = sum(1 for c in run.cells.values() if c.status == "failed")
    return {"path": str(out / "index.html"), "data": str(out / "data.json"), "modality": run.modality,
            "reference": run.reference, "engines": [e.label for e in run.engines], "inputs": len(run.items),
            "buckets": run.buckets(), "cells": cells, "not_run": not_run, "failed": failed,
            "bytes": len(page.encode("utf-8")), "warnings": run.warnings[:20],
            "note": "a view of the run's files: speed/quality numbers are the metrics files' (observed); "
                    "divergence and word errors are computed here (derived)"}
