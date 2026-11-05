"""Clip rounds: every clip engine of a run manifest, R rounds in a Williams order, one server at a time.

The text runner's reasoning (serve/rounds.py) holds for clips too: a GPU's speed drifts, so every round
runs every engine once, in a balanced order. This module reuses, by import, its order design
(round_orders, default_rounds), manifest checks (load_manifest, engine_specs), output parsing
(parse_serve_output, request_rows, cell_status) and summary (summarize), plus the server lifecycle of
servers.py: clean environment, launch in its own process group, readiness, group stop, GPU memory back
to its pre-launch level, and the GPU lock held for the whole run. What differs from text rounds:

  * readiness is ``GET /health`` (the clip HTTP contract) unless the manifest's engine names another
    ``ready_check``; there is no /v1/models lookup and no chat warm-up: bench clip's own warm-up (untimed
    clips at every distinct request shape of the set, before its first timed clip) runs in every cell,
    after each fresh launch;
  * the measurement is ``inferencekit bench clip`` as a subprocess, with ``--first-round k-1`` so round k
    gets the same pre-registered seed on every engine, and ``--extend`` from the engine's previous record
    so its last record holds all its rounds (``compare --metric clip_ms`` takes two of those);
  * per-round values are clip_ms (mean over the round's clips, bench_clip.round_stats) and clips_per_hour,
    per task and 'all';
  * the server log is kept but not run through the text engines' resolved-config parser, whose hard gates
    (CUDA graphs, attention fallback) are rules for LLM servers; what a clip engine says it ran is in
    bench clip's rows (``effective``, self-reported).

Manifest engine fields beyond the text runner's: ``endpoint_kind`` (clip_http by default, or comfyui) and
``workflow`` (comfyui; a path relative to the manifest is resolved there).
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit import proc, stats
from inferencekit.canonical import file_sha256
from inferencekit.gpuinfo import gpu_identity
from inferencekit.ops import bench_clip as bc
from inferencekit.ops.doctor import gpu_processes
from inferencekit.registry import OpError
from inferencekit.serve import clip_client as cc
from inferencekit.serve import rounds as rr
from inferencekit.serve import servers

CLIP_CMD_ENV = "INFERENCEKIT_BENCH_CLIP_CMD"
DIRECTIONS = {"clip_ms": False, "clips_per_hour": True}  # ranked here even where rr.higher_is_better cannot
OUT_DIR = "clip_rounds"


# --------------------------------------------------------------------------- manifest

def clip_specs(manifest: dict, manifest_path: str | os.PathLike, only: str | list[str] | None) -> list[dict]:
    """rr.engine_specs plus the clip fields: endpoint_kind, workflow, and /health as the default ready check."""
    raw = {str(e.get("label") or e.get("name") or ""): e for e in manifest["engines"] if isinstance(e, dict)}
    specs = rr.engine_specs(manifest, only)
    base = Path(manifest_path).resolve().parent
    problems = []
    for s in specs:
        e = raw.get(s["label"]) or {}
        if not e.get("ready_check"):
            s["ready_check"] = cc.HEALTH_PATH
        s["endpoint_kind"] = e.get("endpoint_kind") or "clip_http"
        if s["endpoint_kind"] not in cc.ENDPOINT_KINDS:
            problems.append(f"engine '{s['label']}': endpoint_kind must be one of {list(cc.ENDPOINT_KINDS)}")
        wf = e.get("workflow")
        if wf is not None and not isinstance(wf, str):
            problems.append(f"engine '{s['label']}': 'workflow' must be a path")
        elif wf:
            s["workflow"] = str(Path(wf) if Path(wf).is_absolute() or not (base / wf).is_file() else base / wf)
    if problems:
        raise OpError("manifest engines: " + "; ".join(problems))
    return specs


# --------------------------------------------------------------------------- bench clip

def clip_argv(template: str | None, ctx: dict, extra: str | None) -> list[str]:
    """The ``bench clip`` command for one cell.

    Default: ``<python> -m inferencekit bench clip --base-url ... --clips ... --endpoint-kind ... --label ...
    --run-dir ... --rounds 1 --first-round <k-1> --warmup N``, plus ``--extend`` (the engine's previous
    record), ``--workflow``, ``--engine-version`` and ``--model`` when known, then ``extra`` (the same for
    every engine: seed policy, timeouts, --gpu-name...). A template (``clip_cmd`` or
    $INFERENCEKIT_BENCH_CLIP_CMD) with ``{placeholders}`` is used as written ({base_url} {clips} {label}
    {round} {first_round} {run_dir} {endpoint_kind} {workflow} {extend} {warmup} {model}); a template
    without placeholders replaces only the program part.
    """
    template = template or os.environ.get(CLIP_CMD_ENV) or ""
    words = shlex.split(template) if template else [sys.executable, "-m", "inferencekit", "bench", "clip"]
    if any("{" in w for w in words):
        try:
            argv = [w.format(**ctx) for w in words]
        except (KeyError, IndexError, ValueError) as e:
            raise OpError(f"clip command template: unknown placeholder {e}") from None
    else:
        argv = words + ["--base-url", ctx["base_url"], "--clips", ctx["clips"], "--endpoint-kind",
                        ctx["endpoint_kind"], "--label", ctx["label"], "--run-dir", ctx["run_dir"], "--rounds", "1",
                        "--first-round", str(ctx["first_round"]), "--warmup", str(ctx["warmup"])]
        for flag, key in (("--extend", "extend"), ("--workflow", "workflow"), ("--engine-version", "engine_version"),
                          ("--model", "model")):
            if ctx.get(key):
                argv += [flag, str(ctx[key])]
    return argv + (shlex.split(extra) if extra else [])


def round_metrics(rows: list[dict], k: int) -> dict:
    """``{task | 'all': {metric: {value, unit, stat, n}}}`` for bench clip round ``k`` (from its rows)."""
    mine = [r for r in rows if r.get("round") == k]
    if not mine:
        return {}  # no rows for this round: the cell has no metrics (cell_status says serve_no_metrics)
    out: dict[str, dict] = {}
    for subject, v in bc.round_stats(mine).items():  # the same per-round values as bench clip's data.rounds
        mets = {}
        if v["clip_ms"] is not None:
            mets["clip_ms"] = {"value": v["clip_ms"], "unit": "ms", "stat": "mean", "n": v["n"]}
            mets["clips_per_hour"] = {"value": v["clips_per_hour"], "unit": "clips/h", "stat": "rate", "n": v["n"]}
        mets["failed_requests"] = {"value": float(v["n_failed"]), "unit": "clips", "stat": "count",
                                   "n": v["n"] + v["n_failed"]}
        out[subject] = mets
    return out


# --------------------------------------------------------------------------- one cell

def run_clip_cell(spec: dict, k: int, position: int, cfg: dict) -> dict:
    """Launch, check, measure (bench clip, one round) and stop one engine for round ``k``. Never raises for engine
    failures."""
    run_dir: Path = cfg["run_dir"]
    label = spec["label"]
    base = f"http://{spec['host']}:{spec['port']}"
    log_rel = f"logs/{label}.r{k}.log"
    log_path = run_dir / log_rel
    cell: dict = {"round": k, "position": position, "label": label, "kind": spec["kind"],
                  "endpoint_kind": spec["endpoint_kind"], "status": None, "log": log_rel, "t_start": time.time()}
    if log_path.exists():
        log_path.unlink()
    env, env_info = servers.server_env(spec["env"])
    cell["env"] = env_info
    if servers.port_in_use(spec["host"], spec["port"]):
        cell.update(status="port_busy", detail=f"something already listens on {spec['host']}:{spec['port']}; "
                                               "refusing to benchmark it in place of the engine")
        cell["t_end"] = time.time()
        return cell
    gpu_before = cfg["gpu_reader"]()
    server = None
    try:
        try:
            server = servers.launch(spec["launch"], env, log_path, cwd=spec.get("cwd"))
        except OSError as e:
            cell.update(status="launch_failed", detail=str(e)[:300])
            return cell
        ready = servers.wait_ready(server, base + spec["ready_check"], cfg["startup_timeout_s"], cfg["poll_s"])
        cell["ready"] = ready
        if ready["status"] != "ready":
            cell["status"] = "startup_timeout" if ready["status"] == "timeout" else "exited_early"
            return cell
        ctx = {"base_url": base, "clips": str(cfg["clips_path"]), "label": label, "round": k, "first_round": k - 1,
               "run_dir": str(run_dir), "endpoint_kind": spec["endpoint_kind"], "workflow": spec.get("workflow") or "",
               "engine_version": spec.get("version") or "", "extend": cfg["last_ev"].get(label) or "",
               "warmup": cfg["warmup"], "model": cfg.get("model_id") or ""}
        argv = clip_argv(cfg["clip_cmd"], ctx, cfg["clip_args"])
        child_env = dict(os.environ)
        # Same evidence store as this run, and the GPU lock is already held (by this process).
        root = str(ev.default_root().resolve())
        # The child's record is read from its stdout (parse_serve_output): full JSON, never the terse view.
        child_env.update({"INFERKIT_RUNS_DIR": root, "INFERENCEKIT_RUNS_DIR": root,
                          "INFERENCEKIT_GPU_LOCK_HELD": str(cfg.get("gpu_lock_path") or ""),
                          "INFERENCEKIT_OUTPUT": "json"})
        t0 = time.monotonic()
        res = proc.run(argv, timeout=cfg["clip_timeout_s"], env=child_env)  # a timeout kills its process group
        record = rr.parse_serve_output(res.stdout)
        cell["clip"] = {"returncode": res.returncode, "seconds": round(time.monotonic() - t0, 3),
                        "evidence_id": record.get("evidence_id") if record else None,
                        "stderr_tail": ev.redact_text(res.stderr[-1500:]) if res.returncode != 0 else None}
        rows = rr.request_rows(record, ev.default_root())
        cell["metrics"] = round_metrics(rows or [], k - 1)
        cell["metrics_source"] = f"bench clip requests artifact, round {k - 1} (mean over its clips)"
        if record is not None and res.returncode == 0 and record.get("evidence_id"):
            cfg["last_ev"][label] = record["evidence_id"]
            validity = (record.get("data") or {}).get("claim_validity") or {}
            cell["claim_valid"] = validity.get("valid")
            cell["violations"] = (validity.get("violations") or [])[:5]
            # clips (all rounds so far: the record extends the last one) that reused the fixed negative prompt's
            # encoding; allowed for every engine, stated with the ranking below
            cell["cache_allowed_clips"] = validity.get("cache_allowed_clips")
        cell["status"], detail = rr.cell_status(res.returncode, record, cell["metrics"] if rows else {})
        if detail:
            cell["detail"] = detail
        return cell
    finally:
        if server is not None:
            cell["stop"] = servers.stop(server, cfg["stop_timeout_s"])
            if not cell["stop"]["group_gone"] and cell["status"] == "ok":
                cell["status"] = "stop_failed"
            cell["gpu_memory"] = servers.wait_gpu_memory(gpu_before, cfg["gpu_mem_tolerance_mib"],
                                                         cfg["gpu_mem_wait_s"], reader=cfg["gpu_reader"])
        cell["t_end"] = time.time()


# --------------------------------------------------------------------------- summary

def summarize(cells: list[dict], labels: list[str], resamples: int = stats.DEFAULT_RESAMPLES, seed: int = 0) -> dict:
    """rr.summarize, with clips_per_hour ranked as a rate (rr.higher_is_better does not know '_per_hour')."""
    out = rr.summarize(cells, labels, resamples=resamples, seed=seed, mode="single")
    by_round: dict[tuple[str, str, str], dict[int, float]] = {}
    for c in cells:
        if c.get("status") in rr.STATUSES_OK:
            for bucket, mets in (c.get("metrics") or {}).items():
                for name, m in mets.items():
                    by_round.setdefault((c["label"], bucket, name), {})[c["round"]] = m["value"]
    for name, higher in DIRECTIONS.items():
        if rr.higher_is_better(name) == higher:
            continue  # the text runner ranks it already
        for bucket in sorted({r["bucket"] for r in out["rows"] if r["metric"] == name}):
            rows = [r for r in out["rows"] if r["metric"] == name and r["bucket"] == bucket]
            eligible = [r for r in rows if not r["failed_requests"]]
            best = (max if higher else min)(eligible, key=lambda r: r["median"])["engine"] if eligible else None
            for r in rows:
                r.update(higher_is_better=higher, best=best)
                if r["failed_requests"]:
                    r["excluded_from_best"] = (f"{r['failed_requests']:g} failed clip(s) over {r['n']} round(s): "
                                               "its rate covers only the clips it returned")
                    continue
                if best is None:
                    continue
                b, c = by_round[(best, bucket, name)], by_round[(r["engine"], bucket, name)]
                paired = sorted(set(b) & set(c))
                if r["engine"] == best:
                    r.update(ratio_to_best=1.0, n_pairs=len(paired))
                elif paired:
                    res = stats.paired_ratio_ci([b[i] for i in paired], [c[i] for i in paired], resamples=resamples,
                                                seed=seed)
                    r.update(ratio_to_best=res["ratio"], ratio_ci95=list(res["ci95"]), n_pairs=res["n_pairs"])
    out["definitions"] = {
        "median": "median over rounds of the per-round value (clip_ms: mean over that round's clips of the task, "
                  "or of every clip for 'all', from bench clip's requests artifact; client timer) (observed)",
        "clips_per_hour": "per round: 3,600,000 x clips / sum of their clip_ms, one clip at a time (derived)",
        "ci95": "bootstrap 95% CI of that median, rounds resampled",
        "ratio_to_best": "median(engine) / median(best engine) over rounds both completed, with a paired bootstrap "
                         "95% CI resampling rounds (derived); best = lowest clip_ms, highest clips_per_hour, among "
                         "engines with no failed clip in the task",
        "failed_requests": "clips that failed in a round; an engine with any failure in a task is never best there "
                           "and gets no ratio"}
    return out


def markdown(summary: dict, plan: dict, cells: list[dict]) -> str:
    fmt = rr._fmt
    lines = [f"# bench rounds: clip mode, {plan['rounds']} rounds, {len(plan['engines'])} engines", "",
             f"Order: {plan['design']['design']} design, period {plan['design']['period']}, balanced: "
             f"{'yes' if plan['design']['balanced'] else 'NO'}. Clip set {plan['clips']['name']} (sha256 "
             f"{plan['clips']['sha256'][:12]}..., n={plan['clips']['n']}).", "",
             "clip_ms medians and CIs are [observed] (client timer in `bench clip`, request sent to clip in hand); "
             "clips_per_hour and ratios are [derived]. Stage times are self-reported by the engines and are only in "
             "each engine's bench clip record.", "",
             "| engine | task | metric | median | 95% CI | rounds | ratio to best [derived] | ratio 95% CI |",
             "|---|---|---|---|---|---|---|---|"]
    invalid = {c["label"] for c in cells if c.get("claim_valid") is False}
    for r in summary["rows"]:
        unit = f" {r['unit']}" if r.get("unit") else ""
        mark = " (best)" if r["engine"] == r["best"] else ""
        if r.get("failed_requests"):
            mark += " [FAILURES]"
        if r["engine"] in invalid:
            mark += " [CLAIM INVALID]"
        ratio_ci = "-" if r["engine"] == r["best"] or r["ratio_to_best"] is None else \
            f"{fmt(r['ratio_ci95'][0])} - {fmt(r['ratio_ci95'][1])}"
        lines.append(f"| {r['engine']}{mark} | {r['bucket']} | {r['metric']} | {fmt(r['median'])}{unit} | "
                     f"{fmt(r['ci95'][0])} - {fmt(r['ci95'][1])} | {r['n']} | {fmt(r['ratio_to_best'])} | "
                     f"{ratio_ci} |")
    if not summary["rows"]:
        lines.append("| (no successful rounds) | | | | | | | |")
    failed = [c for c in cells if c.get("status") != "ok"]
    if failed:
        lines += ["", "## Failed cells", "", "| round | engine | status | detail |", "|---|---|---|---|"]
        for c in failed:
            detail = c.get("detail") or (c.get("ready") or {}).get("last") or (c.get("clip") or {}).get("returncode")
            lines.append(f"| {c['round']} | {c['label']} | {c['status']} | {str(detail)[:120]} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- the run

def run_clip_rounds(manifest_path: str, clips_path: str, rounds: int | None = None, engines: str | None = None,
                    run_dir: str | None = None, gpu_lock_path: str | None = None, startup_timeout_s: float = 1800,
                    stop_timeout_s: float = 30, clip_timeout_s: float = 14400, warmup: int = 1,
                    clip_cmd: str | None = None, clip_args: str | None = None, gpu_mem_tolerance_mib: int = 512,
                    gpu_mem_wait_s: float = 60, poll_s: float = 0.5, gpu_reader=None) -> dict:
    """Run every (round, engine) clip cell under the GPU lock; return plan, cells, summary and paths."""
    manifest = rr.load_manifest(manifest_path)
    specs = clip_specs(manifest, manifest_path, engines)
    if not specs:
        raise OpError("no engines selected")
    rounds = rr.default_rounds(len(specs)) if rounds is None else rounds
    if rounds < 1:
        raise OpError("rounds must be >= 1")
    if warmup < 0:
        raise OpError("warmup must be >= 0")
    clipset = bc.load_clip_set(clips_path)  # fail before any launch when the set is bad
    rdir = Path(run_dir) if run_dir else Path(manifest_path).resolve().parent
    rdir.mkdir(parents=True, exist_ok=True)
    labels = [s["label"] for s in specs]
    design = rr.round_orders(labels, rounds)
    plan = {"mode": "clip", "rounds": rounds,
            "engines": [{k: s.get(k) for k in ("label", "name", "kind", "version", "port", "ready_check", "role",
                                                "endpoint_kind")} for s in specs],
            "design": design, "clips": {"name": clipset.name, "sha256": clipset.sha256, "n": len(clipset.clips)},
            "settings": {"startup_timeout_s": startup_timeout_s, "stop_timeout_s": stop_timeout_s,
                         "clip_timeout_s": clip_timeout_s, "warmup": warmup, "clip_args": clip_args,
                         "clip_cmd_custom": bool(clip_cmd or os.environ.get(CLIP_CMD_ENV)),
                         "gpu_mem_tolerance_mib": gpu_mem_tolerance_mib, "gpu_mem_wait_s": gpu_mem_wait_s,
                         "warmup_kind": "bench clip's own warm-up after every launch: `warmup` untimed clip(s) for "
                                        "each distinct request shape of the set (task, size, frames, fps, steps, "
                                        "guidance, audio, image), all before the first timed clip, seeds outside "
                                        "the timed seed policy and prompts no timed clip has"},
            "manifest_sha256": file_sha256(Path(manifest_path))[0]}
    cfg = {"run_dir": rdir, "clips_path": Path(clips_path).resolve(), "startup_timeout_s": startup_timeout_s,
           "stop_timeout_s": stop_timeout_s, "clip_timeout_s": clip_timeout_s, "warmup": warmup, "clip_cmd": clip_cmd,
           "clip_args": clip_args, "gpu_mem_tolerance_mib": gpu_mem_tolerance_mib, "gpu_mem_wait_s": gpu_mem_wait_s,
           "poll_s": poll_s, "gpu_reader": gpu_reader or servers.gpu_memory_used_mib,
           "model_id": (manifest.get("model") or {}).get("id"), "last_ev": {}}
    out_dir = rdir / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    with servers.gpu_lock(gpu_lock_path, owner=f"inferencekit bench rounds (clip) {rdir.name}") as lock:
        plan["gpu_lock"] = lock["path"]
        cfg["gpu_lock_path"] = lock["path"]
        plan["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (out_dir / "plan.json").write_text(json.dumps(ev.redact(plan), indent=2) + "\n")
        by_label = {s["label"]: s for s in specs}
        for k, order in enumerate(design["orders"], 1):
            for pos, label in enumerate(order):
                cell = run_clip_cell(by_label[label], k, pos, cfg)
                cells.append(cell)
                (out_dir / f"{label}.r{k}.json").write_text(json.dumps(ev.redact(cell), indent=2) + "\n")
        plan["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary = summarize(cells, labels)
    plan["clip_evidence_final"] = dict(cfg["last_ev"])  # each engine's bench clip record with all its rounds
    (out_dir / "plan.json").write_text(json.dumps(ev.redact(plan), indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(ev.redact({"plan": plan, "summary": summary}), indent=2) + "\n")
    md = markdown(summary, plan, cells)
    (out_dir / "summary.md").write_text(md)
    return {"plan": plan, "cells": cells, "summary": summary, "markdown": md, "run_dir": rdir, "manifest": manifest,
            "clipset": clipset}


def bench_rounds_clip(call, *, manifest: str, clips: str, rounds: int | None = None, engines: str | None = None,
                      run_dir: str | None = None, gpu_lock: str | None = None, startup_timeout_s: float = 1800,
                      stop_timeout_s: float = 30, clip_timeout_s: float = 14400, warmup: int = 1,
                      clip_args: str | None = None, clip_cmd: str | None = None, gpu_mem_tolerance_mib: int = 512,
                      gpu_reader=None) -> dict:
    """bench rounds in clip mode: run the rounds and store one ``clip_benchmark`` record (the summary)."""
    from inferencekit.ops.bench_rounds import _limitations  # the text runner's cell, order and failure notes

    others = gpu_processes()
    result = run_clip_rounds(manifest, clips, rounds, engines=engines, run_dir=run_dir, gpu_lock_path=gpu_lock,
                             startup_timeout_s=startup_timeout_s, stop_timeout_s=stop_timeout_s,
                             clip_timeout_s=clip_timeout_s, warmup=warmup, clip_cmd=clip_cmd, clip_args=clip_args,
                             gpu_mem_tolerance_mib=gpu_mem_tolerance_mib, gpu_reader=gpu_reader)
    plan, summary, rdir, clipset = result["plan"], result["summary"], Path(result["run_dir"]), result["clipset"]
    labels = [e["label"] for e in plan["engines"]]
    limitations, missing = _limitations({**result, "resolved_config": {}})
    if others:
        limitations.append(f"{len(others)} other compute process(es) were on the GPU when the run started: "
                           "timings may be contended")
    for label in labels:
        bad = sorted({v for c in result["cells"] if c["label"] == label for v in c.get("violations") or []})
        if bad:
            limitations.append(f"CLAIM INVALID: {label}: {'; '.join(bad[:3])}")
    reused = [x for x in labels if any((c.get("cache_allowed_clips") or 0) > 0 for c in result["cells"]
                                       if c["label"] == x)]
    if reused:
        rest = [x for x in labels if x not in reused]
        limitations.append(f"{', '.join(reused)} reused the encoding of a fixed negative prompt "
                           f"({cc.NEGATIVE_PROMPT_CACHE}: computed once, not per clip; allowed for every engine)"
                           + (f"; {', '.join(rest)} encoded it on every clip, so the ranking compares their full text "
                              "encoding with a reused one" if rest else ""))
    limitations.append("server logs kept under logs/ but not parsed for a resolved configuration (the text "
                       "engines' parser and hard gates do not apply to clip engines); each engine's self-reported "
                       "effective settings are in its bench clip record")
    metrics = []
    for r in summary["rows"]:
        subject = f"{r['engine']}/{r['bucket']}"
        metrics.append(ev.metric(r["metric"], r["median"], r["unit"] or "unknown", "median_over_rounds", n=r["n"],
                                 ci95=r["ci95"], subject=subject))
        if r["ratio_to_best"] is not None:
            metrics.append(ev.metric(f"{r['metric']}_ratio_to_best", r["ratio_to_best"], "x", "ratio_of_medians",
                                     n=r["n_pairs"], ci95=r["ratio_ci95"], subject=subject))
    m = result["manifest"].get("model") or {}
    kinds = sorted({e["endpoint_kind"] for e in plan["engines"]})
    ident = ev.identity(gpu=gpu_identity(), engine={"name": "+".join(labels), "flags": {"engines": labels}},
                        model={"id": m.get("id"), "revision": m.get("revision"), "dtype": m.get("precision")},
                        workload={"mode": "single", "batch": 1, "endpoint_kind": "+".join(kinds),
                                  "clip_set": clipset.identity(),
                                  **{k: clipset.uniform(k) for k in bc.SHAPE_KEYS}})
    n_ok = sum(1 for c in result["cells"] if c.get("status") == "ok")
    line = (f"bench rounds (clip): {len(labels)} engines x {plan['rounds']} rounds, {n_ok}/{len(result['cells'])} "
            f"cells ok, Williams order {'balanced' if plan['design']['balanced'] else 'NOT balanced'}")
    heads = [r for r in summary["rows"] if r["engine"] == r["best"] and r["metric"] == "clip_ms" and
             r["bucket"] == "all"]
    if heads:
        line += f"; best clip_ms [all]: {heads[0]['engine']} {heads[0]['median'] / 1000:.3g} s [observed]"
    artifacts = [ev.artifact(rdir / OUT_DIR / name, rdir, role)
                 for name, role in (("summary.md", "summary"), ("summary.json", "summary"), ("plan.json", "plan"))
                 if (rdir / OUT_DIR / name).is_file()]
    artifacts += [ev.artifact(rdir / c["log"], rdir, "server_log") for c in result["cells"]
                  if (rdir / c["log"]).is_file()]
    cells = [{k: c.get(k) for k in ("round", "position", "label", "kind", "endpoint_kind", "status", "ready", "clip",
                                     "stop", "gpu_memory", "env", "log", "metrics_source", "claim_valid",
                                     "violations", "cache_allowed_clips", "t_start", "t_end", "detail")}
             for c in result["cells"]]
    record = ev.build(call, kind=bc.KIND, claim_class="observed", summary=line, identity_=ident, metrics=metrics,
                      artifacts=artifacts, missing=missing, limitations=limitations,
                      data={"source": "bench rounds clip", "run_dir": str(rdir), "plan": plan, "cells": cells,
                            "rows": summary["rows"], "definitions": summary["definitions"],
                            "metric_claim_classes": {"clip_ms": "observed (median over rounds of bench clip's "
                                                                "client timer)",
                                                     "clips_per_hour": "derived", "<metric>_ratio_to_best": "derived"},
                            "failures": summary.get("failures") or [],
                            "clip_evidence": [c["clip"]["evidence_id"] for c in result["cells"]
                                              if (c.get("clip") or {}).get("evidence_id")],
                            "gpu_processes_at_start": len(others), "markdown": result["markdown"]})
    rec = ev.Store().save(record)
    copy = rdir / "evidence" / f"{rec['evidence_id']}.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return rec
