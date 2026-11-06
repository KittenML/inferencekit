"""Fresh-process audio/image cells using the existing client timers and Williams order."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

from inferencekit import evidence as ev, proc
from inferencekit.canonical import file_sha256
from inferencekit.cpuinfo import cpu_identity
from inferencekit.registry import OpError
from inferencekit.serve import resources, round_evidence, rounds as rr, servers

METRICS = {
    "audio": {"audio_ms": "ms", "first_audio_p95_ms": "ms", "rtf": "x"},
    "image": {"image_ms": "ms", "images_per_hour": "images/h"},
}


def mode_for(manifest: dict, requested: str, manifest_path: str) -> str:
    if requested != "auto":
        return requested
    workload = manifest.get("workload") or {}
    modes = workload.get("modes") or []
    if len(modes) == 1:
        return modes[0]
    if len(modes) > 1:
        raise OpError("manifest declares multiple workload modes; select one with --mode")
    modality = workload.get("modality")
    if not modality:
        intake = Path(manifest_path).resolve().parent / "intake.normalized.json"
        if intake.is_file():
            try:
                modality = json.loads(intake.read_text()).get("modality")
            except (ValueError, OSError):
                raise OpError("cannot read intake.normalized.json for automatic modality routing") from None
    if modality:
        mode = {"tts": "audio", "audio": "audio", "image": "image", "video": "clip",
                "text": "single", "llm": "single"}.get(modality)
        if mode is None:
            raise OpError(f"no bench rounds client for modality {modality}; select a supported --mode")
        return mode
    kinds = {e.get("endpoint_kind") for e in manifest.get("engines", [])} - {None}
    if kinds == {"audio_http"}:
        return "audio"
    if kinds == {"image_http"}:
        return "image"
    if kinds and kinds <= {"clip_http", "comfyui"}:
        return "clip"
    return "single"


def media_argv(mode: str, ctx: dict, template: str | None, extra: str | None) -> list[str]:
    # The scheduler owns round indices and endpoint/configuration identity. Extra flags
    # may select client timeouts, quality fixtures or cache policy, not replace the plan.
    extra_words = shlex.split(extra or "")
    protected = {"--base-url", "--fixtures", "--prompts", "--model", "--label", "--rounds", "--first-round",
                 "--extend", "--endpoint-id", "--device", "--hardware-name", "--gpu-name", "--run-dir",
                 "--model-revision", "--model-dtype", "--model-quant", "--dtype", "--quant", "--engine-version",
                 "--seed-base", "--seed-policy", "--seeds", "--sample-rate", "--channels", "--settings", "--warmup",
                 "--gpu-count"}
    if any(w.split("=", 1)[0] in protected for w in extra_words):
        raise OpError("media --serve-args cannot override scheduler identity, rounds, seeds or workload settings")
    words = shlex.split(template) if template else [sys.executable, "-m", "inferencekit", "bench", mode]
    if any("{" in w or "}" in w for w in words):
        raise OpError("audio/image --serve-cmd replaces the program only; scheduler arguments are always appended")
    argv = words + ["--base-url", ctx["base_url"], "--fixtures" if mode == "audio" else "--prompts", ctx["prompts"],
                    "--model", ctx["model"], "--label", ctx["label"], "--rounds", "1",
                    "--first-round", str(ctx["first_round"]), "--warmup", str(ctx["warmup"]),
                    "--seed-base", str(ctx["seed"]), "--device", ctx["device"]]
    if ctx.get("run_dir"):
        argv += ["--run-dir", ctx["run_dir"]]
    for key, flag in (("revision", "--model-revision"), ("engine_version", "--engine-version"),
                      ("extend", "--extend"), ("precision", "--model-dtype" if mode == "audio" else "--dtype"),
                      ("quant", "--model-quant" if mode == "audio" else "--quant")):
        if ctx.get(key):
            argv += [flag, str(ctx[key])]
    argv += ["--hardware-name" if mode == "audio" or ctx["device"] == "cpu" else "--gpu-name", ctx["hardware"]]
    if ctx["device"] == "gpu":
        argv += ["--gpu-count", str(ctx.get("gpu_count", 1))]
    if mode == "audio":
        argv += ["--endpoint-id", ctx["label"], "--sample-rate", str(ctx["sample_rate"]),
                 "--channels", str(ctx["channels"]), "--settings", json.dumps(ctx["settings"])]
    return argv + extra_words


def run_cell(spec: dict, k: int, position: int, cfg: dict) -> dict:
    mode, cpu = cfg["mode"], cfg["device"] == "cpu"
    cell = {"round": k, "position": position, "label": spec["label"], "kind": spec["kind"],
            "status": None, "t_start": time.time(), "log": f"logs/{spec['label']}.r{k}.log"}
    log = cfg["run_dir"] / cell["log"]
    env, cell["env"] = servers.server_env(spec["env"])
    cell["server_env"] = resources.runtime_env(env, cwd=spec.get("cwd"))
    if servers.port_in_use(spec["host"], spec["port"]):
        cell.update(status="port_busy", detail="server port already in use; refusing to measure another process",
                    t_end=time.time())
        return cell
    base = f"http://{spec['host']}:{spec['port']}"
    before = None if cpu else cfg["gpu_reader"]()
    server = monitor = None
    try:
        try:
            server = servers.launch(spec["launch"], env, log, cwd=spec.get("cwd"))
        except OSError as exc:
            cell.update(status="launch_failed", detail=ev.redact_text(str(exc))[:300])
            return cell
        if cpu:
            monitor = resources.ProcessMonitor(server.pgid, env, cwd=spec.get("cwd")).start()
        cell["ready"] = servers.wait_ready(server, base + spec["ready_check"], cfg["startup_timeout_s"], cfg["poll_s"])
        if cell["ready"]["status"] != "ready":
            cell["status"] = "startup_timeout" if cell["ready"]["status"] == "timeout" else "exited_early"
            return cell
        if monitor:
            monitor.settled()
        ctx = {**cfg["client"], "base_url": base, "label": spec["label"], "first_round": k - 1,
               "run_dir": str(cfg["run_dir"].resolve()),
               "engine_version": spec.get("version"), "extend": cfg["last_ev"].get(spec["label"])}
        child_env = dict(os.environ)
        child_env.pop("INFERENCEKIT_GPU_LOCK_HELD" if cpu else "INFERENCEKIT_CPU_LOCK_HELD", None)
        root = str(ev.default_root().resolve())
        child_env.update(INFERKIT_RUNS_DIR=root, INFERENCEKIT_RUNS_DIR=root, INFERENCEKIT_OUTPUT="json")
        child_env[f"INFERENCEKIT_{'CPU' if cpu else 'GPU'}_LOCK_HELD"] = cfg["lock"]
        result = proc.run(media_argv(mode, ctx, cfg["serve_cmd"], cfg["serve_args"]),
                          timeout=cfg["serve_timeout_s"], env=child_env)
        record = rr.parse_serve_output(result.stdout)
        cell[mode] = {"returncode": result.returncode, "evidence_id": (record or {}).get("evidence_id"),
                      "stderr_tail": ev.redact_text(result.stderr[-1500:]) if result.returncode else None}
        cell["metrics"] = {}
        if record and result.returncode == 0 and record.get("kind") == f"{mode}_benchmark":
            rows = [r for r in record.get("data", {}).get("rounds", []) if r.get("index") == k - 1]
            if len(rows) == 1 and not ev.integrity_problem(record):
                row = rows[0]
                mets = {name: {"value": row[name], "unit": unit, "stat": "per_round", "n": cfg["n"]}
                        for name, unit in METRICS[mode].items() if round_evidence.finite(row.get(name))}
                mets["failed_requests"] = {"value": row.get("n_failed", 0), "unit": "requests", "n": cfg["n"]}
                if all(round_evidence.finite(row.get(name)) for name in METRICS[mode]) or (
                        round_evidence.finite(row.get("n_failed")) and row["n_failed"] > 0):
                    cell["metrics"] = {"all": mets}
                cfg["last_ev"][spec["label"]] = record["evidence_id"]
                cell["claim_valid"] = record.get("data", {}).get("claim_validity", {}).get("valid")
        cell["status"], detail = rr.cell_status(result.returncode, record, cell["metrics"])
        if cell.get("claim_valid") is not True and cell["status"] == "ok":
            cell["status"] = "claim_invalid"
        if detail:
            cell["detail"] = detail
        return cell
    finally:
        if server is not None:
            try:
                cell["stop"] = servers.stop(server, cfg["stop_timeout_s"])
            finally:
                if monitor is not None:
                    rr.attach_resources(cell, monitor.stop())
            if not cell["stop"]["group_gone"]:
                cell["status"] = "stop_failed"
            if not cpu:
                cell["gpu_memory"] = servers.wait_gpu_memory(before, cfg["gpu_mem_tolerance_mib"],
                                                              cfg["gpu_mem_wait_s"], reader=cfg["gpu_reader"])
        cell["t_end"] = time.time()


def run_media_rounds(manifest_path, prompts, mode, *, rounds=None, engines=None, run_dir=None,
                     device="auto", cpu_lock=None, gpu_lock=None, startup_timeout_s=1200,
                     stop_timeout_s=30, serve_timeout_s=7200, warmup=1, serve_args=None, serve_cmd=None,
                     gpu_mem_tolerance_mib=512, gpu_mem_wait_s=60, gpu_reader=None, poll_s=0.5):
    manifest = rr.load_manifest(manifest_path)
    device = round_evidence.resource_mode(manifest, device)
    specs = rr.engine_specs(manifest, engines)
    if mode not in METRICS or not specs:
        raise OpError("media rounds requires audio/image mode and at least one engine")
    rounds = rr.default_rounds(len(specs)) if rounds is None else rounds
    if not 1 <= rounds <= 10000 or not 0 <= warmup <= 100:
        raise OpError("rounds must be 1..10000 and warmup 0..100")
    if any(not round_evidence.finite(x) or x <= 0 for x in (startup_timeout_s, serve_timeout_s, poll_s)) or \
            any(not round_evidence.finite(x) or x < 0 for x in (stop_timeout_s, gpu_mem_wait_s, gpu_mem_tolerance_mib)):
        raise OpError("timeouts and polling must be finite positive values; stop/memory waits may be zero")
    raw = {e.get("label", e.get("name")): e for e in manifest["engines"]}
    for spec in specs:
        source = raw[spec["label"]]
        if source.get("endpoint_kind", f"{mode}_http") != f"{mode}_http":
            raise OpError(f"{mode} mode needs {mode}_http endpoints")
        if not source.get("ready_check"):
            spec["ready_check"] = "/health"
    if mode == "audio":
        from inferencekit.ops.bench_audio import load_fixtures
        fixtures, _ = load_fixtures(prompts)
    else:
        from inferencekit.ops.bench_image import load_image_set
        fixtures = load_image_set(prompts)
    workload, model, hardware = (manifest.get(k) or {} for k in ("workload", "model", "hardware"))
    cpu = cpu_identity() if device == "cpu" else None
    hardware_name = cpu.get("model") if cpu else hardware.get("gpu")
    if not hardware_name or not model.get("id"):
        raise OpError("media rounds needs model.id and hardware identity")
    if mode == "image" and workload.get("settings"):
        raise OpError("image settings belong in each image fixture's settings object")
    client = {"prompts": str(Path(prompts).resolve()), "model": model["id"], "revision": model.get("revision"),
              "precision": model.get("precision"), "quant": workload.get("quant"), "hardware": hardware_name,
              "gpu_count": hardware.get("count", 1),
              "device": device, "seed": workload.get("seed", 0), "warmup": warmup,
              "sample_rate": workload.get("sample_rate", 44100), "channels": workload.get("channels", 1),
              "settings": workload.get("settings", {})}
    # Validate all scheduler-owned arguments before launching any server.
    media_argv(mode, {**client, "base_url": "http://127.0.0.1:1", "label": "check", "first_round": 0},
               serve_cmd, serve_args)
    rdir = Path(run_dir) if run_dir else Path(manifest_path).resolve().parent
    out = rdir / f"{mode}_rounds"
    out.mkdir(parents=True, exist_ok=True)
    labels = [s["label"] for s in specs]
    plan = {"mode": mode, "resource_mode": device, "rounds": rounds, "design": rr.round_orders(labels, rounds),
            "engines": [{k: s.get(k) for k in ("label", "name", "kind", "version", "role")} for s in specs],
            "prompts": {"name": Path(prompts).name, "sha256": file_sha256(Path(prompts))[0], "n": len(fixtures)},
            "manifest_sha256": file_sha256(Path(manifest_path))[0], "cpu": cpu}
    cfg = {"mode": mode, "device": device, "run_dir": rdir, "client": client, "last_ev": {}, "n": len(fixtures),
           "startup_timeout_s": startup_timeout_s, "stop_timeout_s": stop_timeout_s,
           "serve_timeout_s": serve_timeout_s, "poll_s": poll_s, "serve_cmd": serve_cmd, "serve_args": serve_args,
           "gpu_reader": gpu_reader or servers.gpu_memory_used_mib, "gpu_mem_wait_s": gpu_mem_wait_s,
           "gpu_mem_tolerance_mib": gpu_mem_tolerance_mib}
    lock = (servers.gpu_lock(cpu_lock, owner=f"inferencekit bench rounds ({mode})", resource="CPU")
            if device == "cpu" else servers.gpu_lock(gpu_lock, owner=f"inferencekit bench rounds ({mode})"))
    cells, failed, cleanup_failed = [], set(), False
    with lock as held:
        cfg["lock"] = held["path"]
        plan[f"{device}_lock"] = held["path"]
        (out / "plan.json").write_text(json.dumps(ev.redact(plan), indent=2) + "\n")
        by_label = {s["label"]: s for s in specs}
        for k, order in enumerate(plan["design"]["orders"], 1):
            for position, label in enumerate(order):
                if cleanup_failed:
                    cell = {"round": k, "position": position, "label": label, "status": "skipped_after_cleanup_failure",
                            "detail": "previous server cleanup failed; no further measurement can be isolated"}
                elif label in failed:
                    cell = {"round": k, "position": position, "label": label, "status": "skipped_after_failure",
                            "detail": "previous cell failed; preserving scheduled indices instead of renumbering"}
                else:
                    cell = run_cell(by_label[label], k, position, cfg)
                    if cell["status"] != "ok":
                        failed.add(label)
                    memory = cell.get("gpu_memory") or {}
                    cleanup_failed = (cell.get("stop", {}).get("group_gone") is False or
                                      memory.get("checked") is True and memory.get("returned") is False)
                cells.append(cell)
                (out / f"{label}.r{k}.json").write_text(json.dumps(ev.redact(cell), indent=2) + "\n")
    plan[f"{mode}_evidence_final"] = round_evidence.finalize_arm_records(cfg["last_ev"], cells, plan, cpu=cpu)
    store = ev.Store()
    evidence_dir = rdir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    for reference in plan[f"{mode}_evidence_final"].values():
        # Standalone per-arm bundles retain hash-matching request artifacts; shared
        # outputs/metrics support review build, whose top-level evidence scan needs JSON copies.
        shutil.copytree(store.run_dir(reference), rdir / "artifacts" / reference, dirs_exist_ok=True)
        (evidence_dir / f"{reference}.json").write_text(json.dumps(store.load(reference), indent=2) + "\n")
    summary = rr.summarize(cells, labels, expected_rounds=rounds)
    summary["definitions"]["median"] = "Median of client-timed per-round values; memory is sampled server process-group RSS."
    lines = [f"# {mode.title()} benchmark", "", "| Engine | Metric | Median | 95% CI | Rounds |",
             "|---|---|---:|---|---:|"]
    for row in summary["rows"]:
        lines.append(f"| {row['engine']} | {row['metric']} | {rr._fmt(row['median'])} {row['unit']} | "
                     f"{rr._fmt(row['ci95'][0])}–{rr._fmt(row['ci95'][1])} | {row['n']}/{rounds} |")
    for label in labels:
        missing = [c["round"] for c in cells if c["label"] == label and c["status"] != "ok"]
        if missing:
            lines += ["", f"{label}: incomplete rounds {missing}; excluded from ranking."]
    lines += ["", "Timing does not establish output quality or sustained reliability."]
    markdown = "\n".join(lines) + "\n"
    (out / "plan.json").write_text(json.dumps(ev.redact(plan), indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(ev.redact({"plan": plan, "summary": summary}), indent=2) + "\n")
    (out / "summary.md").write_text(markdown)
    return {"plan": plan, "cells": cells, "summary": summary, "run_dir": rdir, "manifest": manifest,
            "markdown": markdown, "resolved_config": {}}
