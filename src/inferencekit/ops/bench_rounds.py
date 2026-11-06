"""bench rounds: every engine of a run manifest, R balanced rounds, one server at a time, via bench serve.

For each round (order from a Williams design, see ``serve/rounds.py``) and each engine:
launch the server from the manifest (argv, env, port) under a clean environment and in
its own process group, wait for ``ready_check``, warm up, run ``inferencekit bench serve``
against it as a subprocess, scrape ``/metrics`` before and after, stop the whole process
group, and check GPU memory is back to its pre-launch level. The run holds an exclusive
GPU lock file throughout.

The server log of every cell is kept (``logs/<label>.r<k>.log``) and parsed for the
configuration the engine reports it actually used (``resolved_config/<label>.r<k>.json``):
status RESOLVED / PARTIAL / SELF_REPORTED / UNPROVEN, hard-gate checks (CUDA graphs,
attention fallback, GPU offload), warning lines and unknown settings.

Output: one evidence record (kind ``serve_benchmark``): per engine x bucket x metric, the
median over rounds with a bootstrap 95% CI, and the ratio to the best engine with a paired
bootstrap 95% CI [derived]; plus ``rounds/summary.md`` in the run directory. Single mode's
medians are [observed] (client-side timings of the round's requests); load mode's come
from bench serve's capacity records (N* and the values at N*) and are [derived].

``--extra-body`` (or the manifest's ``workload.extra_body``) times every engine with the same
extra request fields, e.g. sampling settings. ``--ci-target`` turns ``--rounds`` into a maximum
and stops at the first whole Williams period, from ``--min-rounds`` on, where every engine's
decode_tok_s ratio CI is narrow enough; the rule is recorded and named in the limitations.

``--mode clip`` (video) hands the run to ``serve/clip_rounds.py``: the same rounds, launches and lock,
with ``bench clip`` per cell instead of bench serve and ``--prompts`` as the clip set; --warmup counts
untimed clips per distinct request shape (use 1: each is a whole clip). Adaptive stop and the text
request settings (--ref-e2e-ms, --chat-template-kwargs, --extra-body) are refused there.
"""

from __future__ import annotations

import json
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.gpuinfo import gpu_identity
from inferencekit.ops.doctor import gpu_processes
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve import rounds as rr
from inferencekit.serve import media_rounds, round_evidence

KIND = "serve_benchmark"


def _limitations(result: dict) -> tuple[list[str], list[str]]:
    plan, cells, resolved = result["plan"], result["cells"], result["resolved_config"]
    labels = [e["label"] for e in plan["engines"]]
    lim, missing = [], []
    d = plan["design"]
    if not d["balanced"]:
        lim.append(f"order not balanced: {plan['rounds']} rounds is not a multiple of the Williams period "
                   f"{d['period']} for {len(labels)} engines (position and carry-over effects not fully cancelled)")
    if plan["rounds"] < 2:
        lim.append("fewer than 2 rounds: no confidence intervals")
    failures = result["summary"].get("failures") or []
    per_engine_buckets = {}
    for f in failures:
        per_engine_buckets.setdefault(f["engine"], []).append(f["bucket"])
    for f in failures:
        if f["bucket"] == "all" and len(per_engine_buckets[f["engine"]]) > 1:
            continue  # the per-bucket lines already say it
        of = f" of {f['requests']:g}" if isinstance(f.get("requests"), (int, float)) else ""
        lim.append(f"{f['engine']}/{f['bucket']}: {f['failed_requests']:g}{of} requests failed over {f['rounds']} "
                   "round(s); this engine is not ranked there (latencies cover only answered requests)")
    sched = plan.get("load_schedule")
    if sched is not None and not sched.get("shared"):
        lim.append("load schedules differ per engine: concurrency levels were turned into arrival rates with "
                   "each engine's own warm-up e2e, so the same level N offered different loads; pass "
                   "--ref-e2e-ms, or --rates in --serve-args")
    by_status: dict[str, list[str]] = {}
    for c in cells:
        if c.get("status") != "ok":
            by_status.setdefault(c.get("status") or "unknown", []).append(f"{c['label']} r{c['round']}")
    for status, where in sorted(by_status.items()):
        lim.append(f"{len(where)} cell(s) {status}: {', '.join(where[:8])}")
    for label in labels:
        if not any(c["label"] == label and c.get("status") == "ok" for c in cells):
            missing.append(f"{label}: no successful round")
    not_back = [f"{c['label']} r{c['round']}" for c in cells
                if (c.get("gpu_memory") or {}).get("checked") and not c["gpu_memory"].get("returned")]
    if not_back:
        lim.append(f"GPU memory did not return to its pre-launch level after: {', '.join(not_back[:8])} "
                   "(a leftover process may have contended with the next engine)")
    if any(c.get("gpu_memory") and not c["gpu_memory"].get("checked") for c in cells):
        lim.append("GPU memory release not checked (nvidia-smi not available)")
    killed = [f"{c['label']} r{c['round']}" for c in cells if (c.get("stop") or {}).get("method") == "sigkill"]
    if killed:
        lim.append(f"server ignored SIGTERM and was killed: {', '.join(killed[:8])}")
    alive = [f"{c['label']} r{c['round']}" for c in cells if c.get("stop") and not c["stop"].get("group_gone")]
    if alive:
        lim.append(f"processes of the server survived SIGKILL: {', '.join(alive[:8])}")
    for label, info in resolved.items():
        if info["status"] != "RESOLVED":
            lim.append(f"{label}: resolved configuration {info['status']} (raw log kept under logs/)")
        if info.get("self_reported"):
            used = "it is the only source" if info["status"] == "SELF_REPORTED" else \
                "ignored for status and hard gates (the engine's own log lines decide)"
            lim.append(f"{label}: settings self-reported by the engine process, not verified "
                       f"({', '.join(info['self_reported'][:8])}); {used}")
        for v in info["violations"]:
            lim.append(f"HARD GATE: {label}: {v}")
        if len(info.get("attention_backends") or []) > 1:
            lim.append(f"{label}: attention backend differed between rounds: {info['attention_backends']}")
        if info["unknown_settings"]:
            lim.append(f"{label}: engine reported unknown settings: {', '.join(info['unknown_settings'][:6])}")
        if info["warnings"]:
            lim.append(f"{label}: {info['warnings']} distinct warning line(s) in the server log "
                       "(resolved_config/*.json)")
    secret = sorted({n for c in cells for n in (c.get("env") or {}).get("secret_values_in_manifest", [])})
    if secret:
        lim.append(f"manifest env writes secret-looking variable(s) {secret} as values; use null to inherit them "
                   "from the caller instead")
    absent = sorted({n for c in cells for n in (c.get("env") or {}).get("missing", [])})
    if absent:
        lim.append(f"manifest env asks to inherit {absent}, but they were not set")
    for w in (plan.get("extra_body") or {}).get("warnings") or []:
        lim.append(f"extra body sent to every engine: {w}")  # the warning names where the body came from
    adaptive = plan.get("adaptive")
    if adaptive:
        last = (adaptive.get("checks") or [{}])[-1].get("widths") or {}
        widths = ", ".join(f"{lb} {w['width']:.3g}" for lb, w in last.items() if w.get("width") is not None)
        if plan.get("stopped_early"):
            lim.append(f"adaptive stop: stopped after round {plan['rounds']} of at most {adaptive['max_rounds']} "
                       f"because every engine's CI half-width was <= {adaptive['ci_target']:g} ({widths}); stopping "
                       "once a CI looks narrow is optional stopping, so these 95% CIs are slightly too narrow: fix "
                       "the rule before the run (plan.adaptive.rule)")
        else:
            lim.append(f"adaptive stop: the CI target {adaptive['ci_target']:g} was "
                       f"{'met' if adaptive.get('met') else 'NOT met'} by round {plan['rounds']}, the maximum "
                       f"({widths or 'no CI'})")
    return lim, missing


@operation("bench_rounds", ("bench", "rounds"), "Run manifest engines for balanced fresh-process rounds: "
           "text, audio, image or video clients, CPU/GPU locking, cleanup and per-round evidence with CIs.",
           [Arg("manifest", "string", "Run manifest (runs/<run-id>/manifest.json); its 'engines' are launched.",
                required=True),
            Arg("prompts", "string", "Shared JSONL prompt, speech, image or clip fixture set.", required=True),
            Arg("rounds", "integer", "Rounds; each runs every engine once. Default: the smallest multiple of the "
                "Williams period that is at least 3, so the order is balanced (2 engines: 4, 3: 6, 4: 4, 5: 10)."),
            Arg("mode", "string", "Client: single/load text, audio, image or clip. Auto reads manifest modes, "
                "modality, intake modality or endpoint kind; ambiguous modes require a selection.",
                default="auto", choices=("auto", "single", "load", "clip", "audio", "image")),
            Arg("device", "string", "Hardware route; auto reads hardware.backend from the manifest.", default="auto",
                choices=("auto", "cpu", "gpu")),
            Arg("cpu_lock", "string", "CPU-exclusive advisory lock (default $INFERENCEKIT_CPU_LOCK or ~/.cache/inferencekit/cpu.lock)."),
            Arg("engines", "string", "Comma-separated engine labels to run (default: every engine in the manifest)."),
            Arg("run_dir", "string", "Run directory for logs and results (default: the manifest's directory)."),
            Arg("gpu_lock", "string", "Lock file held for the whole run (default: $INFERENCEKIT_GPU_LOCK or "
                "~/.cache/inferencekit/gpu.lock)."),
            Arg("startup_timeout_s", "number", "Seconds to wait for an engine's ready_check to answer 200.",
                default=1200),
            Arg("stop_timeout_s", "number", "Seconds between SIGTERM and SIGKILL of a server's process group.",
                default=30),
            Arg("serve_timeout_s", "number", "Seconds before one bench serve call (--mode clip: one bench clip call) "
                "is abandoned.", default=7200),
            Arg("warmup", "integer", "Untimed warm-up requests per server before bench serve (the set's first "
                "prompts, each salted with a unique prefix tag so they never seed the prefix cache). --mode clip: "
                "untimed clips per distinct request shape (bench clip --warmup); use 1, each is a whole clip.",
                default=2),
            Arg("ref_e2e_ms", "number", "Load mode: unloaded e2e latency (ms) passed as --ref-e2e-ms to every bench "
                "serve call, so every engine gets the same arrival schedule. Default: measured by the first load "
                "cell's warm-up and passed to every later cell (unless --serve-args has --rates, --ref-e2e-ms or "
                "--closed-loop)."),
            Arg("serve_args", "string", "Extra arguments appended to every bench serve call (e.g. the load grid; "
                "--mode clip: to every bench clip call, e.g. '--seed-base 7')."),
            Arg("serve_cmd", "string", "Command that runs bench serve, default '<python> -m inferencekit bench "
                "serve' (or $INFERENCEKIT_BENCH_SERVE_CMD); {base_url} {prompts} {mode} {label} {round} {run_dir} "
                "{model} placeholders are filled in (--mode clip: the bench clip command, or "
                "$INFERENCEKIT_BENCH_CLIP_CMD; placeholders in serve/clip_rounds.py). Stored only as a digest.",
                sensitive=True),
            Arg("gpu_mem_tolerance_mib", "integer", "GPU memory above the pre-launch level still counted as "
                "released.", default=512),
            Arg("chat_template_kwargs", "string", "JSON object passed to every bench serve call (and the warm-up) "
                "as --chat-template-kwargs, e.g. '{\"enable_thinking\": false}'. Default: the manifest's "
                "workload.chat_template_kwargs."),
            Arg("extra_body", "string", "JSON object passed to every bench serve call as --extra-body (merged last "
                "into every timed request), e.g. sampling settings '{\"temperature\": 1.0, \"top_k\": 64, "
                "\"top_p\": 0.95, \"seed\": 1234}'. Default: the manifest's workload.extra_body."),
            Arg("ci_target", "number", "Adaptive stop (single mode): --rounds becomes the maximum (default 12, "
                "rounded up to whole Williams periods), and the run stops after the first whole period, from "
                "--min-rounds on, where every engine's decode_tok_s [all] ratio to the best engine has a 95% CI "
                "half-width at or below this fraction of the ratio (e.g. 0.01 = +/-1%). Default: off, every round "
                "runs."),
            Arg("min_rounds", "integer", "Adaptive stop: never stop before this many rounds (default: the default "
                "--rounds, the smallest whole number of Williams periods that is at least 3). Needs --ci-target.")])
def bench_rounds(call, manifest, prompts, rounds, mode, engines, run_dir, gpu_lock, startup_timeout_s,
                 stop_timeout_s, serve_timeout_s, warmup, ref_e2e_ms, serve_args, serve_cmd, gpu_mem_tolerance_mib,
                 chat_template_kwargs=None, extra_body=None, ci_target=None, min_rounds=None, device="auto", cpu_lock=None):
    declared = rr.load_manifest(manifest)
    device = round_evidence.resource_mode(declared, device)
    mode = media_rounds.mode_for(declared, mode, manifest)
    if device == "cpu" and mode == "load":
        raise OpError("CPU load mode is not supported yet: capacity metrics use GPU denominators; use --mode single")
    if mode in ("clip", "audio", "image"):
        unsupported = [flag for flag, value in (("--ci-target", ci_target), ("--min-rounds", min_rounds),
                                                ("--ref-e2e-ms", ref_e2e_ms),
                                                ("--chat-template-kwargs", chat_template_kwargs),
                                                ("--extra-body", extra_body)) if value is not None]
        if unsupported:
            raise OpError(f"{', '.join(unsupported)} not supported with --mode {mode} (media clients time complete outputs; "
                          "adaptive stop and text request settings do not apply)")
    if mode == "clip":
        from inferencekit.serve.clip_rounds import bench_rounds_clip
        return bench_rounds_clip(call, manifest=manifest, clips=prompts, rounds=rounds, engines=engines,
                                 run_dir=run_dir, gpu_lock=gpu_lock, startup_timeout_s=startup_timeout_s,
                                 stop_timeout_s=stop_timeout_s, clip_timeout_s=serve_timeout_s, warmup=warmup,
                                 clip_args=serve_args, clip_cmd=serve_cmd, gpu_mem_tolerance_mib=gpu_mem_tolerance_mib,
                                 resource_mode=device, cpu_lock_path=cpu_lock)
    others = [] if device == "cpu" else gpu_processes()
    media = mode in ("audio", "image")
    if media:
        result = media_rounds.run_media_rounds(manifest, prompts, mode, rounds=rounds, engines=engines,
                           run_dir=run_dir, device=device, cpu_lock=cpu_lock, gpu_lock=gpu_lock,
                           startup_timeout_s=startup_timeout_s, stop_timeout_s=stop_timeout_s,
                           serve_timeout_s=serve_timeout_s, warmup=warmup, serve_cmd=serve_cmd,
                           serve_args=serve_args, gpu_mem_tolerance_mib=gpu_mem_tolerance_mib)
    else:
        result = rr.run_rounds(manifest, prompts, rounds, mode, engines=engines, run_dir=run_dir,
                           gpu_lock_path=gpu_lock, startup_timeout_s=startup_timeout_s,
                           stop_timeout_s=stop_timeout_s, serve_timeout_s=serve_timeout_s, warmup=warmup,
                           serve_cmd=serve_cmd, serve_args=serve_args, gpu_mem_tolerance_mib=gpu_mem_tolerance_mib,
                           ref_e2e_ms=ref_e2e_ms, chat_template_kwargs=chat_template_kwargs, extra_body=extra_body,
                           ci_target=ci_target, min_rounds=min_rounds, resource_mode=device, cpu_lock_path=cpu_lock)
    plan, summary, rdir = result["plan"], result["summary"], Path(result["run_dir"])
    labels = [e["label"] for e in plan["engines"]]
    limitations, missing = _limitations(result)
    if others:
        limitations.append(f"{len(others)} other compute process(es) were on the GPU when the run started: "
                           "timings may be contended")
    metrics = []
    for r in summary["rows"]:
        subject = f"{r['engine']}/{r['bucket']}"
        metrics.append(ev.metric(r["metric"], r["median"], r["unit"] or "unknown", "median_over_rounds", n=r["n"],
                                 ci95=r["ci95"], subject=subject))
        if r["ratio_to_best"] is not None:
            metrics.append(ev.metric(f"{r['metric']}_ratio_to_best", r["ratio_to_best"], "x", "ratio_of_medians",
                                     n=r["n_pairs"], ci95=r["ratio_ci95"], subject=subject))
    m = result["manifest"].get("model") or {}
    flags = {"engines": labels}
    if plan.get("extra_body"):
        flags["extra_body"] = plan["extra_body"]["value"]  # same body for every engine: part of what was measured
    ident = ev.identity(gpu={} if device == "cpu" else gpu_identity(), cpu=plan.get("cpu"),
                        engine={"name": "+".join(labels), "flags": flags},
                        model={"id": m.get("id"), "revision": m.get("revision"), "dtype": m.get("precision")},
                        workload={"mode": "single" if media else mode, "endpoint_kind": f"{mode}_http" if media else "openai_chat",
                                  "prompt_set": {"name": plan["prompts"]["name"],
                                                 "sha256": plan["prompts"]["sha256"]}})
    n_ok = sum(1 for c in result["cells"] if c.get("status") == "ok")
    load = mode == "load"
    cls = "derived" if load else "observed"
    summary_line = (f"bench rounds ({mode}): {len(labels)} engines x {plan['rounds']} rounds, "
                    f"{n_ok}/{len(result['cells'])} "
                    f"cells ok, Williams order {'balanced' if plan['design']['balanced'] else 'NOT balanced'}")
    adaptive = plan.get("adaptive")
    if adaptive:
        target = f"CI target {adaptive['ci_target']:g}"
        summary_line += (f" (adaptive: stopped early, {target} met)" if plan.get("stopped_early") else
                         f" (adaptive: {target} {'met' if adaptive.get('met') else 'NOT met'} at the maximum)")
    if plan.get("extra_body"):
        summary_line += "; extra body " + json.dumps(plan["extra_body"]["value"], sort_keys=True)[:120]
    heads = [r for r in summary["rows"] if r["engine"] == r["best"]]
    if heads:
        h = heads[0]
        summary_line += f"; best {h['metric']} [{h['bucket']}]: {h['engine']} {h['median']:.4g} [{cls}]"
    output_dir = f"{mode}_rounds" if media else "rounds"
    artifacts = [ev.artifact(rdir / output_dir / name, rdir, role)
                 for name, role in (("summary.md", "summary"), ("summary.json", "summary"), ("plan.json", "plan"))
                 if (rdir / output_dir / name).is_file()]
    for c in result["cells"]:
        log = rdir / c.get("log", "")
        if log.is_file():
            artifacts.append(ev.artifact(log, rdir, "server_log"))
        rc = (c.get("resolved_config") or {}).get("file")
        if rc and (rdir / rc).is_file():
            artifacts.append(ev.artifact(rdir / rc, rdir, "resolved_config"))
    kind = f"{mode}_benchmark" if media else KIND
    v2 = kind in ev.KINDS
    if not v2:
        limitations.append("evidence v2 (kind serve_benchmark) not available in this build: record written "
                           "without schema validation")
    cells = [{k: c.get(k) for k in ("round", "position", "label", "kind", "status", "served_model", "ready",
                                     "warmup", "serve", "stop", "gpu_memory", "resolved_config", "env", "log",
                                     "metrics_source", "load_schedule", "t_start", "t_end", "detail", "resources", "server_env", "audio", "image")}
             for c in result["cells"]]
    if load:
        claim_classes = {"<metric>": "derived (median over rounds of bench serve's capacity record: N* and the "
                                     "values at N*, derived from its observed load levels)",
                         "<metric>_ratio_to_best": "derived"}
    else:
        claim_classes = {"<metric>": "observed (median over rounds of bench serve's client-side timings)",
                         "<metric>_ratio_to_best": "derived"}
    if media:
        claim_classes = {"<latency>": "observed client timer, except simulated first-audio onset (derived)",
                         "<rate>, rtf, <metric>_ratio_to_best": "derived",
                         "peak_rss_mib": "observed sampled server process-group RSS"}
    record = ev.build(call, kind=kind if v2 else "benchmark", claim_class=cls, summary=summary_line,
                      identity_=ident, metrics=metrics, artifacts=artifacts, missing=missing,
                      limitations=limitations,
                      data={"run_dir": str(rdir), "plan": plan, "cells": cells, "rows": summary["rows"],
                            "definitions": summary["definitions"],
                            "metric_claim_classes": claim_classes, "failures": summary.get("failures") or [],
                            "resolved_config": result["resolved_config"],
                            "bench_serve_evidence": [c["serve"]["evidence_id"] for c in result["cells"]
                                                     if (c.get("serve") or {}).get("evidence_id")],
                            **({f"bench_{mode}_evidence": list(plan.get(f"{mode}_evidence_final", {}).values())}
                               if media else {}),
                            "gpu_processes_at_start": None if device == "cpu" else len(others), "markdown": result["markdown"]})
    store = ev.Store()
    if v2:
        rec = store.save(record)
    else:
        record["kind"] = KIND
        rec = ev.finalize(record)
        target = store.run_dir(rec["evidence_id"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "evidence.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    copy = rdir / "evidence" / f"{rec['evidence_id']}.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return rec
