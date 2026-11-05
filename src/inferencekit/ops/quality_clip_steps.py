"""quality clip-steps: is each denoising step of an engine's clip what the reference computes from the same latent?

The final-clip comparison (quality clip) needs dozens of requests on a video model, because tiny exact-kernel
differences grow over the steps (on LTX-2.5 an exact other attention kernel already moves a clip to ~25 dB
PSNR). This check removes the growth: the engine dumps every step of every clip it makes (the contract in
engines/clip/stepdump.py: x_t, the guided output its sampler took, x_{t-1}, the final latent, under
$CLIP_STEP_DUMP_DIR/<clip sha256>.safetensors), and the BF16 reference recomputes each step from the engine's
own x_t (teacher forcing; its own text conditioning from the prompt). So a step's error is that step's alone.

  1. After the clips are made, ``--requests`` of the engine's clips (default 2; bench clip outputs with their
     request bodies) are drawn with ``--audit-seed`` (drawn and recorded when not given). Every step is audited:
     a cache or a skipped guidance pass on any step shows there.
  2. The replay worker (quality/step_replay.py) runs in the reference pipeline's Python (``--python``) and loads
     the reference once (``--pipeline``, ``--pipeline-model``, ``--pipeline-arg``: the reference's launch
     arguments, as the clip server takes them). Per step and latent stream: e = |out_eng - out_ref| / |out_ref|
     and the cosine; then the floor f, the same replay with ``--alt-attention`` (an exact but different
     kernel) on the same inputs: f = |out_alt - out_ref| / |out_ref|.
  3. PASS needs, on every audited request: e <= factor * f + slack at every step and stream (``--factor``
     2, ``--slack`` calibrated on the GPU, below), and every guard:
       dump      present for the clip, named by its sha256, its request equal to bench clip's request body
       layout    streams, shapes and keys of the request's latents; N steps = the steps asked
       schedule  the dump's sigmas equal the reference's (|d| <= ``--sigma-tol``)
       dtypes    each dumped tensor has the reference's dtype for it (no lower-precision latents or outputs)
       chain     x_{t-1} = the reference scheduler's update of (x_t, out_eng) within ``--chain-tol`` (relative
                 L2), each step's x_{t-1} is the next step's x_t and the final latent the last x_{t-1}
                 (exactly), and the first x_t is the reference's own initial noise for the request
       decode    the reference's decoder makes, from the dump's final latent, the clip that was delivered: same
                 frame count and size, mean PSNR >= ``--decode-psnr-db``
     A dump that fails a guard is not judged step by step (FAIL). An alternative kernel identical to the
     reference's (f = 0 everywhere) cannot calibrate the ratio: UNPROVEN unless the engine is exact (e = 0).
The final-clip PSNR of quality clip stays a loose gross-failure gate beside this one.

Calibration (one RTX PRO 6000, LTX-2.5 t2v 768x512x121 with audio, 30 steps, 2 audited clips, 16 Oct 2026):
  * floor (the reference's flash SDPA vs ``_native_efficient`` on the same x_t): 1.1e-2 to 1.5e-1 per step and
    stream, rising towards the last step [measured]; the guidance combination (CFG 3, STG 1, modality 3, rescale
    0.7) amplifies each pass's kernel noise.
  * an exact engine with another kernel (``_native_cudnn``): e / floor 0.81-1.12 over 120 step outputs, PASS
    [measured]. The slack 1e-3 adds at most 5% to the smallest limit (2 x 1.08e-2) [derived].
  * decode check: honest clips decode at 35.1-39.3 dB against the delivered mp4 (H.264 loss) [measured]; clips
    of another exact kernel's trajectory sit at ~25 dB [measured earlier, quality clip floor]: minimum 30 dB.
  * one reused step (step 15 takes step 14's output): FAIL, e / floor 2.8 and 4.1 [measured]. STG dropped on the
    last 3 steps: NOT caught at factor 2 (e / floor at most 1.65: the STG term there is ~9% of the output, the
    floor 6-15%) [measured].
  * time: about 8 min 15 s for 2 audited clips, model load 26 s included [measured].
"""

from __future__ import annotations

import json
import os
import random
import secrets
import sys
from pathlib import Path

import inferencekit
from inferencekit import evidence as ev
from inferencekit.canonical import file_sha256
from inferencekit.ops.quality import _kind
from inferencekit.ops.quality_clip import clip_files, index_mismatch, requests_of
from inferencekit.proc import run
from inferencekit.registry import Arg, OpError, operation

WORKER = Path(__file__).resolve().parents[1] / "quality" / "step_replay.py"
MARK = "INFERKIT_RESULT "
FORMAT = "inferencekit.quality.clip_steps.v1"
SAFETENSORS_DTYPES = {"f64": "float64", "f32": "float32", "bf16": "bfloat16", "f16": "float16"}
DEFAULT_FACTOR = 2.0
DEFAULT_SLACK = 1e-3
DEFAULT_CHAIN_TOL = 1e-5
DEFAULT_SIGMA_TOL = 1e-4
DEFAULT_DECODE_PSNR_DB = 30.0
DEFAULTS_SOURCE = ("RTX PRO 6000, LTX-2.5 t2v 768x512x121, 30 steps, 16 Oct 2026: per-step floor 1.1e-2 to 1.5e-1, "
                   "exact cuDNN engine max e/floor 1.12, honest decode 35.1-39.3 dB, alt-kernel clips ~25 dB")


def _pipeline_args(items) -> dict:
    out = {}
    for text in items or []:
        key, sep, value = str(text).partition("=")
        if not sep or not key:
            raise OpError(f"--pipeline-arg must be KEY=VALUE, got {text!r}")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def _shim(staging: Path) -> Path:
    """A directory holding only the inferencekit package (a link), for the worker's PYTHONPATH: the harness's
    site-packages must not shadow the pipeline venv's packages."""
    shim = staging / "shim"
    shim.mkdir(parents=True, exist_ok=True)
    link = shim / "inferencekit"
    if not link.exists():
        link.symlink_to(Path(inferencekit.__file__).resolve().parent, target_is_directory=True)
    return shim


def run_replay(python: str | None, cfg: dict, staging: Path, timeout_s: float) -> dict:
    work = staging / "worker"
    work.mkdir(parents=True, exist_ok=True)
    cfg_path, out_path, log_path = work / "replay.config.json", work / "replay.results.json", work / "replay.log"
    cfg_path.write_text(json.dumps({**cfg, "out": str(out_path)}))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_shim(staging))] + [p for p in env.get("PYTHONPATH", "").split(
        os.pathsep) if p])
    try:
        r = run([python or sys.executable, str(WORKER), str(cfg_path)], timeout=timeout_s, env=env,
                stdout_path=work / "replay.stdout.txt", stderr_path=log_path)
    finally:
        cfg_path.unlink(missing_ok=True)
        (work / "replay.stdout.txt").unlink(missing_ok=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(MARK)]
    head = json.loads(lines[-1][len(MARK):]) if lines else None
    if not head or not head.get("ok") or not out_path.is_file():
        tail = log_path.read_text(errors="replace")[-1500:] if log_path.is_file() else r.stderr[-1500:]
        why = (head or {}).get("error") or f"exit {r.returncode}"
        raise OpError(f"step replay worker failed under {python} ({why}). It needs torch, safetensors, numpy, PyAV "
                      "and the reference pipeline: pass the reference's interpreter with --python." +
                      (f"\nlog tail:\n{tail}" if tail.strip() else ""))
    doc = json.loads(out_path.read_text())
    out_path.unlink()
    return doc


def _check(ok: bool, reason: str) -> dict:
    return {"verdict": "PASS" if ok else "FAIL", "reason": reason}


def judge(item: dict, steps: int, factor: float, slack: float, chain_tol: float, sigma_tol: float,
          decode_db: float) -> dict:
    """{verdict, checks, worst} of one audited request's worker result."""
    checks: dict[str, dict] = {}
    if item.get("error"):
        checks["dump"] = _check(False, item["error"])
        return {"verdict": "FAIL", "checks": checks}
    diff = item.get("request_diff") or []
    checks["dump"] = _check(not diff, "request matches" if not diff else f"the dump's request differs in {diff}")
    layout = item.get("layout") or []
    checks["layout"] = _check(not layout, "; ".join(layout) or f"{item.get('n_steps')} steps, streams "
                              f"{item.get('streams')}")
    sch = item.get("schedule") or {}
    ref = sch.get("ref_sigmas") or []
    sig, sig_n = sch.get("sigma") or [], sch.get("sigma_next") or []
    if not ref:
        checks["schedule"] = {"verdict": "NOT RUN", "reason": "not replayed"}
    elif len(ref) - 1 != len(sig) or len(sig) != steps:
        checks["schedule"] = _check(False, f"{len(sig)} steps dumped, the reference's schedule has {len(ref) - 1} "
                                    f"for {steps} asked")
    else:
        d = max([abs(a - b) for a, b in zip(sig, ref[:-1])] + [abs(a - b) for a, b in zip(sig_n, ref[1:])] + [0.0])
        checks["schedule"] = _check(len(sig_n) == len(sig) and d <= sigma_tol,
                                    f"max |sigma - reference sigma| {d:.3g} (tolerance {sigma_tol:g})")
    if "steps" not in item:  # not replayed: a layout or schedule failure says why
        checks.setdefault("schedule", {"verdict": "NOT RUN", "reason": "not replayed"})
        v = "FAIL" if any(c["verdict"] == "FAIL" for c in checks.values()) else "UNPROVEN"
        return {"verdict": v, "checks": checks}
    bad_dt = []
    for s, fields in (item.get("ref_dtypes") or {}).items():
        for f, want in fields.items():
            got = (item.get("dump_dtypes") or {}).get(s, {}).get(f)
            if SAFETENSORS_DTYPES.get(str(got).lower(), got) != want:
                bad_dt.append(f"{s}.{f} {SAFETENSORS_DTYPES.get(str(got).lower(), got)} (reference {want})")
    checks["dtypes"] = _check(not bad_dt, "; ".join(bad_dt) or "as the reference's")
    rows = item["steps"]
    cont = [w for w, v in item.get("continuity") or [] if v != 0.0]
    chain = max((r["chain"] for r in rows), default=0.0)
    init = {s: v for s, v in (item.get("initial") or {}).items() if v > chain_tol}
    why = []
    if chain > chain_tol:
        worst = max(rows, key=lambda r: r["chain"])
        why.append(f"x_next is not the reference scheduler's update of the dumped output (max relative error "
                   f"{chain:.3g} at {worst['stream']} step {worst['i']}; tolerance {chain_tol:g})")
    if cont:
        why.append(f"{len(cont)} latent(s) differ from the next step's start (e.g. {cont[0]})")
    if init:
        why.append(f"the first latent is not the request's initial noise ({init})")
    checks["chain"] = _check(not why, "; ".join(why) or f"max relative error {chain:.3g}")
    dec = item.get("decode") or {}
    if dec.get("psnr_db") is None:
        checks["decode"] = _check(False, f"the delivered clip has {dec.get('frames_clip')} frames of "
                                  f"{dec.get('size_clip')}, the reference's decode {dec.get('frames_ref')} of "
                                  f"{dec.get('size_ref')}")
    else:
        same = dec.get("frames_clip") == dec.get("frames_ref")
        checks["decode"] = _check(same and dec["psnr_db"] >= decode_db,
                                  f"{dec.get('frames_clip')} vs {dec.get('frames_ref')} frames; mean PSNR "
                                  f"{dec['psnr_db']:.2f} dB vs the reference's decode of the final latent "
                                  f"(minimum {decode_db:g})")
    over, worst, unfloored = [], None, 0
    for r in rows:
        f = r.get("floor")
        if f is None:
            unfloored += 1
            continue
        r["limit"] = factor * f + slack
        r["ratio"] = r["e"] / f if f > 0 else (0.0 if r["e"] == 0 else None)
        if r["e"] > r["limit"]:
            over.append(r)
        if worst is None or r["e"] - r["limit"] > worst["e"] - worst["limit"]:
            worst = r
    exact = all(r["e"] == 0 for r in rows)
    floor_zero = rows and all((r.get("floor") or 0) == 0 for r in rows)
    if unfloored:
        checks["steps"] = {"verdict": "UNPROVEN", "reason": f"{unfloored} step(s) have no floor (alt replay failed)"}
    elif over:
        w = max(over, key=lambda r: r["e"] - r["limit"])
        checks["steps"] = _check(False, f"{len(over)} of {len(rows)} step outputs beyond {factor:g} x floor + "
                                 f"{slack:g} (worst {w['stream']} step {w['i']}: e {w['e']:.3g}, floor "
                                 f"{w['floor']:.3g}, cosine {w['cos']})")
    elif floor_zero and not exact:
        checks["steps"] = {"verdict": "UNPROVEN", "reason": "the alternative kernel's outputs equal the reference's "
                           "(floor 0): it cannot calibrate the ratio (was --alt-attention applied?)"}
    else:
        checks["steps"] = _check(True, f"all {len(rows)} step outputs within {factor:g} x floor + {slack:g}" +
                                 (" (bit-identical)" if exact else f"; max e {max(r['e'] for r in rows):.3g}"))
    from inferencekit import stats

    v = stats.combine_claims(c["verdict"] for c in checks.values())
    return {"verdict": v, "checks": checks, "worst": worst}


@operation("quality_clip_steps", ("quality", "clip-steps"),
           "Per-step check of a video engine against the BF16 reference: for audited clips drawn after the clips "
           "are made, each denoising step's guided output (from the engine's per-step dump) against the reference's "
           "output from the same latent, judged against an alternative exact attention kernel's floor, with "
           "schedule, dtype, chain and decode guards.",
           [Arg("outputs", "string", "The engine's clips: a bench clip outputs/<label> directory (its request bodies "
                "in inputs/<label>/).", required=True),
            Arg("dumps", "string", "The engine's dump directory (its CLIP_STEP_DUMP_DIR).", required=True),
            Arg("pipeline", "string", "The reference clip pipeline, as the clip server names it (diffusers_ltx or "
                "MODULE:FUNCTION).", required=True),
            Arg("pipeline_model", "string", "The reference's --model (folder or hub id).", required=True),
            Arg("pipeline_arg", "array", "The reference's --pipeline-arg KEY=VALUE (repeat; JSON values).",
                items="string"),
            Arg("python", "string", "Python of the reference pipeline's venv (torch, safetensors, numpy, PyAV); "
                "default: this interpreter."),
            Arg("task", "string", "The reference pipeline variant.", default="t2v", choices=("t2v", "i2v")),
            Arg("device", "string", "The reference's --device.", default="cuda"),
            Arg("dtype", "string", "The reference's --dtype.", default="bf16"),
            Arg("offload", "string", "The reference's --offload.", default="none",
                choices=("none", "model", "sequential")),
            Arg("attention", "string", "The reference's --attention (reference mode sets its own)."),
            Arg("alt_attention", "string", "The exact but different attention backend of the floor replay.",
                default="_native_efficient"),
            Arg("requests", "integer", "Clips to audit (every step of each).", default=2),
            Arg("audit_seed", "integer", "Seed of the draw of audited clips (default: drawn now and recorded)."),
            Arg("factor", "number", "Pass: e <= factor x floor + slack at every step.", default=DEFAULT_FACTOR),
            Arg("slack", "number", "Absolute slack on the relative error e.", default=DEFAULT_SLACK),
            Arg("chain_tol", "number", "Relative L2 tolerance of the chain check.", default=DEFAULT_CHAIN_TOL),
            Arg("sigma_tol", "number", "Absolute tolerance on each sigma.", default=DEFAULT_SIGMA_TOL),
            Arg("decode_psnr_db", "number", "Minimum mean PSNR of the delivered clip vs the reference's decode of the "
                "final latent.", default=DEFAULT_DECODE_PSNR_DB),
            Arg("model", "string", "Model id for the record's identity."),
            Arg("run_dir", "string", "Run directory: the score also goes to <run-dir>/quality/clip_steps.json and "
                "the record to <run-dir>/evidence/."),
            Arg("timeout_s", "number", "Seconds the replay worker may take in all.", default=3600)])
def quality_clip_steps(call, outputs, dumps, pipeline, pipeline_model, pipeline_arg, python, task, device, dtype,
                       offload, attention, alt_attention, requests, audit_seed, factor, slack, chain_tol, sigma_tol,
                       decode_psnr_db, model, run_dir, timeout_s):
    if requests < 1:
        raise OpError("--requests must be >= 1")
    out_dir, dump_dir = Path(outputs), Path(dumps)
    if not dump_dir.is_dir():
        raise OpError(f"--dumps {dump_dir}: not a directory")
    files = clip_files(out_dir)
    bodies = requests_of(out_dir)
    if not bodies:
        raise OpError(f"{out_dir}: no request bodies (bench clip keeps them in inputs/<label>/): the replay needs "
                      "each clip's prompt, shape, steps and seed")
    not_timed = index_mismatch(out_dir, files)
    candidates = sorted(k for k in files if k in bodies)
    seed = audit_seed if audit_seed is not None else secrets.randbits(32)
    picked = sorted(random.Random(seed).sample(candidates, min(requests, len(candidates))))
    from inferencekit.engines.clip import stepdump

    reqs = [{"key": k, "body": bodies[k], "clip": str(files[k].resolve()),
             "dump": str(stepdump.dump_path(dump_dir, file_sha256(files[k])[0]).resolve())} for k in picked]
    cfg = {"pipeline": {"name": pipeline, "model": pipeline_model, "task": task, "device": device, "dtype": dtype,
                        "offload": offload, "attention": attention, "extra": _pipeline_args(pipeline_arg)},
           "alt_attention": alt_attention, "requests": reqs}
    store = ev.Store()
    with store.staging_dir() as staging:
        doc = run_replay(python, cfg, staging, timeout_s)
        audited = {}
        for it in doc["items"]:
            j = judge(it, int(bodies[it["key"]]["steps"]), factor, slack, chain_tol, sigma_tol, decode_psnr_db)
            if it["key"] in not_timed:
                j["checks"]["outputs_index"] = _check(False, not_timed[it["key"]])
                j["verdict"] = "FAIL"
            audited[it["key"]] = {**j, "decode": it.get("decode"), "seconds": it.get("seconds"),
                                  "dump_bytes": it.get("dump_bytes"), "n_steps": it.get("n_steps")}
        from inferencekit import stats

        verdict = stats.combine_claims(a["verdict"] for a in audited.values())
        caps = []
        if len(picked) < requests:
            caps.append(f"only {len(picked)} clip(s) with request bodies to audit, {requests} asked")
            verdict = "UNPROVEN" if verdict == "PASS" else verdict
        rows = [{"key": it["key"], **r} for it in doc["items"] for r in it.get("steps") or []]
        jl = staging / "clip_steps.jsonl"
        jl.write_text("".join(json.dumps(r) + "\n" for r in rows))
        artifacts = [ev.artifact(jl, staging, "clip_steps")]
        log = staging / "worker" / "replay.log"
        if log.is_file() and log.stat().st_size:
            log.write_text(ev.redact_text(log.read_text(errors="replace")))
            artifacts.append(ev.artifact(log, staging, "worker_log"))
        judged = [r for r in rows if r.get("floor") is not None]
        ratios = [r["ratio"] for r in judged if r.get("ratio") is not None]
        label = out_dir.name
        result = {"verdict": verdict, "label": label, "audit_seed": seed, "audited": audited,
                  "candidates": len(candidates), "requests": requests, "caps": caps,
                  "rule": {"factor": factor, "slack": slack, "chain_tol": chain_tol, "sigma_tol": sigma_tol,
                           "decode_psnr_db": decode_psnr_db, "alt_attention": alt_attention,
                           "defaults_source": DEFAULTS_SOURCE},
                  "max_ratio": max(ratios) if ratios else None,
                  "max_e": max((r["e"] for r in rows), default=None),
                  "max_floor": max((r["floor"] for r in judged), default=None),
                  "worker": {"load_s": doc.get("load_s"), "total_s": doc.get("total_s"),
                             "describe": doc.get("describe")}}
        mets = [ev.metric("step_error_max", result["max_e"], "relative_l2", "max", n=len(rows), subject=label),
                ev.metric("step_floor_max", result["max_floor"], "relative_l2", "max", n=len(judged), subject=label),
                ev.metric("step_error_over_floor_max", result["max_ratio"], "ratio", "max", n=len(ratios),
                          subject=label),
                ev.metric("replay_seconds", doc.get("total_s"), "s", "total", n=len(picked))]
        summary = (f"quality clip-steps {verdict}: {label}, {len(picked)} audited clip(s), {len(rows)} step outputs; "
                   f"max e/floor {result['max_ratio']:.3g}" if result["max_ratio"] is not None else
                   f"quality clip-steps {verdict}: {label}, {len(picked)} audited clip(s)")
        kind, extra = _kind("clip_steps")
        rec = ev.build(call, kind=kind, claim_class="observed", summary=summary,
                       identity_=ev.identity(engine={"name": label}, model={"id": model}),
                       metrics=mets, artifacts=artifacts, missing=[] if rows else ["no step was replayed"],
                       limitations=caps + ["audio and video latents are judged; the decoded audio is not compared"],
                       data={**extra, **result, "format": FORMAT})
        if run_dir:
            q = Path(run_dir) / "quality"
            q.mkdir(parents=True, exist_ok=True)
            (q / "clip_steps.json").write_text(json.dumps(result, indent=1) + "\n")
        return store.save(rec, staging)
