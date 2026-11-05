"""The replay worker of ``quality clip-steps``: runs inside the reference pipeline's own Python, on its GPU.

Usage: python step_replay.py CONFIG.json  (inferencekit importable: the harness puts a path holding only the
inferencekit package first on PYTHONPATH). Prints one line ``INFERKIT_RESULT {...}`` on stdout; results go to the
config's ``out``; logs to stderr. Needs torch, safetensors, numpy and PyAV (the reference's venv has them).

Config: {"pipeline": {"name", "model", "task", "device", "dtype", "offload", "attention", "extra"},
         "alt_attention": backend name, "out": path,
         "requests": [{"key", "body": the request, "clip": the delivered clip's path, "dump": its dump's path}]}

The reference pipeline is loaded once (the clip server's factory: ``--pipeline`` names), then for each request:
  1. the dump's layout against the request (stepdump.layout_problems) and its recorded request;
  2. continuity inside the dump: each step's x_next equals the next step's x, the final latent the last x_next;
  3. replay with the reference's own backend (pipeline.replay: every step starts from the dump's x_t): per step
     and stream, e = |out_eng - out_ref| / |out_ref| (L2), cosine, the chain error |x_next_eng - update(x_eng,
     out_eng)| / |x_next_eng| with the reference's scheduler, the reference's sigmas and dtypes; the reference's
     own initial latent against the dump's first x; the frames the reference decodes from the dump's final
     latent. A dump with another step count than the reference's schedule stops at step 0;
  4. the same replay with ``alt_attention`` (an exact but different kernel): the floor f = |out_alt - out_ref| /
     |out_ref| on the same inputs;
  5. the decode check: the delivered clip's decoded frames (PyAV) against the reference's decode, PSNR per frame.
Judging is the harness's job (ops/quality_clip_steps.py); this file only measures.
"""

import json
import math
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

MARK = "INFERKIT_RESULT "
PSNR_CAP_DB = 100.0


class _Stop(Exception):
    pass


def log(msg):
    print(f"step_replay: {msg}", file=sys.stderr, flush=True)


def _rel(a, b):
    """|a - b| / |b| (L2, float64); 0 when both are zero."""
    a, b = a.double(), b.double()
    nb = float(b.norm())
    d = float((a - b).norm())
    return d / nb if nb else (0.0 if d == 0 else math.inf)


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    n = float(a.norm() * b.norm())
    return float((a * b).sum()) / n if n else None


def continuity(dump):
    """[(what, rel diff)] for x_next[i] vs x[i+1] and final vs the last x_next, per stream (0.0 = equal)."""
    out = []
    n = dump.n_steps
    for s in dump.streams:
        for i in range(n):
            a = dump.get(f"step.{i}.{s}.x_next")
            nxt = f"step.{i + 1}.{s}.x" if i + 1 < n else f"final.{s}"
            if not dump.has(nxt) or not dump.has(f"step.{i}.{s}.x_next"):
                continue
            b = dump.get(nxt)
            out.append((f"{s} step {i} x_next vs {nxt}", 0.0 if a.shape == b.shape and bool((a == b).all())
                        else (_rel(a, b) if a.shape == b.shape else math.inf)))
    return out


def decode_clip(path):
    import av

    c = av.open(path)
    try:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        return [f.to_ndarray(format="rgb24") for f in c.decode(s)]
    finally:
        c.close()


def decode_check(frames_ref, clip_path):
    import numpy as np

    ref = np.clip(np.round(np.asarray(frames_ref, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    got = decode_clip(clip_path)
    out = {"frames_ref": int(ref.shape[0]), "frames_clip": len(got), "size_ref": [int(ref.shape[2]), int(ref.shape[1])],
           "size_clip": [int(got[0].shape[1]), int(got[0].shape[0])] if got else None}
    if not got or tuple(got[0].shape) != tuple(ref.shape[1:]):
        out["psnr_db"] = None
        return out
    vals = []
    for a, b in zip(ref, got):
        mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        vals.append(PSNR_CAP_DB if mse == 0 else min(PSNR_CAP_DB, 10 * math.log10(255.0 ** 2 / mse)))
    out["psnr_db"] = sum(vals) / len(vals)
    out["psnr_db_min_frame"] = min(vals)
    return out


def one_request(pipe, req, alt):
    from inferencekit.engines.clip import stepdump
    from inferencekit.engines.clip.server import parse_job

    body = dict(req["body"])
    if body.get("task") == "i2v" or body.get("image") is not None:
        return {"key": req["key"], "error": "i2v replay is not supported (the request's image bytes are not kept)"}
    job = parse_job({k: v for k, v in body.items() if k in stepdump.REQUEST_FIELDS})
    res = {"key": req["key"], "error": None}
    if not os.path.isfile(req["dump"]):
        res["error"] = "no dump for this clip (the engine wrote none under its sha256)"
        return res
    dump = stepdump.DumpReader(req["dump"])
    shapes = {k: list(v) for k, v in pipe.noise_shapes(job).items()}
    res["layout"] = stepdump.layout_problems(dump, shapes, job.steps)
    want = stepdump.request_of(body)
    got = stepdump.request_of(dump.request) if isinstance(dump.request, dict) else {}
    res["request_diff"] = [k for k in stepdump.COMPARED_FIELDS if got.get(k) != want.get(k)]
    res["n_steps"], res["streams"] = dump.n_steps, dump.streams
    res["dump_dtypes"] = {s: {f: dump.dtype(f"step.0.{s}.{f}").lower() for f in stepdump.FIELDS
                              if dump.has(f"step.0.{s}.{f}")} for s in dump.streams}
    res["dump_bytes"] = os.path.getsize(req["dump"])
    if res["layout"] and any(("missing" in p or "shape" in p or "streams" in p) for p in res["layout"]):
        return res  # cannot be replayed step for step
    res["continuity"] = [[w, v] for w, v in continuity(dump)]
    sig = dump.get("sigma").tolist() if dump.has("sigma") else []
    sig_n = dump.get("sigma_next").tolist() if dump.has("sigma_next") else []
    sched = {}

    def on_schedule(sigmas):
        sched["ref"] = sigmas
        if len(sigmas) - 1 != dump.n_steps:
            raise _Stop()

    ref_out, rows = {}, {}

    def on_step(i, stream, sigma, sigma_next, out, chain):
        eng = dump.get(f"step.{i}.{stream}.out").to(out.device)
        x_next = dump.get(f"step.{i}.{stream}.x_next").to(out.device)
        rows[(i, stream)] = {"i": i, "stream": stream, "sigma_ref": sigma, "sigma_next_ref": sigma_next,
                             "e": _rel(eng, out), "cos": _cos(eng, out), "chain": _rel(chain, x_next),
                             "norm_ref": float(out.double().norm())}
        ref_out[(i, stream)] = out.detach().float().cpu()

    t0 = time.perf_counter()
    try:
        rep = pipe.replay(job, dump, on_step, on_schedule=on_schedule)
    except _Stop:
        res["schedule"] = {"ref_sigmas": sched.get("ref"), "sigma": sig, "sigma_next": sig_n, "stopped": True}
        res["seconds"] = {"replay": round(time.perf_counter() - t0, 3)}
        return res
    t1 = time.perf_counter()
    res["schedule"] = {"ref_sigmas": sched.get("ref"), "sigma": sig, "sigma_next": sig_n, "stopped": False}
    res["ref_dtypes"] = rep["dtypes"]
    res["initial"] = {s: _rel(dump.get(f"step.0.{s}.x").float(), t) for s, t in rep["initial"].items()
                      if dump.has(f"step.0.{s}.x")}
    res["decode"] = decode_check(rep["frames"], req["clip"])
    t2 = time.perf_counter()
    floors = {}

    def on_alt(i, stream, sigma, sigma_next, out, chain):
        ref = ref_out.get((i, stream))
        if ref is not None:
            floors[(i, stream)] = _rel(out.float(), ref.to(out.device))

    pipe.replay(job, dump, on_alt, attention=alt, decode=False)
    t3 = time.perf_counter()
    for k, row in rows.items():
        row["floor"] = floors.get(k)
    res["steps"] = [rows[k] for k in sorted(rows)]
    res["seconds"] = {"replay": round(t1 - t0, 3), "decode_check": round(t2 - t1, 3), "alt_replay": round(t3 - t2, 3)}
    return res


def main(argv):
    cfg = json.loads(open(argv[1]).read())
    t0 = time.time()
    try:
        from inferencekit.engines.clip.server import PipelineOptions, load_factory

        pc = cfg["pipeline"]
        opts = PipelineOptions(model=pc["model"], task=pc.get("task") or "t2v", device=pc.get("device") or "cuda",
                               dtype=pc.get("dtype") or "bf16", offload=pc.get("offload") or "none",
                               attention=pc.get("attention"), extra=dict(pc.get("extra") or {}))
        pipe = load_factory(pc["name"])(opts)
        load_s = time.time() - t0
        log(f"pipeline loaded in {load_s:.1f} s")
        items = []
        for req in cfg["requests"]:
            t = time.time()
            try:
                items.append(one_request(pipe, req, cfg["alt_attention"]))
            except Exception as e:  # noqa: BLE001 - one request's failure is its result
                traceback.print_exc(file=sys.stderr)
                items.append({"key": req["key"], "error": f"{type(e).__name__}: {e}"[:500]})
            log(f"{req['key']}: {time.time() - t:.1f} s")
        doc = {"items": items, "describe": pipe.describe(), "load_s": round(load_s, 3),
               "total_s": round(time.time() - t0, 3)}
        with open(cfg["out"], "w") as f:
            json.dump(doc, f)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        print(MARK + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}), flush=True)
        return 1
    print(MARK + json.dumps({"ok": True, "n": len(items), "seconds": round(time.time() - t0, 3)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
