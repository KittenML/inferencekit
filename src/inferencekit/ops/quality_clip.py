"""quality clip: does a faster video engine still make the reference pipeline's clips?

Inputs are directories of clips: a ``bench clip`` outputs directory (``outputs/<label>/r<round>/<id>.mp4``)
or any directory of ``<id>.<ext>`` files. Clips pair up by their path relative to the directory, so the
engine and the reference must have been asked for the same clips in the same rounds:

  --reference        the BF16 reference pipeline's clips
  --reference-repeat the reference run a second time with the same requests: the run-to-run A/A floor
  --reference-alt    the reference with another exact attention backend (e.g. SDPA math instead of flash):
                     the kernel A/A floor, how far an exact but different kernel already moves the clip
  --outputs          each engine's clips (one directory per engine)

Every clip pair is scored frame by frame by ``quality/video_metrics.py``, which runs as a worker inside the
video pipeline's own Python (``--python``; PyAV and numpy, and torch + lpips for LPIPS), because the harness
carries none of those packages. Metrics per frame: PSNR (dB, RGB, capped at 100 dB for identical frames),
SSIM (luma, Gaussian 11x11) and LPIPS (v0.1, AlexNet); each clip's value is its mean over frames, and the
request is the bootstrap unit: frames inside a clip are not independent, and neither are the clips of one
request made again in later rounds (bench clip's ``--seed-policy fixed`` asks for the same clips every
round; a deterministic pipeline makes them byte for byte). Clips whose kept request bodies match on every
REQUEST_KEYS field are one unit; without request bodies, every round of a clip id is one unit.

Same inputs, or the comparison means nothing: the engine's clip and the reference's must come from the same
prompt, image, shape, steps, guidance, seed and initial noise. When the directories are bench clip outputs,
the request bodies it kept (``inputs/<label>/<key>.json``) are compared field by field and a clip whose
request differs is left out (the verdict is then at best UNPROVEN). Identical initial noise is the
engines' job (the clip contract's ``noise_seed``); a pipeline that draws its noise differently shows up as
an engine-vs-reference distance far above the floor.

Verdicts (``--fidelity``):
  lossless (default)  per metric, a paired bootstrap CI over clips of "engine vs reference" minus "floor vs
                      reference" (oriented so positive = worse), against pre-registered margins
                      (``--margin-psnr-db``, ``--margin-ssim``, ``--margin-lpips``): PASS when the whole CI is
                      within the margin, FAIL when all of it is beyond, else UNPROVEN
                      (``quality.metrics.margin_verdict``). The floor is the alternative-backend run when
                      given, else the repeat. A deterministic reference repeats itself exactly; an exact floor
                      cannot tell a different-but-exact kernel from a lossy one, so whenever the floor used is
                      bit-identical to the reference (the repeat, or an alternative-backend run whose switch
                      was not applied) an engine that is not bit-exact gets UNPROVEN, never PASS or FAIL.
  lossy               a declared quality budget (``--budget``, the full-reference bound: psnr_db_min, ssim_min,
                      lpips_max on the engine's mean vs the reference, judged on the CI over clips) and a
                      human spot-check record (``--spot-check``): a lossy PASS needs both; "not_acceptable"
                      in the spot check is a FAIL. Lossy results are a separate track, reported as
                      "N x at the stated quality cost", never ranked against lossless ones.
  gross               a loose gross-failure gate beside the per-step check (quality clip-steps), which judges
                      losslessness: the lossless comparison with the same margins (set them loose), but only a
                      CI entirely beyond the margin FAILs, anything else PASSes (an exact floor still gives
                      UNPROVEN: it needs --reference-alt).
Any clip whose frame size or frame count differs from the reference's is a FAIL (the engine did not make
the clip it was asked for; gates clip checks the decoded shape against the request), and so is an engine
clip whose sha256 differs from the one its directory's ``index.json`` (bench clip's outputs index, when
present) recorded: it is not the clip that was timed. A requested metric that could not be computed (e.g.
LPIPS without its weights) counts as UNPROVEN.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit import stats
from inferencekit.canonical import file_sha256
from inferencekit.ops.quality import _kind
from inferencekit.proc import run, which
from inferencekit.quality import metrics as qm
from inferencekit.registry import Arg, OpError, operation

WORKER = Path(__file__).resolve().parents[1] / "quality" / "video_metrics.py"
MARK = "INFERKIT_RESULT "
VIDEO_EXT = (".mp4", ".webm", ".mkv", ".mov", ".avi")
METRICS = ("psnr", "ssim", "lpips")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False}
UNITS = {"psnr": "dB", "ssim": "index", "lpips": "distance"}
# Evidence metric names (the review page and a re-run read psnr_db; the flag is --margin-psnr-db).
NAMES = {"psnr": "psnr_db", "ssim": "ssim", "lpips": "lpips"}
ROUND_KEY = re.compile(r"^r(\d+)/(.+)$")  # bench clip: r<round>/<id>
BUDGET = {"psnr_db_min": ("psnr", "min"), "ssim_min": ("ssim", "min"), "lpips_max": ("lpips", "max")}
SPOT_VERDICTS = ("acceptable", "not_acceptable")
# Request fields the engine's clip and the reference's must share (bench clip's inputs/<label>/<key>.json).
REQUEST_KEYS = ("task", "prompt", "image", "width", "height", "num_frames", "fps", "steps", "guidance_scale",
                "seed", "noise_seed", "audio")
FORMAT = "inferencekit.quality.clip.v1"
PSNR_CAP_DB = 100.0
FFPROBE_ENTRIES = ("stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate,"
                   "channels,duration:frame=stream_index,media_type,width,height,best_effort_timestamp_time,"
                   "nb_samples:format=format_name,duration")


# --------------------------------------------------------------------------- the worker and decoding

def run_worker(python: str | None, cfg: dict, work: Path, name: str, timeout_s: float) -> dict:
    """Run video_metrics.py under ``python`` (the video pipeline's interpreter); return its results file.

    Paths in ``cfg`` are only in the config file, which is deleted afterwards; the results carry keys."""
    python = python or sys.executable
    work.mkdir(parents=True, exist_ok=True)
    cfg_path, out_path = work / f"{name}.config.json", work / f"{name}.results.json"
    log_path, stdout_path = work / f"{name}.log", work / f"{name}.stdout.txt"
    cfg_path.write_text(json.dumps({**cfg, "out": str(out_path)}))
    try:
        r = run([python, str(WORKER), str(cfg_path)], timeout=timeout_s, stdout_path=stdout_path,
                stderr_path=log_path)
    finally:
        cfg_path.unlink(missing_ok=True)
    if log_path.is_file():
        log_path.write_text(ev.redact_text(log_path.read_text(errors="replace")))
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(MARK)]
    stdout_path.unlink(missing_ok=True)
    head = json.loads(lines[-1][len(MARK):]) if lines else None
    if not head or not head.get("ok") or not out_path.is_file():
        tail = log_path.read_text(errors="replace")[-1200:] if log_path.is_file() else r.stderr[-1200:]
        why = (head or {}).get("error") or f"exit {r.returncode}"
        raise OpError(f"video metrics worker failed under {python} ({why}). It needs PyAV and numpy (and torch + "
                      "lpips for LPIPS): pass the video pipeline's interpreter with --python." +
                      (f"\nlog tail:\n{tail}" if tail.strip() else ""))
    doc = json.loads(out_path.read_text())
    doc["head"] = head
    return doc


def _rate(text) -> float | None:
    try:
        num, _, den = str(text).partition("/")
        v = float(num) / float(den or 1)
        return v if v > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def probe_ffprobe(path: Path, timeout_s: float = 900) -> dict:
    """Decode ``path`` with ffprobe (``-show_entries frame=...`` decodes every frame); same shape as the
    worker's probe: what decoding gives, beside what the header claims."""
    exe = which("ffprobe")
    if not exe:
        return {"ok": False, "error": "ffprobe not found"}
    r = run([exe, "-v", "error", "-show_entries", FFPROBE_ENTRIES, "-of", "json", str(path)], timeout=timeout_s)
    if r.returncode != 0:
        return {"ok": False, "error": f"ffprobe exit {r.returncode}: {r.stderr.strip()[:300]}"}
    try:
        doc = json.loads(r.stdout)
    except ValueError:
        return {"ok": False, "error": "ffprobe printed no JSON"}
    streams, frames = doc.get("streams") or [], doc.get("frames") or []
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    au = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = doc.get("format") or {}
    out = {"ok": True, "decoder": "ffprobe", "video": None, "audio": None,
           "header": {"format": fmt.get("format_name"), "duration_s": _num(fmt.get("duration"))}}
    if vs is not None:
        out["header"]["video"] = {"codec": vs.get("codec_name"), "width": vs.get("width"),
                                  "height": vs.get("height"), "frames": int(vs["nb_frames"])
                                  if str(vs.get("nb_frames", "")).isdigit() else None,
                                  "fps": _rate(vs.get("avg_frame_rate"))}
        vf = [f for f in frames if f.get("stream_index") == vs.get("index")]
        sizes = sorted({(f.get("width"), f.get("height")) for f in vf})
        times = sorted(t for t in (_num(f.get("best_effort_timestamp_time")) for f in vf) if t is not None)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 and times[-1] > times[0] else None
        w, h = sizes[0] if sizes else (None, None)
        out["video"] = {"frames": len(vf), "width": w, "height": h, "sizes": [list(s) for s in sizes], "fps": fps,
                        "first_s": times[0] if times else None, "last_s": times[-1] if times else None,
                        "duration_s": len(vf) / fps if fps else None}
    if au is not None:
        rate = int(au["sample_rate"]) if str(au.get("sample_rate", "")).isdigit() else None
        out["header"]["audio"] = {"codec": au.get("codec_name"), "sample_rate": rate, "channels": au.get("channels"),
                                  "duration_s": _num(au.get("duration"))}
        samples = sum(int(f.get("nb_samples") or 0) for f in frames if f.get("stream_index") == au.get("index"))
        out["audio"] = {"samples": samples, "sample_rate": rate, "duration_s": samples / rate if rate else None}
    return out


def probe_clips(files: list[tuple[str, Path]], python: str | None, work: Path, timeout_s: float) -> tuple[dict, str]:
    """Decode each clip: with the worker (PyAV) when ``python`` is given, else ffprobe. ({key: probe}, decoder)."""
    if python:
        doc = run_worker(python, {"task": "probe", "files": [{"key": k, "path": str(p)} for k, p in files]},
                         work, "probe", timeout_s)
        return {it["key"]: it for it in doc["items"]}, "pyav"
    if not which("ffprobe"):
        raise OpError("nothing to decode clips with: install ffprobe (ffmpeg) or pass --python (a venv with PyAV)")
    return {k: probe_ffprobe(p, timeout_s) for k, p in files}, "ffprobe"


# --------------------------------------------------------------------------- clip directories

def clip_files(d: Path) -> dict[str, Path]:
    """{key: path} for every clip under ``d``; key = the path relative to ``d`` without its extension."""
    if not d.is_dir():
        raise OpError(f"{d}: not a directory of clips")
    out: dict[str, Path] = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            key = p.relative_to(d).with_suffix("").as_posix()
            if key in out:
                raise OpError(f"{d}: two clips share the key {key!r} ({out[key].name}, {p.name})")
            out[key] = p
    if not out:
        raise OpError(f"{d}: no clips ({', '.join(VIDEO_EXT)})")
    return out


def requests_of(d: Path) -> dict[str, dict]:
    """bench clip's request bodies for an ``outputs/<label>`` directory: ``inputs/<label>/<key>.json``."""
    if d.parent.name != "outputs":
        return {}
    root = d.parent.parent / "inputs" / d.name
    out = {}
    for p in sorted(root.rglob("*.json")) if root.is_dir() else []:
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            out[p.relative_to(root).with_suffix("").as_posix()] = doc
    return out


def request_diff(a: dict | None, b: dict | None) -> list[str] | None:
    """Request fields that differ (None when either side has no request body to compare)."""
    if a is None or b is None:
        return None
    return [k for k in REQUEST_KEYS if a.get(k) != b.get(k)]


def split_key(key: str) -> tuple[int | None, str]:
    """(round, clip id) of a clip key: bench clip's ``r<round>/<id>``, else (None, the key)."""
    m = ROUND_KEY.match(key)
    return (int(m.group(1)), m.group(2)) if m else (None, key)


def request_groups(keys, bodies: list[dict[str, dict]]) -> dict[str, str]:
    """{key: request identity}; clips of one identity are one bootstrap unit, on every side alike.

    The identity is the REQUEST_KEYS values of the first request body found for the key (the reference's
    first: the sides' bodies must match it to be scored). Without any body it is the clip id, so every
    round of a clip counts once: rounds may have repeated the same request, and a unit too many narrows the
    CI (by about sqrt(rounds) when every round repeats), a unit too few only widens it."""
    out = {}
    for k in keys:
        body = next((b[k] for b in bodies if k in b), None)
        out[k] = ("request " + json.dumps({f: body.get(f) for f in REQUEST_KEYS}, sort_keys=True)
                  if body is not None else "clip " + split_key(k)[1])
    return out


def index_mismatch(d: Path, files: dict[str, Path]) -> dict[str, str]:
    """{key: why} for clips whose sha256 differs from ``d/index.json`` (bench clip's outputs index, when the
    directory has one): the file is not the clip that was timed."""
    idx = d / "index.json"
    if not idx.is_file():
        return {}
    try:
        listed = (json.loads(idx.read_text()) or {}).get("files") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    out = {}
    for key, p in files.items():
        want = listed.get(p.relative_to(d).as_posix()) if isinstance(listed, dict) else None
        if isinstance(want, str) and file_sha256(p)[0] != want:
            out[key] = "sha256 differs from index.json: not the clip that was timed"
    return out


# --------------------------------------------------------------------------- scoring

def _units(items: dict[str, dict], metric: str, groups: dict[str, str], keys=None) -> list[tuple[float, int]]:
    """Per request (sum over its clips' frames, frames): the request is the bootstrap unit. Units come in the
    order of ``keys`` (default: the items'), so two sides scored over the same keys pair up unit by unit."""
    acc: dict[str, list] = {}
    for k in items if keys is None else keys:
        vs = [v for v in (items[k].get("per_frame") or {}).get(metric) or [] if v is not None]
        u = acc.setdefault(groups.get(k, k), [0.0, 0])
        u[0] += sum(vs)
        u[1] += len(vs)
    return [(s, n) for s, n in acc.values()]


def _scored(items: dict[str, dict], metric: str) -> int:
    return sum(1 for it in items.values() if any(v is not None for v in (it.get("per_frame") or {}).get(metric) or []))


def _exact(it: dict) -> bool:
    return bool(it.get("frames_compared")) and it.get("identical_frames") == it.get("frames_compared")


def lossless_check(eng: dict[str, dict], flo: dict[str, dict], keys: list[str], groups: dict[str, str], metric: str,
                   margin: float, resamples: int, seed: int) -> dict:
    d = qm.paired_delta_ci(_units(eng, metric, groups, keys), _units(flo, metric, groups, keys), resamples, seed)
    lo, hi = d["ci95"]
    if HIGHER_IS_BETTER[metric]:  # worse = floor - engine
        worse = (-d["delta"] if d["delta"] is not None else None, -hi if hi is not None else None,
                 -lo if lo is not None else None)
    else:
        worse = (d["delta"], lo, hi)
    v, why = qm.margin_verdict(worse[1], worse[2], margin)
    both = [k for k in keys if _scored({k: eng[k]}, metric) and _scored({k: flo[k]}, metric)]
    return {"verdict": v, "reason": why, "worse_than_floor": worse[0], "ci95": [worse[1], worse[2]],
            "margin": margin, "n_clips": len(both), "n_requests": d["n_prompts"]}


def budget_check(eng: dict[str, dict], groups: dict[str, str], metric: str, side: str, bound: float, resamples: int,
                 seed: int) -> dict:
    units = _units(eng, metric, groups)
    value = qm.pooled(units)
    lo, hi = qm.cluster_ci(units, resamples, seed)
    if lo is None or hi is None:
        v, why = "UNPROVEN", "no confidence interval (need >= 2 distinct requests scored)"
    elif side == "min":
        v = "PASS" if lo >= bound else "FAIL" if hi < bound else "UNPROVEN"
        why = f"CI [{lo:.4g}, {hi:.4g}] vs minimum {bound:g}"
    else:
        v = "PASS" if hi <= bound else "FAIL" if lo > bound else "UNPROVEN"
        why = f"CI [{lo:.4g}, {hi:.4g}] vs maximum {bound:g}"
    return {"verdict": v, "reason": why, "value": value, "ci95": [lo, hi], "bound": bound, "side": side,
            "n_clips": _scored(eng, metric), "n_requests": len([u for u in units if u[1]])}


def summary_of(items: dict[str, dict], groups: dict[str, str], metric: str, resamples: int, seed: int) -> dict:
    units = _units(items, metric, groups)
    return {"value": qm.pooled(units), "ci95": list(qm.cluster_ci(units, resamples, seed)),
            "n_clips": _scored(items, metric), "n_requests": len([u for u in units if u[1]]),
            "n_frames": sum(u[1] for u in units)}


def load_spot_check(path: str | None) -> dict | None:
    """The human spot-check record: {reviewer, date, clips: [...], verdict: acceptable|not_acceptable, notes?}."""
    if not path:
        return None
    p = Path(path)
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise OpError(f"--spot-check {path}: not a readable JSON file ({type(e).__name__})") from None
    problems = []
    if not isinstance(doc, dict):
        raise OpError(f"--spot-check {path}: must be a JSON object")
    for k in ("reviewer", "date"):
        if not isinstance(doc.get(k), str) or not doc[k].strip():
            problems.append(f"{k} must be non-empty text")
    if doc.get("verdict") not in SPOT_VERDICTS:
        problems.append(f"verdict must be one of {list(SPOT_VERDICTS)}")
    if not isinstance(doc.get("clips"), list) or not doc["clips"] or not all(isinstance(c, str) for c in doc["clips"]):
        problems.append("clips must list the clip keys the reviewer watched")
    if problems:
        raise OpError(f"--spot-check {path}: " + "; ".join(problems))
    sha = ev.artifact(p, None, "spot_check")["sha256"]
    return {"sha256": sha, "reviewer": doc["reviewer"], "date": doc["date"], "verdict": doc["verdict"],
            "clips": doc["clips"], "notes": str(doc.get("notes") or "")[:2000]}


def _budget_arg(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        doc = json.loads(text)
    except ValueError:
        raise OpError("--budget must be a JSON object, e.g. {\"lpips_max\": 0.05, \"psnr_db_min\": 30}") from None
    if not isinstance(doc, dict) or not doc:
        raise OpError("--budget must be a non-empty JSON object")
    bad = [k for k in doc if k not in BUDGET]
    if bad:
        raise OpError(f"--budget: unknown key(s) {bad}; known: {sorted(BUDGET)}")
    for k, v in doc.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise OpError(f"--budget {k} must be a number")
    return doc


# --------------------------------------------------------------------------- the operation

@operation("quality_clip", ("quality", "clip"),
           "Video clip quality vs the BF16 reference pipeline: per-frame PSNR, SSIM and LPIPS of each engine's "
           "clips (same prompt, steps, seed and initial noise), against an A/A floor (the reference repeated, or "
           "run with another exact attention backend) with bootstrap CIs over clips; lossless margins or a lossy "
           "budget plus a human spot check.",
           [Arg("reference", "string", "The reference pipeline's clips: a bench clip outputs/<label> directory or "
                "any directory of <id>.<ext> clips.", required=True),
            Arg("outputs", "array", "Each engine's clip directory, laid out like --reference.", required=True),
            Arg("reference_repeat", "string", "The reference run a second time with the same requests (the "
                "run-to-run A/A floor)."),
            Arg("reference_alt", "string", "The reference with another exact attention backend (the kernel A/A "
                "floor; used as the floor when given)."),
            Arg("python", "string", "Python of the video pipeline's venv (PyAV, numpy; torch + lpips for LPIPS): "
                "the metrics worker runs there. Default: this interpreter."),
            Arg("metrics", "array", "Metrics to compute and judge: psnr, ssim, lpips.", default=list(METRICS),
                items="string"),
            Arg("fidelity", "string", "lossless: judged against the A/A floor with margins; lossy: judged against "
                "--budget, plus a human --spot-check; gross: the lossless comparison as a loose gate (only a CI "
                "entirely beyond the margin fails).", default="lossless", choices=("lossless", "lossy", "gross")),
            Arg("margin_psnr_db", "number", "Pre-registered margin: max PSNR drop vs the floor (dB).", default=1.0),
            Arg("margin_ssim", "number", "Pre-registered margin: max SSIM drop vs the floor.", default=0.01),
            Arg("margin_lpips", "number", "Pre-registered margin: max LPIPS increase vs the floor.", default=0.01),
            Arg("budget", "string", "Lossy: the declared full-reference bound, JSON with any of psnr_db_min, "
                "ssim_min, lpips_max (on the engine's mean vs the reference)."),
            Arg("spot_check", "string", "Lossy: the human spot-check record, JSON {reviewer, date, clips: [keys], "
                "verdict: acceptable|not_acceptable, notes?}."),
            Arg("device", "string", "Where LPIPS runs (auto = the GPU when the worker sees one).", default="auto",
                choices=("auto", "cpu", "cuda")),
            Arg("frame_stride", "integer", "Score every n-th frame (1 = every frame).", default=1),
            Arg("model", "string", "Model id for the record's identity."),
            Arg("run_dir", "string", "Run directory: the score is also written to <run-dir>/quality/clip_score.json "
                "and the record to <run-dir>/evidence/."),
            Arg("resamples", "integer", "Bootstrap resamples.", default=stats.DEFAULT_RESAMPLES),
            Arg("seed", "integer", "Bootstrap seed.", default=0),
            Arg("timeout_s", "number", "Seconds the metrics worker may take in all.", default=14400)])
def quality_clip(call, reference, outputs, reference_repeat, reference_alt, python, metrics, fidelity,
                 margin_psnr_db, margin_ssim, margin_lpips, budget, spot_check, device, frame_stride, model,
                 run_dir, resamples, seed, timeout_s):
    metrics = [m.lower() for m in (metrics or [])]
    if not metrics or any(m not in METRICS for m in metrics):
        raise OpError(f"--metrics must be a non-empty subset of {list(METRICS)}")
    if frame_stride < 1:
        raise OpError("frame_stride must be >= 1")
    budget_doc = _budget_arg(budget)
    if fidelity == "lossy" and not budget_doc:
        raise OpError("--fidelity lossy needs --budget (the declared full-reference bound)")
    if fidelity in ("lossless", "gross") and (budget_doc or spot_check):
        raise OpError("--budget and --spot-check are for --fidelity lossy")
    for k in budget_doc or {}:
        if BUDGET[k][0] not in metrics:
            raise OpError(f"--budget {k} needs {BUDGET[k][0]} in --metrics")
    spot = load_spot_check(spot_check)
    margins = {"psnr": margin_psnr_db, "ssim": margin_ssim, "lpips": margin_lpips}

    ref_dir = Path(reference)
    ref = clip_files(ref_dir)
    ref_req = requests_of(ref_dir)
    floors = {name: Path(p) for name, p in (("alt", reference_alt), ("repeat", reference_repeat)) if p}
    resolved = {ref_dir.resolve()}
    for name, p in floors.items():
        if p.resolve() in resolved:
            raise OpError(f"--reference-{name} must be another run of the reference, not the same directory")
        resolved.add(p.resolve())
    engines: dict[str, Path] = {}
    for p in map(Path, outputs or []):
        if p.resolve() in resolved:
            raise OpError(f"{p}: the same directory as a reference run")
        label = p.name
        if label in engines:
            raise OpError(f"two --outputs directories are both named {label!r}; rename one")
        engines[label] = p
    if not engines:
        raise OpError("--outputs: give at least one engine's clip directory")

    sides = {f"floor:{n}": d for n, d in floors.items()} | {f"engine:{lab}": d for lab, d in engines.items()}
    files = {s: clip_files(d) for s, d in sides.items()}
    reqs = {s: requests_of(d) for s, d in sides.items()}
    groups = request_groups(ref, [ref_req] + list(reqs.values()))
    not_timed = {s: index_mismatch(sides[s], files[s]) for s in sides if s.startswith("engine:")}
    pairs, plan = [], {}  # plan[side] = {key: {"pair": id} | {"missing": True} | {"request_diff": [...]}}
    for side, have in files.items():
        plan[side] = {}
        for key, ref_path in ref.items():
            if key not in have:
                plan[side][key] = {"missing": True}
                continue
            diff = request_diff(ref_req.get(key), reqs[side].get(key))
            if diff:
                plan[side][key] = {"request_diff": diff}
                continue
            pid = f"{side}|{key}"
            pairs.append({"key": pid, "a": str(ref_path), "b": str(have[key])})
            plan[side][key] = {"pair": pid, "request_checked": diff is not None}
        extra = sorted(set(have) - set(ref))
        if extra:
            plan[side]["__extra__"] = extra

    store = ev.Store()
    with store.staging_dir() as staging:
        doc = run_worker(python, {"task": "compare", "pairs": pairs, "metrics": metrics, "device": device,
                                  "frame_stride": frame_stride, "psnr_cap_db": PSNR_CAP_DB, "lpips_net": "alex"},
                         staging / "worker", "compare", timeout_s)
        by_pair = {it["key"]: it for it in doc["items"]}
        computed = [m for m in metrics if not (m == "lpips" and doc.get("lpips_error"))]
        result = _score(plan, by_pair, ref, floors, engines, metrics, computed, margins, fidelity, budget_doc, spot,
                        resamples, seed, doc.get("lpips_error"), groups, not_timed)
        rows = staging / "clip_metrics.jsonl"
        rows.write_text("".join(json.dumps(it) + "\n" for it in doc["items"]))
        artifacts = [ev.artifact(rows, staging, "clip_metrics")]
        log = staging / "worker" / "compare.log"
        if log.is_file() and log.stat().st_size:
            artifacts.append(ev.artifact(log, staging, "worker_log"))
        for leftover in (staging / "worker").glob("*.results.json"):
            leftover.unlink()
        return _save(call, store, staging, result, artifacts, doc, model, ref_req, run_dir)


def _side_items(plan_side: dict, by_pair: dict) -> dict[str, dict]:
    return {k: by_pair[v["pair"]] for k, v in plan_side.items()
            if k != "__extra__" and "pair" in v and v["pair"] in by_pair}


def _per_clip(items: dict[str, dict]) -> list[dict]:
    """[{id, round, psnr_db, ssim, lpips}] per scored clip (its mean over frames), in key order."""
    out = []
    for k, it in sorted(items.items()):
        r, cid = split_key(k)
        mean = it.get("mean") or {}
        out.append({"id": cid, "round": r, **{NAMES[m]: mean.get(m) for m in METRICS}})
    return out


def _score(plan, by_pair, ref, floors, engines, metrics, computed, margins, fidelity, budget, spot, resamples, seed,
           lpips_error, groups, not_timed) -> dict:
    floor_side = "floor:alt" if "alt" in floors else "floor:repeat" if "repeat" in floors else None
    floor_items = {s: _side_items(plan[s], by_pair) for s in plan if s.startswith("floor:")}
    floor_info = {}
    for s, items in floor_items.items():
        ok = {k: it for k, it in items.items() if it.get("ok")}
        floor_info[s.split(":", 1)[1]] = {
            "n_clips": len(ok), "exact_clips": sum(1 for it in ok.values() if _exact(it)),
            "missing": sorted(k for k, v in plan[s].items() if k != "__extra__" and v.get("missing")),
            "request_mismatch": {k: v["request_diff"] for k, v in plan[s].items() if k != "__extra__"
                                 and v.get("request_diff")},
            "metrics": {m: summary_of(ok, groups, m, resamples, seed) for m in computed}}
    out_engines = {}
    for label in engines:
        side = f"engine:{label}"
        items = _side_items(plan[side], by_pair)
        missing = sorted(k for k, v in plan[side].items() if k != "__extra__" and v.get("missing"))
        mismatch = {k: v["request_diff"] for k, v in plan[side].items() if k != "__extra__" and v.get("request_diff")}
        unchecked = sum(1 for k, v in plan[side].items() if k != "__extra__" and "pair" in v
                        and not v.get("request_checked"))
        changed = {k: why for k, why in not_timed.get(side, {}).items() if k in items}
        shape_bad = {k: (it.get("error") or f"{it.get('frames_b')} frames vs the reference's {it.get('frames_a')}")
                     for k, it in items.items() if k not in changed and (not it.get("ok")
                                                                          or it.get("frame_count_differs"))}
        good = {k: it for k, it in items.items() if k not in shape_bad and k not in changed}
        checks, caps = {}, []
        if changed:
            k0 = next(iter(changed))
            checks["outputs_index"] = {"verdict": "FAIL", "reason": f"{len(changed)} clip file(s) are not the clips "
                                       f"bench clip timed (e.g. {k0}: {changed[k0]})"}
        if shape_bad:
            k0 = next(iter(shape_bad))
            checks["shape"] = {"verdict": "FAIL", "reason": f"{len(shape_bad)} clip(s) differ from the reference in "
                               f"frame size or frame count (e.g. {k0}: {shape_bad[k0]})"}
        if missing:
            caps.append(f"{len(missing)} reference clip(s) have no engine clip (e.g. {missing[0]})")
        if mismatch:
            k0 = next(iter(mismatch))
            caps.append(f"{len(mismatch)} clip(s) were made from a different request than the reference's (e.g. {k0}: "
                        f"{', '.join(mismatch[k0])}) and were not scored")
        block = {"n_clips_scored": len(good), "n_requests_scored": len({groups.get(k, k) for k in good}),
                 "missing": missing, "request_mismatch": mismatch, "requests_unchecked": unchecked,
                 "shape_mismatch": shape_bad, "not_timed": changed, "extra_clips": plan[side].get("__extra__", []),
                 "metrics": {m: summary_of(good, groups, m, resamples, seed) for m in computed},
                 "exact_clips": sum(1 for it in good.values() if _exact(it)), "per_clip": _per_clip(good)}
        if fidelity in ("lossless", "gross"):
            if floor_side is None:
                caps.append("no A/A floor (pass --reference-repeat and/or --reference-alt)")
                for m in metrics:
                    checks[m] = {"verdict": "NOT RUN", "reason": "no A/A floor"}
            else:
                fl = floor_items[floor_side]
                keys = [k for k in good if k in fl and fl[k].get("ok") and not fl[k].get("frame_count_differs")]
                for m in metrics:
                    if m not in computed:
                        checks[m] = {"verdict": "NOT RUN", "reason": f"not computed ({lpips_error})"}
                        continue
                    checks[m] = lossless_check(good, fl, keys, groups, m, margins[m], resamples, seed)
                exact_floor = keys and all(_exact(fl[k]) for k in keys)
                exact_engine = keys and all(_exact(good[k]) for k in keys)
                if fidelity == "gross" and not (exact_floor and not exact_engine):
                    for m, c in checks.items():
                        if m in METRICS and c["verdict"] == "UNPROVEN" and None not in (c.get("ci95") or [None]):
                            c["verdict"] = "PASS"
                            c["reason"] = f"gross gate, not entirely beyond the margin: {c['reason']}"
                if exact_floor and not exact_engine:  # whichever floor: exact cannot judge a drifting kernel
                    for m, c in checks.items():
                        if m in METRICS and c["verdict"] in ("PASS", "FAIL"):
                            c["verdict_before_exact_floor_rule"] = c["verdict"]
                            c["verdict"] = "UNPROVEN"
                    if floor_side == "floor:alt":
                        caps.append("the --reference-alt run is bit-identical to the reference (was the attention "
                                    "backend switch applied?), so the A/A floor is exact and cannot separate an "
                                    "exact-but-different kernel from a lossy one")
                    else:
                        caps.append("the A/A floor is exact (the reference repeats itself bit for bit), so it "
                                    "cannot separate an exact-but-different kernel from a lossy one: pass "
                                    "--reference-alt (the reference with another exact attention backend)")
                block["floor"] = floor_side.split(":", 1)[1]
                block["n_clips_judged"] = len(keys)
                if len(keys) < len(good):
                    caps.append(f"{len(good) - len(keys)} scored clip(s) have no usable floor clip (missing, made "
                                "from another request, or a different frame count) and were not judged")
        else:
            for k, (m, sidek) in BUDGET.items():
                if k in (budget or {}):
                    if m not in computed:
                        checks[f"budget.{k}"] = {"verdict": "NOT RUN", "reason": f"{m} not computed ({lpips_error})"}
                    else:
                        checks[f"budget.{k}"] = budget_check(good, groups, m, sidek, budget[k], resamples, seed)
            if spot is None:
                checks["spot_check"] = {"verdict": "UNPROVEN", "reason": "lossy: no human spot-check record "
                                        "(--spot-check)"}
            else:
                checks["spot_check"] = {"verdict": "PASS" if spot["verdict"] == "acceptable" else "FAIL",
                                        "reason": f"{spot['reviewer']} ({spot['date']}): {spot['verdict']}, "
                                                  f"{len(spot['clips'])} clip(s) watched"}
            if floor_side is not None:  # for the reader: how far the floor already is, no verdict
                fl = floor_items[floor_side]
                block["vs_floor"] = {m: summary_of({k: fl[k] for k in good if k in fl and fl[k].get("ok")}, groups,
                                                   m, resamples, seed) for m in computed}
        verdict = stats.combine_claims("UNPROVEN" if c["verdict"] == "NOT RUN" else c["verdict"]
                                       for c in checks.values())
        if verdict in ("PASS", "NOT RUN") and caps:
            verdict = "UNPROVEN"
        block.update({"verdict": verdict, "checks": checks, "caps": caps,
                      "reason": "; ".join([f"{k}: {c['reason']}" for k, c in checks.items()] + caps)})
        out_engines[label] = block
    overall = stats.combine_claims(b["verdict"] for b in out_engines.values())
    return {"verdict": overall, "fidelity": fidelity, "engines": out_engines, "floors": floor_info,
            "floor_used": floor_side.split(":", 1)[1] if floor_side else None, "reference_clips": len(ref),
            "reference_requests": len(set(groups.values())),
            "margins": {m: margins[m] for m in metrics}, "budget": budget, "spot_check": spot, "metrics": metrics,
            "computed": computed, "lpips_error": lpips_error}


def _uniform(reqs: dict[str, dict], key: str):
    vals = {json.dumps(r.get(key), sort_keys=True) for r in reqs.values()}
    return next(iter(reqs.values())).get(key) if reqs and len(vals) == 1 else None


def _save(call, store, staging, result, artifacts, doc, model, ref_req, run_dir) -> dict:
    mets = []
    for label, b in result["engines"].items():  # n = distinct requests, the bootstrap unit of the CI
        for m, s in b["metrics"].items():
            mets.append(ev.metric(NAMES[m], s["value"], UNITS[m], "mean_over_frames", n=s["n_requests"],
                                  ci95=s["ci95"], subject=label))
        for m, c in b["checks"].items():
            if c.get("worse_than_floor") is not None:
                mets.append(ev.metric(f"{NAMES[m]}_worse_than_floor", c["worse_than_floor"], UNITS[m],
                                      "paired_delta", n=c["n_requests"], ci95=c["ci95"], subject=label))
    for name, f in result["floors"].items():
        for m, s in f["metrics"].items():
            mets.append(ev.metric(NAMES[m], s["value"], UNITS[m], "mean_over_frames", n=s["n_requests"],
                                  ci95=s["ci95"], subject=f"floor:{name}"))
    lim = [f"{label}: {c}" for label, b in result["engines"].items() for c in b["caps"]]
    if result["lpips_error"]:
        lim.append(f"LPIPS not computed: {result['lpips_error'][:200]}")
    if result["fidelity"] == "lossy":
        lim.append("lossy track: report as 'N x at the stated quality cost'; never ranked against lossless results")
    unchecked = sum(b["requests_unchecked"] for b in result["engines"].values())
    if unchecked:
        lim.append(f"{unchecked} clip pair(s) had no bench clip request body on one side: same prompt, seed and "
                   "steps were not checked for them")
    workload = {k: _uniform(ref_req, k) for k in ("task", "width", "height", "num_frames", "fps", "steps",
                                                  "guidance_scale")} if ref_req else {}
    labels = list(result["engines"])
    summary = f"quality clip {result['verdict']} ({result['fidelity']}, floor {result['floor_used'] or 'none'}): " + \
        ", ".join(f"{lab} {b['verdict']}" + (f" (PSNR {b['metrics']['psnr']['value']:.2f} dB)"
                                               if b["metrics"].get("psnr", {}).get("value") is not None else "")
                  for lab, b in result["engines"].items())
    kind, extra = _kind("clip_quality")
    worker = {k: doc.get(k) for k in ("versions", "device", "ssim_backend", "frame_stride", "psnr_cap_db")}
    data = {**extra, **result, "format": FORMAT, "worker": worker,
            "definitions": {"psnr_db": "10 log10(255^2 / MSE) per frame over RGB24, capped at 100 dB; clip = mean "
                                       "over frames",
                            "ssim": "luma (BT.601), 11x11 Gaussian sigma 1.5, K1 0.01, K2 0.03",
                            "lpips": "LPIPS v0.1, AlexNet backbone, RGB in [-1, 1]",
                            "worse_than_floor": "engine-vs-reference minus floor-vs-reference, oriented so positive "
                                                "is worse; paired bootstrap 95% CI over requests",
                            "request": "the bootstrap unit: clips whose request bodies match on every REQUEST_KEYS "
                                       "field (or, without bodies, every round of one clip id) count once; metric n "
                                       "is the number of requests",
                            "per_clip": "each scored clip's mean over frames; id and round from its r<round>/<id> "
                                        "key (round null for other layouts)"}}
    if len(labels) == 1:  # the review page and a re-run read the engine and its clips from here
        data["label"] = labels[0]
        data["per_clip"] = result["engines"][labels[0]]["per_clip"]
    rec = ev.build(call, kind=kind, claim_class="observed", summary=summary,
                   identity_=ev.identity(engine={"name": labels[0] if len(labels) == 1 else None},
                                         model={"id": model}, workload=workload),
                   metrics=mets, artifacts=artifacts, missing=[] if any(
                       b["n_clips_scored"] for b in result["engines"].values()) else ["no clip pair was scored"],
                   limitations=lim, data=data)
    if run_dir:  # the store also copies the record to <run-dir>/evidence/
        out = Path(run_dir) / "quality"
        out.mkdir(parents=True, exist_ok=True)
        (out / "clip_score.json").write_text(json.dumps(result, indent=1) + "\n")
    return store.save(rec, staging)

