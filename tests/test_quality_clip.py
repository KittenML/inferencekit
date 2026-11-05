"""quality clip and its metrics worker (quality/video_metrics.py) on tiny synthetic clips (CPU only, no GPU).

The clips are numpy frames written with PyAV, so these tests need a Python with PyAV and numpy: the one in
INFERENCEKIT_VIDEO_PYTHON, else this interpreter. The harness itself has neither (CI skips these tests).
Clips are lossless (ffv1 in Matroska) where a test needs exact pixels, and H.264 + AAC in MP4 for decoding.
"""

import importlib
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from inferencekit import registry
from inferencekit.registry import OpError


def _video_python():
    for py in (os.environ.get("INFERENCEKIT_VIDEO_PYTHON"), sys.executable):
        if py and subprocess.run([py, "-c", "import av, numpy"], capture_output=True).returncode == 0:
            return py
    return None


PY = _video_python()
needs_video = pytest.mark.skipif(PY is None, reason="needs a Python with PyAV and numpy (INFERENCEKIT_VIDEO_PYTHON)")

# Run under PY: writes the clips described by a JSON list.
WRITER = r'''
import json, sys, fractions
import numpy as np, av

def frames_of(s):
    rng = np.random.default_rng(s["seed"])
    h, w, n = s["h"], s["w"], s["frames"]
    grid = rng.uniform(40, 215, (4, 4, 3))
    ys, xs = np.linspace(0, 3, h)[:, None], np.linspace(0, 3, w)[None, :]
    y0, x0 = np.floor(ys).astype(int).clip(0, 2), np.floor(xs).astype(int).clip(0, 2)
    fy, fx = (ys - y0)[..., None], (xs - x0)[..., None]
    img = (grid[y0, x0] * (1 - fy) * (1 - fx) + grid[y0 + 1, x0] * fy * (1 - fx)
           + grid[y0, x0 + 1] * (1 - fy) * fx + grid[y0 + 1, x0 + 1] * fy * fx)
    nrng = np.random.default_rng(s.get("noise_seed", 0))
    out = []
    for t in range(n):
        f = np.roll(img, 2 * t, axis=1) + s.get("tint", 0.0)
        if s.get("noise"):
            f = f + nrng.integers(-s["noise"], s["noise"] + 1, f.shape)
        out.append(np.clip(np.rint(f), 0, 255).astype(np.uint8))
    return out

def write(s):
    frames = frames_of(s)
    c = av.open(s["path"], "w")
    if s["path"].endswith(".mkv"):
        v = c.add_stream("ffv1", rate=fractions.Fraction(s["fps"]).limit_denominator(1001))
        v.width, v.height, v.pix_fmt = s["w"], s["h"], "bgr0"
        conv = lambda f: av.VideoFrame.from_ndarray(f, format="rgb24").reformat(format="bgr0")
    else:
        v = c.add_stream("libx264", rate=fractions.Fraction(s["fps"]).limit_denominator(1001))
        v.width, v.height, v.pix_fmt = s["w"], s["h"], "yuv420p"
        v.options = {"crf": "18", "threads": "1"}
        conv = lambda f: av.VideoFrame.from_ndarray(f, format="rgb24")
    a = None
    if s.get("audio_s"):
        a = c.add_stream("aac", rate=48000)
        a.layout = "mono"
    for f in frames:
        for p in v.encode(conv(f)):
            c.mux(p)
    for p in v.encode():
        c.mux(p)
    if a is not None:
        n = int(s["audio_s"] * 48000)
        x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / 48000)).astype(np.float32)[None]
        for i in range(0, n, 1024):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(x[:, i:i + 1024]), format="flt", layout="mono")
            fr.sample_rate, fr.pts = 48000, i
            for p in a.encode(fr):
                c.mux(p)
        for p in a.encode():
            c.mux(p)
    c.close()

for s in json.load(open(sys.argv[1])):
    write(s)
'''


def write_clips(tmp_path: Path, specs: list[dict]) -> None:
    script, spec = tmp_path / "writer.py", tmp_path / f"spec-{len(list(tmp_path.glob('spec-*')))}.json"
    script.write_text(WRITER)
    for s in specs:
        Path(s["path"]).parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps([{"w": 64, "h": 48, "frames": 6, "fps": 24, **s} for s in specs]))
    subprocess.run([PY, str(script), str(spec)], check=True, capture_output=True)


@pytest.fixture(scope="module")
def qc():
    """Import quality clip; leave the registry as it was (the contract test lists every op), and drop the module
    too, so a later import registers it again (a cached module does not)."""
    before = set(registry.load_all())
    mod = importlib.import_module("inferencekit.ops.quality_clip")
    if "quality_clip" not in registry.REGISTRY:
        mod = importlib.reload(mod)
    yield mod
    for name in set(registry.REGISTRY) - before:
        module = registry.REGISTRY.pop(name).handler.__module__
        sys.modules.pop(module, None)
        pkg, _, attr = module.rpartition(".")
        if pkg in sys.modules and attr in vars(sys.modules[pkg]):
            delattr(sys.modules[pkg], attr)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def make_run(tmp_path, name, n=4, frames=6, **kw) -> Path:
    """A directory of n lossless clips c0..c(n-1); kw are added to every clip's spec (noise, tint...)."""
    d = tmp_path / name
    specs = []
    for i in range(n):
        s = {"path": str(d / f"c{i}.mkv"), "seed": 100 + i, "frames": frames, **kw}
        if "noise" in kw:
            s["noise_seed"] = zlib.crc32(f"{name}:{i}".encode()) & 0xFFFF
        specs.append(s)
    write_clips(tmp_path, specs)
    return d


def score(store, ref, outputs, metrics=("psnr", "ssim"), **kw):
    args = {"reference": str(ref), "outputs": [str(o) for o in outputs], "python": PY, "metrics": list(metrics),
            "resamples": 400, **{k: str(v) if isinstance(v, Path) else v for k, v in kw.items()}}
    return registry.call("quality_clip", args)


# --------------------------------------------------------------------------- the worker

@needs_video
def test_worker_scores_identical_noisy_and_resized_clips(qc, tmp_path):
    write_clips(tmp_path, [{"path": str(tmp_path / "a.mkv"), "seed": 1},
                           {"path": str(tmp_path / "b.mkv"), "seed": 1},
                           {"path": str(tmp_path / "n.mkv"), "seed": 1, "noise": 3, "noise_seed": 5},
                           {"path": str(tmp_path / "s.mkv"), "seed": 1, "w": 32},
                           {"path": str(tmp_path / "f.mkv"), "seed": 1, "frames": 4}])
    pairs = [{"key": k, "a": str(tmp_path / "a.mkv"), "b": str(tmp_path / f"{k}.mkv")} for k in "bnsf"]
    doc = qc.run_worker(PY, {"task": "compare", "pairs": pairs, "metrics": ["psnr", "ssim", "mad"]},
                        tmp_path / "w", "t", 300)
    by = {it["key"]: it for it in doc["items"]}
    assert by["b"]["ok"] and by["b"]["identical_frames"] == 6 and by["b"]["mean"] == {"psnr": 100.0, "ssim": 1.0,
                                                                                     "mad": 0.0}
    n = by["n"]
    assert n["ok"] and n["identical_frames"] == 0 and 35 < n["mean"]["psnr"] < 50 and n["mean"]["ssim"] < 1.0
    assert 1.0 < n["mean"]["mad"] < 2.5  # uniform integers in [-3, 3]: mean |x| = 12/7, a little less after clipping
    assert not by["s"]["ok"] and "frame size differs" in by["s"]["error"]
    assert by["f"]["ok"] and by["f"]["frame_count_differs"] and (by["f"]["frames_a"], by["f"]["frames_b"]) == (6, 4)
    assert not list((tmp_path / "w").glob("*.config.json"))  # the config (local paths) is not kept


@needs_video
def test_worker_ssim_matches_the_reference_formula(qc, tmp_path):
    """SSIM against an independent float64 implementation of Wang et al. (2004) with the same window."""
    write_clips(tmp_path, [{"path": str(tmp_path / "a.mkv"), "seed": 2},
                           {"path": str(tmp_path / "b.mkv"), "seed": 2, "noise": 9, "noise_seed": 1}])
    check = tmp_path / "check.py"
    check.write_text(r'''
import sys, importlib.util, numpy as np, av
spec = importlib.util.spec_from_file_location("vm", sys.argv[1]); vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)
a = next(iter(av.open(sys.argv[2]).decode(video=0))).to_ndarray(format="rgb24")
b = next(iter(av.open(sys.argv[3]).decode(video=0))).to_ndarray(format="rgb24")
L = np.array([0.299, 0.587, 0.114])
ya, yb = a.astype(float) @ L, b.astype(float) @ L
g = np.exp(-((np.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2)); w2 = np.outer(g, g) / np.outer(g, g).sum()
H, W = ya.shape
def filt(x):
    return np.array([[np.sum(w2 * x[i:i + 11, j:j + 11]) for j in range(W - 10)] for i in range(H - 10)])
c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
ma, mb = filt(ya), filt(yb)
va, vb, cov = filt(ya * ya) - ma ** 2, filt(yb * yb) - mb ** 2, filt(ya * yb) - ma * mb
ref = np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))
print(abs(vm.Scorer(["ssim"], "cpu", 100, "alex").ssim(a, b) - ref))
''')
    r = subprocess.run([PY, str(check), str(qc.WORKER), str(tmp_path / "a.mkv"), str(tmp_path / "b.mkv")],
                       capture_output=True, text=True, check=True)
    assert float(r.stdout.strip()) < 1e-9


@needs_video
def test_worker_probe_and_ffprobe_agree_on_decoded_facts(qc, tmp_path):
    write_clips(tmp_path, [{"path": str(tmp_path / "a.mp4"), "seed": 3, "frames": 9, "audio_s": 9 / 24},
                           {"path": str(tmp_path / "b.mp4"), "seed": 3, "frames": 7, "w": 32, "fps": 12}])
    files = [("a", tmp_path / "a.mp4"), ("b", tmp_path / "b.mp4")]
    pyav, dec = qc.probe_clips(files, PY, tmp_path / "w", 300)
    assert dec == "pyav"
    a, b = pyav["a"], pyav["b"]
    assert (a["video"]["width"], a["video"]["height"], a["video"]["frames"]) == (64, 48, 9)
    assert abs(a["video"]["fps"] - 24) < 1e-6 and abs(a["audio"]["duration_s"] - 9 / 24) < 0.05
    assert (b["video"]["width"], b["video"]["frames"]) == (32, 7) and abs(b["video"]["fps"] - 12) < 1e-6
    assert b["audio"] is None
    if qc.which("ffprobe") is None:
        pytest.skip("no ffprobe on PATH")
    ff, dec = qc.probe_clips(files, None, tmp_path / "w", 300)
    assert dec == "ffprobe"
    for k in ("a", "b"):
        for f in ("width", "height", "frames"):
            assert ff[k]["video"][f] == pyav[k]["video"][f], (k, f)
        assert abs(ff[k]["video"]["fps"] - pyav[k]["video"]["fps"]) < 1e-3  # ffprobe prints 6 decimals
    assert abs(ff["a"]["audio"]["duration_s"] - pyav["a"]["audio"]["duration_s"]) < 0.05
    assert ff["a"]["header"]["video"]["frames"] == 9


def test_worker_reports_missing_modules(qc, tmp_path):
    """Under an interpreter without PyAV (this harness in CI) the op says what to pass instead."""
    if PY == sys.executable:
        pytest.skip("this interpreter has PyAV")
    with pytest.raises(OpError, match="--python"):
        qc.run_worker(sys.executable, {"task": "probe", "files": [{"key": "a", "path": "x.mp4"}]}, tmp_path, "t", 60)


# --------------------------------------------------------------------------- lossless verdicts

@needs_video
def test_lossless_pass_against_the_kernel_floor(qc, store, tmp_path):
    ref = make_run(tmp_path, "ref")
    alt = make_run(tmp_path, "alt", noise=2)  # an exact but different attention kernel: tiny drift
    rep = make_run(tmp_path, "rep")  # deterministic: the repeat is bit-exact
    eng = make_run(tmp_path, "fast", noise=2)
    rec = score(store, ref, [eng], reference_alt=alt, reference_repeat=rep, run_dir=tmp_path / "run")
    d = rec["data"]
    assert rec["kind"] == ("clip_quality" if "clip_quality" in qc.ev.KINDS else "correctness")
    assert d["verdict"] == "PASS", d["engines"]["fast"]["reason"]
    assert d["floor_used"] == "alt" and d["floors"]["repeat"]["exact_clips"] == 4
    e = d["engines"]["fast"]
    assert e["n_clips_judged"] == 4 and e["checks"]["psnr"]["n_clips"] == 4
    assert abs(e["checks"]["psnr"]["worse_than_floor"]) < 1.0
    # The names the review page and a re-run read: psnr_db, data.label, data.per_clip.
    assert {m["name"] for m in rec["metrics"]} >= {"psnr_db", "ssim", "psnr_db_worse_than_floor"}
    assert next(m for m in rec["metrics"] if m["name"] == "psnr_db" and m["subject"] == "fast")["unit"] == "dB"
    assert d["label"] == "fast" and [r["id"] for r in d["per_clip"]] == ["c0", "c1", "c2", "c3"]
    assert all(r["round"] is None and 35 < r["psnr_db"] < 60 and r["ssim"] < 1 for r in d["per_clip"])
    assert (tmp_path / "run" / "quality" / "clip_score.json").is_file()
    assert (tmp_path / "run" / "evidence" / f"{rec['evidence_id']}.json").is_file()
    rows = (store.run_dir(rec["evidence_id"]) / "clip_metrics.jsonl").read_text().splitlines()
    assert len(rows) == 12 and str(tmp_path) not in "".join(rows)  # keys only, no local paths


@needs_video
def test_lossless_fail_when_the_engine_is_far_beyond_the_floor(qc, store, tmp_path):
    ref = make_run(tmp_path, "ref")
    alt = make_run(tmp_path, "alt", noise=1)
    lossy = make_run(tmp_path, "lossy", noise=24)
    d = score(store, ref, [lossy], reference_alt=alt)["data"]
    e = d["engines"]["lossy"]
    assert d["verdict"] == "FAIL" and e["checks"]["psnr"]["verdict"] == "FAIL"
    assert e["checks"]["psnr"]["worse_than_floor"] > 5


@needs_video
def test_gross_gate_fails_only_far_beyond_the_floor(qc, store, tmp_path):
    """--fidelity gross (beside quality clip-steps): a straddling CI passes, a gross failure fails, an exact floor
    cannot judge a drifting engine."""
    ref, alt = make_run(tmp_path, "ref"), make_run(tmp_path, "alt", noise=1)
    near, far = make_run(tmp_path, "near", noise=1), make_run(tmp_path, "far", noise=24)
    d = score(store, ref, [near, far], reference_alt=alt, fidelity="gross", margin_psnr_db=0.0, margin_ssim=0.0)
    e = d["data"]["engines"]
    assert e["near"]["verdict"] == "PASS" and "gross gate" in e["near"]["checks"]["psnr"]["reason"]
    assert e["far"]["verdict"] == "FAIL" and e["far"]["checks"]["psnr"]["verdict"] == "FAIL"
    rep, drift = make_run(tmp_path, "rep"), make_run(tmp_path, "drift", noise=2)
    d = score(store, ref, [drift], reference_repeat=rep, fidelity="gross", margin_psnr_db=6.0, margin_ssim=0.2)
    assert d["data"]["engines"]["drift"]["verdict"] == "UNPROVEN"


@needs_video
def test_an_exact_floor_cannot_pass_or_fail_a_different_kernel(qc, store, tmp_path):
    ref, rep = make_run(tmp_path, "ref"), make_run(tmp_path, "rep")
    drift, same = make_run(tmp_path, "drift", noise=2), make_run(tmp_path, "same")
    d = score(store, ref, [drift, same], reference_repeat=rep)["data"]
    assert d["floor_used"] == "repeat"
    assert d["engines"]["drift"]["verdict"] == "UNPROVEN"
    assert d["engines"]["drift"]["checks"]["psnr"]["verdict_before_exact_floor_rule"] == "FAIL"
    assert "--reference-alt" in d["engines"]["drift"]["reason"]
    assert d["engines"]["same"]["verdict"] == "PASS"  # bit-exact engine: exact floor is enough
    assert "label" not in d and "per_clip" not in d  # two engines: per engine only
    assert len(d["engines"]["drift"]["per_clip"]) == 4


@needs_video
def test_an_exact_alt_floor_is_no_better_than_an_exact_repeat(qc, store, tmp_path):
    """--reference-alt bit-identical to the reference (the backend switch was not applied): same rule."""
    ref, alt = make_run(tmp_path, "ref"), make_run(tmp_path, "alt")
    eng = make_run(tmp_path, "eng", noise=2)
    rec = score(store, ref, [eng], reference_alt=alt)
    e = rec["data"]["engines"]["eng"]
    assert rec["data"]["floor_used"] == "alt" and rec["data"]["floors"]["alt"]["exact_clips"] == 4
    assert e["verdict"] == "UNPROVEN"
    assert e["checks"]["psnr"]["verdict_before_exact_floor_rule"] == "FAIL"
    assert any("--reference-alt run is bit-identical" in x for x in rec["limitations"])


@needs_video
def test_shape_missing_clips_and_no_floor(qc, store, tmp_path):
    ref, alt = make_run(tmp_path, "ref"), make_run(tmp_path, "alt", noise=2)
    short = make_run(tmp_path, "short", frames=5, noise=2)
    part = make_run(tmp_path, "part", n=3, noise=2)
    d = score(store, ref, [short, part], reference_alt=alt)["data"]
    assert d["engines"]["short"]["verdict"] == "FAIL" and d["engines"]["short"]["checks"]["shape"]["verdict"] == "FAIL"
    assert d["engines"]["part"]["verdict"] == "UNPROVEN" and d["engines"]["part"]["missing"] == ["c3"]
    d = score(store, ref, [part])["data"]
    assert d["verdict"] == "UNPROVEN" and "no A/A floor" in d["engines"]["part"]["reason"]
    thin = make_run(tmp_path, "thin", n=3, noise=2)  # a floor run that lacks c3
    full = make_run(tmp_path, "full", noise=2)
    e = score(store, ref, [full], reference_alt=thin)["data"]["engines"]["full"]
    assert e["n_clips_judged"] == 3 and e["verdict"] == "UNPROVEN" and "no usable floor clip" in e["reason"]


@needs_video
def test_request_mismatch_from_bench_clip_layout(qc, store, tmp_path):
    """bench clip keeps request bodies under inputs/<label>/: a clip made from another seed is not scored."""
    def bench_dir(name, seeds, noise=0):
        root = tmp_path / f"run-{name}"
        specs = [{"path": str(root / "outputs" / name / "r0" / f"c{i}.mkv"), "seed": 100 + i, "noise": noise,
                  "noise_seed": i + 7} for i in range(3)]
        write_clips(tmp_path, specs)
        for i in range(3):
            p = root / "inputs" / name / "r0" / f"c{i}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"task": "t2v", "prompt": f"clip {i}", "width": 64, "height": 48,
                                     "num_frames": 6, "fps": 24, "steps": 8, "guidance_scale": 3.0,
                                     "seed": seeds[i]}))
        return root / "outputs" / name

    ref, alt = bench_dir("ref", [1, 1, 1]), bench_dir("alt", [1, 1, 1], noise=2)
    eng = bench_dir("eng", [1, 2, 1], noise=2)
    rec = score(store, ref, [eng], reference_alt=alt)
    e = rec["data"]["engines"]["eng"]
    assert e["request_mismatch"] == {"r0/c1": ["seed"]} and e["n_clips_scored"] == 2
    assert e["verdict"] == "UNPROVEN" and "different request" in e["reason"]
    assert rec["identity"]["workload"].get("steps") in (8, None)  # written once the clip workload fields exist


def bench_layout(tmp_path, name, rounds, noise=0, n=4) -> Path:
    """bench clip's layout (outputs/<name>/r<k>/c<i>.mkv + inputs/<name>/r<k>/c<i>.json) where every round repeats
    the same requests (--seed-policy fixed) and a deterministic pipeline makes the same clips again."""
    root = tmp_path / f"run-{name}-{rounds}"
    specs = []
    for r in range(rounds):
        for i in range(n):
            specs.append({"path": str(root / "outputs" / name / f"r{r}" / f"c{i}.mkv"), "seed": 100 + i,
                          "noise": noise, "noise_seed": zlib.crc32(f"{name}:{i}".encode()) & 0xFFFF})
            p = root / "inputs" / name / f"r{r}" / f"c{i}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"task": "t2v", "prompt": f"clip {i}", "width": 64, "height": 48,
                                     "num_frames": 6, "fps": 24, "steps": 8, "guidance_scale": 3.0, "seed": 5}))
    write_clips(tmp_path, specs)
    return root / "outputs" / name


@needs_video
def test_repeated_requests_are_one_bootstrap_unit(qc, store, tmp_path):
    """Two rounds of the same requests: twice the clips, the same information, so the same CI as one round."""
    out = {}
    for rounds in (1, 2):
        ref = bench_layout(tmp_path, "ref", rounds)
        alt = bench_layout(tmp_path, "alt", rounds, noise=2)
        eng = bench_layout(tmp_path, "eng", rounds, noise=3)
        rec = score(store, ref, [eng], reference_alt=alt)
        out[rounds] = rec
    one, two = (out[r]["data"]["engines"]["eng"] for r in (1, 2))
    for e, clips in ((one, 4), (two, 8)):
        assert e["checks"]["psnr"]["n_clips"] == clips and e["checks"]["psnr"]["n_requests"] == 4
        assert e["metrics"]["psnr"]["n_clips"] == clips and e["metrics"]["psnr"]["n_requests"] == 4
    for m in ("psnr", "ssim"):
        assert two["checks"][m]["ci95"] == pytest.approx(one["checks"][m]["ci95"], abs=1e-9)
        assert two["metrics"][m]["ci95"] == pytest.approx(one["metrics"][m]["ci95"], abs=1e-9)
    assert two["n_requests_scored"] == 4 and out[2]["data"]["reference_requests"] == 4
    assert next(m for m in out[2]["metrics"] if m["name"] == "psnr_db" and m["subject"] == "eng")["n"] == 4
    assert [(r["round"], r["id"]) for r in out[2]["data"]["per_clip"]][:5] == [(0, "c0"), (0, "c1"), (0, "c2"),
                                                                              (0, "c3"), (1, "c0")]
    groups = qc.request_groups(["r0/c0", "r1/c0", "r0/c1", "c2"], [{"r0/c0": {"seed": 1}, "r1/c0": {"seed": 2}}])
    assert len({groups["r0/c0"], groups["r1/c0"]}) == 2  # another seed: another request
    assert groups["r0/c1"] == "clip c1" and groups["c2"] == "clip c2"  # no body: by clip id


@needs_video
def test_an_engine_clip_that_is_not_the_timed_clip_fails(qc, store, tmp_path):
    """bench clip's outputs/<label>/index.json lists each timed clip's sha256: a swapped file is a FAIL."""
    ref, alt, eng = make_run(tmp_path, "ref"), make_run(tmp_path, "alt", noise=2), make_run(tmp_path, "eng", noise=2)
    sha = {p.name: qc.file_sha256(p)[0] for p in sorted(eng.glob("*.mkv"))}
    (eng / "index.json").write_text(json.dumps({"label": "eng", "files": {**sha, "c1.mkv": sha["c0.mkv"]}}))
    d = score(store, ref, [eng], reference_alt=alt)["data"]
    e = d["engines"]["eng"]
    assert d["verdict"] == e["verdict"] == "FAIL" and e["checks"]["outputs_index"]["verdict"] == "FAIL"
    assert list(e["not_timed"]) == ["c1"] and e["n_clips_scored"] == 3
    (eng / "index.json").write_text(json.dumps({"label": "eng", "files": sha}))
    assert score(store, ref, [eng], reference_alt=alt)["data"]["engines"]["eng"]["not_timed"] == {}


# --------------------------------------------------------------------------- lossy track

@needs_video
def test_lossy_budget_and_human_spot_check(qc, store, tmp_path):
    ref, eng = make_run(tmp_path, "ref"), make_run(tmp_path, "nvfp4", noise=4)
    budget = json.dumps({"psnr_db_min": 30, "ssim_min": 0.5})
    d = score(store, ref, [eng], fidelity="lossy", budget=budget)["data"]
    e = d["engines"]["nvfp4"]
    assert e["checks"]["budget.psnr_db_min"]["verdict"] == "PASS"
    assert e["checks"]["spot_check"]["verdict"] == "UNPROVEN" and d["verdict"] == "UNPROVEN"
    spot = tmp_path / "spot.json"
    spot.write_text(json.dumps({"reviewer": "R. Viewer", "date": "2026-10-15", "clips": ["c0", "c2"],
                                "verdict": "acceptable", "notes": "no visible artefacts"}))
    rec = score(store, ref, [eng], fidelity="lossy", budget=budget, spot_check=spot)
    assert rec["data"]["verdict"] == "PASS" and rec["data"]["spot_check"]["sha256"]
    assert any("lossy track" in x for x in rec["limitations"])
    spot.write_text(json.dumps({"reviewer": "R. Viewer", "date": "2026-10-15", "clips": ["c1"],
                                "verdict": "not_acceptable"}))
    assert score(store, ref, [eng], fidelity="lossy", budget=budget, spot_check=spot)["data"]["verdict"] == "FAIL"
    tight = json.dumps({"psnr_db_min": 60})
    assert score(store, ref, [eng], fidelity="lossy", budget=tight)["data"]["engines"]["nvfp4"]["checks"][
        "budget.psnr_db_min"]["verdict"] == "FAIL"


def test_argument_errors(qc, store, tmp_path):
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "c0.mp4").write_bytes(b"x")
    base = {"reference": str(tmp_path / "ref"), "outputs": [str(tmp_path / "ref")]}
    for extra, msg in (({"fidelity": "lossy"}, "needs --budget"),
                       ({"fidelity": "lossy", "budget": '{"kl_max": 1}'}, "unknown key"),
                       ({"budget": '{"lpips_max": 0.1}'}, "for --fidelity lossy"),
                       ({"metrics": ["vmaf"]}, "subset"),
                       ({"fidelity": "lossy", "budget": '{"lpips_max": 0.1}', "metrics": ["psnr"]}, "needs lpips"),
                       ({}, "same directory as a reference")):
        with pytest.raises(OpError, match=msg):
            registry.call("quality_clip", {**base, **extra})
    spot = tmp_path / "bad.json"
    spot.write_text(json.dumps({"reviewer": "", "verdict": "fine"}))
    with pytest.raises(OpError, match="reviewer"):
        qc.load_spot_check(str(spot))


# --------------------------------------------------------------------------- LPIPS (imports torch)

def test_lpips_scores_and_a_missing_backbone_is_unproven(qc, store, tmp_path, monkeypatch):
    # Checked here, not at collection: importing torch is heavy, and only this test needs it.
    if PY is None or subprocess.run([PY, "-c", "import lpips, torch"], capture_output=True).returncode != 0:
        pytest.skip("needs torch and lpips in the video Python")
    write_clips(tmp_path, [{"path": str(tmp_path / "a.mkv"), "seed": 4},
                           {"path": str(tmp_path / "b.mkv"), "seed": 4, "noise": 20, "noise_seed": 2}])
    check = tmp_path / "lp.py"  # LPIPS with an untrained backbone: exercises the code path, not the metric
    check.write_text(r'''
import sys, importlib.util, av, lpips, torch
spec = importlib.util.spec_from_file_location("vm", sys.argv[1]); vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)
s = vm.Scorer(["psnr"], "cpu", 100, "alex")
s.metrics.append("lpips"); s.torch, s.dev = torch, torch.device("cpu")
s.lpips = lpips.LPIPS(net="alex", pnet_rand=True, verbose=False).eval()
a = next(iter(av.open(sys.argv[2]).decode(video=0))).to_ndarray(format="rgb24")
b = next(iter(av.open(sys.argv[3]).decode(video=0))).to_ndarray(format="rgb24")
t = vm.Scorer(["ssim"], "cpu", 100, "alex", ssim_backend="torch")  # the GPU path's maths, run on the CPU
print(s.frame(a, a)["lpips"], s.frame(a, b)["lpips"], t.ssim_backend, abs(t.ssim(a, b) - s.ssim(a, b)))
''')
    r = subprocess.run([PY, str(check), str(qc.WORKER), str(tmp_path / "a.mkv"), str(tmp_path / "b.mkv")],
                       capture_output=True, text=True, check=True)
    same, diff, backend, ssim_gap = r.stdout.split()
    assert float(same) == 0.0 and float(diff) > 0.0
    assert backend == "torch" and float(ssim_gap) < 1e-9  # torch SSIM == numpy SSIM
    # No network for the backbone download: LPIPS is NOT RUN, so the verdict cannot be PASS.
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch-home"))
    for v in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        monkeypatch.setenv(v, "http://127.0.0.1:9")
    ref, alt, eng = make_run(tmp_path, "ref", n=2), make_run(tmp_path, "alt", n=2), make_run(tmp_path, "e", n=2)
    d = score(store, ref, [eng], metrics=("psnr", "ssim", "lpips"), reference_alt=alt, device="cpu")["data"]
    assert d["lpips_error"] and d["engines"]["e"]["checks"]["lpips"]["verdict"] == "NOT RUN"
    assert d["worker"]["ssim_backend"] == "numpy"  # no GPU here: SSIM stays in numpy
    assert d["engines"]["e"]["verdict"] == "UNPROVEN"
