"""gates clip against a fake clip engine that makes real (tiny) H.264 clips, honest or cheating (no GPU).

The fake engine is this file run as a script (``python tests/test_gates_clip.py --serve --port P``) under a
Python with PyAV and numpy (INFERENCEKIT_VIDEO_PYTHON, else this interpreter); everything above the
``__main__`` guard is stdlib only. ``POST /_mode`` switches how it cheats. The gate logic tests further down
use synthetic probes and need neither PyAV nor a server.
"""

import argparse
import base64
import hashlib
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- the fake clip engine

MODES = ("honest", "noisy", "ignore_steps", "fewer_steps_counter", "ignore_seed", "random_seed", "ignore_prompt",
         "ignore_cfg", "cache", "reported_cache", "lowres", "fewer_frames", "no_audio")


def render(req: dict, mode: str) -> bytes:
    """One clip as MP4 bytes: a smooth field from the seed, tinted by the prompt, contrast from the guidance
    scale, grain that shrinks with the step count, panning over the frames."""
    import fractions
    import io

    import av
    import numpy as np

    seed = 7 if mode == "ignore_seed" else random.getrandbits(31) if mode == "random_seed" else req["seed"]
    steps = 4 if mode == "ignore_steps" else req["steps"]
    g = 3.0 if mode == "ignore_cfg" else req["guidance_scale"]
    prompt = "" if mode == "ignore_prompt" else req["prompt"]
    w, h, n = req["width"], req["height"], req["num_frames"]
    if mode == "lowres":
        w, h = w // 2, h // 2
    if mode == "fewer_frames":
        n -= 2
    rng = np.random.default_rng(seed)
    grid = rng.uniform(40, 215, (4, 4, 3))
    ys, xs = np.linspace(0, 3, h)[:, None], np.linspace(0, 3, w)[None, :]
    y0, x0 = np.floor(ys).astype(int).clip(0, 2), np.floor(xs).astype(int).clip(0, 2)
    fy, fx = (ys - y0)[..., None], (xs - x0)[..., None]
    img = (grid[y0, x0] * (1 - fy) * (1 - fx) + grid[y0 + 1, x0] * fy * (1 - fx)
           + grid[y0, x0 + 1] * (1 - fy) * fx + grid[y0 + 1, x0 + 1] * fy * fx)
    tint = np.array([b % 41 - 20 for b in hashlib.sha256(prompt.encode()).digest()[:3]], dtype=float)
    grain = rng.uniform(-1, 1, (h, w, 3)) * (120.0 / steps)
    jitter = np.zeros((n, h, w, 3))
    if mode == "noisy":  # a few pixels off by one on every call, like a non-deterministic reduction order
        free = np.random.default_rng()
        jitter = free.integers(-1, 2, (n, h, w, 3)) * (free.random((n, h, w, 1)) < 0.03)
    buf = io.BytesIO()
    c = av.open(buf, "w", format="mp4")
    v = c.add_stream("libx264", rate=fractions.Fraction(req["fps"]).limit_denominator(1001))
    v.width, v.height, v.pix_fmt = w, h, "yuv420p"
    v.options = {"crf": "12", "threads": "1"}
    a = None
    if req.get("audio") and mode != "no_audio":
        a = c.add_stream("aac", rate=48000)
        a.layout = "mono"
    for t in range(n):
        f = (np.roll(img, 2 * t, axis=1) - 128) * (1 + 0.25 * g) + 128 + tint + grain + jitter[t]
        for p in v.encode(av.VideoFrame.from_ndarray(np.clip(np.rint(f), 0, 255).astype(np.uint8), format="rgb24")):
            c.mux(p)
    for p in v.encode():
        c.mux(p)
    if a is not None:
        total = int(round(req["num_frames"] / req["fps"] * 48000))
        x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(total) / 48000)).astype(np.float32)[None]
        for i in range(0, total, 1024):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(x[:, i:i + 1024]), format="flt", layout="mono")
            fr.sample_rate, fr.pts = 48000, i
            for p in a.encode(fr):
                c.mux(p)
        for p in a.encode():
            c.mux(p)
    c.close()
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200 if self.path == "/health" else 404, b'{"ok": true}')

    def do_POST(self):
        srv = self.server
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        if self.path == "/_mode":
            srv.mode, srv.cache = body["mode"], {}
            self._send(200, b"{}")
            return
        if self.path != "/v1/clips":
            self._send(404, b"{}")
            return
        mode = srv.mode
        key = json.dumps(body, sort_keys=True)
        if mode == "cache" and key in srv.cache:
            self._send(200, srv.cache[key])  # an output cache: no work, no wait
            return
        t0 = time.perf_counter()
        steps = 4 if mode == "ignore_steps" else body["steps"]
        branches = 2 if body["guidance_scale"] > 1 else 1
        time.sleep(srv.delay_ms * steps * branches / 1000.0)
        video = render(body, mode)
        ms = (time.perf_counter() - t0) * 1000
        effective = {"steps_run": body["steps"] // 2 if mode == "fewer_steps_counter" else body["steps"],
                     "width": body["width"], "height": body["height"], "frames": body["num_frames"],
                     "fps": body["fps"], "dtype": "bf16", "attention": "sdpa"}
        out = json.dumps({"video_b64": base64.b64encode(video).decode(), "format": "mp4", "effective": effective,
                          "stage_ms": {"denoise": 0.9 * ms, "vae_decode": 0.1 * ms},
                          "cache": ["text_encode"] if mode == "reported_cache" else []}).encode()
        if mode == "cache":
            srv.cache[key] = out
        self._send(200, out)


def _serve_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--delay-ms", type=float, default=30.0)
    a = p.parse_args(argv)
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), _Handler)
    httpd.mode, httpd.cache, httpd.delay_ms = "honest", {}, a.delay_ms
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"fake clip engine on {a.port}", flush=True)
    sys.stdin.read()  # until the test closes our stdin
    httpd.shutdown()


if __name__ == "__main__":
    _serve_main(sys.argv[1:])
    raise SystemExit(0)

# --------------------------------------------------------------------------- tests

import importlib  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

cc = pytest.importorskip("inferencekit.serve.clip_client", reason="gates clip needs the clip client")

from inferencekit import registry  # noqa: E402
from inferencekit.proc import which  # noqa: E402
from inferencekit.registry import OpError  # noqa: E402

THIS = str(Path(__file__).resolve())


def _video_python():
    for py in (os.environ.get("INFERENCEKIT_VIDEO_PYTHON"), sys.executable):
        if py and subprocess.run([py, "-c", "import av, numpy"], capture_output=True).returncode == 0:
            return py
    return None


PY = _video_python()
needs_video = pytest.mark.skipif(PY is None, reason="needs a Python with PyAV and numpy (INFERENCEKIT_VIDEO_PYTHON)")


def registered(module: str, op: str):
    """Import an operation module so that its operation is in the registry, even when an earlier test removed
    it after importing the module (a cached module does not register again on import)."""
    mod = importlib.import_module(module)
    if op not in registry.REGISTRY:
        mod = importlib.reload(mod)
    return mod


def forget_new_ops(before: set) -> None:
    """Remove the operations registered since ``before`` and their modules (and the package attributes that
    ``from inferencekit.ops import x`` would return), so a later import registers them again."""
    for name in set(registry.REGISTRY) - before:
        module = registry.REGISTRY.pop(name).handler.__module__
        sys.modules.pop(module, None)
        pkg, _, attr = module.rpartition(".")
        if pkg in sys.modules and attr in vars(sys.modules[pkg]):
            delattr(sys.modules[pkg], attr)


@pytest.fixture(scope="module")
def gc():
    """Import gates clip (and bench clip for the static test); leave the registry as it was (the contract test
    lists every op)."""
    before = set(registry.load_all())
    registered("inferencekit.ops.bench_clip", "bench_clip")
    registered("inferencekit.ops.quality_clip", "quality_clip")
    mod = registered("inferencekit.ops.gates_clip", "gates_clip")
    yield mod
    forget_new_ops(before)


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="module")
def engine():
    """The fake engine in a child process (under PY); yields a function that sets its mode."""
    if PY is None:
        pytest.skip("needs a Python with PyAV and numpy (INFERENCEKIT_VIDEO_PYTHON)")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([PY, THIS, "--serve", "--port", str(port), "--delay-ms", "30"], stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            try:
                urllib.request.urlopen(url + "/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("fake clip engine did not start")

        def set_mode(mode):
            req = urllib.request.Request(url + "/_mode", data=json.dumps({"mode": mode}).encode(), method="POST")
            urllib.request.urlopen(req, timeout=5).read()
            return url

        yield set_mode
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def clip_set(tmp_path, **kw) -> Path:
    row = {"id": "t2v-0", "task": "t2v", "prompt": "a red fox runs through fresh snow at dawn", "width": 64,
           "height": 48, "num_frames": 9, "fps": 24, "steps": 4, "guidance_scale": 3.0, **kw}
    p = tmp_path / "clips.jsonl"
    p.write_text(json.dumps(row) + "\n")
    return p


def gates(url, clips, **kw):
    args = {"base_url": url, "clips": str(clips), "python": PY, "seed": 1234, **kw}
    return registry.call("gates_clip", args)


def verdicts(rec):
    return {k: g["verdict"] for k, g in rec["data"]["gates"].items()}


# --------------------------------------------------------------------------- live, honest

@needs_video
def test_honest_engine_passes_every_gate(gc, engine, store, tmp_path):
    url = engine("honest")
    rec = gates(url, clip_set(tmp_path, audio=True), threshold=0.25, run_dir=str(tmp_path))
    assert rec["kind"] == gc.KIND == "clip_gates"
    assert rec["data"]["verdict"] == "PASS", {k: g["reason"] for k, g in rec["data"]["gates"].items()}
    assert set(verdicts(rec)) == set(gc.GATES)
    d = rec["data"]
    assert d["decoder"] == "pyav" and d["pixels_compared"] and d["repeat_floor_mad"] == 0.0
    assert d["probe_seed"] == 1234 and d["probe_clip"] == "t2v-0"
    steps = d["gates"]["steps_honoured"]["metrics"]
    assert steps["step_time_ratio_median"] > 1.5 and steps["steps_reported"] == 9
    root = store.run_dir(rec["evidence_id"])
    rows = [json.loads(x) for x in (root / "probes.jsonl").read_text().splitlines()]
    assert len(rows) == 9 and {r["kind"] for r in rows} == {"base", "replay", "half_steps", "other_seed", "prompt",
                                                            "cfg"}
    assert all(r["decoded"]["video"]["frames"] == 9 for r in rows)
    prompt = next(r for r in rows if r["kind"] == "prompt")
    assert prompt["request"]["prompt"] != rows[0]["request"]["prompt"] and len(prompt["request"]["prompt"].split()) \
        == len(rows[0]["request"]["prompt"].split())
    index = json.loads((root / "outputs" / "clip_http" / "index.json").read_text())
    assert len(index["files"]) == 9 and all((root / "outputs" / "clip_http" / k).is_file() for k in index["files"])
    assert url not in json.dumps(rec) and url.rsplit(":", 1)[1] not in (root / "probes.jsonl").read_text()
    assert (tmp_path / "evidence" / f"{rec['evidence_id']}.json").is_file()


@needs_video
def test_nondeterministic_engine_passes_on_pixels(gc, engine, store, tmp_path):
    """Grain that differs on every call (non-deterministic kernels): the repeat floor is above zero."""
    rec = gates(engine("noisy"), clip_set(tmp_path), threshold=0.25)
    v = verdicts(rec)
    assert rec["data"]["repeat_floor_mad"] > 0
    assert v["seed_honoured"] == v["prompt_sensitivity"] == v["cfg_honoured"] == "PASS", rec["summary"]


@needs_video
def test_without_python_pixels_are_not_compared(gc, engine, store, tmp_path):
    if which("ffprobe") is None:
        pytest.skip("no ffprobe on PATH")
    rec = gates(engine("honest"), clip_set(tmp_path), python=None, threshold=0.25)
    v = verdicts(rec)
    assert rec["data"]["decoder"] == "ffprobe" and not rec["data"]["pixels_compared"]
    assert v["resolution_frames"] == "PASS" and v["prompt_sensitivity"] == "UNPROVEN"
    assert rec["data"]["verdict"] == "UNPROVEN"
    assert any("bytes only" in x for x in rec["limitations"])


@needs_video
def test_a_broken_python_keeps_the_probes(gc, engine, store, tmp_path):
    """--python without PyAV: the clips are still decoded with ffprobe and compared by bytes; nothing is lost."""
    if subprocess.run([sys.executable, "-c", "import av"], capture_output=True).returncode == 0:
        pytest.skip("this interpreter has PyAV")
    if which("ffprobe") is None:
        pytest.skip("no ffprobe on PATH")
    rec = gates(engine("honest"), clip_set(tmp_path), python=sys.executable, threshold=0.25)
    assert rec["data"]["decoder"].startswith("ffprobe (the --python worker failed")
    assert verdicts(rec)["resolution_frames"] == "PASS" and rec["data"]["verdict"] == "UNPROVEN"
    assert any("pixel comparison failed" in x for x in rec["limitations"])


# --------------------------------------------------------------------------- live, cheating

CHEATS = [("ignore_steps", "steps_honoured"), ("fewer_steps_counter", "steps_honoured"),
          ("ignore_seed", "seed_honoured"), ("random_seed", "seed_honoured"),
          ("ignore_prompt", "prompt_sensitivity"), ("ignore_cfg", "cfg_honoured"), ("cache", "no_cache"),
          ("reported_cache", "no_cache"), ("lowres", "resolution_frames"), ("fewer_frames", "resolution_frames"),
          ("no_audio", "resolution_frames")]


@needs_video
@pytest.mark.parametrize("mode,gate", CHEATS)
def test_each_cheat_fails_its_gate(gc, engine, store, tmp_path, mode, gate):
    rec = gates(engine(mode), clip_set(tmp_path, audio=True))
    g = rec["data"]["gates"][gate]
    assert g["verdict"] == "FAIL", (mode, g["reason"])
    assert rec["data"]["verdict"] == "FAIL"


@needs_video
def test_cheat_reasons_name_the_problem(gc, engine, store, tmp_path):
    rec = gates(engine("lowres"), clip_set(tmp_path))
    assert "decoded 32x24, asked 64x48" in rec["data"]["gates"]["resolution_frames"]["reason"]
    rec = gates(engine("cache"), clip_set(tmp_path))
    assert "replays are" in rec["data"]["gates"]["no_cache"]["reason"]
    rec = gates(engine("random_seed"), clip_set(tmp_path))
    assert "does not reproduce" in rec["data"]["gates"]["seed_honoured"]["reason"]


def test_an_engine_that_is_not_up_is_refused(gc, store, tmp_path):
    """No engine on the port: health fails before any probe; the op refuses to run."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(OpError, match="not ready"):
        gates(f"http://127.0.0.1:{port}", clip_set(tmp_path))


# --------------------------------------------------------------------------- a bench clip record

@needs_video
def test_bench_clip_record_is_decoded_and_checked(gc, engine, store, tmp_path):
    url = engine("honest")
    clips = clip_set(tmp_path)
    bench = registry.call("bench_clip", {"base_url": url, "clips": str(clips), "rounds": 2, "warmup": 0,
                                         "gpu_name": "Test GPU", "resamples": 100})
    rec = registry.call("gates_clip", {"bench_evidence": bench["evidence_id"], "python": PY})
    v = verdicts(rec)
    assert v["resolution_frames"] == "PASS" and v["no_cache"] == "PASS" and v["steps_honoured"] == "NOT RUN"
    assert rec["data"]["verdict"] == "PASS"  # live gates were not asked for (no --base-url)
    assert rec["data"]["gates"]["resolution_frames"]["metrics"]["clips_decoded"] == 2
    bench = registry.call("bench_clip", {"base_url": engine("fewer_frames"), "clips": str(clips), "rounds": 1,
                                         "warmup": 0, "gpu_name": "Test GPU", "resamples": 100})
    rec = registry.call("gates_clip", {"bench_evidence": bench["evidence_id"], "python": PY})
    assert verdicts(rec)["resolution_frames"] == "FAIL"
    assert "decoded 7 frames, asked 9" in rec["data"]["gates"]["resolution_frames"]["reason"]
    bench = registry.call("bench_clip", {"base_url": engine("honest"), "clips": str(clips), "rounds": 1, "warmup": 0,
                                         "gpu_name": "Test GPU", "resamples": 100, "save_outputs": "hash_only"})
    rec = registry.call("gates_clip", {"bench_evidence": bench["evidence_id"], "python": PY})
    g = rec["data"]["gates"]["resolution_frames"]
    assert g["verdict"] == "UNPROVEN" and rec["data"]["verdict"] == "UNPROVEN"
    assert "hash_only" in g["reason"] and g["metrics"]["clips_undecoded"] == 1


def _bench(url, clips, **kw):
    return registry.call("bench_clip", {"base_url": url, "clips": str(clips), "rounds": 1, "warmup": 0,
                                        "gpu_name": "Test GPU", "resamples": 100, **kw})


@needs_video
def test_an_extended_bench_record_is_decoded_in_every_round(gc, engine, store, tmp_path):
    """bench clip --extend (and bench rounds' clip mode) keeps each round's clips with the record that timed it:
    the last record's rows cover every round, and every round's clip is found and decoded."""
    url, clips = engine("honest"), clip_set(tmp_path)
    first = _bench(url, clips)
    last = _bench(url, clips, extend=first["evidence_id"])
    assert last["data"]["outputs_from"] == [first["evidence_id"]]
    rec = registry.call("gates_clip", {"bench_evidence": last["evidence_id"], "python": PY})
    g = rec["data"]["gates"]["resolution_frames"]
    assert g["verdict"] == "PASS" and g["metrics"]["clips_decoded"] == 2, g["reason"]
    assert sorted(r["key"] for r in g["details"]) == ["bench:r0/t2v-0", "bench:r1/t2v-0"]
    # --max-decode: a random subset, the same one for the same probe seed.
    def pick(seed):
        rec = registry.call("gates_clip", {"bench_evidence": last["evidence_id"], "python": PY, "max_decode": 1,
                                           "seed": seed})
        g = rec["data"]["gates"]["resolution_frames"]
        assert g["metrics"]["clips_decoded"] == 1 and g["metrics"]["clips_left_out"] == 1
        assert any("--max-decode" in x for x in g["limitations"])
        return g["details"][0]["key"]

    picks = [pick(seed) for seed in range(10)]
    assert set(picks) == {"bench:r0/t2v-0", "bench:r1/t2v-0"} and pick(0) == picks[0]
    # A round whose clip file is gone: that clip is undecoded, so the gate cannot pass.
    old = store.run_dir(first["evidence_id"]) / "outputs" / "clip_http" / "r0" / "t2v-0.mp4"
    old.unlink()
    rec = registry.call("gates_clip", {"bench_evidence": last["evidence_id"], "python": PY})
    g = rec["data"]["gates"]["resolution_frames"]
    assert g["verdict"] == "UNPROVEN" and g["metrics"]["clips_undecoded"] == 1 and "1 clip(s)" in g["reason"]
    # ... unless the run directory still has it (bench clip --run-dir writes every round there too).
    run = tmp_path / "run"
    a = _bench(url, clips, label="eng", run_dir=str(run))
    b = _bench(url, clips, label="eng", run_dir=str(run), extend=a["evidence_id"])
    (store.run_dir(a["evidence_id"]) / "outputs" / "eng" / "r0" / "t2v-0.mp4").unlink()
    rec = registry.call("gates_clip", {"bench_evidence": b["evidence_id"], "python": PY})
    assert rec["data"]["gates"]["resolution_frames"]["metrics"]["clips_decoded"] == 2


@needs_video
def test_edited_bench_evidence_is_refused_or_failed(gc, engine, store, tmp_path):
    url, clips = engine("honest"), clip_set(tmp_path)
    bench = _bench(url, clips, rounds=2)
    root = store.run_dir(bench["evidence_id"])
    # A swapped clip file: not the clip that was timed.
    r0, r1 = (root / "outputs" / "clip_http" / f"r{k}" / "t2v-0.mp4" for k in (0, 1))
    keep = r0.read_bytes()
    r0.write_bytes(r1.read_bytes())
    rec = registry.call("gates_clip", {"bench_evidence": bench["evidence_id"], "python": PY})
    g = rec["data"]["gates"]["resolution_frames"]
    assert g["verdict"] == "FAIL" and "not the clip that was timed" in g["reason"], g["reason"]
    r0.write_bytes(keep)
    # An edited requests.jsonl (a cache report removed, a duplicate hash changed...): refused.
    req = root / "requests.jsonl"
    text = req.read_text()
    req.write_text(text.replace('"cache": []', '"cache": null', 1))
    with pytest.raises(OpError, match="requests.jsonl does not match"):
        registry.call("gates_clip", {"bench_evidence": bench["evidence_id"], "python": PY})
    req.write_text(text)
    # An edited record: refused (its id no longer matches its content).
    ev_path = root / "evidence.json"
    doc = json.loads(ev_path.read_text())
    doc["data"]["claim_validity"]["valid"] = not doc["data"]["claim_validity"]["valid"]
    ev_path.write_text(json.dumps(doc))
    with pytest.raises(OpError, match="does not match its content"):
        registry.call("gates_clip", {"bench_evidence": bench["evidence_id"], "python": PY})


def test_argument_errors(gc, store, tmp_path):
    for args, msg in (({}, "--base-url"), ({"base_url": "http://127.0.0.1:9"}, "needs --clips"),
                      ({"base_url": "http://127.0.0.1:9", "clips": "x", "pairs": 0}, "pairs"),
                      ({"base_url": "http://127.0.0.1:9", "clips": "x", "endpoint_kind": "comfyui"}, "--workflow")):
        with pytest.raises(OpError, match=msg):
            registry.call("gates_clip", args)


# --------------------------------------------------------------------------- gate logic on synthetic probes

def _req(**kw):
    base = {"id": "x", "task": "t2v", "prompt": "a fox in snow", "width": 64, "height": 48, "num_frames": 9,
            "fps": 24, "steps": 8, "guidance_scale": 3.0, "seed": 1}
    return cc.ClipRequest(**{**base, **kw})


def _probe(gc, key, kind, ms, sha, pair=None, steps=8, effective=None, cache=None, **req):
    res = cc.ClipResult(id=key, ok=True, t_send_ns=0, t_done_ns=int(ms * 1e6), cache=cache or [],
                        effective={"steps_run": steps} if effective is None else effective)
    res.video = b""
    return gc.Probe(key, kind, _req(steps=steps, **req), pair=pair, res=res, sha=sha)


def _judge(gc, ms=None, shas=None, dist=None, steps_run=None):
    ms = {"base0": 1000, "replay0": 1000, "half0": 560, "base1": 1000, "replay1": 1000, "half1": 560, "seed": 1000,
          "prompt": 1000, "cfg": 600, **(ms or {})}
    shas = {k: k for k in ms} | (shas or {})
    ps = {}
    for k in ms:
        kind = {"base": "base", "repl": "replay", "half": "half_steps"}.get(k[:4], k if k != "seed" else "other_seed")
        pair = int(k[-1]) if k[-1].isdigit() else None
        steps = 4 if k.startswith("half") else 8
        ps[k] = _probe(gc, k, kind, ms[k], shas[k], pair=pair, steps=steps,
                       effective={"steps_run": (steps_run or {}).get(k, steps)})
    return gc.Judge(ps, dist, 2)


def _dist(same=0.4, other=30.0, **kw):
    d = {"same0": same, "same1": same, "steps0": 12.0, "steps1": 12.0, "seed_base1": other, "seed": other,
         "prompt": 9.0, "cfg": 8.0, **kw}
    return {k: {"ok": True, "mad": v, "psnr": 30.0} for k, v in d.items()}


def test_one_word_change_and_guidance(gc):
    rng = random.Random(0)
    for _ in range(50):
        new, how = gc.one_word_change("A quiet harbour at dusk, boats rocking", rng)
        old, got = "A quiet harbour at dusk, boats rocking".split(" "), new.split(" ")
        changed = [i for i, (a, b) in enumerate(zip(old, got)) if a != b]
        assert len(got) == len(old) and len(changed) == 1 and got[changed[0]].lower() in gc.SWAP_WORDS
    new, how = gc.one_word_change("a b c", rng)
    assert how["inserted"] and new.endswith("a b c")
    assert gc.alt_guidance(3.0) == 1.0 and gc.alt_guidance(1.0) == 4.0 and gc.alt_guidance(0.0) == 3.0


def test_probe_order_separates_each_replay_from_its_request(gc):
    clipset = argparse.Namespace(images={})
    clip = {"id": "c", "task": "t2v", "prompt": "a fox in snow", "width": 64, "height": 48, "num_frames": 9,
            "fps": 24, "steps": 8, "guidance_scale": 3.0}
    for s in range(20):
        rng = random.Random(s)
        probes = gc.plan_probes(clip, clipset, rng, 2)
        assert len({p.req.seed for p in probes if p.kind in ("base", "other_seed")}) == 3  # fresh distinct seeds
        seq = gc.order(probes, rng)
        pos = {p.key: i for i, p in enumerate(seq)}
        assert all(pos[f"replay{i}"] > pos[f"base{i}"] + 1 for i in range(2))
        assert {p.req.steps for p in probes if p.kind == "half_steps"} == {4}


def _png() -> bytes:
    """A valid 2x1 RGB PNG."""
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00\x00\x80\xff")) + chunk(b"IEND", b""))


def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    import zlib

    out, at = [], 8
    while at < len(data):
        n = int.from_bytes(data[at:at + 4], "big")
        kind, body = data[at + 4:at + 8], data[at + 8:at + 8 + n]
        assert int.from_bytes(data[at + 8 + n:at + 12 + n], "big") == zlib.crc32(kind + body)
        out.append((kind, body))
        at += 12 + n
    return out


def test_image_variant_is_the_same_picture_as_another_file(gc):
    img = _png()
    v = gc.image_variant(img, "inferencekit probe group 1")
    assert v != img and v != gc.image_variant(img, "inferencekit probe group 2")
    chunks = _png_chunks(v)  # every CRC checks
    assert [c for c in chunks if c[0] != b"tEXt"] == _png_chunks(img)  # IHDR, IDAT, IEND unchanged
    assert (b"tEXt", b"Comment\x00inferencekit probe group 1") in chunks and chunks[0][0] == b"IHDR"
    jfif = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00\x02\xff\xd9"
    v = gc.image_variant(jfif, "tag")
    assert v == jfif[:20] + b"\xff\xfe\x00\x05tag" + jfif[20:]  # a COM segment after APP0
    assert gc.image_variant(b"\xff\xd8\xff\xdb\x00\x02", "t") == b"\xff\xd8\xff\xfe\x00\x03t\xff\xdb\x00\x02"
    assert gc.image_variant(b"RIFF\x00\x00\x00\x00WEBPVP8 ", "tag") is None


@pytest.mark.parametrize("task", ["t2v", "i2v"])
def test_probes_never_follow_one_with_the_same_prompt_or_image(gc, task):
    """An engine that keeps the last run's encodings (ComfyUI's node cache) must have nothing to reuse: no
    probe follows one with the same prompt or image file, while the probes compared with each other still
    differ only in what each gate changes."""
    img = _png()
    clipset = argparse.Namespace(images={"c": (img, hashlib.sha256(img).hexdigest())} if task == "i2v" else {})
    clip = {"id": "c", "task": task, "prompt": "a fox in snow", "width": 64, "height": 48, "num_frames": 9,
            "fps": 24, "steps": 8, "guidance_scale": 3.0}
    fields = ("prompt", "seed", "guidance_scale", "steps", "image", "width", "height", "num_frames", "fps")
    for pairs in (2, 3):
        for s in range(20):
            rng = random.Random(s)
            probes = gc.plan_probes(clip, clipset, rng, pairs)
            by = {p.key: p for p in probes}
            seq = gc.order(probes, rng)
            assert gc.neighbour_clashes(seq) == [], [p.key for p in seq]
            for i in range(pairs):
                base = by[f"base{i}"].req
                assert by[f"replay{i}"].req == gc.dataclasses.replace(base, id=f"replay{i}")
                assert by[f"half{i}"].req == gc.dataclasses.replace(base, id=f"half{i}", steps=4)
            assert (by["seed"].vs, by["prompt"].vs, by["cfg"].vs) == ("base0", "base0", f"base{pairs - 1}")
            for key, changed in (("seed", {"seed"}), ("cfg", {"guidance_scale"}),
                                 ("prompt", {"prompt", "image"} if task == "i2v" else {"prompt"})):
                a, b = by[by[key].vs].req, by[key].req
                assert {f for f in fields if getattr(a, f) != getattr(b, f)} == changed, key
            assert by["base0"].req.prompt == clip["prompt"] and by["base1"].req.prompt == clip["prompt"] + " (probe 2)"
            assert len({by[f"base{i}"].req.prompt for i in range(pairs)}) == pairs
            if task == "i2v":
                assert by["base0"].req.image == img
                assert len({p.req.image for p in probes}) == pairs + 1  # one file per group, one for the prompt probe
            warm = gc.warmup_request(by["base0"].req, 0, 5, probes)
            assert warm.prompt not in {p.req.prompt for p in probes}
            assert task == "t2v" or warm.image not in {p.req.image for p in probes}
    probes = gc.plan_probes(clip, clipset, random.Random(0), 1)  # five probes share one group: they cannot be apart
    assert len(gc.neighbour_clashes(gc.order(probes, random.Random(0)))) == 3


def _comfy_gates(gc, tmp_path, pairs):
    """gates clip (i2v, no --python) against the fake ComfyUI, with an image encoder that does not see the
    prompt, as LTX-2's in-place image conditioning does."""
    from fakes_clip.fake_comfyui import ComfyConfig, FakeComfyUI, ltx_workflow

    doc = ltx_workflow(i2v=True)
    g = doc["workflow"]
    g["13"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["1", 2], "image": ["12", 0],
                                                                 "latent": ["5", 0], "strength": 1.0}}
    g["3"]["inputs"]["latent_image"] = ["13", 0]
    doc["allow_cached"] = ["7"]  # the fixed negative prompt's encoder
    wf = tmp_path / f"wf{pairs}.json"
    wf.write_text(json.dumps(doc))
    (tmp_path / "cat.png").write_bytes(_png())
    clips = clip_set(tmp_path, task="i2v", image="cat.png")
    with FakeComfyUI(ComfyConfig(step_s=0.01)) as srv:
        return registry.call("gates_clip", {"base_url": srv.base_url, "clips": str(clips), "endpoint_kind": "comfyui",
                                            "workflow": str(wf), "seed": 1234, "pairs": pairs, "threshold": 1.0,
                                            "label": f"comfy{pairs}"})


def test_gates_clip_against_comfyui_leaves_no_encoding_to_reuse(gc, store, tmp_path):
    rec = _comfy_gates(gc, tmp_path, 2)
    nc = rec["data"]["gates"]["no_cache"]
    assert nc["verdict"] == "PASS", nc["reason"]
    assert rec["data"]["neighbour_clashes"] == [] and len(rec["data"]["probe_order"]) == 9
    rows = [json.loads(x) for x in (store.run_dir(rec["evidence_id"]) / "probes.jsonl").read_text().splitlines()]
    assert len(rows) == 9 and all(r["ok"] and r["cache"] == [] for r in rows), [(r["key"], r["cache"]) for r in rows]
    assert any("fixed negative prompt" in x for x in nc["limitations"])
    assert rec["data"]["warmup"][0]["prompt"].endswith("(warm-up)")
    assert {r["request"]["image"]["sha256"] for r in rows if r["kind"] == "base"} != {None}
    # One pair: its five probes cannot be kept apart, so ComfyUI reuses encodings and no_cache names why.
    one = _comfy_gates(gc, tmp_path, 1)["data"]["gates"]["no_cache"]
    assert one["verdict"] == "FAIL" and "ran right after a probe with the same prompt or image file" in one["reason"]
    assert "use --pairs 2 or more" in one["reason"] and one["metrics"]["neighbour_clashes"] == 3


def test_differs_classifies_against_the_repeat_floor(gc):
    j = _judge(gc, dist=_dist(same=1.0, prompt=2.0, cfg=0.9, seed=30.0))
    assert j.floor == 1.0
    assert j.differs("seed", "base0", "seed")["state"] == "differs"
    assert j.differs("prompt", "base0", "prompt")["state"] == "small"  # 2.0 <= 3 x floor
    assert j.differs("cfg", "base0", "cfg")["state"] == "within_floor"
    j = _judge(gc, dist=_dist(), shas={"cfg": "base0"})
    assert j.differs("cfg", "base0", "cfg")["state"] == "same_bytes"
    j = _judge(gc, dist=None)
    assert j.floor is None and j.differs("prompt", "base0", "prompt")["state"] == "bytes_only"


def test_steps_gate_on_timing_counter_and_outputs(gc):
    assert gc.gate_steps(_judge(gc, dist=_dist()), 8)["verdict"] == "PASS"
    g = gc.gate_steps(_judge(gc, ms={"half0": 990, "half1": 1000}, dist=_dist()), 8)
    assert g["verdict"] == "FAIL" and "does not change the work" in g["reason"]
    g = gc.gate_steps(_judge(gc, ms={"half0": 930, "half1": 940}, dist=_dist()), 8)
    assert g["verdict"] == "UNPROVEN"  # 1.07x: neither clearly scaling nor clearly flat
    g = gc.gate_steps(_judge(gc, dist=_dist(), steps_run={"base1": 6}), 8)
    assert g["verdict"] == "FAIL" and "steps_run != asked" in g["reason"]
    assert gc.gate_steps(_judge(gc, dist=_dist()), 1)["verdict"] == "NOT RUN"


def test_seed_gate(gc):
    assert gc.gate_seed(_judge(gc, dist=_dist()))["verdict"] == "PASS"
    g = gc.gate_seed(_judge(gc, dist=_dist(same=20.0, other=30.0)))
    assert g["verdict"] == "FAIL" and "does not reproduce" in g["reason"]
    g = gc.gate_seed(_judge(gc, dist=_dist(), shas={"seed": "base0"}))
    assert g["verdict"] == "FAIL" and "byte-identical" in g["reason"]
    assert gc.gate_seed(_judge(gc, dist=None))["verdict"] == "UNPROVEN"


def test_no_cache_gate(gc):
    assert gc.gate_no_cache(_judge(gc, dist=_dist()), 0.10, None)["verdict"] == "PASS"
    g = gc.gate_no_cache(_judge(gc, ms={"replay0": 500, "replay1": 400}, dist=_dist()), 0.10, None)
    assert g["verdict"] == "FAIL" and "faster" in g["reason"]
    g = gc.gate_no_cache(_judge(gc, ms={"replay0": 500}, dist=_dist()), 0.10, None)
    assert g["verdict"] == "PASS"  # one fast replay of two is not "typical"
    g = gc.gate_no_cache(_judge(gc, dist=_dist(), shas={"prompt": "base0"}), 0.10, None)
    assert g["verdict"] == "FAIL" and "byte-identical" in g["reason"]
    static = {"n": 4, "duplicates": [["a@seed1", "b@seed1"]], "cached": []}
    assert gc.gate_no_cache(None, 0.10, static)["verdict"] == "FAIL"
    assert gc.gate_no_cache(None, 0.10, None)["verdict"] == "NOT RUN"


def test_static_cache_groups_by_request(gc):
    rows = [{"id": "a", "seed": 1, "round": 0, "ok": True, "out_sha256": "s1"},
            {"id": "a", "seed": 1, "round": 1, "ok": True, "out_sha256": "s1"},  # same request again (fixed seed)
            {"id": "b", "seed": 1, "round": 0, "ok": True, "out_sha256": "s2", "cache": ["vae"]}]
    out = gc.static_cache({"rows": rows})
    assert out["duplicates"] == [] and out["cached"] == ["b r0"]
    rows.append({"id": "c", "seed": 1, "round": 0, "ok": True, "out_sha256": "s2"})
    assert gc.static_cache({"rows": rows})["duplicates"] == [["b@seed1", "c@seed1"]]


def test_decoded_checks(gc):
    asked = {"width": 64, "height": 48, "num_frames": 9, "fps": 24}

    def dec(w=64, h=48, n=9, fps=24.0, audio=None, sizes=None, header=None):
        return {"ok": True, "video": {"width": w, "height": h, "frames": n, "fps": fps, "sizes": sizes or [[w, h]]},
                "audio": audio, "header": {"video": header or {}}}

    assert gc.check_decoded(dec(), asked, None) == []
    assert gc.check_decoded(dec(fps=23.5), asked, None)[0].startswith("decoded timestamps give 23.5 fps")
    assert "decoded 8 frames" in gc.check_decoded(dec(n=8), asked, None)[0]
    assert "changes inside the clip" in gc.check_decoded(dec(sizes=[[64, 48], [32, 24]]), asked, None)[0]
    assert "none was decoded" in gc.check_decoded(dec(), asked, True)[0]
    assert gc.check_decoded(dec(audio={"duration_s": 0.40}), asked, True) == []
    assert "audio is 0.200 s" in gc.check_decoded(dec(audio={"duration_s": 0.2}), asked, True)[0]
    assert gc.header_claims(dec(n=7, header={"frames": 9})) == ["header frames 9 vs decoded 7"]
