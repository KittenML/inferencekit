"""quality clip-steps on a tiny LTX-2-shaped pipeline (CPU): the per-step dump contract, the replay, the verdicts.

Engines are diffusers_ltx's real wrapper (dumps, replay) around tests/fakes_clip/fake_steps.py, served by the real
clip server code (ClipServer._run: clip, sha256, dump), laid out like bench clip outputs. Needs a Python with torch,
safetensors, numpy, PyAV and Diffusers (its encode_video muxes the clips): INFERENCEKIT_VIDEO_PYTHON, else this one.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from inferencekit import registry

ROOT = Path(__file__).resolve().parents[1]
PATHS = os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])


def _video_python():
    for py in (os.environ.get("INFERENCEKIT_VIDEO_PYTHON"), sys.executable):
        if py and subprocess.run([py, "-c", "import torch, safetensors, av, numpy, diffusers.utils"],
                                 capture_output=True).returncode == 0:
            return py
    return None


PY = _video_python()
pytestmark = pytest.mark.skipif(PY is None, reason="needs torch, safetensors, PyAV, numpy and diffusers "
                                                   "(INFERENCEKIT_VIDEO_PYTHON)")
STEPS = 6
BODIES = [{"task": "t2v", "prompt": p, "width": 64, "height": 64, "num_frames": 9, "fps": 24, "steps": STEPS,
           "guidance_scale": 3.0, "seed": s, "audio": False}
          for p, s in (("a red fox in snow", 11), ("a harbour at dusk", 12), ("rain on a window", 13))]
# label -> (attention, knobs): the honest engines and the shortcuts
ENGINES = {"reference": ("native", {}), "honest": ("_native_cudnn", {}), "fewer_steps": ("native", {"fewer_steps": 1}),
           "skip_guidance": ("_native_cudnn", {"skip_guidance_steps": [4]}),
           "reuse": ("_native_cudnn", {"reuse_steps": [2]}), "lowp": ("lowp", {}),
           "fewer_frames": ("native", {"fewer_frames": 8}), "other_sigmas": ("native", {"other_sigmas": True})}

ENGINE = r'''
import json, shutil, sys
from pathlib import Path
from inferencekit.engines.clip.server import ClipServer, PipelineOptions, parse_job
from fakes_clip.fake_steps import make_pipeline

cfg = json.loads(sys.argv[1])
run = Path(cfg["run_dir"])
for label, (attention, knobs) in cfg["engines"].items():
    opts = PipelineOptions(model="fake", device="cpu", attention=attention, extra=knobs)
    srv = ClipServer(lambda: make_pipeline(opts), port=0, out_dir=run / "srv" / label, keep=0, stage_sync=False,
                     step_dump_dir=run / "dumps" / label)
    srv._load_pipeline()
    assert srv.pipeline is not None, srv.load_error
    files = {}
    for i, body in enumerate(cfg["bodies"]):
        ans = srv._run(parse_job(body))
        dst = run / "outputs" / label / "r1" / f"c{i}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ans["video_path"], dst)
        files[f"r1/c{i}.mp4"] = ans["video_sha256"]
        req = run / "inputs" / label / "r1" / f"c{i}.json"
        req.parent.mkdir(parents=True, exist_ok=True)
        req.write_text(json.dumps(body))
        assert ans["effective"]["step_dump_bytes"] > 0 and "step_dump" in ans["stage_ms"]
    (run / "outputs" / label / "index.json").write_text(json.dumps({"files": files}))
    srv.httpd.server_close()
'''

TAMPER = r'''
import json, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
path, keys, eps = sys.argv[1], json.loads(sys.argv[2]), float(sys.argv[3])
with safe_open(path, framework="pt") as f:
    meta, tensors = f.metadata(), {k: f.get_tensor(k) for k in f.keys()}
for k in keys:
    tensors[k] = tensors[k] + eps
save_file(tensors, path, metadata=meta)
'''


def _env():
    return {**os.environ, "PYTHONPATH": PATHS, "CUDA_VISIBLE_DEVICES": ""}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    run = tmp_path_factory.mktemp("clipsteps")
    r = subprocess.run([PY, "-c", ENGINE, json.dumps({"run_dir": str(run), "engines": ENGINES, "bodies": BODIES})],
                       capture_output=True, text=True, env=_env(), timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return run


@pytest.fixture(autouse=True)
def worker_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "tests"))  # the worker imports the fake reference factory
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def steps(run: Path, label: str, dumps: Path | None = None, **kw):
    registry.load_all()
    args = {"outputs": str(run / "outputs" / label), "dumps": str(dumps or run / "dumps" / label),
            "pipeline": "fakes_clip.fake_steps:make_pipeline", "pipeline_model": "fake", "python": PY,
            "device": "cpu", "attention": "native", "audit_seed": 7, "slack": 1e-6, "decode_psnr_db": 30.0,
            "timeout_s": 600, **kw}
    rec = registry.call("quality_clip_steps", args)
    return rec["data"], rec


def test_honest_engine_passes_and_the_record_holds_every_step(run_dir):
    data, rec = steps(run_dir, "honest")
    assert data["verdict"] == "PASS", json.dumps(data["audited"], indent=1)[:3000]
    assert len(data["audited"]) == 2 and data["audit_seed"] == 7
    for a in data["audited"].values():
        assert {c: v["verdict"] for c, v in a["checks"].items()} == {
            "dump": "PASS", "layout": "PASS", "schedule": "PASS", "dtypes": "PASS", "chain": "PASS", "decode": "PASS",
            "steps": "PASS"}
        assert a["n_steps"] == STEPS and a["decode"]["psnr_db"] > 30
    assert 0 < data["max_floor"] and data["max_ratio"] <= 1.0 + 1e-9  # the same exact kernel as the floor's
    assert rec["kind"] in ("clip_steps", "correctness")
    assert {m["name"] for m in rec["metrics"]} >= {"step_error_max", "step_floor_max", "step_error_over_floor_max"}


def test_the_reference_against_itself_is_bit_identical(run_dir):
    data, _ = steps(run_dir, "reference")
    assert data["verdict"] == "PASS" and data["max_e"] == 0.0
    assert all("bit-identical" in a["checks"]["steps"]["reason"] for a in data["audited"].values())


@pytest.mark.parametrize("label, check", [("fewer_steps", "layout"), ("other_sigmas", "schedule"), ("skip_guidance", "steps"), ("reuse", "steps"),
                                          ("lowp", "steps"), ("fewer_frames", "layout")])
def test_shortcuts_fail(run_dir, label, check):
    data, _ = steps(run_dir, label, audit_seed=1, requests=3)
    assert data["verdict"] == "FAIL"
    assert all(a["checks"][check]["verdict"] == "FAIL" for a in data["audited"].values()), \
        {k: a["checks"] for k, a in data["audited"].items()}


def _dump_of(run: Path, label: str, i: int) -> Path:
    sha = json.loads((run / "outputs" / label / "index.json").read_text())["files"][f"r1/c{i}.mp4"]
    return run / "dumps" / label / f"{sha}.safetensors"


def test_a_dump_whose_latents_are_not_the_updates_fails_the_chain(run_dir, tmp_path):
    fake = tmp_path / "dumps"
    fake.mkdir()
    for i in range(len(BODIES)):
        src = _dump_of(run_dir, "honest", i)
        (fake / src.name).write_bytes(src.read_bytes())
        keys = ["step.2.video.x_next", "step.3.video.x"]  # still continuous, but not the update of step 2
        r = subprocess.run([PY, "-c", TAMPER, str(fake / src.name), json.dumps(keys), "1e-3"], capture_output=True,
                           text=True, env=_env())
        assert r.returncode == 0, r.stderr
    data, _ = steps(run_dir, "honest", dumps=fake)
    assert data["verdict"] == "FAIL"
    assert all(a["checks"]["chain"]["verdict"] == "FAIL" and "scheduler's update" in a["checks"]["chain"]["reason"]
               for a in data["audited"].values())


def test_a_dump_of_another_request_fails(run_dir, tmp_path):
    fake = tmp_path / "dumps"
    fake.mkdir()
    for i in range(len(BODIES)):
        other = _dump_of(run_dir, "honest", (i + 1) % len(BODIES))
        (fake / _dump_of(run_dir, "honest", i).name).write_bytes(other.read_bytes())
    data, _ = steps(run_dir, "honest", dumps=fake)
    assert data["verdict"] == "FAIL"
    assert all(a["checks"]["dump"]["verdict"] == "FAIL" for a in data["audited"].values())


def test_an_honest_dump_beside_a_cached_clip_fails_the_decode_check(run_dir, tmp_path):
    """Each slot delivers another request's clip (a clip from a cache) with an honest dump of its own request named
    by that clip's sha256: every step and the chain pass, but the reference does not decode the delivered clip."""
    n = len(BODIES)
    run = tmp_path / "run"
    (run / "outputs" / "cached" / "r1").mkdir(parents=True)
    (run / "inputs" / "cached" / "r1").mkdir(parents=True)
    fake = tmp_path / "dumps"
    fake.mkdir()
    honest = json.loads((run_dir / "outputs" / "honest" / "index.json").read_text())["files"]
    files = {}
    for i in range(n):
        j = (i + 1) % n
        clip = (run_dir / "outputs" / "honest" / "r1" / f"c{j}.mp4").read_bytes()
        (run / "outputs" / "cached" / "r1" / f"c{i}.mp4").write_bytes(clip)
        files[f"r1/c{i}.mp4"] = honest[f"r1/c{j}.mp4"]
        (run / "inputs" / "cached" / "r1" / f"c{i}.json").write_text(json.dumps(BODIES[i]))
        (fake / f"{honest[f'r1/c{j}.mp4']}.safetensors").write_bytes(_dump_of(run_dir, "honest", i).read_bytes())
    (run / "outputs" / "cached" / "index.json").write_text(json.dumps({"files": files}))
    data, _ = steps(run, "cached", dumps=fake)
    assert data["verdict"] == "FAIL"
    for a in data["audited"].values():
        assert a["checks"]["decode"]["verdict"] == "FAIL", a["checks"]
        assert a["checks"]["steps"]["verdict"] == "PASS" and a["checks"]["chain"]["verdict"] == "PASS"


def test_a_missing_dump_fails(run_dir, tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    data, _ = steps(run_dir, "honest", dumps=empty)
    assert data["verdict"] == "FAIL"
    assert all("no dump" in a["checks"]["dump"]["reason"] for a in data["audited"].values())


def test_an_alternative_kernel_identical_to_the_reference_cannot_judge(run_dir):
    data, _ = steps(run_dir, "honest", alt_attention="native")
    assert data["verdict"] == "UNPROVEN"
    assert all("floor 0" in a["checks"]["steps"]["reason"] for a in data["audited"].values())


def test_audited_clips_follow_the_seed(run_dir):
    a, _ = steps(run_dir, "reference", audit_seed=3)
    b, _ = steps(run_dir, "reference", audit_seed=3)
    assert sorted(a["audited"]) == sorted(b["audited"])

