"""Video clip adapters: the clip HTTP server, the LTX-2 pipeline factories and the ComfyUI client.

No GPU and no model: fake pipelines (fakes_clip/fake_pipelines.py) and a fake ComfyUI server
(fakes_clip/fake_comfyui.py). Tests that need torch, Diffusers or bench clip's client module skip when
those are not installed.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import http.client
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes_clip.fake_comfyui import ComfyConfig, FakeComfyUI, ltx_workflow
from fakes_clip.fake_pipelines import FakeClipPipeline, fake_noise

from inferencekit.engines.clip import comfyui as cu
from inferencekit.engines.clip import server as cs
from inferencekit.registry import OpError

ROOT = Path(__file__).resolve().parents[1]
JOB = {"task": "t2v", "prompt": "a red fox runs through snow", "width": 768, "height": 512, "num_frames": 97,
       "fps": 24, "steps": 6, "guidance_scale": 3.0, "seed": 11}
SMALL = {**JOB, "width": 64, "height": 32, "num_frames": 17}  # torch tests: real frame arrays stay small


# --------------------------------------------------------------------------- helpers


class Running:
    """A ClipServer serving in a thread; ``post`` and ``get`` return (status, JSON)."""

    def __init__(self, pipe=None, load=None, **kw):
        kw.setdefault("noise_fn", fake_noise)
        self.pipe = pipe or FakeClipPipeline()
        self.srv = cs.ClipServer(load or (lambda: self.pipe), port=0, **kw)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        self.srv.start_loading().join(5)
        return self

    def __exit__(self, *exc):
        self.srv.close()

    def _req(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=10)
        try:
            data = json.dumps(body).encode() if body is not None else None
            conn.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read() or b"{}")
        finally:
            conn.close()

    def post(self, body):
        return self._req("POST", "/v1/clips", body)

    def get(self, path="/health"):
        return self._req("GET", path)


def workflow_file(tmp_path, doc=None, name="wf.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc or ltx_workflow()))
    return p


# --------------------------------------------------------------------------- request parsing and the stage clock


def test_parse_job_defaults_noise_seed_to_seed_and_refuses_unknown_fields():
    job = cs.parse_job(dict(JOB))
    assert job.noise_seed == 11 and job.image is None and job.audio is None
    assert cs.parse_job({**JOB, "noise_seed": 5}).noise_seed == 5
    for bad, msg in (({**JOB, "negative_prompt": "x"}, "unknown field"), ({k: v for k, v in JOB.items() if k != "steps"},
                     "missing"), ({**JOB, "task": "v2v"}, "task"), ({**JOB, "steps": 2.5}, "steps"),
                     ({**JOB, "steps": True}, "steps"), ({**JOB, "prompt": " "}, "prompt"),
                     ({**JOB, "image": "aGk="}, "i2v only"), ({**JOB, "task": "i2v"}, "needs an image"),
                     ({**JOB, "task": "i2v", "image": "***"}, "base64"), ({**JOB, "audio": "yes"}, "audio")):
        with pytest.raises(cs.BadRequest, match=msg):
            cs.parse_job(bad)
    job = cs.parse_job({**JOB, "task": "i2v", "image": base64.b64encode(b"PNGDATA").decode(), "audio": False})
    assert job.image == b"PNGDATA" and job.audio is False


def test_stage_clock_folds_nested_spans_and_counts_steps():
    syncs = []
    clock = cs.StageClock(lambda: syncs.append(1))
    with clock.span("text_encode"):
        with clock.span("inner"):
            time.sleep(0.002)
    clock.denoise_start()
    for _ in range(3):
        time.sleep(0.001)
        clock.step()
    with clock.span("vae_decode"):
        pass
    out = clock.result()
    assert set(out) == {"text_encode", "denoise", "denoise_steps", "vae_decode"}
    assert clock.steps == 3 and clock.counted and len(out["denoise_steps"]) == 3
    assert out["denoise"] == pytest.approx(sum(out["denoise_steps"]), abs=0.01)
    assert out["text_encode"] >= 2.0 and syncs  # the device is synced around stages
    # no transformer hook: the loop starts where the last span ended
    c2 = cs.StageClock()
    with c2.span("text_encode"):
        pass
    assert not c2.counted  # nothing counted yet: steps_run would be null, not 0
    c2.step()
    assert c2.steps == 1 and c2.result()["denoise_steps"][0] >= 0
    c3 = cs.StageClock()
    c3.set_steps(4)  # a pipeline that counts another way
    assert (c3.steps, c3.counted) == (4, True)


def test_cuda_sync_syncs_the_pipeline_device(monkeypatch):
    """torch.cuda.synchronize() with no argument syncs the current device; cuda:1 must be named."""
    from types import SimpleNamespace

    synced = []
    fake_torch = SimpleNamespace(device=lambda d: ("device", d),
                                 cuda=SimpleNamespace(synchronize=lambda device=None: synced.append(device)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert cs.cuda_sync("cpu") is None
    cs.cuda_sync("cuda:1")()
    cs.cuda_sync("cuda")()
    assert synced == [("device", "cuda:1"), ("device", "cuda")]


def test_timed_hook_calls_through_without_a_clock():
    class Obj:
        def work(self, x):
            return x * 2

    o, holder = Obj(), {"clock": None}
    cs.hook_method(o, "work", "stage", lambda: holder["clock"])
    assert o.work(3) == 6
    holder["clock"] = clock = cs.StageClock()
    assert o.work(4) == 8 and "stage" in clock.result()
    assert Obj().work(1) == 2  # the class is untouched


# --------------------------------------------------------------------------- the server


def test_server_answers_a_clip_with_counted_steps_stages_and_hash(tmp_path):
    with Running(out_dir=tmp_path / "clips") as r:
        assert r.get() == (200, {"status": "ok", "pipeline": "fake", "busy": False})
        status, doc = r.post({**JOB, "noise_seed": 99})
    assert status == 200, doc
    path = Path(doc["video_path"])
    assert path.parent == (tmp_path / "clips").resolve() and doc["format"] == "mp4" and doc["cache"] == []
    assert doc["video_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc["video_bytes"] == path.stat().st_size
    eff = doc["effective"]
    assert (eff["steps_run"], eff["width"], eff["height"], eff["frames"], eff["fps"]) == (6, 768, 512, 97, 24)
    assert eff["dtype"] == "bf16" and eff["attention"] == "fake-sdpa" and eff["fake"] is True
    assert eff["noise_seed"] == 99 and eff["noise_sha256"] == fake_noise(99, {"video": (1, 4, 13, 16, 24)})[1]
    st = doc["stage_ms"]
    assert {"text_encode", "denoise", "denoise_steps", "vae_decode", "mux"} <= set(st) and "inner" not in st
    assert len(st["denoise_steps"]) == 6
    job, noise = r.pipe.calls[0]
    assert job.noise_seed == 99 and job.seed == 11 and noise == {"video": (99, (1, 4, 13, 16, 24))}


def test_server_reports_steps_the_pipeline_ran_not_steps_asked(tmp_path):
    with Running(FakeClipPipeline(skip_steps=2), out_dir=tmp_path) as r:
        status, doc = r.post(JOB)
    assert status == 200 and doc["effective"]["steps_run"] == 4


def test_server_errors_keep_it_serving(tmp_path):
    pipe = FakeClipPipeline()
    with Running(pipe, out_dir=tmp_path) as r:
        assert r.post({**JOB, "width": 700})[0] == 400  # the pipeline's shape check
        assert r.post({**JOB, "extra": 1})[0] == 400
        assert r.get("/nope")[0] == 404
        pipe.fail = "CUDA out of memory"
        status, doc = r.post(JOB)
        assert status == 500 and "CUDA out of memory" in doc["error"]
        pipe.fail = None
        assert r.post(JOB)[0] == 200
        conn = http.client.HTTPConnection("127.0.0.1", r.srv.port, timeout=5)
        conn.request("POST", "/v1/clips", body=b"{not json", headers={"Content-Type": "application/json"})
        assert conn.getresponse().status == 400
        conn.close()


def test_server_b64_answer_and_keep(tmp_path):
    with Running(out_dir=tmp_path, answer="b64", keep=2) as r:
        docs = [r.post({**JOB, "seed": s})[1] for s in (1, 2, 3)]
    assert all("video_path" not in d for d in docs)
    assert hashlib.sha256(base64.b64decode(docs[-1]["video_b64"])).hexdigest() == docs[-1]["video_sha256"]
    assert len(list(tmp_path.glob("clip-*.mp4"))) == 2  # the oldest was removed
    assert len({d["video_sha256"] for d in docs}) == 3


def test_health_is_503_while_loading_and_one_clip_runs_at_a_time(tmp_path):
    loaded, gate = threading.Event(), threading.Event()
    pipe = FakeClipPipeline(gate=gate)

    def load():
        loaded.wait(5)
        return pipe

    srv = cs.ClipServer(load, port=0, out_dir=tmp_path, noise_fn=fake_noise)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        srv.start_loading()
        assert srv.health()[0] == 503 and srv.clip(JOB)[0] == 503
        loaded.set()
        for _ in range(100):
            if srv.pipeline is not None:
                break
            time.sleep(0.01)
        results = []
        threads = [threading.Thread(target=lambda s=s: results.append(srv.clip({**JOB, "seed": s}))) for s in (1, 2)]
        for t in threads:
            t.start()
        time.sleep(0.05)
        status, doc = srv.health()
        assert status == 200 and doc["busy"] is True and len(pipe.calls) == 1  # the second clip waits
        gate.set()
        for t in threads:
            t.join(5)
        assert [s for s, _ in results] == [200, 200]
    finally:
        srv.close()


def test_load_failure_stops_the_server(tmp_path):
    def load():
        raise RuntimeError("no such model")

    srv = cs.ClipServer(load, port=0, out_dir=tmp_path, log=io.StringIO())
    srv.start_loading()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive() and "no such model" in srv.load_error
    assert srv.health() == (500, {"status": "failed", "error": "RuntimeError: no such model"})
    srv.httpd.server_close()


def test_server_cli_runs_a_pipeline_factory(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])}
    proc = subprocess.Popen([sys.executable, "-m", "inferencekit.engines.clip.server", "--pipeline",
                             "fakes_clip.fake_pipelines:make_pipeline", "--model", "none", "--port", "0",
                             "--device", "cpu", "--out-dir", str(tmp_path), "--pipeline-arg", "skip_steps=1"],
                            stderr=subprocess.PIPE, text=True, env=env, cwd=tmp_path)
    try:
        line = proc.stderr.readline()
        port = int(line.split("listening on 127.0.0.1:")[1].split(",")[0])
        resolved = proc.stderr.readline()
        assert json.loads(resolved)["inferencekit_resolved_config"]["pipeline"] == "fake"
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", "/v1/clips", body=json.dumps(JOB).encode(), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        doc = json.loads(resp.read())
        conn.close()
        assert resp.status == 200 and doc["effective"]["steps_run"] == 5 and Path(doc["video_path"]).is_file()
    finally:
        proc.terminate()
        proc.wait(10)
        proc.stderr.close()


def test_server_against_bench_clip_client(tmp_path):
    cc = pytest.importorskip("inferencekit.serve.clip_client")
    from inferencekit.serve.client import Endpoint

    with Running(out_dir=tmp_path) as r:
        backend = cc.make_backend("clip_http", Endpoint.parse(f"http://127.0.0.1:{r.srv.port}"))
        assert backend.health()[0] == 200
        req = cc.ClipRequest(id="c1", task="t2v", prompt=JOB["prompt"], width=768, height=512, num_frames=97, fps=24,
                             steps=6, guidance_scale=3.0, seed=11, noise_seed=7)
        res = backend.generate(req)
    assert res.ok, res.error
    assert res.via == "video_path" and res.e2e_ms > 0 and res.cache == []
    assert res.effective["steps_run"] == 6 and res.effective["noise_seed"] == 7
    assert cc.stage_sum_ms(res.stage_ms) <= res.e2e_ms


# --------------------------------------------------------------------------- noise and the Diffusers LTX-2 factory


def test_initial_noise_is_drawn_on_the_cpu_in_order():
    torch = pytest.importorskip("torch")
    from inferencekit.engines.clip import noise

    shapes = {"video": (1, 4, 2, 3, 5), "audio": (1, 2, 6, 4)}
    a, b = noise.initial_noise(42, shapes), noise.initial_noise(42, shapes)
    g = torch.Generator("cpu").manual_seed(42)
    assert torch.equal(a["video"], torch.randn((1, 4, 2, 3, 5), generator=g, dtype=torch.float32))
    assert torch.equal(a["audio"], torch.randn((1, 2, 6, 4), generator=g, dtype=torch.float32))
    assert a["video"].device.type == "cpu" and a["video"].dtype == torch.float32
    assert noise.digest(a) == noise.digest(b) != noise.digest(noise.initial_noise(43, shapes))
    assert noise.digest(a) != noise.digest({"audio": a["audio"], "video": a["video"]})  # order matters


def _ltx(pipe, **extra):
    import torch

    from inferencekit.engines.clip.diffusers_ltx import LTXDiffusers

    task = "i2v" if extra.pop("i2v", False) else "t2v"
    opts = cs.PipelineOptions(model="fake/ltx", task=task, device="cpu", extra=extra)
    return LTXDiffusers(pipe, opts, torch, {"pipeline": "diffusers:Fake"})


def _png() -> bytes:
    PIL = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    PIL.new("RGB", (64, 32), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def test_diffusers_ltx_maps_the_request_hooks_stages_and_checks_noise():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from fakes_clip.fake_pipelines import fake_ltx2

    from inferencekit.engines.clip import noise

    pipe = fake_ltx2()
    ad = _ltx(pipe, negative_prompt="blurry", **{"call.decode_timestep": 0.05})
    job = cs.parse_job({**SMALL, "noise_seed": 5})
    shapes = ad.noise_shapes(job)
    assert shapes == {"video": (1, 4, 3, 1, 2), "audio": (1, 2, round(17 / 24 * 25), 4)}
    clock = cs.StageClock()
    out = ad.generate(job, noise.initial_noise(5, shapes), clock)
    kw = pipe.last_kwargs
    assert (kw["prompt"], kw["negative_prompt"], kw["height"], kw["width"], kw["num_frames"], kw["frame_rate"],
            kw["steps"], kw["guidance_scale"], kw["decode_timestep"]) == (JOB["prompt"], "blurry", 32, 64, 17, 24.0, 6,
                                                                           3.0, 0.05)
    assert kw["generator"].device.type == "cpu"
    assert clock.steps == 6 and pipe.transformer.calls == 6
    st = clock.result()
    assert {"text_encode", "prepare_latents", "denoise", "vae_decode", "postprocess", "audio_decode"} <= set(st)
    assert (out.width, out.height, out.frames, out.fps) == (64, 32, 17, 24.0)
    assert out.audio is not None and out.audio_sample_rate == 24000
    assert out.effective["noise_check"] is True and out.effective["noise_checked"] == ["audio", "video"]
    d = ad.describe()
    assert d["dtype"] == "bf16" and d["task"] == "t2v" and d["call_kwargs"] == {"decode_timestep": 0.05}
    assert d["negative_prompt_sha256"] == hashlib.sha256(b"blurry").hexdigest()
    assert _ltx(fake_ltx2()).describe()["negative_prompt_sha256"] is None
    # hooks call straight through outside a clip
    assert pipe.encode_prompt("x").shape[0] == 1

    # offloading restores each module's forward after a call: the module hooks must survive that
    for mod in (pipe.transformer, pipe.connectors, pipe.vocoder):
        mod.forward = type(mod).forward.__get__(mod)
    starts = []

    class Clock(cs.StageClock):
        def denoise_start(self):
            starts.append(1)
            super().denoise_start()

    clock2 = Clock()
    ad.generate(job, noise.initial_noise(5, shapes), clock2)
    assert starts and clock2.steps == 6 and {"text_encode", "audio_decode"} <= set(clock2.result())
    assert clock2._depth == 0


def test_diffusers_ltx_noise_check_fails_when_the_pipeline_draws_other_noise():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from fakes_clip.fake_pipelines import fake_ltx2

    from inferencekit.engines.clip import noise

    ad = _ltx(fake_ltx2(wrong_noise=True))
    job = cs.parse_job(dict(SMALL))
    out = ad.generate(job, noise.initial_noise(job.noise_seed, ad.noise_shapes(job)), cs.StageClock())
    assert out.effective["noise_check"] is False


def test_diffusers_ltx_i2v_checks_noise_outside_the_image_tokens_and_refuses_bad_shapes():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from fakes_clip.fake_pipelines import fake_ltx2

    from inferencekit.engines.clip import noise

    pipe = fake_ltx2(i2v=True, with_audio=False)
    ad = _ltx(pipe, i2v=True)
    job = cs.parse_job({**SMALL, "task": "i2v", "image": base64.b64encode(_png()).decode(), "audio": False})
    shapes = ad.noise_shapes(job)
    assert list(shapes) == ["video"]
    out = ad.generate(job, noise.initial_noise(job.noise_seed, shapes), cs.StageClock())
    assert pipe.last_kwargs["image"].size == (64, 32) and out.effective["noise_check"] is True and out.audio is None
    for bad, msg in (({**JOB, "width": 760}, "multiples of 32"), ({**JOB, "num_frames": 96}, "8k\\+1"),
                     ({**JOB, "fps": 23.976}, "whole number")):
        with pytest.raises(cs.BadRequest, match=msg):
            _ltx(fake_ltx2()).noise_shapes(cs.parse_job(bad))
    with pytest.raises(cs.BadRequest, match="loaded the i2v pipeline"):
        ad.noise_shapes(cs.parse_job(dict(JOB)))


def test_diffusers_ltx_refuses_call_overrides_of_request_fields():
    torch = pytest.importorskip("torch")
    from fakes_clip.fake_pipelines import fake_ltx2

    from inferencekit.engines.clip.diffusers_ltx import LTXDiffusers

    opts = cs.PipelineOptions(model="m", device="cpu", extra={"call.num_inference_steps": 4})
    with pytest.raises(ValueError, match="may not set"):
        LTXDiffusers(fake_ltx2(), opts, torch)


def test_diffusers_ltx_reference_mode_refuses_what_changes_numbers(tmp_path):
    from inferencekit.engines.clip.diffusers_ltx import check_options, make_pipeline

    for extra, kw in (({"vae_tiling": True}, {}), ({"call.stg_scale": 0.0}, {}), ({}, {"dtype": "fp16"}),
                      ({}, {"attention": "sage"}), ({"pipeline_class": "LTX2ConditionPipeline"}, {})):
        opts = cs.PipelineOptions(model="unused", device="cpu", extra={"reference": True, **extra}, **kw)
        with pytest.raises(ValueError, match="reference mode"):  # refused before any import or load
            make_pipeline(opts)
    with pytest.raises(ValueError, match="unknown --pipeline-arg"):
        make_pipeline(cs.PipelineOptions(model="unused", device="cpu", extra={"steps": 4}))
    ref = {"reference": True, "default_negative_prompt": True}
    assert check_options(cs.PipelineOptions(model=str(tmp_path), extra=ref, attention="native")) is True
    assert check_options(cs.PipelineOptions(model="m", attention="flash")) is False
    with pytest.raises(ValueError, match="explicit negative prompt"):  # Diffusers would encode an empty one
        check_options(cs.PipelineOptions(model=str(tmp_path), extra={"reference": True}))
    assert check_options(cs.PipelineOptions(model=str(tmp_path), extra={"reference": True, "negative_prompt": ""}))


def test_diffusers_ltx_reference_mode_pins_native_attention():
    """Diffusers' default backend comes from DIFFUSERS_ATTN_BACKEND: reference mode must not depend on it."""
    torch = pytest.importorskip("torch")
    from fakes_clip.fake_pipelines import fake_ltx2

    from inferencekit.engines.clip.diffusers_ltx import LTXDiffusers, configure

    pipe = fake_ltx2()
    opts = configure(pipe, cs.PipelineOptions(model="m", device="cpu", extra={"reference": True}), reference=True)
    assert opts.attention == "native" and pipe.placed == ("to", "cpu")
    assert pipe.transformer.attention_backend == pipe.connectors.attention_backend == "native"
    assert LTXDiffusers(pipe, opts, torch).describe()["attention"] == "native"
    other = fake_ltx2()  # outside reference mode, no --attention leaves Diffusers' default (reported by describe)
    opts = configure(other, cs.PipelineOptions(model="m", device="cuda:1", offload="model"), reference=False)
    assert opts.attention is None and other.placed == ("model_offload", "cuda:1")
    assert getattr(other.transformer, "attention_backend", None) is None
    flash = fake_ltx2()
    assert configure(flash, cs.PipelineOptions(model="m", device="cpu", attention="flash"), False).attention == "flash"
    assert flash.transformer.attention_backend == flash.connectors.attention_backend == "flash"


def test_real_diffusers_ltx2_draws_the_recipe_noise_with_a_cpu_generator():
    """Calls Diffusers' own prepare_latents / prepare_audio_latents (t2v and i2v) with a CPU generator."""
    torch = pytest.importorskip("torch")
    ltx2 = pytest.importorskip("diffusers.pipelines.ltx2")
    from types import SimpleNamespace

    from inferencekit.engines.clip import noise

    T2V, I2V = ltx2.LTX2Pipeline, ltx2.LTX2ImageToVideoPipeline
    shapes = {"video": (1, 8, 3, 2, 3), "audio": (1, 2, 10, 4)}
    ref = noise.initial_noise(77, shapes)

    class Dist:
        def mode(self):
            return torch.zeros(1, 8, 1, 2, 3)

    vae = SimpleNamespace(latents_mean=torch.zeros(8), latents_std=torch.ones(8), config=SimpleNamespace(scaling_factor=1.0),
                          encode=lambda img: SimpleNamespace(latent_dist=Dist()))
    audio_vae = SimpleNamespace(latents_mean=torch.zeros(8), latents_std=torch.ones(8))
    for cls in (T2V, I2V):
        me = SimpleNamespace(vae_spatial_compression_ratio=32, vae_temporal_compression_ratio=8,
                             transformer_spatial_patch_size=1, transformer_temporal_patch_size=1,
                             audio_vae_mel_compression_ratio=4, vae=vae, audio_vae=audio_vae,
                             _pack_latents=cls._pack_latents, _pack_audio_latents=cls._pack_audio_latents,
                             _normalize_latents=cls._normalize_latents,
                             _normalize_audio_latents=cls._normalize_audio_latents,
                             _create_noised_state=cls._create_noised_state)
        g = torch.Generator("cpu").manual_seed(77)
        args = dict(batch_size=1, num_channels_latents=8, height=64, width=96, num_frames=17, noise_scale=0.0,
                    dtype=torch.float32, device=torch.device("cpu"), generator=g, latents=None)
        if cls is I2V:
            lat, mask = cls.prepare_latents(me, image=torch.zeros(1, 3, 64, 96), **args)
            free = mask[0] == 0
        else:
            lat = cls.prepare_latents(me, **args)
            free = torch.ones(lat.shape[1], dtype=torch.bool)
        packed = T2V._pack_latents(ref["video"])
        assert torch.equal(lat[:, free], packed[:, free]), cls.__name__
        aud = cls.prepare_audio_latents(me, 1, 2, 10, 16, 0.0, torch.float32, torch.device("cpu"), g, None)
        assert torch.equal(aud, T2V._pack_audio_latents(ref["audio"])), cls.__name__


def test_official_ltx_refuses_cleanly_without_the_package(monkeypatch):
    from inferencekit.engines.clip import official_ltx

    monkeypatch.setitem(sys.modules, "ltx_pipelines", None)  # import fails even where it is installed
    with pytest.raises(ImportError, match="ltx_pipelines"):
        official_ltx.make_pipeline(cs.PipelineOptions(model="m", extra={"builder": "x:y"}))


def test_official_ltx_maps_arguments_by_name_and_counts_steps():
    torch = pytest.importorskip("torch")
    from inferencekit.engines.clip.official_ltx import OfficialLTX

    class Denoiser(torch.nn.Module):
        def forward(self, x):
            return x

    class Pipe:
        def __init__(self):
            self.model = Denoiser()
            self.got = None

        def __call__(self, prompt, height, width, num_frames, frame_rate, num_inference_steps, cfg_guidance_scale, seed):
            self.got = dict(prompt=prompt, height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
                            steps=num_inference_steps, cfg=cfg_guidance_scale, seed=seed)
            for _ in range(num_inference_steps * 2):
                self.model(torch.zeros(1))
            return torch.zeros(1, num_frames, height, width, 3), None

    pipe = Pipe()
    ad = OfficialLTX(pipe, cs.PipelineOptions(model="m", device="cpu", extra={"step_module": "model",
                                                                              "forwards_per_step": 2}), torch)
    job = cs.parse_job({**JOB, "width": 64, "height": 32, "num_frames": 9, "noise_seed": 3})
    assert ad.noise_shapes(job) == {}
    clock = cs.StageClock()
    out = ad.generate(job, {}, clock)
    assert pipe.got == dict(prompt=JOB["prompt"], height=32, width=64, num_frames=9, frame_rate=24.0, steps=6, cfg=3.0,
                            seed=3)
    assert clock.steps == 6 and (out.width, out.height, out.frames) == (64, 32, 9)
    assert out.effective == {"noise_check": False, "generator": "package seed"}
    assert "generate" in clock.result()

    class NoSteps:
        def __call__(self, prompt, height, width, num_frames, frame_rate, guidance_scale, seed):
            return None

    with pytest.raises(cs.BadRequest, match="num_inference_steps"):
        OfficialLTX(NoSteps(), cs.PipelineOptions(model="m", device="cpu"), torch).generate(job, {}, cs.StageClock())


class _PackageWritesFile:
    """An official-package-shaped call with no step counter: it takes a seed and writes the mp4 itself."""

    def __call__(self, prompt, height, width, num_frames, frame_rate, num_inference_steps, guidance_scale, seed,
                 output_path):
        Path(output_path).write_bytes(b"\x00\x00\x00\x18ftypmp42" + hashlib.sha256(f"{prompt}{seed}".encode()).digest())


def _official_clip(tmp_path, monkeypatch) -> dict:
    from inferencekit.engines.clip.official_ltx import OfficialLTX

    monkeypatch.chdir(tmp_path)  # the package writes its file in the working directory first
    ad = OfficialLTX(_PackageWritesFile(), cs.PipelineOptions(model="m", device="cpu"), None)  # no torch needed
    assert ad.describe()["steps_counted"] is False
    with Running(ad, out_dir=tmp_path / "clips") as r:
        status, doc = r.post(JOB)
    assert status == 200, doc
    return doc


def test_server_reports_null_steps_when_nothing_counted_them(tmp_path, monkeypatch):
    """An honest engine whose steps nobody counted reports steps_run null (not checked), never 0."""
    eff = _official_clip(tmp_path, monkeypatch)["effective"]
    assert eff["steps_run"] is None and eff["noise_sha256"] is None and eff["noise_check"] is False
    assert (eff["width"], eff["height"], eff["frames"]) == (768, 512, 97)


def test_bench_clip_sees_no_step_mismatch_when_steps_were_not_counted(bench_clip_registered, tmp_path, monkeypatch):
    from inferencekit.ops import bench_clip as bc  # through the fixture: the registry is restored afterwards

    eff = _official_clip(tmp_path, monkeypatch)["effective"]
    mismatch, checked = bc.check_effective(eff, JOB)
    assert mismatch == [] and checked == 4  # width, height, frames, fps; steps_run is not checked


# --------------------------------------------------------------------------- ComfyUI


@pytest.mark.parametrize("cls, stage", [
    ("KSampler", "denoise"), ("KSamplerAdvanced", "denoise"), ("SamplerCustomAdvanced", "denoise"),
    ("LTXVBaseSampler", "denoise"), ("KSamplerSelect", "other"), ("SamplerEulerAncestral", "other"),
    ("BasicScheduler", "other"), ("LTXVScheduler", "other"), ("RandomNoise", "other"), ("CFGGuider", "other"),
    ("EmptyLTXVLatentVideo", "other"), ("LTXVEmptyLatentAudio", "other"), ("ModelSamplingLTXV", "other"),
    ("CLIPTextEncode", "text_encode"), ("LTXVConditioning", "text_encode"), ("LTXAVTextEncoderLoader", "load"),
    ("CheckpointLoaderSimple", "load"), ("LoadImage", "load"), ("LatentUpscaleModelLoader", "load"),
    ("LTXVImgToVideo", "image_encode"), ("VAEEncode", "image_encode"), ("LTXVLatentUpsampler", "upscale"),
    ("VAEDecode", "vae_decode"), ("VAEDecodeTiled", "vae_decode"), ("LTXVAudioVAEDecode", "audio_decode"),
    ("VHS_VideoCombine", "mux"), ("SaveVideo", "mux"), ("CreateVideo", "mux")])
def test_comfyui_stage_of_node_classes(cls, stage):
    assert cu.stage_of(cls) == stage
    assert (stage in cu.COSTLY) == (stage not in ("other", "load"))


def test_workflow_validation_and_defaults(tmp_path):
    wf = cu.Workflow.load(workflow_file(tmp_path))
    assert wf.sha256 == hashlib.sha256((tmp_path / "wf.json").read_bytes()).hexdigest()
    assert wf.allow_cached == {"1", "2", "5"} and wf.sampler_nodes == {"3"} and wf.output_node == "9"
    custom = cu.Workflow.parse(ltx_workflow(custom_sampler=True))
    assert custom.sampler_nodes == {"3"}  # SamplerCustomAdvanced, not KSamplerSelect
    assert custom.allow_cached == {"1", "2", "5", "21", "22", "23", "24"}  # loaders and constant helpers
    doc = ltx_workflow()
    doc["allow_cached"] = ["7"]  # adds to the default
    assert cu.Workflow.parse(doc).allow_cached == {"1", "2", "5", "7"}
    assert cu.Workflow.parse(doc).fixed_encoders == {"7"} and wf.fixed_encoders == frozenset()
    custom_doc = ltx_workflow(custom_sampler=True)
    custom_doc["allow_cached"] = ["7"]  # feeds LTXVConditioning.negative
    assert cu.Workflow.parse(custom_doc).fixed_encoders == {"7"}
    # among the costly nodes, only a fixed negative-prompt encoder may be allowed
    for ids, why in ((["6"], "node 6 \\(CLIPTextEncode\\) has a bound input"), (["8"], "vae_decode node"),
                     (["3"], "denoise node")):
        with pytest.raises(cu.WorkflowError, match=why):
            cu.Workflow.parse({**ltx_workflow(), "allow_cached": ids})
    custom_doc["allow_cached"] = ["20"]  # LTXVConditioning: its frame_rate is bound
    with pytest.raises(cu.WorkflowError, match="node 20 .*has a bound input"):
        cu.Workflow.parse(custom_doc)
    fixed_pos = ltx_workflow()
    fixed_pos["workflow"]["13"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "style", "clip": ["2", 0]}}
    fixed_pos["workflow"]["3"]["inputs"]["positive"] = ["13", 0]
    fixed_pos["allow_cached"] = ["13"]  # fixed text, but not a negative prompt
    with pytest.raises(cu.WorkflowError, match="feeds no 'negative' input \\(\\['3.positive'\\]\\)"):
        cu.Workflow.parse(fixed_pos)
    g = wf.build({"prompt": "fox", "seed": 4, "steps": 8, "width": 512, "height": 320, "num_frames": 49, "fps": 25,
                  "guidance_scale": 4.0})
    assert g["6"]["inputs"]["text"] == "fox" and g["3"]["inputs"]["steps"] == 8 and g["5"]["inputs"]["length"] == 49
    assert wf.graph["6"]["inputs"]["text"] == ""  # the pinned graph is not modified
    with pytest.raises(cu.WorkflowError, match="no binding for it"):
        wf.build({"audio": True})
    base = ltx_workflow()
    for change, msg in ((lambda d: d.update(format="x"), "format"),
                        (lambda d: d["bindings"].pop("guidance_scale"), "missing \\['guidance_scale'\\]"),
                        (lambda d: d["bindings"].update(cfg=[["3", "cfg"]]), "unknown \\['cfg'\\]"),
                        (lambda d: d["bindings"].update(steps=[["3", "nope"]]), "not an input"),
                        (lambda d: d["bindings"].update(steps=[["99", "steps"]]), "not an input"),
                        (lambda d: d.update(output_node="42"), "output_node"),
                        (lambda d: d.update(allow_cached=["77"]), "allow_cached"),
                        (lambda d: d["workflow"]["3"].pop("class_type"), "API format")):
        doc = json.loads(json.dumps(base))
        change(doc)
        with pytest.raises(cu.WorkflowError, match=msg):
            cu.Workflow.parse(doc)


def _values(**kw):
    v = {"prompt": "a fox", "seed": 1, "steps": 5, "width": 768, "height": 512, "num_frames": 97, "fps": 24,
         "guidance_scale": 3.0}
    v.update(kw)
    return v


def _client(srv, **kw):
    port = int(srv.base_url.rsplit(":", 1)[1])
    kw.setdefault("timeout_s", 10.0)
    return cu.ComfyUIClient("127.0.0.1", port, **kw)


def test_comfyui_run_over_websocket_times_and_observes(tmp_path):
    wf = cu.Workflow.load(workflow_file(tmp_path))
    with FakeComfyUI(ComfyConfig()) as srv:
        run = _client(srv).run(wf, _values())
        sent = srv.prompts[-1]
        expected = srv.files[srv.history[run.prompt_id]["outputs"]["9"]["gifs"][0]["filename"]]
    assert run.ok, run.error
    assert run.websocket and run.video == expected and run.format == "mp4"
    assert run.t_send_ns < run.t_connected_ns < run.t_done_ns <= run.t_end_ns
    assert sent["6"]["inputs"]["text"] == "a fox" and sent["3"]["inputs"]["seed"] == 1
    assert run.progress == {"3": 5} and run.cached == []
    assert {"queue_wait", "text_encode", "denoise", "vae_decode", "mux", "load", "fetch"} <= set(run.stage_ms)
    assert run.stage_ms["denoise"] > 0


def test_comfyui_without_websocket_polls_history_and_still_sees_the_cache(tmp_path):
    wf = cu.Workflow.load(workflow_file(tmp_path))
    with FakeComfyUI(ComfyConfig(websocket=False, history_delay_s=0.02)) as srv:
        client = _client(srv, poll_s=0.01)
        first = client.run(wf, _values())
        second = client.run(wf, _values(seed=2))  # same prompt text: ComfyUI skips both text encoders
    assert first.ok and second.ok and not first.websocket and first.progress == {}
    assert set(second.cached) == {"1", "2", "5", "6", "7"}  # loaders, the empty latent, both text encoders
    assert "denoise" not in first.stage_ms and "fetch" in first.stage_ms
    eff = cu.observed(first, wf)
    assert "steps_run" not in eff and "404" in eff["websocket"] and eff["steps_unchecked"].startswith("no websocket")


def test_comfyui_custom_sampler_helpers_cached_on_later_clips_are_not_flagged(tmp_path):
    """KSamplerSelect, BasicScheduler... are constants ComfyUI reuses on every clip after the first."""
    doc = ltx_workflow(custom_sampler=True)
    strict = cu.Workflow.parse(doc)
    doc["allow_cached"] = ["7"]  # the fixed negative prompt's encoder
    lenient = cu.Workflow.parse(doc)
    with FakeComfyUI() as srv:
        client = _client(srv)
        first = client.run(strict, _values(steps=4))
        second = client.run(strict, _values(prompt="a cat sleeps", seed=2, steps=4))
        sent = srv.prompts[-1]
    assert first.ok and second.ok, (first.error, second.error)
    assert sent["23"]["inputs"]["noise_seed"] == 2 and sent["22"]["inputs"]["steps"] == 4
    assert {"21", "22", "7"} <= set(second.cached)
    flagged = lambda wf, run: sorted(n for n in run.cached if n not in wf.allow_cached)  # noqa: E731
    assert flagged(strict, second) == ["7"] and flagged(lenient, second) == []
    eff = cu.observed(second, lenient)
    assert eff["steps_run"] == 4 and eff["websocket"] == "connected" and "steps_unchecked" not in eff
    assert eff["cached_allowed"] == ["1", "2", "21", "22", "5", "7"]
    assert {"text_encode", "denoise", "vae_decode", "mux"} <= set(first.stage_ms)


def test_comfyui_sampler_without_progress_leaves_steps_run_out(tmp_path):
    wf = cu.Workflow.load(workflow_file(tmp_path))
    with FakeComfyUI(ComfyConfig(progress=False)) as srv:
        run = _client(srv).run(wf, _values())
    assert run.ok and run.websocket and run.progress == {}
    eff = cu.observed(run, wf)
    assert "steps_run" not in eff
    assert eff["steps_unchecked"] == "no progress from sampler node(s) ['3'] on the websocket"
    nosampler = cu.Workflow.parse({**ltx_workflow(), "sampler_nodes": []})
    assert "steps_run" not in cu.observed(run, nosampler)


def test_comfyui_api_key_goes_on_the_websocket_upgrade_too(tmp_path):
    wf = cu.Workflow.load(workflow_file(tmp_path))
    with FakeComfyUI(ComfyConfig(api_key="k3y")) as srv:
        port = int(srv.base_url.rsplit(":", 1)[1])
        with pytest.raises(ConnectionError, match="401"):
            cu.WebSocket("127.0.0.1", port, "/ws?clientId=x", timeout=5.0)
        run = _client(srv, api_key="k3y").run(wf, _values(steps=3))
    assert run.ok and run.websocket and run.ws_note == "connected" and run.progress == {"3": 3}
    assert cu.observed(run, wf)["steps_run"] == 3


def test_comfyui_backend_flags_cached_timed_nodes_and_counts_steps(tmp_path):
    cc = pytest.importorskip("inferencekit.serve.clip_client")
    from inferencekit.serve.client import Endpoint

    path = workflow_file(tmp_path)
    with FakeComfyUI(ComfyConfig(skip_steps=1)) as srv:
        cu.register()
        backend = cc.make_backend("comfyui", Endpoint.parse(srv.base_url), timeout_s=10.0, workflow=str(path))
        assert backend.health() == (200, "comfyui 0.3.99-fake")
        req = cc.ClipRequest(id="c1", task="t2v", prompt="a fox", width=768, height=512, num_frames=97, fps=24,
                             steps=6, guidance_scale=3.0, seed=1)
        a = backend.generate(req)
        b = backend.generate(cc.ClipRequest(**{**req.__dict__, "seed": 2}))
        c = backend.generate(cc.ClipRequest(**{**req.__dict__, "seed": 2}))  # an exact repeat: all cached
        d = backend.describe()
    assert a.ok and a.e2e_ms > 0 and a.cache == [] and a.via == "view"
    assert a.effective["steps_run"] == 5 and a.effective["frames"] == 97  # the fake skipped a step
    assert a.effective["websocket"] == "connected"
    assert b.cache == ["6", "7"] and b.effective["cached_allowed"] == ["1", "2", "5"]
    assert {"3", "8", "9"} <= set(c.cache)
    assert "steps_run" not in c.effective and "no progress" in c.effective["steps_unchecked"]  # the sampler was cached
    assert d["workflow_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert d["comfyui_version"] == "0.3.99-fake" and d["devices"] == ["cuda:0 Fake GPU"]


def test_comfyui_allow_cached_lets_a_fixed_negative_prompt_through(tmp_path):
    cc = pytest.importorskip("inferencekit.serve.clip_client")
    from inferencekit.serve.client import Endpoint

    doc = ltx_workflow()
    doc["allow_cached"] = ["7"]  # added to the default (loaders and the empty latent)
    path = workflow_file(tmp_path, doc)
    with FakeComfyUI() as srv:
        backend = cu.ComfyUIBackend(Endpoint.parse(srv.base_url), timeout_s=10.0, workflow=str(path))
        reqs = [cc.ClipRequest(id=f"c{i}", task="t2v", prompt=p, width=768, height=512, num_frames=97, fps=24,
                               steps=4, guidance_scale=3.0, seed=i) for i, p in enumerate(("a fox", "a cat"))]
        results = [backend.generate(r) for r in reqs]
    assert [r.cache for r in results] == [[], []]
    assert results[1].effective["cached_allowed"] == ["1", "2", "5", "7"]
    assert [r.cache_allowed for r in results] == [[], ["negative_text_encode"]]  # bench clip states the reuse


def test_comfyui_i2v_uploads_the_image_and_noise_seed_goes_to_the_sampler(tmp_path):
    cc = pytest.importorskip("inferencekit.serve.clip_client")
    from inferencekit.serve.client import Endpoint

    path = workflow_file(tmp_path, ltx_workflow(i2v=True))
    with FakeComfyUI() as srv:
        backend = cu.ComfyUIBackend(Endpoint.parse(srv.base_url), timeout_s=10.0, workflow=str(path))
        res = backend.generate(cc.ClipRequest(id="i1", task="i2v", prompt="a fox", width=768, height=512, num_frames=97,
                                              fps=24, steps=3, guidance_scale=3.0, seed=1, image=b"PNGBYTES",
                                              noise_seed=31))
        sent, uploads = srv.prompts[-1], dict(srv.uploads)
        bad = backend.generate(cc.ClipRequest(id="i2", task="t2v", prompt="x", width=768, height=512, num_frames=97,
                                              fps=24, steps=3, guidance_scale=3.0, seed=1, audio=True))
    assert res.ok, res.error
    name = sent["12"]["inputs"]["image"]
    assert uploads[name] == b"PNGBYTES" and "upload" in res.stage_ms
    assert sent["3"]["inputs"]["seed"] == 31  # no noise_seed binding: the sampler's seed is its noise seed
    assert not bad.ok and "no binding" in bad.error


@pytest.mark.parametrize("cfg, msg", [(ComfyConfig(fail_node="8"), "node 8 \\(VAEDecode\\): fake failure"),
                                      (ComfyConfig(output_key="images", hang=True), "no result within")])
def test_comfyui_failures_are_results_not_exceptions(tmp_path, cfg, msg):
    wf = cu.Workflow.load(workflow_file(tmp_path))
    with FakeComfyUI(cfg) as srv:
        run = _client(srv, timeout_s=0.5).run(wf, _values())
        sent = (list(srv.interrupts), list(srv.deleted))
    assert not run.ok and run.video is None
    assert re.search(msg, run.error), run.error
    # a timed-out prompt that runs is interrupted by its id and dropped from the queue
    assert sent == (([run.prompt_id], [run.prompt_id]) if cfg.hang else ([], []))


def test_comfyui_timeout_while_queued_drops_the_prompt_and_interrupts_nothing(tmp_path):
    """Another client's prompt runs; ours waits. On our timeout it must not run later, nor stop theirs."""
    wf = cu.Workflow.load(workflow_file(tmp_path))
    with FakeComfyUI(ComfyConfig(hang=True)) as srv:
        theirs = srv.queue_prompt(wf.graph, None)
        for _ in range(100):
            if srv.running == theirs:
                break
            time.sleep(0.01)
        run = _client(srv, timeout_s=0.5).run(wf, _values())
        sent, still_running = (list(srv.interrupts), list(srv.deleted)), srv.running
    assert not run.ok and "no result within" in run.error
    assert sent == ([], [run.prompt_id]) and still_running == theirs and run.prompt_id not in srv.history


def test_comfyui_core_save_video_outputs_are_found(tmp_path):
    wf = cu.Workflow.load(workflow_file(tmp_path, ltx_workflow(output_class="SaveVideo")))
    with FakeComfyUI(ComfyConfig(output_key="images")) as srv:
        run = _client(srv).run(wf, _values())
    assert run.ok and run.video.startswith(b"\x00\x00\x00\x18ftyp")


# --------------------------------------------------------------------------- bench clip end to end (after the merge)


def test_bench_clip_finds_the_comfyui_kind_in_a_fresh_interpreter(tmp_path):
    """Merge gate: with only bench clip's client imported (nothing imports this adapter first),
    make_backend("comfyui") must reach ComfyUIBackend. Skips until the video client is in this build."""
    pytest.importorskip("inferencekit.serve.clip_client")
    code = ("import sys\n"
            "from inferencekit.serve import clip_client as cc\n"
            "from inferencekit.serve.client import Endpoint\n"
            "assert 'inferencekit.engines.clip.comfyui' not in sys.modules\n"
            "b = cc.make_backend('comfyui', Endpoint.parse('http://127.0.0.1:1'), workflow=sys.argv[1])\n"
            "print(type(b).__module__, type(b).__name__)\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code, str(workflow_file(tmp_path))], capture_output=True, text=True,
                         env=env, cwd=tmp_path, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["inferencekit.engines.clip.comfyui", "ComfyUIBackend"]


@pytest.fixture(scope="module")
def bench_clip_registered():
    """bench clip in the registry for these tests (skips until the video client is in this build); the
    registry is left as it was, since the contract test lists every registered operation."""
    pytest.importorskip("inferencekit.serve.clip_client")
    import importlib

    from inferencekit import registry

    before = "bench_clip" in registry.load_all()
    mod = pytest.importorskip("inferencekit.ops.bench_clip")
    if "bench_clip" not in registry.REGISTRY:  # imported earlier and its entry removed by another test module
        importlib.reload(mod)
    cu.register()
    yield registry
    if not before:  # as if never imported: a later import registers it again
        import inferencekit.ops

        registry.REGISTRY.pop("bench_clip", None)
        sys.modules.pop("inferencekit.ops.bench_clip", None)
        if hasattr(inferencekit.ops, "bench_clip"):
            delattr(inferencekit.ops, "bench_clip")


@pytest.fixture
def bench_clip_op(bench_clip_registered, tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))
    return lambda **args: bench_clip_registered.call("bench_clip", {"gpu_name": "Test GPU", "resamples": 200, **args})


def _clip_set(tmp_path) -> Path:
    rows = [{"id": f"t2v-{i}", "task": "t2v", "prompt": p, "width": 64, "height": 32, "num_frames": 17, "fps": 24,
             "steps": 4, "guidance_scale": 3.0} for i, p in enumerate(("a red fox runs through snow", "a cat sleeps"))]
    p = tmp_path / "clips.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_bench_clip_over_the_clip_server_is_a_valid_claim(store, bench_clip_op, tmp_path):
    with Running(out_dir=tmp_path / "srv") as r:
        rec = bench_clip_op(base_url=f"http://127.0.0.1:{r.srv.port}", clips=str(_clip_set(tmp_path)), rounds=2,
                            warmup=1)
    claim = rec["data"]["claim_validity"]
    assert claim["valid"] is True, claim
    art = next(a for a in rec["artifacts"] if a["role"] == "requests")
    rows = [json.loads(x) for x in (store.run_dir(rec["evidence_id"]) / art["path"]).read_text().splitlines()]
    assert len(rows) == 4 and all(x["ok"] and x["effective_checked"] == 5 and x["stage_sum_ok"] for x in rows)


def test_bench_clip_over_comfyui_allows_only_the_fixed_negative_prompt(store, bench_clip_op, tmp_path):
    """ComfyUI encodes a fixed negative prompt once. Listed in allow_cached, that reuse is allowed (and stated);
    the positive prompt's encoder is never reused, since the warm-up has prompts of its own and neighbouring
    clips differ."""
    strict = workflow_file(tmp_path, ltx_workflow(), "strict.json")
    doc = ltx_workflow()
    doc["allow_cached"] = ["7"]  # the negative prompt's encoder, on top of the loaders and the empty latent
    lenient = workflow_file(tmp_path, doc, "lenient.json")
    with FakeComfyUI() as srv:
        recs = [bench_clip_op(base_url=srv.base_url, clips=str(_clip_set(tmp_path)), endpoint_kind="comfyui",
                              workflow=str(wf), rounds=2, warmup=1, label=wf.stem) for wf in (strict, lenient)]
        doc["allow_cached"] = ["6", "7"]  # the positive prompt's encoder too: refused
        with pytest.raises(OpError, match="only the fixed negative prompt's encoder may be reused"):
            bench_clip_op(base_url=srv.base_url, clips=str(_clip_set(tmp_path)), endpoint_kind="comfyui",
                          workflow=str(workflow_file(tmp_path, doc, "both.json")), rounds=1, warmup=1)
    bad, good = (r["data"]["claim_validity"] for r in recs)
    assert bad["valid"] is False and len(bad["violations"]) == 4, bad
    assert all(v.endswith("engine reports cached stage(s) ['7']") for v in bad["violations"]), bad
    assert good["valid"] is True and good["cache_allowed_clips"] == 4, good
    assert recs[1]["data"]["backend"]["fixed_encoders"] == ["7"]
    assert any("fixed negative prompt" in x for x in recs[1]["limitations"])


# --------------------------------------------------------------------------- GPU-run fixes (RTX PRO 6000)


class _BuilderWritesFile(_PackageWritesFile):
    """A builder object (official_ltx:one_stage-shaped) that counts its own steps and says whether it muxed audio."""

    def __init__(self, audio: bool):
        self.steps_run, self.audio_written, self.audio = None, None, audio

    def describe(self) -> dict:
        return {"video_guider": {"stg_scale": 1.0}, "negative_prompt_sha256": "ab" * 32}

    def __call__(self, prompt, height, width, num_frames, frame_rate, num_inference_steps, guidance_scale, seed,
                 output_path):
        super().__call__(prompt, height, width, num_frames, frame_rate, num_inference_steps, guidance_scale, seed,
                         output_path)
        self.steps_run, self.audio_written = num_inference_steps, self.audio


@pytest.mark.parametrize("audio", [True, False])
def test_official_ltx_takes_steps_and_audio_from_a_builder_that_writes_the_file(tmp_path, monkeypatch, audio):
    """On the GPU host the one-stage builder wrote an mp4 with an AAC track while effective.audio said false:
    the server only knew of audio it muxed itself. A builder's own steps_run / audio_written now count."""
    from inferencekit.engines.clip.official_ltx import OfficialLTX

    monkeypatch.chdir(tmp_path)
    ad = OfficialLTX(_BuilderWritesFile(audio), cs.PipelineOptions(model="m", device="cpu"), None)
    d = ad.describe()
    assert d["steps_counted"] is True and d["video_guider"] == {"stg_scale": 1.0}
    assert d["negative_prompt_sha256"] == "ab" * 32  # the builder's own negative prompt, not the adapter's None
    with Running(ad, out_dir=tmp_path / "clips") as r:
        status, doc = r.post(JOB)
    assert status == 200, doc
    eff = doc["effective"]
    assert eff["steps_run"] == JOB["steps"] and eff["audio"] is audio


def test_official_ltx_audio_is_unknown_when_the_package_writes_the_file_unobserved(tmp_path, monkeypatch):
    eff = _official_clip(tmp_path, monkeypatch)["effective"]
    assert eff["audio"] is None  # was False: a file with an audio track was reported as silent


def _fake_ltx_package(monkeypatch, tmp_path):
    """Just enough of ltx_pipelines / ltx_core for official_ltx.one_stage: records what the pipeline got."""
    import types

    torch = pytest.importorskip("torch")
    calls = {}

    @dataclasses.dataclass(frozen=True)
    class Guider:
        cfg_scale: float = 3.0
        stg_scale: float = 1.0
        rescale_scale: float = 0.7
        modality_scale: float = 3.0
        skip_step: int = 0
        stg_blocks: tuple = (29,)

    class Params:
        video_guider_params = Guider()
        audio_guider_params = Guider(cfg_scale=7.0)

    class Stage:
        video_scale_factors = "kept"

        def __call__(self, *, denoiser, sigmas, steps):
            for i in range(steps):
                denoiser("transformer", None, None, sigmas, i)
            return "states"

    class Pipe:
        def __init__(self, model_paths, loras, device):
            calls["init"] = (model_paths, tuple(loras), str(device))
            self.stage = Stage()

        def __call__(self, **kw):
            calls["call"] = kw
            assert self.stage.video_scale_factors == "kept"  # the counting wrapper passes attributes through
            self.stage(denoiser=lambda *a: None, sigmas=None, steps=kw["num_inference_steps"])
            if kw["images"]:
                calls["image_file_existed"] = Path(kw["images"][0][0]).exists()
            return SimpleNamespace(video="frames", audio="wave", num_frames=kw["num_frames"], tiling_config=None)

    class ModelPaths:
        @staticmethod
        def from_monolith(ckpt, gemma_root):
            return ("monolith", ckpt, gemma_root)

        @staticmethod
        def from_split(**kw):
            return ("split", tuple(sorted(kw.items())))

    def encode_video(video, fps, audio, output_path, video_chunks_number):
        calls["encode"] = (video, fps, audio, video_chunks_number)
        Path(output_path).write_bytes(b"mp4")

    mods = {
        "ltx_pipelines": types.ModuleType("ltx_pipelines"),
        "ltx_pipelines.ti2vid_one_stage": types.SimpleNamespace(TI2VidOneStagePipeline=Pipe),
        "ltx_pipelines.utils": types.ModuleType("ltx_pipelines.utils"),
        "ltx_pipelines.utils.constants": types.SimpleNamespace(detect_params=lambda path: Params()),
        "ltx_pipelines.utils.model_paths": types.SimpleNamespace(ModelPaths=ModelPaths),
        "ltx_pipelines.utils.args": types.SimpleNamespace(DEFAULT_NEGATIVE_PROMPT="ugly, blurry"),
        "ltx_pipelines.utils.media_io": types.SimpleNamespace(encode_video=encode_video),
        "ltx_pipelines.utils.types": types.SimpleNamespace(ImageConditioningInput=lambda *a: a),
        "ltx_core": types.ModuleType("ltx_core"),
        "ltx_core.model": types.ModuleType("ltx_core.model"),
        "ltx_core.model.video_vae": types.SimpleNamespace(get_video_chunks_number=lambda n, t: 1),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return torch, calls


def test_official_ltx_one_stage_builder_runs_the_package_pipeline_with_its_defaults(tmp_path, monkeypatch):
    torch, calls = _fake_ltx_package(monkeypatch, tmp_path)
    from inferencekit.engines.clip import official_ltx as ol

    with pytest.raises(ValueError, match="gemma_root"):
        ol.one_stage(model="ckpt.safetensors", device="cpu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="BF16 only"):
        ol.one_stage(model="ckpt.safetensors", device="cpu", dtype=torch.float16, gemma_root="g")
    built = ol.one_stage(model="ckpt.safetensors", device="cpu", dtype=torch.bfloat16, gemma_root="gemma")
    assert calls["init"] == (("monolith", "ckpt.safetensors", "gemma"), (), "cpu")
    d = built.describe()
    assert d["video_guider"]["stg_scale"] == 1.0 and d["audio_guider"]["cfg_scale"] == 7.0
    assert d["negative_prompt_sha256"] == hashlib.sha256(b"ugly, blurry").hexdigest()
    monkeypatch.chdir(tmp_path)
    ad = ol.OfficialLTX(built, cs.PipelineOptions(model="ckpt.safetensors", device="cpu"), torch)
    with Running(ad, out_dir=tmp_path / "clips") as r:
        status, doc = r.post(JOB)
    assert status == 200, doc
    kw = calls["call"]
    assert kw["video_guider_params"].cfg_scale == JOB["guidance_scale"]  # the request's guidance is the video CFG
    assert kw["video_guider_params"].stg_scale == 1.0 and kw["audio_guider_params"].cfg_scale == 7.0
    assert (kw["negative_prompt"], kw["seed"], kw["num_inference_steps"], kw["images"]) == ("ugly, blurry", 11,
                                                                                            JOB["steps"], [])
    assert calls["encode"] == ("frames", 24, "wave", 1)
    assert doc["effective"]["steps_run"] == JOB["steps"] and doc["effective"]["audio"] is True
    split = ol.one_stage(model="m", device="cpu", dtype=torch.bfloat16, transformer="t.safetensors",
                         text_encoder="te.safetensors", video_vae="v.safetensors", audio_vae="a.safetensors",
                         negative_prompt="none of it")
    assert calls["init"][0][0] == "split" and split.negative_prompt == "none of it"
    # i2v: the image goes to the package as a file at frame 0, strength 1.0, removed afterwards
    i2v = ol.OfficialLTX(built, cs.PipelineOptions(model="m", task="i2v", device="cpu"), torch)
    job = cs.parse_job({**JOB, "task": "i2v", "image": base64.b64encode(_png()).decode()})
    i2v.generate(job, {}, cs.StageClock())
    img = calls["call"]["images"][0]
    assert img[1:] == (0, 1.0) and calls["image_file_existed"] and not Path(img[0]).exists()


def test_diffusers_ltx_options_for_a_checkpoint_with_a_full_transformer(tmp_path):
    """LTX-2.5-Diffusers loads its distilled transformer/ by default: a reference must name transformer_full."""
    from inferencekit.engines.clip.diffusers_ltx import check_options

    model = tmp_path / "ltx25"
    (model / "transformer_full").mkdir(parents=True)
    (tmp_path / "ltx2").mkdir()
    ref = {"reference": True, "default_negative_prompt": True}
    with pytest.raises(ValueError, match="transformer_subfolder=transformer_full"):
        check_options(cs.PipelineOptions(model=str(model), extra=ref))
    assert check_options(cs.PipelineOptions(model=str(model), extra={
        **ref, "transformer_subfolder": "transformer_full",
        "scheduler": {"use_dynamic_shifting": True, "shift_terminal": 0.1}}))
    for sched in (None, {"use_dynamic_shifting": True}):  # the distilled scheduler's sigmas with the full DiT
        with pytest.raises(ValueError, match="scheduler overrides that go with it"):
            check_options(cs.PipelineOptions(model=str(model), extra={
                **ref, "transformer_subfolder": "transformer_full", **({"scheduler": sched} if sched else {})}))
    assert check_options(cs.PipelineOptions(model=str(tmp_path / "ltx2"), extra=ref)) is True  # no transformer_full
    assert check_options(cs.PipelineOptions(model=str(model))) is False  # outside reference mode: allowed
    with pytest.raises(ValueError, match="not both"):
        check_options(cs.PipelineOptions(model="m", extra={"negative_prompt": "x", "default_negative_prompt": True}))
    with pytest.raises(ValueError, match="JSON object"):
        check_options(cs.PipelineOptions(model="m", extra={"scheduler": "dynamic"}))


def test_diffusers_ltx_reference_mode_looks_a_hub_id_up_in_the_hf_cache(tmp_path, monkeypatch):
    """The quality checker and the engine under test load LTX-2.5-Diffusers by hub id from the Hugging Face cache, not from a folder: the
    distilled-transformer guard must look there too (offline), and a reference whose checkpoint it cannot see is
    refused rather than assumed safe."""
    import types

    from inferencekit.engines.clip.diffusers_ltx import _has_full_transformer, check_options

    snap = tmp_path / "hub" / "models--Lightricks--LTX-2.5-Diffusers" / "snapshots" / "da0b6909"
    (snap / "transformer_full").mkdir(parents=True)
    plain = tmp_path / "hub" / "models--Lightricks--LTX-2" / "snapshots" / "dfcc2108"
    plain.mkdir(parents=True)
    calls = []

    def snapshot_download(repo_id, revision=None, local_files_only=False, **kw):
        calls.append((repo_id, revision, local_files_only))
        found = {"Lightricks/LTX-2.5-Diffusers": snap, "Lightricks/LTX-2": plain}.get(repo_id)
        if found is None or not local_files_only:
            raise FileNotFoundError(repo_id)  # stands in for LocalEntryNotFoundError
        return str(found)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    ref = {"reference": True, "default_negative_prompt": True, "revision": "da0b6909"}
    with pytest.raises(ValueError, match="transformer_subfolder=transformer_full"):
        check_options(cs.PipelineOptions(model="Lightricks/LTX-2.5-Diffusers", extra=ref))
    assert calls[-1] == ("Lightricks/LTX-2.5-Diffusers", "da0b6909", True)  # offline lookup at the pinned revision
    assert check_options(cs.PipelineOptions(model="Lightricks/LTX-2", extra=ref)) is True
    with pytest.raises(ValueError, match="cannot tell whether"):
        check_options(cs.PipelineOptions(model="Lightricks/not-cached", extra=ref))
    assert check_options(cs.PipelineOptions(model="Lightricks/not-cached")) is False  # outside reference mode
    assert _has_full_transformer("Lightricks/not-cached") is None
    assert _has_full_transformer(str(snap)) is True


def _fake_diffusers(monkeypatch, scheduler_config):
    """Fake ``diffusers`` and ``torch`` modules for make_pipeline: just what it touches (from_pretrained, the
    transformer class, the scheduler's from_config, DEFAULT_NEGATIVE_PROMPT) and what LTXDiffusers hooks."""
    import types

    loads = []

    class Hookable:
        def __init__(self, **config):
            self.config = SimpleNamespace(**config)

        def register_forward_pre_hook(self, fn):
            pass

        def register_forward_hook(self, fn):
            pass

        def set_attention_backend(self, name):
            self.attention_backend = name

    class Transformer(Hookable):
        dtype = "torch.bfloat16"

        @classmethod
        def from_pretrained(cls, model, subfolder=None, **kw):
            loads.append(("transformer", model, subfolder, kw))
            t = cls(in_channels=128)
            t.subfolder = subfolder
            return t

    class Config(dict):
        pass

    class Scheduler:
        def __init__(self, config):
            self.config = Config(config)

        @classmethod
        def from_config(cls, config, **overrides):
            return cls({**config, **overrides})

    class Pipe:
        def __init__(self, transformer):
            self.transformer = transformer or Transformer(in_channels=128)
            self.connectors = Hookable()
            self.scheduler = Scheduler(scheduler_config)
            self.vae = SimpleNamespace(decode=lambda *a, **k: None)
            self.video_processor = SimpleNamespace(postprocess_video=lambda *a, **k: None)
            self.audio_vae = SimpleNamespace(decode=lambda *a, **k: None)
            self.vocoder = Hookable(output_sampling_rate=24000)

        def encode_prompt(self, *a, **k):
            return None

        def prepare_latents(self, *a, **k):
            return None

        def prepare_audio_latents(self, *a, **k):
            return None

        def to(self, device):
            self.device = device
            return self

    class LTX2Pipeline:
        _optional_components = ["processor", "prompt_enhancer", "duration_head"]  # as diffusers @7ecb1e1b

        @classmethod
        def from_pretrained(cls, model, **kw):
            loads.append(("pipeline", model, None, kw))
            return Pipe(kw.get("transformer"))

    utils = types.ModuleType("diffusers.pipelines.ltx2.utils")
    utils.DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry"
    diffusers = types.ModuleType("diffusers")
    diffusers.__version__ = "0.41.0.dev0"
    diffusers.LTX2Pipeline = LTX2Pipeline
    diffusers.LTX2VideoTransformer3DModel = Transformer
    torch = types.ModuleType("torch")
    torch.__version__, torch.bfloat16 = "2.13.0", "torch.bfloat16"
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    for name, mod in (("diffusers", diffusers), ("diffusers.pipelines", types.ModuleType("diffusers.pipelines")),
                      ("diffusers.pipelines.ltx2", types.ModuleType("diffusers.pipelines.ltx2")),
                      ("diffusers.pipelines.ltx2.utils", utils), ("torch", torch)):
        monkeypatch.setitem(sys.modules, name, mod)
    return loads, utils.DEFAULT_NEGATIVE_PROMPT


def test_diffusers_ltx_make_pipeline_loads_the_full_transformer_scheduler_and_negative_prompt(tmp_path, monkeypatch):
    """The LTX-2.5-Diffusers branch of make_pipeline (never run on LTX-2.5: the download is gated), with fake
    diffusers and torch modules: transformer_full/ is loaded and handed to from_pretrained, the scheduler gets its
    overrides, the negative prompt is Diffusers' DEFAULT_NEGATIVE_PROMPT, and describe() reports all three (what
    a re-run compares with the reference manifest's expect_config)."""
    from inferencekit.engines.clip.diffusers_ltx import make_pipeline

    loads, default_neg = _fake_diffusers(monkeypatch, {"use_dynamic_shifting": False, "shift_terminal": None,
                                                       "shift": 1.0, "base_shift": 0.95})
    model = tmp_path / "ltx25"
    (model / "transformer_full").mkdir(parents=True)
    ad = make_pipeline(cs.PipelineOptions(model=str(model), device="cpu", extra={
        "reference": True, "transformer_subfolder": "transformer_full", "default_negative_prompt": True,
        "revision": "da0b6909", "scheduler": {"use_dynamic_shifting": True, "shift_terminal": 0.1}}))
    kinds = [x[0] for x in loads]
    assert kinds == ["transformer", "pipeline"]
    assert loads[0][1:3] == (str(model), "transformer_full") and loads[0][3]["torch_dtype"] == "torch.bfloat16"
    assert "revision" not in loads[0][3] and "revision" not in loads[1][3]  # a local folder is that revision
    assert loads[1][3]["transformer"].subfolder == "transformer_full"
    # the optional components a clip never uses are not loaded (the 10 GB prompt enhancer, its processor, the
    # duration head), and the transformer load gets none of them
    assert {k: loads[1][3][k] for k in ("prompt_enhancer", "processor", "duration_head")} == dict.fromkeys(
        ("prompt_enhancer", "processor", "duration_head"))
    assert not {"prompt_enhancer", "processor", "duration_head"} & set(loads[0][3])
    assert ad.pipe.transformer.subfolder == "transformer_full"
    cfg = ad.pipe.scheduler.config
    assert cfg["use_dynamic_shifting"] is True and cfg["shift_terminal"] == 0.1 and cfg["base_shift"] == 0.95
    assert ad.negative_prompt == default_neg
    assert ad.pipe.transformer.attention_backend == ad.pipe.connectors.attention_backend == "native"
    d = ad.describe()
    assert d["reference"] is True and d["transformer_subfolder"] == "transformer_full"
    assert d["scheduler"] == {"class": "Scheduler", "use_dynamic_shifting": True, "shift_terminal": 0.1, "shift": 1.0}
    assert d["negative_prompt_sha256"] == hashlib.sha256(default_neg.encode()).hexdigest()
    assert d["attention"] == "native" and d["pipeline"] == "diffusers:LTX2Pipeline"
    assert d["skipped_components"] == ["prompt_enhancer", "processor", "duration_head"] and d["loaded_from"] == "folder"
    # without transformer_subfolder (a checkpoint with no transformer_full/): the default transformer, as shipped
    loads.clear()
    plain = tmp_path / "ltx2"
    plain.mkdir()
    ad = make_pipeline(cs.PipelineOptions(model=str(plain), device="cpu", extra={"reference": True,
                                                                                 "negative_prompt": "blurry"}))
    assert [x[0] for x in loads] == ["pipeline"] and "transformer" not in loads[0][3]
    assert ad.describe()["transformer_subfolder"] == "transformer" and ad.negative_prompt == "blurry"
    assert ad.pipe.scheduler.config["use_dynamic_shifting"] is False


def test_diffusers_ltx_make_pipeline_loads_a_pinned_hub_id_from_its_cache_snapshot(tmp_path, monkeypatch):
    """On a GPU host, real LTX-2.5-Diffusers by hub id with HF_HUB_OFFLINE=1: Diffusers' sharded loader asked
    the Hub for the file list and failed (OfflineModeIsEnabled). A hub id pinned to a revision that is in the HF
    cache is now loaded from its snapshot folder (no revision argument, no Hub call); one that is not cached is
    left to from_pretrained, with its revision; an unpinned hub id online is not looked up, and neither is a branch or
    tag name such as ``main`` (the cache's ref for it may point at an older snapshot).
    describe() records the revision and the snapshot's commit, which a reference manifest's expect_config can pin."""
    import types

    from inferencekit.engines.clip.diffusers_ltx import _cached_snapshot, make_pipeline

    loads, _ = _fake_diffusers(monkeypatch, {"use_dynamic_shifting": False, "shift_terminal": None})
    snap = tmp_path / "hub" / "models--Lightricks--LTX-2.5-Diffusers" / "snapshots" / ("da0b6909" * 5)
    (snap / "transformer_full").mkdir(parents=True)
    (snap / "model_index.json").write_text("{}")
    calls = []

    def snapshot_download(repo_id, revision=None, local_files_only=False, **kw):
        calls.append((repo_id, revision, local_files_only))
        if repo_id != "Lightricks/LTX-2.5-Diffusers" or not local_files_only:
            raise FileNotFoundError(repo_id)  # stands in for LocalEntryNotFoundError
        return str(snap)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    extra = {"reference": True, "transformer_subfolder": "transformer_full", "default_negative_prompt": True,
             "revision": "da0b6909" * 5, "scheduler": {"use_dynamic_shifting": True, "shift_terminal": 0.1}}
    ad = make_pipeline(cs.PipelineOptions(model="Lightricks/LTX-2.5-Diffusers", device="cpu", extra=extra))
    assert [x[:3] for x in loads] == [("transformer", str(snap), "transformer_full"), ("pipeline", str(snap), None)]
    assert "revision" not in loads[0][3] and "revision" not in loads[1][3]
    assert ("Lightricks/LTX-2.5-Diffusers", "da0b6909" * 5, True) in calls
    d = ad.describe()
    assert d["model"] == "Lightricks/LTX-2.5-Diffusers" and d["loaded_from"] == "hf_cache_snapshot"
    assert d["revision"] == "da0b6909" * 5 and d["snapshot_commit"] == "da0b6909" * 5
    # not in the cache: from_pretrained gets the hub id and the revision (it may download)
    loads.clear()
    make_pipeline(cs.PipelineOptions(model="Lightricks/other", device="cpu", extra={"revision": "abc"}))
    assert loads[0][1] == "Lightricks/other" and loads[0][3]["revision"] == "abc"
    # unpinned, or a branch name: looked up only offline
    n = len(calls)
    assert _cached_snapshot("Lightricks/LTX-2.5-Diffusers") is None and len(calls) == n
    assert _cached_snapshot("Lightricks/LTX-2.5-Diffusers", "main") is None and len(calls) == n
    assert _cached_snapshot("Lightricks/LTX-2.5-Diffusers", "da0b6909") is None and len(calls) == n  # short sha
    loads.clear()
    ad = make_pipeline(cs.PipelineOptions(model="Lightricks/LTX-2.5-Diffusers", device="cpu", extra={"revision": "main"}))
    assert loads[0][1] == "Lightricks/LTX-2.5-Diffusers" and loads[0][3]["revision"] == "main"
    assert ad.describe()["loaded_from"] == "hub" and ad.describe()["snapshot_commit"] is None
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert _cached_snapshot("Lightricks/LTX-2.5-Diffusers") == str(snap)
    assert _cached_snapshot("Lightricks/LTX-2.5-Diffusers", "main") == str(snap)
    assert _cached_snapshot(str(snap)) == str(snap)
    # a snapshot that is not a whole pipeline (no model_index.json) is not used
    (snap / "model_index.json").unlink()
    assert _cached_snapshot("Lightricks/LTX-2.5-Diffusers", "da0b6909" * 5) is None


def test_comfyui_workflow_declares_a_fixed_audio_track(tmp_path):
    """LTX-2's ComfyUI graphs always decode an audio track and have no input to switch it, so a clip set with
    "audio": true failed every clip on the GPU host ("the request sets audio but the workflow has no binding").
    A workflow file now declares "audio": true: a request asking for audio is served, one asking for none is
    refused, and the record says so."""
    cc = pytest.importorskip("inferencekit.serve.clip_client")
    from inferencekit.serve.client import Endpoint

    doc = ltx_workflow()
    doc["audio"] = True
    path = workflow_file(tmp_path, doc)
    with FakeComfyUI() as srv:
        backend = cu.ComfyUIBackend(Endpoint.parse(srv.base_url), timeout_s=10.0, workflow=str(path))
        base = dict(task="t2v", width=768, height=512, num_frames=97, fps=24, steps=3, guidance_scale=3.0, seed=1)
        ok = backend.generate(cc.ClipRequest(id="a1", prompt="a fox", audio=True, **base))
        unset = backend.generate(cc.ClipRequest(id="a2", prompt="a cat", **base))
        off = backend.generate(cc.ClipRequest(id="a3", prompt="an owl", audio=False, **base))
        d = backend.describe()
    assert ok.ok and ok.effective["audio"] is True, ok.error
    assert unset.ok and unset.effective["audio"] is True
    assert not off.ok and "always makes audio=true" in off.error
    assert d["audio"] is True
    for bad, message in (({"audio": "yes"}, "audio must be true"),
                         ({"audio": True, "bindings": {**doc["bindings"], "audio": [["9", "format"]]}}, "not both")):
        with pytest.raises(cu.WorkflowError, match=message):
            cu.Workflow.parse({**doc, **bad})
    assert cu.Workflow.parse(ltx_workflow()).audio is None  # undeclared: an audio request still needs a binding


@pytest.mark.parametrize("argv, mode", [(["main.py", "--listen", "127.0.0.1", "--port=8188"], "ram"),
                                        (["main.py", "--cache-classic"], "classic"),
                                        (["main.py", "--cache-lru", "40"], "lru"),
                                        (["main.py", "--cache-none"], "none"), (None, None)])
def test_bench_clip_records_comfyui_cache_mode_and_warns_unless_classic(store, bench_clip_op, tmp_path, argv, mode):
    """On the GPU host ComfyUI 0.40's default cache (RAM pressure) kept every earlier prompt's text encoding, so
    rounds 2 and 3 of a clip set reused them (CLAIM INVALID); --cache-classic keeps only the last prompt's outputs.
    The record now names the cache mode (from /system_stats argv) and warns for the modes that keep earlier
    prompts' outputs (ram, lru); --cache-none keeps nothing, so it gets no warning."""
    doc = ltx_workflow()
    doc["allow_cached"] = ["7"]
    with FakeComfyUI(ComfyConfig(argv=argv)) as srv:
        rec = bench_clip_op(base_url=srv.base_url, clips=str(_clip_set(tmp_path)), endpoint_kind="comfyui",
                            workflow=str(workflow_file(tmp_path, doc)), rounds=2, warmup=1)
    backend = rec["data"]["backend"]
    assert backend.get("cache_mode") == mode
    if argv is not None:
        assert backend["launch_flags"] == [a.split("=")[0] for a in argv if a.startswith("--")]  # names, no values
    warned = [x for x in rec["limitations"] if "--cache-classic" in x]
    assert bool(warned) == (mode in ("ram", "lru")), rec["limitations"]
    assert cu.cache_mode(["--cache-none"]) == "none"
