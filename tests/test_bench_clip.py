"""bench clip, the clip HTTP client, clip rounds and compare on clip records, against a fake clip server (no GPU).

The fake server below is stdlib only. Tests run it in a thread; the clip rounds test launches this file as
a script (``python tests/test_bench_clip.py --serve --port P ...``), the way bench rounds launches engines.
"""

import argparse
import base64
import hashlib
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --------------------------------------------------------------------------- the fake clip server


class _Handler(BaseHTTPRequestHandler):
    """POST /v1/clips and GET /health. ``server.cfg``: delay_ms, i2v_delay_ms, mode, delivery, out_dir, ctype."""

    def log_message(self, *args):
        pass

    def _send(self, status, body: bytes, ctype="application/json", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b'{"ok": true}')
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        cfg = self.server.cfg
        if self.path != "/v1/clips":
            self._send(404, b'{"error": "not found"}')
            return
        t0 = time.perf_counter()
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        self.server.requests.append(req)
        if "[fail]" in req["prompt"]:
            self._send(500, b'{"error": "engine failed at 127.0.0.1:9"}')
            return
        delay = cfg["i2v_delay_ms"] if req["task"] == "i2v" and cfg["i2v_delay_ms"] is not None else cfg["delay_ms"]
        time.sleep(delay / 1000.0)
        mode = cfg["mode"]
        seed = None if mode == "cache_bytes" else req["seed"]  # cache_bytes: the same clip whatever the seed
        key = json.dumps([req["prompt"], seed, req["width"], req["height"], req["num_frames"], req["steps"],
                          hashlib.sha256(req.get("image", "").encode()).hexdigest()])
        video = b"FAKEMP4" + hashlib.sha256(key.encode()).digest() * 8
        elapsed = (time.perf_counter() - t0) * 1000
        stage = {"text_encode": 0.1 * elapsed, "denoise": 0.7 * elapsed, "vae_decode": 0.1 * elapsed,
                 "total": elapsed, "denoise_steps": [0.7 * elapsed / req["steps"]] * req["steps"]}
        if mode == "overstage":
            stage["denoise"] = 1000.0 * elapsed + 5000.0
        effective = {"steps_run": req["steps"] // 2 if mode == "fewer_steps" else req["steps"],
                     "width": req["width"], "height": req["height"], "frames": req["num_frames"], "fps": req["fps"],
                     "dtype": "bf16", "attention": "sdpa"}
        meta = {"format": "mp4", "stage_ms": stage, "effective": None if mode == "no_effective" else effective,
                "cache": {"reported_cache": ["text_encode"], "neg_cache": ["negative_text_encode"]}.get(mode, [])}
        sha = hashlib.sha256(video).hexdigest()
        if cfg["delivery"] == "raw":
            meta.update(sha256=sha, bytes=len(video))  # the short names of video_sha256 / video_bytes
            if cfg["ctype"]:
                del meta["format"]  # the Content-Type subtype is all the client has to name the file
            self._send(200, video, ctype=cfg["ctype"] or "video/mp4", headers={"X-Clip-Meta": json.dumps(meta)})
            return
        if cfg["delivery"] == "path":
            p = Path(cfg["out_dir"]) / f"clip-{len(self.server.requests)}.mp4"
            # early: answers while the file is half written (the announced hash is the finished clip's)
            p.write_bytes(video[: len(video) // 2] if mode == "early" else video)
            meta["video_path"] = str(p)
            if mode != "path_nohash":
                meta.update(video_sha256=sha, video_bytes=len(video))
        else:
            meta["video_b64"] = base64.b64encode(video).decode()
            if mode == "wrong_hash":
                meta.update(video_sha256="0" * 64, video_bytes=len(video))
        self._send(200, json.dumps(meta).encode())


class FakeClipServer:
    def __init__(self, port=0, delay_ms=5.0, mode="honest", delivery="b64", out_dir=None, ctype=None,
                 i2v_delay_ms=None):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self.httpd.cfg = {"delay_ms": delay_ms, "i2v_delay_ms": i2v_delay_ms, "mode": mode, "delivery": delivery,
                          "out_dir": out_dir, "ctype": ctype}
        self.httpd.requests = []
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    @property
    def requests(self):
        return self.httpd.requests

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def _serve_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--delay-ms", type=float, default=5.0)
    p.add_argument("--mode", default="honest")
    p.add_argument("--label", default="")
    a = p.parse_args(argv)
    srv = FakeClipServer(a.port, a.delay_ms, a.mode)
    print(f"fake clip server {a.label} on {a.port}", flush=True)
    srv.httpd.serve_forever()


if __name__ == "__main__":
    _serve_main(sys.argv[1:])
    raise SystemExit(0)

# --------------------------------------------------------------------------- tests

import pytest  # noqa: E402

from inferencekit import evidence as ev  # noqa: E402
from inferencekit import registry  # noqa: E402
from inferencekit.registry import Call, OpError, Operation  # noqa: E402
from inferencekit.serve import clip_client as cc  # noqa: E402
from inferencekit.serve.client import Endpoint  # noqa: E402

THIS = str(Path(__file__).resolve())
REPO = Path(__file__).resolve().parents[1]
# A bench clip command template (clip_cmd): the clip rounds test runs bench clip through it; the default
# '-m inferencekit bench clip' runs in test_bench_rounds_mode_clip_runs_clip_rounds_with_the_default_bench_clip.
CLIP_CMD = (f"{sys.executable} -c 'import sys, inferencekit.ops.bench_clip; from inferencekit.cli import main; "
            "raise SystemExit(main())' bench clip")


@pytest.fixture(scope="module")
def mods():
    """Import bench clip and clip rounds; leave the registry as it was (the contract test lists every op)."""
    before = "bench_clip" in registry.load_all()
    from inferencekit.ops import bench_clip
    from inferencekit.ops import compare
    from inferencekit.serve import clip_rounds

    yield argparse.Namespace(bc=bench_clip, cr=clip_rounds, compare=compare)
    if not before:
        registry.REGISTRY.pop("bench_clip", None)


@pytest.fixture(autouse=True)
def endpoint_key(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))


def write_clips(tmp_path: Path, extra: list[dict] | None = None, name="clips") -> Path:
    (tmp_path / "img").mkdir(exist_ok=True)
    (tmp_path / "img" / "cat.png").write_bytes(b"\x89PNG fake image bytes")
    rows = [{"id": "t2v-0", "task": "t2v", "prompt": "a red fox runs through snow", "width": 64, "height": 48,
             "num_frames": 9, "fps": 24, "steps": 4, "guidance_scale": 3.0},
            {"id": "i2v-0", "task": "i2v", "prompt": "the cat turns its head", "image": "img/cat.png", "width": 64,
             "height": 48, "num_frames": 9, "fps": 24, "steps": 4, "guidance_scale": 3.0}] + (extra or [])
    p = tmp_path / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def bench(srv, clips, **kw):
    args = {"base_url": srv.base_url, "clips": str(clips), "gpu_name": "Test GPU", "rounds": 2, "resamples": 200,
            **kw}
    return registry.call("bench_clip", args)


def rows_of(store, rec):
    art = next(a for a in rec["artifacts"] if a["role"] == "requests")
    return [json.loads(x) for x in (store.run_dir(rec["evidence_id"]) / art["path"]).read_text().splitlines()]


def metric(rec, name, subject="all"):
    return next(m for m in rec["metrics"] if m["name"] == name and m.get("subject") == subject)


# --------------------------------------------------------------------------- schema

def _call():
    return Call(Operation("bench_clip", ("bench", "clip"), "t", (), lambda c, **k: {}), {},
                started="2026-01-01T00:00:00.000Z")


def test_schema_accepts_clip_kinds_and_workloads():
    w = {"mode": "single", "endpoint_kind": "clip_http", "task": "t2v", "width": 1280, "height": 704,
         "num_frames": 121, "fps": 24, "steps": 30, "guidance_scale": 3.0, "audio": True,
         "seed_policy": {"name": "per_round", "base": 0, "rule": "r"}, "seeds": [1, 2],
         "clip_set": {"name": "c", "sha256": "a" * 64}}
    for kind in ("clip_benchmark", "clip_gates", "clip_quality"):
        rec = ev.finalize(ev.build(_call(), kind=kind, claim_class="observed", summary="s",
                                   identity_=ev.identity(workload=w)))
        assert ev.validate(rec) == [], kind
        assert rec["identity"]["workload"]["clip_set"]["sha256"] == "a" * 64
    bad = ev.finalize(ev.build(_call(), kind="clip_benchmark", claim_class="observed", summary="s",
                               identity_=ev.identity(workload={**w, "task": "v2v", "seed_policy": {"name": "x"}})))
    errors = " ".join(ev.validate(bad))
    assert "task" in errors and "seed_policy" in errors


def test_every_published_record_still_validates():
    if not (REPO / "published").is_dir():
        pytest.skip("no published/ records in this repository")
    n = 0
    for p in sorted((REPO / "published").rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("schema") in ev.SCHEMA_FILES and "evidence_id" in d:
            n += 1
            assert ev.validate(d) == [], p
    assert n > 0


# --------------------------------------------------------------------------- clip set and seeds

def test_clip_set_is_checked_and_images_are_read(mods, tmp_path):
    cs = mods.bc.load_clip_set(write_clips(tmp_path))
    assert [c["id"] for c in cs.clips] == ["t2v-0", "i2v-0"]
    assert cs.images["i2v-0"][1] == hashlib.sha256(b"\x89PNG fake image bytes").hexdigest()
    assert cs.uniform("width") == 64 and cs.uniform("task") is None  # mixed tasks
    bad = [{"id": "t2v-0", "task": "t2v", "prompt": "dup", "width": 64, "height": 48, "num_frames": 9, "fps": 24,
            "steps": 4, "guidance_scale": 3.0},
           {"id": "x", "task": "i2v", "prompt": "no image", "width": 0, "height": 48, "num_frames": 9, "fps": 0,
            "steps": 4, "guidance_scale": -1},
           {"id": "y/z", "task": "v2v", "prompt": "", "width": 64, "height": 48, "num_frames": 9, "fps": 24,
            "steps": 4, "guidance_scale": 1}]
    with pytest.raises(OpError) as e:
        mods.bc.load_clip_set(write_clips(tmp_path, bad, name="bad"))
    msg = str(e.value)
    for part in ("id appears twice", "needs an image", "width must be", "fps must be", "guidance_scale must be",
                 "task must be", "prompt must be", "names a file"):
        assert part in msg, part


def test_seed_policies_are_deterministic_and_vary_per_round(mods):
    s = mods.bc.round_seed
    assert s("per_round", 0, None, 3) == s("per_round", 0, None, 3)
    assert len({s("per_round", 0, None, r) for r in range(20)}) == 20
    assert s("per_round", 0, None, 1) != s("per_round", 1, None, 1)
    assert all(0 <= s("per_round", 7, None, r) < 2 ** 31 for r in range(10))
    assert [s("fixed", 42, None, r) for r in range(3)] == [42, 42, 42]
    assert [s("list", 0, [5, 6, 7], r) for r in range(3)] == [5, 6, 7]
    assert mods.bc.warmup_seed(0, 0) not in {s("per_round", 0, None, r) for r in range(50)}


# --------------------------------------------------------------------------- the client

@pytest.mark.parametrize("delivery", ["b64", "path", "raw"])
def test_client_gets_the_clip_and_self_reports(tmp_path, delivery):
    with FakeClipServer(delay_ms=20, delivery=delivery, out_dir=str(tmp_path)) as srv:
        client = cc.make_backend("clip_http", Endpoint.parse(srv.base_url))
        assert client.health()[0] == 200
        req = cc.ClipRequest(id="a", task="i2v", prompt="p", width=64, height=48, num_frames=9, fps=24, steps=4,
                             guidance_scale=3.0, seed=11, image=b"img")
        res = client.generate(req)
        sent = srv.requests[-1]
    assert res.ok and res.video.startswith(b"FAKEMP4") and res.via == {"raw": "body"}.get(delivery, f"video_{delivery}")
    assert res.e2e_ms >= 20.0 and res.format == "mp4"
    assert res.effective["steps_run"] == 4 and res.cache == []
    assert cc.stage_sum_ms(res.stage_ms) < res.e2e_ms  # total and per-step list are not summed
    assert sent["seed"] == 11 and base64.b64decode(sent["image"]) == b"img" and sent["task"] == "i2v"


@pytest.mark.parametrize("delivery,mode,needle", [
    ("path", "early", "answered before the file was complete"),  # still writing when it answered
    ("path", "path_nohash", "must carry video_sha256 and video_bytes"),
    ("b64", "wrong_hash", "sha256 is not the announced one"),
])
def test_client_fails_clips_that_are_not_the_announced_bytes(tmp_path, delivery, mode, needle):
    with FakeClipServer(delivery=delivery, mode=mode, out_dir=str(tmp_path)) as srv:
        res = cc.ClipHTTP(Endpoint.parse(srv.base_url)).generate(
            cc.ClipRequest(id="a", task="t2v", prompt="p", width=8, height=8, num_frames=1, fps=8, steps=1,
                           guidance_scale=1, seed=0))
    assert not res.ok and needle in res.error, res.error
    assert res.e2e_ms is None and res.video is None  # a short read is a failed clip, never a short time


def test_raw_answer_content_type_cannot_name_a_path(tmp_path):
    evil = "video/../../../../escaped"
    with FakeClipServer(delivery="raw", ctype=evil) as srv:
        res = cc.ClipHTTP(Endpoint.parse(srv.base_url)).generate(
            cc.ClipRequest(id="a", task="t2v", prompt="p", width=8, height=8, num_frames=1, fps=8, steps=1,
                           guidance_scale=1, seed=0))
    assert res.ok and res.format == "bin" and res.via == "body"
    assert cc._clip_format("mp4") == "mp4" and cc._clip_format("../x") == "bin" and cc._clip_format("mp٤") == "bin"


def test_client_failures_are_results_with_scrubbed_errors():
    with FakeClipServer() as srv:
        client = cc.ClipHTTP(Endpoint.parse(srv.base_url))
        res = client.generate(cc.ClipRequest(id="a", task="t2v", prompt="[fail]", width=8, height=8, num_frames=1,
                                             fps=8, steps=1, guidance_scale=1, seed=0))
    assert not res.ok and res.status == 500 and res.e2e_ms is None and res.video is None
    assert "127.0.0.1" not in res.error and "HTTP 500" in res.error
    port = srv.httpd.server_address[1]
    dead = cc.ClipHTTP(Endpoint.parse(f"http://127.0.0.1:{port}")).generate(
        cc.ClipRequest(id="a", task="t2v", prompt="x", width=8, height=8, num_frames=1, fps=8, steps=1,
                       guidance_scale=1, seed=0))
    assert not dead.ok and "127.0.0.1" not in dead.error


def test_comfyui_kind_reaches_its_adapter_module():
    ep = Endpoint.parse("http://127.0.0.1:1")
    with pytest.raises(ValueError, match="needs --workflow"):
        cc.make_backend("comfyui", ep)  # the adapter (engines/clip/comfyui.py) is found; it needs a workflow
    with pytest.raises(LookupError, match="unknown endpoint kind"):
        cc.make_backend("nope", ep)
    with pytest.raises(ValueError, match="workflow"):
        cc.make_backend("clip_http", ep, workflow="w.json")


# --------------------------------------------------------------------------- bench clip

def test_bench_clip_times_clips_and_keeps_hashed_outputs(mods, store, tmp_path):
    clips = write_clips(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    with FakeClipServer(delay_ms=10) as srv:
        rec = bench(srv, clips, rounds=3, warmup=1, label="eng-a", run_dir=str(run), model="org/fake-video")
        sent = list(srv.requests)
    assert rec["kind"] == "clip_benchmark" and ev.validate(rec) == [] and rec["claim_class"] == "observed"
    rows = rows_of(store, rec)
    assert len(rows) == 6 and all(r["ok"] for r in rows)
    seeds = {r: mods.bc.round_seed("per_round", 0, None, r) for r in range(3)}
    assert [r["seed"] for r in rows] == [seeds[0]] * 2 + [seeds[1]] * 2 + [seeds[2]] * 2
    # one untimed warm-up per shape (t2v, then i2v with its image), both before the first timed clip
    assert len(sent) == 8 and [x["task"] for x in sent[:2]] == ["t2v", "i2v"] and "image" in sent[1]
    assert [x["seed"] for x in sent[:2]] == [mods.bc.warmup_seed(0, 0), mods.bc.warmup_seed(0, 1)]
    assert sent[2]["seed"] == seeds[0]
    # the warm-up's prompts are its own: no timed clip can reuse an encoding the warm-up made
    assert [x["prompt"] for x in sent[:2]] == ["a red fox runs through snow (warm-up)",
                                                "the cat turns its head (warm-up)"]
    assert not {x["prompt"] for x in sent[:2]} & {x["prompt"] for x in sent[2:]}
    assert rec["data"]["order"]["ids"] == ["t2v-0", "i2v-0"] and rec["data"]["order"]["reordered"] is False
    warm = rec["data"]["warmup"]
    assert warm["per_shape"] == 1 and [(x["task"], x["image"], x["clip"]) for x in warm["shapes"]] == [
        ("t2v", False, "t2v-0"), ("i2v", True, "i2v-0")]
    assert [(x["id"], x["shape"], x["ok"]) for x in warm["clips"]] == [("t2v-0", 0, True), ("i2v-0", 1, True)]
    assert all(r["stage_sum_ok"] is True and r["effective_checked"] == 5 for r in rows)
    w = rec["identity"]["workload"]
    assert w["seeds"] == [seeds[0], seeds[1], seeds[2]] and w["seed_policy"]["name"] == "per_round"
    assert w["width"] == 64 and w.get("task") is None and w["endpoint_kind"] == "clip_http"
    assert w["clip_set"]["sha256"] == hashlib.sha256(clips.read_bytes()).hexdigest()
    assert rec["data"]["claim_validity"]["valid"] is True
    # a round's clip_ms is the mean over its clips; the record's value is the median over rounds
    means = sorted((rows[2 * k]["e2e_ms"] + rows[2 * k + 1]["e2e_ms"]) / 2 for k in range(3))
    assert metric(rec, "clip_ms")["value"] == pytest.approx(means[1]) and metric(rec, "clip_ms")["value"] >= 10.0
    assert metric(rec, "clip_ms")["ci95"][0] is not None and metric(rec, "clip_ms")["n"] == 3
    per_round = sorted(3_600_000 * 2 / (rows[2 * k]["e2e_ms"] + rows[2 * k + 1]["e2e_ms"]) for k in range(3))
    assert metric(rec, "clips_per_hour")["value"] == pytest.approx(per_round[1])
    assert metric(rec, "clip_ms", "t2v")["n"] == 3 and metric(rec, "stage_denoise_ms")["stat"] == "median_self_reported"
    assert metric(rec, "clip_ms", "i2v")["value"] == pytest.approx(sorted(r["e2e_ms"] for r in rows[1::2])[1])
    assert [x["index"] for x in rec["data"]["rounds"]] == [0, 1, 2] and rec["data"]["rounds"][0]["seed"] == seeds[0]
    r0 = rec["data"]["rounds"][0]
    assert r0["clip_ms_t2v"] == rows[0]["e2e_ms"] and r0["clip_ms_i2v"] == rows[1]["e2e_ms"]
    assert r0["clip_ms"] == pytest.approx((rows[0]["e2e_ms"] + rows[1]["e2e_ms"]) / 2)
    assert r0["clips_per_hour_i2v"] == pytest.approx(3_600_000 / rows[1]["e2e_ms"])
    assert rec["data"]["per_task"]["all"]["clip_ms"]["per_clip"]["n"] == 6
    # outputs: every timed clip, hashed, in the record dir and the run dir; inputs carry the image hash only
    d = store.run_dir(rec["evidence_id"])
    index = json.loads((d / "outputs" / "eng-a" / "index.json").read_text())["files"]
    assert len(index) == 6
    for rel, sha in index.items():
        assert hashlib.sha256((d / "outputs" / "eng-a" / rel).read_bytes()).hexdigest() == sha
        assert (run / "outputs" / "eng-a" / rel).is_file()
    assert {r["out_sha256"] for r in rows} == set(index.values())
    body = json.loads((run / "inputs" / "eng-a" / "r0" / "i2v-0.json").read_text())
    assert body["image"]["sha256"] == hashlib.sha256(b"\x89PNG fake image bytes").hexdigest()
    assert (run / "evidence" / f"{rec['evidence_id']}.json").is_file()
    assert "127.0.0.1" not in json.dumps(rec)  # the engine URL is stored only as a digest of its HMAC tag
    assert set(rec["data"]["request"]["args"]["base_url"]) == {"digest"}


@pytest.mark.parametrize("mode,needle", [("fewer_steps", "steps_run 2 (asked 4)"),
                                         ("cache_bytes", "byte-identical clips"),
                                         ("reported_cache", "cached stage")])
def test_bench_clip_flags_engines_that_skip_work(mods, store, tmp_path, mode, needle):
    with FakeClipServer(mode=mode) as srv:
        rec = bench(srv, write_clips(tmp_path))
    claim = rec["data"]["claim_validity"]
    assert claim["valid"] is False and any(needle in v for v in claim["violations"]), claim
    assert "CLAIM INVALID" in rec["summary"] and any(needle in x for x in rec["limitations"])


def test_reusing_the_fixed_negative_prompt_encoding_is_allowed_and_stated(mods, store, tmp_path):
    with FakeClipServer(mode="neg_cache") as srv:
        rec = bench(srv, write_clips(tmp_path))
    claim = rec["data"]["claim_validity"]
    assert claim["valid"] is True and claim["cache_allowed_clips"] == 4, claim
    assert any("fixed negative prompt" in x and "Allowed for every engine" in x for x in rec["limitations"])
    rows = rows_of(store, rec)
    assert all(r["cache"] == [] and r["cache_allowed"] == ["negative_text_encode"] for r in rows)
    with FakeClipServer() as srv:
        plain = bench(srv, write_clips(tmp_path), label="plain")
    assert plain["data"]["claim_validity"]["cache_allowed_clips"] == 0
    assert not any("fixed negative prompt" in x for x in plain["limitations"])
    cmp = registry.call("compare", {"baseline": plain["evidence_id"], "candidate": rec["evidence_id"],
                                    "metric": "clip_ms", "resamples": 200})
    assert any("only the candidate reused its fixed negative prompt" in x for x in cmp["limitations"])


def test_clips_are_ordered_so_neighbours_differ(mods, store, tmp_path):
    """Neighbouring clips with the same prompt or image would let an engine reuse an encoding (ComfyUI does):
    bench clip runs an order without such neighbours when one exists, and names those it cannot avoid."""
    base = {"task": "t2v", "width": 64, "height": 48, "num_frames": 9, "fps": 24, "steps": 4, "guidance_scale": 3.0}
    rows = [{**base, "id": f"c{i}", "prompt": p} for i, p in enumerate(("fox", "fox", "cat", "cat"))]
    path = tmp_path / "same.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    clipset = mods.bc.load_clip_set(path)
    order = [c["id"] for c in mods.bc.timed_order(clipset)]
    assert mods.bc.clashes(clipset.clips, clipset) and not mods.bc.clashes(mods.bc.timed_order(clipset), clipset)
    with FakeClipServer() as srv:
        rec = bench(srv, path, rounds=2, warmup=1)
        sent = [x["prompt"] for x in srv.requests[1:]]  # after the one warm-up (one shape)
    assert rec["data"]["order"] == {**rec["data"]["order"], "ids": order, "reordered": True}
    assert all(a != b for a, b in zip(sent, sent[1:])), sent
    assert not any("neighbouring clips" in x for x in rec["limitations"])
    # images count too, and a clash that no order avoids is named
    (tmp_path / "img").mkdir(exist_ok=True)
    (tmp_path / "img" / "a.png").write_bytes(b"\x89PNG a")
    i2v = [{**base, "task": "i2v", "id": f"i{i}", "prompt": f"p{i}", "image": "img/a.png"} for i in range(2)]
    path = tmp_path / "img.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in i2v))
    with FakeClipServer() as srv:
        rec = bench(srv, path, rounds=1, warmup=1)
    assert any("1 pair(s) of neighbouring clips share a prompt or an image (e.g. i0 -> i1)" in x
               for x in rec["limitations"])
    assert rec["data"]["warmup"]["last_image_is_first_clips"] is True
    assert any("last warm-up clip used the first timed clip's image" in x for x in rec["limitations"])
    one = tmp_path / "one.jsonl"
    one.write_text(json.dumps({**base, "id": "solo", "prompt": "fox"}) + "\n")
    with FakeClipServer() as srv:
        rec = bench(srv, one, rounds=2, warmup=1)
    assert any("solo -> solo" in x for x in rec["limitations"])


def test_warm_up_does_not_end_on_the_first_timed_clips_image(mods, tmp_path):
    (tmp_path / "img").mkdir()
    for n in ("a", "b"):
        (tmp_path / "img" / f"{n}.png").write_bytes(f"\x89PNG {n}".encode())
    base = {"task": "i2v", "width": 64, "height": 48, "num_frames": 9, "fps": 24, "steps": 4, "guidance_scale": 3.0}
    rows = [{**base, "id": "x", "prompt": "p0", "image": "img/a.png"},
            {**base, "id": "y", "prompt": "p1", "image": "img/b.png"}]
    path = tmp_path / "c.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cs = mods.bc.load_clip_set(path)
    plan = mods.bc.warm_plan(mods.bc.warm_shapes(cs), mods.bc.timed_order(cs), cs)
    assert [c["id"] for _, c in plan] == ["y"]  # one shape; its first clip x has the first timed clip's image
    assert mods.bc.warmup_prompt("p0", {"p0", "p0 (warm-up)"}) == "p0 (warm-up 2)"


def test_bench_clip_notes_stage_overrun_missing_effective_and_failures(mods, store, tmp_path):
    with FakeClipServer(mode="overstage") as srv:
        rec = bench(srv, write_clips(tmp_path))
    assert rec["data"]["claim_validity"]["valid"] is True  # the client timer is untouched
    assert any("sum to more than the client-timed" in x for x in rec["limitations"])
    assert all(r["stage_sum_ok"] is False for r in rows_of(store, rec))
    with FakeClipServer(mode="no_effective") as srv:
        rec = bench(srv, write_clips(tmp_path), save_outputs="hash_only")
    assert any("reported no effective" in x for x in rec["limitations"])
    assert not any(a["role"] == "outputs_index" for a in rec["artifacts"])
    assert all(r["out_sha256"] for r in rows_of(store, rec))
    fail = [{"id": "bad-0", "task": "t2v", "prompt": "[fail] this one", "width": 64, "height": 48, "num_frames": 9,
             "fps": 24, "steps": 4, "guidance_scale": 3.0}]
    with FakeClipServer() as srv:
        rec = bench(srv, write_clips(tmp_path, fail, name="withfail"), warmup=0)
    assert rec["data"]["rounds"][0]["n_failed"] == 1 and "2 failed" in rec["summary"]
    assert any("first timed clip of each shape may include compilation" in x for x in rec["limitations"])
    assert rec["data"]["warmup"]["clips"] == [] and len(rec["data"]["warmup"]["shapes"]) == 2


def test_bench_clip_saves_outputs_inside_its_directories(mods, store, tmp_path):
    """An engine-chosen Content-Type must not choose where the clip is written."""
    run = tmp_path / "run"
    run.mkdir()
    with FakeClipServer(delivery="raw", ctype="video/../../../../escaped") as srv:
        rec = bench(srv, write_clips(tmp_path), label="lbl", run_dir=str(run), rounds=1)
    rows = rows_of(store, rec)
    assert all(r["ok"] and r["format"] == "bin" for r in rows)
    d = store.run_dir(rec["evidence_id"])
    index = json.loads((d / "outputs" / "lbl" / "index.json").read_text())["files"]
    assert sorted(index) == ["r0/i2v-0.bin", "r0/t2v-0.bin"]
    for root in (d, run):
        saved = [x for x in (root / "outputs").rglob("*") if x.is_file()]
        assert saved and all(x.resolve().is_relative_to((root / "outputs" / "lbl").resolve()) for x in saved)
        assert not (root / "outputs" / "escaped").exists()
    for rel in ("outputs/lbl/r0/a.../../../../escaped", "/abs/x.mp4"):
        with pytest.raises(OpError, match="refusing to write"):
            mods.bc._place(b"x", rel, [tmp_path / "root"])
    assert not (tmp_path / "root").exists()


def test_bench_clip_refuses_bad_arguments_and_a_dead_engine(mods, store, tmp_path):
    clips = write_clips(tmp_path)
    with FakeClipServer() as srv:
        with pytest.raises(OpError, match="needs --seeds"):
            bench(srv, clips, seed_policy="list")
        with pytest.raises(OpError, match="needs seed index 2"):
            bench(srv, clips, seed_policy="list", seeds=[1, 2], rounds=3)
        with pytest.raises(OpError, match="is for --seed-policy list"):
            bench(srv, clips, seeds=[1])
        with pytest.raises(OpError, match="needs --workflow"):
            bench(srv, clips, endpoint_kind="comfyui")
        rec = bench(srv, clips, seed_policy="list", seeds=[5, 9])
        assert [r["seed"] for r in rows_of(store, rec)] == [5, 5, 9, 9]
        assert rec["identity"]["workload"]["seed_policy"]["base"] is None
        rec = bench(srv, clips, seed_policy="fixed", seed_base=3)
        assert any("--seed-policy fixed" in x for x in rec["limitations"])
        port = srv.httpd.server_address[1]
    with pytest.raises(OpError, match="engine not ready"):
        registry.call("bench_clip", {"base_url": f"http://127.0.0.1:{port}", "clips": str(clips)})


def test_extend_alternates_engines_and_keeps_round_seeds(mods, store, tmp_path):
    clips = write_clips(tmp_path)
    with FakeClipServer(delay_ms=30) as slow, FakeClipServer(delay_ms=5) as fast:
        a = b = None
        for _ in range(3):  # A B A B A B: interleaved in time
            a = bench(slow, clips, rounds=1, label="slow", **({"extend": a["evidence_id"]} if a else {}))
            b = bench(fast, clips, rounds=1, label="fast", **({"extend": b["evidence_id"]} if b else {}))
        with pytest.raises(OpError, match="these differ"):
            bench(slow, clips, rounds=1, label="slow", seed_base=9, extend=a["evidence_id"])
        with pytest.raises(OpError, match="already has rounds up to 2"):
            bench(slow, clips, rounds=1, label="slow", first_round=1, extend=a["evidence_id"])
        gap = bench(fast, clips, rounds=1, label="fast", first_round=5, extend=b["evidence_id"])
    assert [r["index"] for r in a["data"]["rounds"]] == [0, 1, 2] and len(rows_of(store, a)) == 6
    assert [r["seed"] for r in a["data"]["rounds"]] == [r["seed"] for r in b["data"]["rounds"]]
    assert a["data"]["extends"] and len(a["data"]["outputs_from"]) == 2
    assert [r["index"] for r in gap["data"]["rounds"]] == [0, 1, 2, 5]
    assert gap["data"]["rounds"][-1]["seed"] == mods.bc.round_seed("per_round", 0, None, 5)

    # compare: same requests, interleaved rounds, the fast engine is faster
    cmp = registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"],
                                    "metric": "clip_ms", "resamples": 500})
    assert cmp["data"]["interleaved"] is True and cmp["data"]["verdict"] == "improved", cmp["summary"]
    assert cmp["metrics"][0]["value"] < 0.6
    rate = registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"],
                                     "metric": "clips_per_hour", "resamples": 500})
    assert rate["data"]["verdict"] == "improved" and rate["data"]["higher_is_better"] is True
    assert rate["metrics"][1]["unit"] == "clips/h"
    tok = registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"]})
    assert tok["data"]["verdict"] == "inconclusive"
    assert "not measured by bench clip" in tok["data"]["decision"]["reason"]


def test_compare_sees_a_change_to_one_task_of_a_mixed_set(mods, store, tmp_path):
    """One t2v and one i2v clip per round; only i2v gets faster. A median over the round's two clips is the
    t2v clip in both records (within noise); the round mean and the per-task metric both show the change."""
    clips = write_clips(tmp_path)
    with FakeClipServer(delay_ms=20, i2v_delay_ms=150) as old, FakeClipServer(delay_ms=20, i2v_delay_ms=50) as new:
        a = b = None
        for _ in range(3):  # A B A B A B
            a = bench(old, clips, rounds=1, label="old", **({"extend": a["evidence_id"]} if a else {}))
            b = bench(new, clips, rounds=1, label="new", **({"extend": b["evidence_id"]} if b else {}))
    assert a["identity"]["workload"].get("task") is None  # mixed set

    def cmp(metric):
        return registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"],
                                         "metric": metric, "resamples": 500})

    whole, i2v, rate = cmp("clip_ms"), cmp("clip_ms_i2v"), cmp("clips_per_hour_i2v")
    assert whole["data"]["verdict"] == "improved" and whole["metrics"][0]["value"] < 0.75, whole["summary"]
    assert any("mixes tasks" in x and "clip_ms_i2v" in x for x in whole["limitations"])
    assert i2v["data"]["verdict"] == "improved" and i2v["metrics"][0]["value"] < 0.5, i2v["summary"]
    assert not any("mixes tasks" in x for x in i2v["limitations"])
    assert rate["data"]["verdict"] == "improved" and rate["data"]["higher_is_better"] is True
    assert rate["metrics"][1]["unit"] == "clips/h"
    t2v = cmp("clip_ms_t2v")  # unchanged task: no verdict asserted (two 20 ms sleeps, scheduler noise decides)
    assert t2v["data"]["pairs"] == 3 and t2v["metrics"][0]["value"] is not None


def test_compare_refuses_clip_records_that_differ_or_cheat(mods, store, tmp_path):
    clips = write_clips(tmp_path)
    with FakeClipServer() as honest, FakeClipServer(mode="fewer_steps") as cheat:
        base = bench(honest, clips, label="ref")
        bad = bench(cheat, clips, label="cand")
        other_seeds = bench(honest, clips, label="cand2", seed_base=1)
        other_set = bench(honest, write_clips(tmp_path, [{"id": "t2v-1", "task": "t2v", "prompt": "more", "width": 64,
                                                          "height": 48, "num_frames": 9, "fps": 24, "steps": 4,
                                                          "guidance_scale": 3.0}], name="other"), label="cand3")

    def reason(cand):
        out = registry.call("compare", {"baseline": base["evidence_id"], "candidate": cand["evidence_id"],
                                        "metric": "clip_ms", "resamples": 200})
        assert out["data"]["verdict"] == "inconclusive"
        return out["data"]["decision"]["reason"]

    assert "claim invalid" in reason(bad) and "steps_run" in reason(bad)
    r = reason(other_seeds)
    assert "different seed policies" in r and "paired round(s) used different seeds" in r
    assert "different clip sets" in reason(other_set)


def test_comfyui_backends_plug_in_through_register_backend(mods, store, tmp_path, monkeypatch):
    calls = {}

    class FakeComfy:
        endpoint_kind = "comfyui"

        def __init__(self, ep, timeout_s, workflow):
            calls["workflow"] = workflow
            self.inner = cc.ClipHTTP(ep, timeout_s)

        def health(self):
            return self.inner.health()

        def generate(self, request):
            return self.inner.generate(request)

        def describe(self):
            return {"endpoint_kind": "comfyui", "nodes": "mapped"}

        def close(self):
            calls["closed"] = True

    monkeypatch.setitem(cc._BACKENDS, "comfyui", FakeComfy)
    wf = tmp_path / "wf.json"
    wf.write_text('{"3": {"class_type": "KSampler"}}')
    with FakeClipServer() as srv:
        rec = bench(srv, write_clips(tmp_path), endpoint_kind="comfyui", workflow=str(wf))
    assert calls == {"workflow": str(wf), "closed": True}
    assert rec["identity"]["workload"]["endpoint_kind"] == "comfyui"
    assert rec["data"]["backend"]["workflow_sha256"] == hashlib.sha256(wf.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- clip rounds


@pytest.mark.parametrize("cleanup", [
    {"stop": {"group_gone": False}, "status": "startup_timeout"},
    {"stop": {"group_gone": True}, "gpu_memory": {"checked": True, "returned": False}, "status": "ok"},
])
def test_cleanup_failure_skips_every_remaining_clip_cell(mods, tmp_path, monkeypatch, cleanup):
    clips = write_clips(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"engines": [{"label": label, "name": "custom", "port": _free_port(),
                                                 "launch": ["unused"]} for label in ("a", "b")]}))
    launched = []

    def fake_cell(spec, k, position, cfg):
        launched.append(spec["label"])
        return {"round": k, "position": position, "label": spec["label"], "kind": "generic", **cleanup}

    monkeypatch.setattr(mods.cr, "run_clip_cell", fake_cell)
    result = mods.cr.run_clip_rounds(str(manifest), str(clips), rounds=4, gpu_lock_path=str(tmp_path / "lock"))
    assert launched == ["a"] and len(result["cells"]) == 8
    assert all(c["status"] == "skipped_after_cleanup_failure" for c in result["cells"][1:])
    assert result["plan"]["aborted_after_cleanup_failure"]["label"] == "a"
    assert result["cells"][0]["status"] != "ok"
    assert "Incomplete rounds" in result["markdown"]

def test_clip_argv_default_and_template(mods):
    ctx = {"base_url": "http://127.0.0.1:9", "clips": "/c.jsonl", "label": "a", "round": 2, "first_round": 1,
           "run_dir": "/r", "endpoint_kind": "clip_http", "workflow": "", "engine_version": "1.0", "extend": "ev_x",
           "warmup": 1, "model": ""}
    argv = mods.cr.clip_argv(None, ctx, "--seed-base 7")
    assert argv[1:5] == ["-m", "inferencekit", "bench", "clip"]
    joined = " ".join(argv)
    assert "--first-round 1" in joined and "--extend ev_x" in joined and "--engine-version 1.0" in joined
    assert argv[-2:] == ["--seed-base", "7"] and "--workflow" not in argv and "--model" not in argv
    assert mods.cr.clip_argv("run {label} {first_round}", ctx, None) == ["run", "a", "1"]
    with pytest.raises(OpError, match="placeholder"):
        mods.cr.clip_argv("run {nope}", ctx, None)


def test_clip_summary_ranks_clips_per_hour_as_a_rate(mods):
    cells = []
    for k in (1, 2, 3):
        for label, ms in (("a", 100.0 + k), ("b", 50.0 + k)):
            rows = [{"round": k - 1, "task": "t2v", "ok": True, "e2e_ms": ms}]
            cells.append({"round": k, "label": label, "status": "ok", "metrics": mods.cr.round_metrics(rows, k - 1)})
    s = mods.cr.summarize(cells, ["a", "b"], resamples=200)
    rate = {r["engine"]: r for r in s["rows"] if r["metric"] == "clips_per_hour" and r["bucket"] == "all"}
    clip = {r["engine"]: r for r in s["rows"] if r["metric"] == "clip_ms" and r["bucket"] == "all"}
    assert rate["a"]["best"] == "b" and rate["a"]["higher_is_better"] is True
    assert rate["a"]["ratio_to_best"] == pytest.approx(rate["a"]["median"] / rate["b"]["median"])
    assert clip["a"]["best"] == "b" and clip["a"]["ratio_to_best"] > 1.9
    assert {r["bucket"] for r in s["rows"]} == {"t2v", "all"}


def test_clip_metrics_median_over_an_even_number_of_rounds_is_the_usual_median(mods):
    """With 2 rounds the nearest-rank "median" was the faster round for clip_ms and the
    slower one for clips_per_hour (the 2-round reference record quoted 111.50 s per clip and 31.97 clips/h, i.e.
    112.61 s). Both are now the usual median, as in bench rounds' summary: the mean of the two middle rounds."""
    rows = [{"round": k, "task": "t2v", "ok": True, "e2e_ms": ms, "stage_ms": {}}
            for k, ms in ((0, 100_000.0), (0, 100_000.0), (1, 110_000.0), (1, 110_000.0))]
    metrics, table = mods.bc.clip_metrics(rows, ["t2v"], 1, 200)
    t = table["t2v"]
    assert t["rounds"] == 2 and t["clip_ms"]["median"] == pytest.approx(105_000.0)
    assert t["clips_per_hour"]["median"] == pytest.approx((36.0 + 3600 / 110) / 2)
    odd = rows + [{"round": 2, "task": "t2v", "ok": True, "e2e_ms": 130_000.0, "stage_ms": {}}]
    _, table = mods.bc.clip_metrics(odd, ["t2v"], 1, 200)
    assert table["t2v"]["clip_ms"]["median"] == pytest.approx(110_000.0)  # odd: the middle round, as before
    assert table["t2v"]["clips_per_hour"]["median"] == pytest.approx(3600 / 110)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_clip_rounds_interleave_engines_with_shared_seeds(mods, store, tmp_path, monkeypatch):
    # The caller's terse view must not reach the bench clip children, whose stdout is parsed as a record.
    monkeypatch.setenv("INFERENCEKIT_OUTPUT", "terse")
    monkeypatch.setattr(mods.cr, "gpu_processes", lambda: [])
    monkeypatch.setattr(mods.cr, "gpu_identity", lambda: {})
    clips = write_clips(tmp_path)
    run = tmp_path / "runs" / "clip-run"
    run.mkdir(parents=True)
    engines = []
    for label, delay, mode in (("slow", 25, "honest"), ("fast", 5, "neg_cache")):
        port = _free_port()
        engines.append({"name": "custom", "label": label, "version": "0.0.1", "install": [], "env": {}, "port": port,
                        "launch": [sys.executable, THIS, "--serve", "--port", str(port), "--delay-ms", str(delay),
                                   "--label", label, "--mode", mode], "role": "custom"})
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "model": {"id": "org/fake-video"}, "engines": engines}))
    rec = mods.cr.bench_rounds_clip(_call(), manifest=str(manifest), clips=str(clips), rounds=2,
                                    gpu_lock=str(tmp_path / "gpu.lock"), startup_timeout_s=30, stop_timeout_s=5,
                                    clip_cmd=CLIP_CMD, clip_args="--gpu-name 'Test GPU' --resamples 200",
                                    gpu_reader=lambda: None)
    assert rec["kind"] == "clip_benchmark" and ev.validate(rec) == [] and rec["data"]["source"] == "bench rounds clip"
    plan, cells = rec["data"]["plan"], rec["data"]["cells"]
    assert [c["status"] for c in cells] == ["ok"] * 4, [(c["status"], c.get("detail"), c.get("clip")) for c in cells]
    assert [[c["label"] for c in cells if c["round"] == k] for k in (1, 2)] == [["slow", "fast"], ["fast", "slow"]]
    assert all(c["stop"]["group_gone"] for c in cells)
    rows = {(r["engine"], r["metric"], r["bucket"]): r for r in rec["data"]["rows"]}
    assert rows[("slow", "clip_ms", "all")]["best"] == "fast"
    assert rows[("slow", "clips_per_hour", "all")]["best"] == "fast"
    assert rows[("slow", "clip_ms", "all")]["ratio_to_best"] > 1.5
    # only fast reused its fixed negative prompt's encoding: the ranking record says so
    assert [c["cache_allowed_clips"] for c in cells if c["label"] == "fast"] == [2, 4]
    assert all(c["cache_allowed_clips"] == 0 for c in cells if c["label"] == "slow")
    assert any(x.startswith("fast reused the encoding of a fixed negative prompt") and "slow encoded it on every clip"
               in x for x in rec["limitations"]), rec["limitations"]
    final = plan["clip_evidence_final"]
    a, b = (store.load(final[x]) for x in ("slow", "fast"))
    assert [r["index"] for r in a["data"]["rounds"]] == [0, 1] == [r["index"] for r in b["data"]["rounds"]]
    assert [r["seed"] for r in a["data"]["rounds"]] == [r["seed"] for r in b["data"]["rounds"]]
    assert (run / "clip_rounds" / "summary.md").read_text().startswith("# bench rounds: clip mode")
    cmp = registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"], "metric": "clip_ms",
                                    "resamples": 200})
    assert cmp["data"]["interleaved"] is True and cmp["data"]["pairs"] == 2


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process-group RSS")
def test_clip_cpu_rounds_capture_process_memory_and_cpu_identity(mods, store, tmp_path, monkeypatch):
    monkeypatch.setattr(mods.cr, "gpu_processes", lambda: pytest.fail("CPU scheduler probed GPU processes"))
    monkeypatch.setattr(mods.cr, "gpu_identity", lambda: pytest.fail("CPU scheduler probed GPU identity"))
    clips = write_clips(tmp_path)
    run = tmp_path / "cpu-clips"
    run.mkdir()
    port = _free_port()
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "model": {"id": "org/fake-video"},
        "hardware": {"backend": "cpu"}, "engines": [{"name": "custom", "label": "cpu-video", "version": "0.0.1",
        "env": {"OMP_NUM_THREADS": "2"}, "port": port, "launch": [sys.executable, THIS, "--serve", "--port",
        str(port), "--label", "cpu-video"], "role": "custom"}]}))
    # The child fails if its evidence builder attempts GPU discovery. Its timer still
    # exercises the normal fake clip endpoint and the real bench-clip operation.
    command = (f"{sys.executable} -c 'import inferencekit.ops.bench_clip as b; "
               "b.gpu_identity=lambda: (_ for _ in ()).throw(AssertionError(\"GPU probe\")); "
               "from inferencekit.cli import main; raise SystemExit(main())' bench clip")
    rec = mods.cr.bench_rounds_clip(_call(), manifest=str(manifest), clips=str(clips), rounds=1,
        resource_mode="cpu", cpu_lock_path=str(tmp_path / "cpu.lock"), startup_timeout_s=30, stop_timeout_s=3,
        clip_cmd=command, clip_args="--resamples 100", gpu_reader=lambda: pytest.fail("CPU GPU-memory read"))
    assert rec["identity"]["cpu"]["architecture"] and rec["identity"]["gpu"]["name"] is None
    cell = rec["data"]["cells"][0]
    assert cell["status"] == "ok", cell
    assert cell["resources"]["peak_rss_mib"] > 0 and cell["stop"]["group_gone"]
    final = store.load(rec["data"]["plan"]["clip_evidence_final"]["cpu-video"])
    assert final["data"]["rounds"][0]["peak_rss_mib"] > 0
    assert final["identity"]["cpu"] == rec["identity"]["cpu"]
    assert not any("gpu identity" in item for item in final["coverage"]["missing"])
    assert final["identity"]["engine"]["flags"]["server_env"]["controls"]["OMP_NUM_THREADS"] == "2"


def test_bench_rounds_mode_clip_runs_clip_rounds_with_the_default_bench_clip(mods, store, tmp_path, monkeypatch):
    """bench rounds --mode clip hands the run to serve/clip_rounds.py, and each cell runs the registered
    'inferencekit bench clip' (no command template)."""
    monkeypatch.setattr(mods.cr, "gpu_processes", lambda: [])
    monkeypatch.setattr(mods.cr, "gpu_identity", lambda: {})
    monkeypatch.setattr(mods.cr.servers, "gpu_memory_used_mib", lambda: None)
    monkeypatch.delenv(mods.cr.CLIP_CMD_ENV, raising=False)
    clips = write_clips(tmp_path)
    run = tmp_path / "runs" / "clip-run"
    run.mkdir(parents=True)
    port = _free_port()
    engine = {"name": "custom", "label": "only", "version": "0.0.1", "install": [], "env": {}, "port": port,
              "launch": [sys.executable, THIS, "--serve", "--port", str(port), "--label", "only"], "role": "custom"}
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "model": {"id": "org/fake-video"}, "engines": [engine]}))
    args = {"manifest": str(manifest), "prompts": str(clips), "mode": "clip", "rounds": 1, "warmup": 1,
            "gpu_lock": str(tmp_path / "gpu.lock"), "startup_timeout_s": 30, "stop_timeout_s": 5,
            "serve_args": "--gpu-name 'Test GPU' --resamples 200"}
    rec = registry.call("bench_rounds", args)
    assert rec["kind"] == "clip_benchmark" and rec["data"]["source"] == "bench rounds clip"
    cell = rec["data"]["cells"][0]
    assert cell["status"] == "ok", (cell.get("detail"), cell.get("clip"))
    assert rec["data"]["plan"]["settings"]["clip_cmd_custom"] is False
    final = store.load(rec["data"]["plan"]["clip_evidence_final"]["only"])
    assert final["data"]["source"] == "bench clip" and final["data"]["claim_validity"]["valid"] is True
    for extra, flag in (({"ci_target": 0.01}, "--ci-target"), ({"extra_body": '{"temperature": 1}'}, "--extra-body"),
                        ({"ref_e2e_ms": 100.0}, "--ref-e2e-ms")):
        with pytest.raises(OpError, match=f"{flag}.*not supported with --mode clip"):
            registry.call("bench_rounds", {**args, **extra})
