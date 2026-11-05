"""A clip HTTP server around a video pipeline: POST /v1/clips and GET /health. Standard library only.

Run it with the pipeline's own interpreter (inferencekit installed there with ``pip install --no-deps``,
or on PYTHONPATH), so inferencekit never shares a virtual environment with the pipeline::

    python -m inferencekit.engines.clip.server --pipeline diffusers_ltx --model DIR --port 8000 \\
        [--task t2v|i2v] [--dtype bf16] [--offload none|model|sequential] [--attention native] \\
        [--out-dir DIR] [--answer path|b64] [--pipeline-arg KEY=VALUE ...]

The contract (the client side is inferencekit.serve.clip_client)::

  GET  /health    200 {"status": "ok"} when a clip can start; 503 while the pipeline loads.
  POST /v1/clips  {task, prompt, image?, width, height, num_frames, fps, steps, guidance_scale, seed,
                   noise_seed?, audio?}; any other field is refused (400), so a client never believes a
                   setting was applied when it was not.
     -> {video_path | video_b64, format, video_sha256, video_bytes, stage_ms, effective, cache}; the file is
        complete and closed before the answer is sent (the client checks its size and sha256)

What the server does, whatever the pipeline:
  * Initial noise: drawn on the CPU from ``noise_seed`` (``seed`` when absent) with noise.initial_noise
    and handed to the pipeline, so every engine that follows the recipe starts from the same tensors on
    any GPU. ``effective.noise_sha256`` fingerprints them.
  * ``effective.steps_run`` is counted by the pipeline's step callback (StageClock.step), never copied
    from the request, and is null when nothing counted the steps (bench clip then says they were not
    checked, instead of seeing 0); width, height, frames and fps come from the decoded output the
    pipeline returns.
  * ``stage_ms``: disjoint stages timed by hooks on the pipeline's parts (noise, text_encode, denoise,
    vae_decode, audio_decode, mux, ...), each after a device sync when --stage-sync is on (the default),
    plus ``denoise_steps`` (one entry per step; a list, so the client does not add it to the sum). These
    are self-reported: the client's clock is the only published timer.
  * No output cache: every request runs the pipeline, and ``cache`` is always [].
  * Per-step dumps (stepdump.py, the contract quality clip-steps reads): a pipeline that returns a
    ``Generated.step_dump`` (a StepRecorder) gets it written to ``<dump dir>/<video_sha256>.safetensors``
    before the answer (stage ``step_dump``). The dump dir is ``--step-dump-dir``, else $CLIP_STEP_DUMP_DIR,
    else ``<out-dir>/step-dumps``, where only the last ``--keep`` dumps are kept like the clips; a dump dir
    that was given is never pruned (the checker picks the requests it checks after the clips are made).
  * One clip at a time (a lock); /health still answers while a clip runs.

A pipeline is any object with ``describe()``, ``noise_shapes(job)``, ``generate(job, noise, clock)`` and
``write(output, path, clock)`` (see ClipPipeline); it may bring its own ``make_noise(noise_seed, shapes)``
-> (tensors, sha256), and a pipeline that draws its own noise returns no shapes (noise_sha256 is then
null). ``--pipeline`` names a module in this package (diffusers_ltx, official_ltx) or
``MODULE:FUNCTION``; the factory gets a PipelineOptions.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import functools
import hashlib
import importlib
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

TASKS = ("t2v", "i2v")
CLIPS_PATH = "/v1/clips"
HEALTH_PATH = "/health"
MAX_BODY = 64 << 20  # an i2v image in base64 fits easily
FIELDS = {"task", "prompt", "image", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "seed",
          "noise_seed", "audio"}
REQUIRED = ("task", "prompt", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "seed")
PIPELINES = {"diffusers_ltx": "inferencekit.engines.clip.diffusers_ltx",
             "official_ltx": "inferencekit.engines.clip.official_ltx"}


class BadRequest(ValueError):
    """The request is malformed or asks for something this pipeline cannot do (HTTP 400)."""


@dataclass(frozen=True)
class ClipJob:
    task: str
    prompt: str
    width: int
    height: int
    num_frames: int
    fps: float
    steps: int
    guidance_scale: float
    seed: int
    noise_seed: int  # the request's noise_seed, else its seed
    image: bytes | None = field(default=None, repr=False)
    audio: bool | None = None


def _int(doc: dict, key: str, lo: int, hi: int) -> int:
    v = doc[key]
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise BadRequest(f"{key} must be an integer in [{lo}, {hi}], got {v!r}")
    return v


def _num(doc: dict, key: str, lo: float, hi: float) -> float:
    v = doc[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise BadRequest(f"{key} must be a number in [{lo}, {hi}], got {v!r}")
    return float(v)


def parse_job(doc) -> ClipJob:
    """The request body as a ClipJob; BadRequest for anything missing, unknown or out of range."""
    if not isinstance(doc, dict):
        raise BadRequest("the body must be a JSON object")
    unknown = sorted(set(doc) - FIELDS)
    if unknown:
        raise BadRequest(f"unknown field(s) {unknown}; this server takes {sorted(FIELDS)}")
    missing = [k for k in REQUIRED if doc.get(k) is None]
    if missing:
        raise BadRequest(f"missing field(s) {missing}")
    task = doc["task"]
    if task not in TASKS:
        raise BadRequest(f"task must be one of {TASKS}, got {task!r}")
    if not isinstance(doc["prompt"], str) or not doc["prompt"].strip():
        raise BadRequest("prompt must be a non-empty string")
    image = None
    if doc.get("image") is not None:
        if task != "i2v":
            raise BadRequest("image is for task i2v only")
        try:
            image = base64.b64decode(doc["image"], validate=True)
        except (TypeError, ValueError, binascii.Error):
            raise BadRequest("image must be base64 of the image file's bytes") from None
    if task == "i2v" and not image:
        raise BadRequest("task i2v needs an image")
    audio = doc.get("audio")
    if audio is not None and not isinstance(audio, bool):
        raise BadRequest(f"audio must be true or false, got {audio!r}")
    seed = _int(doc, "seed", 0, 2**63 - 1)
    noise_seed = _int(doc, "noise_seed", 0, 2**63 - 1) if doc.get("noise_seed") is not None else seed
    return ClipJob(task=task, prompt=doc["prompt"], width=_int(doc, "width", 1, 16384),
                   height=_int(doc, "height", 1, 16384), num_frames=_int(doc, "num_frames", 1, 100000),
                   fps=_num(doc, "fps", 1e-3, 1000.0), steps=_int(doc, "steps", 1, 10000),
                   guidance_scale=_num(doc, "guidance_scale", 0.0, 1000.0), seed=seed, noise_seed=noise_seed,
                   image=image, audio=audio)


# --------------------------------------------------------------------------- stage timing


class StageClock:
    """Stage times of one clip (ms, ``time.perf_counter_ns``), each taken after ``sync()``.

    ``span(name)`` times a disjoint stage; a span opened inside another is folded into the outer one, so a
    hooked method that calls another hooked method is never counted twice. The denoise loop is timed by
    marks: ``denoise_start()`` (the first transformer call) and ``step()`` (the end of each step, from the
    pipeline's step callback). ``steps`` counts the step() calls: the steps the pipeline really ran.
    ``counted`` says whether anything counted them (step() was called, or the pipeline used set_steps);
    when it is False the server reports steps_run null, not 0.
    """

    def __init__(self, sync: Callable[[], None] | None = None):
        self.sync = sync or (lambda: None)
        self.stage_ms: dict[str, float] = {}
        self.step_ms: list[float] = []
        self.steps = 0
        self.counted = False
        self._depth = 0
        self._t0 = 0
        self._t_denoise0: int | None = None
        self._t_last: int | None = None  # last step mark, or the end of the last span

    def begin(self) -> None:
        """Open a stage (``end`` closes it); for module pre/post hooks, where a with block does not fit."""
        self._depth += 1
        if self._depth == 1:
            self.sync()
            self._t0 = time.perf_counter_ns()

    def end(self, name: str) -> None:
        self._depth -= 1
        if self._depth == 0:
            self.sync()
            t1 = time.perf_counter_ns()
            self.stage_ms[name] = self.stage_ms.get(name, 0.0) + (t1 - self._t0) / 1e6
            self._t_last = t1

    @contextmanager
    def span(self, name: str):
        self.begin()
        try:
            yield
        finally:
            self.end(name)

    def denoise_start(self) -> None:
        if self._t_denoise0 is None and not self._depth:
            self.sync()
            self._t_denoise0 = self._t_last = time.perf_counter_ns()

    def step(self) -> None:
        self.sync()
        t = time.perf_counter_ns()
        if self._t_denoise0 is None:  # no transformer hook: the loop started when the last span ended
            self._t_denoise0 = self._t_last if self._t_last is not None else t
            self._t_last = self._t_denoise0
        self.step_ms.append((t - self._t_last) / 1e6)
        self._t_last = t
        self.steps += 1
        self.counted = True

    def set_steps(self, n: int) -> None:
        """The steps run, for a pipeline that counts them another way (e.g. denoiser forward calls)."""
        self.steps, self.counted = int(n), True

    def result(self) -> dict:
        out = {k: round(v, 3) for k, v in self.stage_ms.items()}
        if self.step_ms:
            out["denoise"] = round(sum(self.step_ms), 3)
            out["denoise_steps"] = [round(v, 3) for v in self.step_ms]
        return out


def timed(fn: Callable, stage: str, current: Callable[[], StageClock | None]) -> Callable:
    """``fn`` timed as ``stage`` on the clock ``current()`` returns (called through when there is none)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        clock = current()
        if clock is None:
            return fn(*args, **kwargs)
        with clock.span(stage):
            return fn(*args, **kwargs)

    return wrapper


def hook_method(obj, attr: str, stage: str, current: Callable[[], StageClock | None]) -> None:
    """Replace ``obj.attr`` on this instance by a timed wrapper (the class is left alone)."""
    setattr(obj, attr, timed(getattr(obj, attr), stage, current))


# --------------------------------------------------------------------------- pipelines


@dataclass
class Generated:
    """What a pipeline produced, before muxing. width/height/frames/fps describe the decoded frames."""

    video: Any
    width: int
    height: int
    frames: int
    fps: float
    audio: Any = None
    audio_sample_rate: int | None = None
    effective: dict = field(default_factory=dict)  # extra facts for ``effective`` (e.g. noise checks)
    step_dump: Any = None  # a stepdump.StepRecorder: written to the dump dir, named by the clip's sha256


class ClipPipeline(Protocol):
    def describe(self) -> dict:
        """Static facts for the resolved-config line and ``effective``: pipeline, model, dtype, attention..."""

    def noise_shapes(self, job: ClipJob) -> dict:
        """{name: shape} of the initial noise for ``job``, in draw order. BadRequest if the shape is invalid."""

    def generate(self, job: ClipJob, noise: dict, clock: StageClock) -> Generated:
        """Run the pipeline; call clock.step() once per denoise step (or clock.set_steps(n) once) and time
        stages on ``clock``. A pipeline that does neither gets steps_run null: not counted."""

    def write(self, out: Generated, path: Path, clock: StageClock) -> None:
        """Encode ``out`` into the container file at ``path`` (timed by the server as stage ``mux``)."""


@dataclass
class PipelineOptions:
    model: str
    task: str = "t2v"  # the pipeline variant to load; a request for the other task is refused
    device: str = "cuda"
    dtype: str = "bf16"
    offload: str = "none"  # none | model | sequential
    attention: str | None = None  # Diffusers attention backend name; None keeps the pipeline's default
    extra: dict = field(default_factory=dict)  # --pipeline-arg KEY=VALUE


def load_factory(name: str) -> Callable[[PipelineOptions], ClipPipeline]:
    mod_name, _, fn_name = PIPELINES.get(name, name).partition(":")
    return getattr(importlib.import_module(mod_name), fn_name or "make_pipeline")


def cuda_sync(device: str) -> Callable[[], None] | None:
    """torch.cuda.synchronize of ``device`` (imported lazily); None for a device that is not CUDA.

    The device is passed on: with no argument torch syncs the current device (cuda:0 unless set), so a
    pipeline on cuda:1 would be timed by kernel launches, not by the GPU finishing them.
    """
    if not str(device).startswith("cuda"):
        return None
    import torch

    return functools.partial(torch.cuda.synchronize, torch.device(device))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_noise(noise_seed: int, shapes: dict) -> tuple[dict, str]:
    from inferencekit.engines.clip import noise

    tensors = noise.initial_noise(noise_seed, shapes)
    return tensors, noise.digest(tensors)


class ClipServer:
    """Binds first (so a port clash fails fast), then loads the pipeline in a thread; /health is 503 until then."""

    def __init__(self, load: Callable[[], ClipPipeline], *, host: str = "127.0.0.1", port: int = 8000,
                 out_dir: str | Path = "clips", answer: str = "path", keep: int = 16, stage_sync: bool = True,
                 sync: Callable[[], None] | None = None,
                 noise_fn: Callable[[int, dict], tuple[dict, str]] | None = None, log=sys.stderr,
                 step_dump_dir: str | Path | None = None):
        if answer not in ("path", "b64"):
            raise ValueError("answer must be path or b64")
        self._load, self.answer, self.keep, self.noise_fn, self.log = load, answer, keep, noise_fn, log
        self.sync = sync if stage_sync else None
        self.stage_sync = stage_sync
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        from inferencekit.engines.clip import stepdump

        given = step_dump_dir or os.environ.get(stepdump.ENV) or None
        self.step_dump_dir = Path(given or self.out_dir / "step-dumps").resolve()
        self._prune_dumps = not given
        self._dumps: list[Path] = []
        self.pipeline: ClipPipeline | None = None
        self.load_error: str | None = None
        self.facts: dict = {}
        self._lock = threading.Lock()
        self._n = 0
        self._written: list[Path] = []
        self.httpd = ThreadingHTTPServer((host, port), _handler(self))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]

    def start_loading(self) -> threading.Thread:
        t = threading.Thread(target=self._load_pipeline, name="clip-pipeline-load", daemon=True)
        t.start()
        return t

    def _load_pipeline(self) -> None:
        try:
            pipe = self._load()
            self.facts = dict(pipe.describe() or {})
            self.facts["stage_sync"] = self.stage_sync
            print(json.dumps({"inferencekit_resolved_config": self.facts}, sort_keys=True), file=self.log, flush=True)
            self.pipeline = pipe
        except Exception as e:  # noqa: BLE001 - reported, then the process stops
            self.load_error = f"{type(e).__name__}: {e}"
            traceback.print_exc(file=self.log)
            print(f"clip server: pipeline load failed: {self.load_error}", file=self.log, flush=True)
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()

    def health(self) -> tuple[int, dict]:
        if self.pipeline is not None:
            return 200, {"status": "ok", "pipeline": self.facts.get("pipeline"), "busy": self._lock.locked()}
        if self.load_error:
            return 500, {"status": "failed", "error": self.load_error}
        return 503, {"status": "loading"}

    def clip(self, doc) -> tuple[int, dict]:
        if self.pipeline is None:
            return self.health()
        try:
            job = parse_job(doc)
        except BadRequest as e:
            return 400, {"error": str(e)}
        with self._lock:
            try:
                return 200, self._run(job)
            except BadRequest as e:
                return 400, {"error": str(e)}
            except Exception as e:  # noqa: BLE001 - one failed clip must not stop the server
                traceback.print_exc(file=self.log)
                return 500, {"error": f"{type(e).__name__}: {e}"[:2000]}

    def _run(self, job: ClipJob) -> dict:
        pipe = self.pipeline
        shapes = pipe.noise_shapes(job)
        clock = StageClock(self.sync)
        noise, noise_sha = {}, None  # a pipeline that draws its own noise asks for none
        if shapes:
            make = self.noise_fn or getattr(pipe, "make_noise", None) or _default_noise
            with clock.span("noise"):
                noise, noise_sha = make(job.noise_seed, shapes)
        out = pipe.generate(job, noise, clock)
        self._n += 1
        path = self.out_dir / f"clip-{time.strftime('%Y%m%d-%H%M%S')}-{self._n:05d}.mp4"
        with clock.span("mux"):
            pipe.write(out, path, clock)
        sha, size = file_sha256(path), path.stat().st_size
        self._remember(path)
        dump_bytes = None
        if getattr(out, "step_dump", None) is not None:
            from inferencekit.engines.clip import stepdump

            dump = stepdump.dump_path(self.step_dump_dir, sha)
            with clock.span("step_dump"):
                dump_bytes = out.step_dump.save(dump)
            if self._prune_dumps:
                self._dumps.append(dump)
                while self.keep and len(self._dumps) > self.keep:
                    self._dumps.pop(0).unlink(missing_ok=True)
        effective = {"steps_run": clock.steps if clock.counted else None, "width": out.width, "height": out.height,
                     "frames": out.frames, "fps": out.fps, "dtype": self.facts.get("dtype"),
                     "attention": self.facts.get("attention"), "audio": out.audio is not None, "noise_seed": job.noise_seed, "noise_sha256": noise_sha,
                     "step_dump_bytes": dump_bytes, **out.effective}
        answer = {"format": "mp4", "video_sha256": sha, "video_bytes": size, "stage_ms": clock.result(),
                  "effective": effective, "cache": []}
        if self.answer == "b64":
            answer["video_b64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        else:
            answer["video_path"] = str(path)
        return answer

    def _remember(self, path: Path) -> None:
        """Keep the last ``keep`` clips on disk (0 keeps all); the client copies what it needs."""
        self._written.append(path)
        while self.keep and len(self._written) > self.keep:
            old = self._written.pop(0)
            try:
                old.unlink()
            except OSError:
                pass

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.2)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _handler(server: ClipServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "inferencekit-clip/1"

        def log_message(self, fmt, *args):  # one short line per request on stderr
            print(f"clip server: {self.command} {self.path} {fmt % args}", file=server.log, flush=True)

        def _send(self, status: int, doc: dict) -> None:
            body = json.dumps(doc).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == HEALTH_PATH:
                self._send(*server.health())
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            if self.path.split("?")[0] != CLIPS_PATH:
                self._send(404, {"error": f"unknown path {self.path}"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 < n <= MAX_BODY:
                self._send(400 if n <= 0 else 413, {"error": f"Content-Length must be 1..{MAX_BODY}"})
                return
            try:
                doc = json.loads(self.rfile.read(n))
            except ValueError as e:
                self._send(400, {"error": f"the body is not JSON: {e}"})
                return
            self._send(*server.clip(doc))

    return Handler


def _pipeline_arg(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"--pipeline-arg must be KEY=VALUE, got {text!r}")
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m inferencekit.engines.clip.server", description=__doc__.split("\n")[0])
    p.add_argument("--pipeline", required=True, help="diffusers_ltx, official_ltx, or MODULE:FUNCTION")
    p.add_argument("--model", required=True, help="model directory or hub id")
    p.add_argument("--task", choices=TASKS, default="t2v", help="the pipeline variant to load")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", help="bf16 (the reference), fp16 or fp32")
    p.add_argument("--offload", choices=("none", "model", "sequential"), default="none")
    p.add_argument("--attention", default=None, help="attention backend (Diffusers names, e.g. native, flash)")
    p.add_argument("--out-dir", default="clips")
    p.add_argument("--answer", choices=("path", "b64"), default="path",
                   help="path: the client reads the file (same machine); b64: the clip in the JSON answer")
    p.add_argument("--keep", type=int, default=16, help="clips kept on disk (0 keeps all)")
    p.add_argument("--step-dump-dir", default=None,
                   help="per-step dumps (default $CLIP_STEP_DUMP_DIR, else <out-dir>/step-dumps, pruned like clips)")
    p.add_argument("--stage-sync", choices=("on", "off"), default="on",
                   help="sync the device around each timed stage (off: stage times measure launches only)")
    p.add_argument("--pipeline-arg", type=_pipeline_arg, action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args(argv)
    opts = PipelineOptions(model=a.model, task=a.task, device=a.device, dtype=a.dtype, offload=a.offload,
                           attention=a.attention, extra=dict(a.pipeline_arg))
    factory = load_factory(a.pipeline)
    stage_sync = a.stage_sync == "on"
    srv = ClipServer(lambda: factory(opts), host=a.host, port=a.port, out_dir=a.out_dir, answer=a.answer,
                     keep=a.keep, stage_sync=stage_sync, sync=cuda_sync(a.device) if stage_sync else None,
                     step_dump_dir=a.step_dump_dir)
    print(f"clip server: listening on {a.host}:{srv.port}, loading {a.pipeline}", file=sys.stderr, flush=True)
    srv.start_loading()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.httpd.server_close()
    return 1 if srv.load_error else 0


if __name__ == "__main__":
    sys.exit(main())
