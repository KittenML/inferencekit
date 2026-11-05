"""Clients for video clip engines: the clip HTTP contract, and the interface other backends implement.

The clip HTTP contract
----------------------
  GET  /health      200 when the engine can take a clip (the ready check of bench rounds' clip cells).
  POST /v1/clips    JSON body:
      task            "t2v" | "i2v"
      prompt          the text prompt
      image           i2v only: the conditioning image, base64 of the image file's bytes
      width, height, num_frames, fps, steps, guidance_scale, seed
      noise_seed      optional: seed of the initial noise, when it is given apart from ``seed``
      audio           optional: true/false, sent only when the clip set says so
  The answer is JSON:
      video_b64       the encoded clip, base64; or
      video_path      a file the server wrote on this machine (accepted only from a local server); the
                      answer must then also carry video_sha256 and video_bytes
      video_sha256    sha256 (hex) and size in bytes of the finished clip: required with video_path, checked
      video_bytes     whenever given (``sha256`` and ``bytes`` are read as the same two fields)
      format          container, e.g. "mp4" (default "mp4"); letters and digits only, else "bin"
      stage_ms        {stage: ms}, the engine's own timings (SELF-REPORTED): e.g. text_encode, denoise,
                      vae_decode, audio_decode, mux. Disjoint stages; a "total"/"e2e"/"wall" key is kept but
                      not summed. Lists (e.g. per-step times) are kept and not summed either.
      effective       what the engine says it ran (SELF-REPORTED): steps_run, width, height, frames, fps,
                      dtype, attention, audio
      cache           optional: names of stages the engine served from a cache (must be empty for a timed clip),
                      except "negative_text_encode": the encoding of the engine's fixed negative prompt (a
                      launch setting, the same text for every clip) computed once and reused. That one reuse
                      is allowed for every engine (ComfyUI makes it through its node cache) and stated in the
                      record's limitations; anything else listed makes the claim invalid.
  An answer may instead be the raw clip (Content-Type video/*), with the other fields as JSON in an
  ``X-Clip-Meta`` header. The subtype (video/mp4 -> "mp4") is a format like any other: it names the saved
  file, so anything but letters and digits becomes "bin".

The timer
---------
The client's clock is the only timer for published numbers. ``t_send`` is taken just before the
connection is opened, ``t_done`` once the client holds the whole clip: the response is read and its
base64 decoded, or the file at ``video_path`` is read. A server cannot answer before the clip is
complete and finish it later: the bytes of a raw or base64 answer are the response itself, and the file
behind ``video_path`` must match the video_sha256 and video_bytes the answer announced, so a server that
answers while it is still encoding, muxing or writing the file gets a failed clip, not a shorter time (a
server closes the file before it answers). That check, hashing and saving happen after ``t_done``, so
they are not timed. Every clock is ``time.perf_counter_ns()`` (monotonic).

Other backends
--------------
``ClipBackend`` is the interface bench clip drives: ``health()``, ``generate(request)``, ``describe()``,
``close()``. ``ClipHTTP`` implements the contract above (endpoint kind ``clip_http``). A ComfyUI
backend (endpoint kind ``comfyui``) lives in ``inferencekit.engines.clip.comfyui`` when installed; that
module calls ``register_backend("comfyui", factory)`` on import, and ``make_backend`` imports it on
first use. A ComfyUI backend must keep the same timer (``t_send`` before POST /prompt, ``t_done`` when
the output bytes are in hand from /view), put the node ids ComfyUI reports in ``execution_cached`` for
timed nodes into ``ClipResult.cache`` (a reused fixed negative-prompt encoding into ``cache_allowed``),
and fill ``effective`` from what it submitted and the progress it saw (steps_run = sampler progress
steps seen). Its ``workflow`` option is the API-format workflow
file with the node mapping; ``describe()`` returns that file's sha256.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from inferencekit.serve.client import Endpoint, scrub

TASKS = ("t2v", "i2v")
ENDPOINT_KINDS = ("clip_http", "comfyui")  # what bench clip can drive; in_process is for ops that run the pipeline
CLIPS_PATH = "/v1/clips"
HEALTH_PATH = "/health"
DEFAULT_TIMEOUT_S = 1800.0
MAX_ERROR_BODY = 2000
TOTAL_KEYS = ("total", "total_ms", "e2e", "e2e_ms", "wall", "wall_ms")  # in stage_ms, but not stages
NEGATIVE_PROMPT_CACHE = "negative_text_encode"  # the fixed negative prompt's encoding, computed once and reused
ALLOWED_CACHE = frozenset({NEGATIVE_PROMPT_CACHE})  # reuse every engine may report without invalidating a clip


@dataclass(frozen=True)
class ClipRequest:
    """One clip to generate: a clip-set row plus the round's seed."""

    id: str
    task: str
    prompt: str
    width: int
    height: int
    num_frames: int
    fps: float
    steps: int
    guidance_scale: float
    seed: int
    image: bytes | None = field(default=None, repr=False)  # i2v: the conditioning image file's bytes
    noise_seed: int | None = None
    audio: bool | None = None

    def body(self) -> dict:
        """The JSON body of POST /v1/clips."""
        b = {"task": self.task, "prompt": self.prompt, "width": self.width, "height": self.height,
             "num_frames": self.num_frames, "fps": self.fps, "steps": self.steps,
             "guidance_scale": self.guidance_scale, "seed": self.seed}
        if self.image is not None:
            b["image"] = base64.b64encode(self.image).decode("ascii")
        if self.noise_seed is not None:
            b["noise_seed"] = self.noise_seed
        if self.audio is not None:
            b["audio"] = self.audio
        return b


@dataclass
class ClipResult:
    id: str
    ok: bool = False
    error: str | None = None
    status: int | None = None
    t_send_ns: int = 0
    t_connected_ns: int | None = None
    t_done_ns: int | None = None  # the whole clip is in the client's hands
    t_end_ns: int = 0
    video: bytes | None = field(default=None, repr=False)
    format: str = "mp4"
    stage_ms: dict | None = None  # self-reported
    effective: dict | None = None  # self-reported
    cache: list | None = None  # stages served from a cache: any makes the claim invalid
    cache_allowed: list | None = None  # reuse that is allowed and stated (ALLOWED_CACHE)
    via: str | None = None  # how the clip arrived: video_b64, video_path or body

    @property
    def e2e_ms(self) -> float | None:
        return (self.t_done_ns - self.t_send_ns) / 1e6 if self.ok and self.t_done_ns is not None else None

    @property
    def connect_ms(self) -> float | None:
        return (self.t_connected_ns - self.t_send_ns) / 1e6 if self.t_connected_ns is not None else None


class ClipBackend(Protocol):
    """What bench clip drives. ``generate`` never raises for engine failures: it returns ok=False."""

    endpoint_kind: str

    def health(self) -> tuple[int | None, str]:
        """(HTTP status or None, short detail): 200 means ready."""

    def generate(self, request: ClipRequest) -> ClipResult:
        """Generate one clip, client-timed (see the module docstring for the timer)."""

    def describe(self) -> dict:
        """Facts for the evidence record, e.g. {"workflow_sha256": ...}; never secrets or host names."""

    def close(self) -> None:
        ...


def stage_sum_ms(stage_ms: dict | None) -> float | None:
    """Sum of the numeric stages (totals and per-step lists left out); None when there are none."""
    if not isinstance(stage_ms, dict):
        return None
    vals = [float(v) for k, v in stage_ms.items() if k not in TOTAL_KEYS and isinstance(v, (int, float))
            and not isinstance(v, bool)]
    return sum(vals) if vals else None


def _clip_format(value) -> str:
    """The engine's format name as a file extension: up to 8 ASCII letters and digits, else "bin"."""
    f = str(value or "mp4").lower().lstrip(".")
    return f if f.isascii() and f.isalnum() and len(f) <= 8 else "bin"


def _announced(doc: dict) -> tuple | None:
    """(sha256, bytes) of the finished clip as the answer states them, or None when it states neither."""
    sha = doc.get("video_sha256", doc.get("sha256"))
    size = doc.get("video_bytes", doc.get("bytes"))
    return None if sha is None and size is None else (sha, size)


def _mismatch(video: bytes, announced: tuple, via: str | None) -> str | None:
    """Why the clip in hand is not the one the answer announced (None when it is)."""
    sha, size = announced
    bad = []
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size != len(video)):
        bad.append(f"{len(video)} bytes in hand, {str(size)[:20]} announced")
    if sha is not None and (not isinstance(sha, str) or hashlib.sha256(video).hexdigest() != sha.strip().lower()):
        bad.append("its sha256 is not the announced one")
    if not bad:
        return None
    late = ": the server answered before the file was complete" if via == "video_path" else ""
    return f"the clip in hand is not the one the answer describes ({'; '.join(bad)}){late}"


class ClipHTTP:
    """The clip HTTP contract over plain http.client (one connection per request, like a fresh client)."""

    endpoint_kind = "clip_http"

    def __init__(self, ep: Endpoint, timeout_s: float = DEFAULT_TIMEOUT_S, workflow: str | None = None):
        if workflow:
            raise ValueError("--workflow is for the comfyui backend; the clip HTTP contract takes none")
        self.ep, self.timeout_s = ep, timeout_s

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        if self.ep.scheme == "https":
            return http.client.HTTPSConnection(self.ep.host, self.ep.port, timeout=timeout)
        return http.client.HTTPConnection(self.ep.host, self.ep.port, timeout=timeout)

    def _headers(self, extra: dict | None = None) -> dict:
        h = {"Accept": "application/json, video/*", **(extra or {})}
        if self.ep.api_key:
            h["Authorization"] = f"Bearer {self.ep.api_key}"
        return h

    def health(self) -> tuple[int | None, str]:
        conn = self._connection(10.0)
        try:
            conn.request("GET", self.ep.prefix + HEALTH_PATH, headers=self._headers())
            resp = conn.getresponse()
            body = resp.read(MAX_ERROR_BODY).decode(errors="replace")
            return resp.status, scrub(body.strip(), self.ep)[:200]
        except (OSError, http.client.HTTPException) as e:
            return None, scrub(f"{type(e).__name__}: {e}", self.ep)
        finally:
            conn.close()

    def generate(self, request: ClipRequest) -> ClipResult:
        res = ClipResult(id=request.id)
        payload = json.dumps(request.body()).encode()
        conn = self._connection(self.timeout_s)
        announced = None
        res.t_send_ns = time.perf_counter_ns()
        try:
            conn.connect()
            res.t_connected_ns = time.perf_counter_ns()
            conn.request("POST", self.ep.prefix + CLIPS_PATH, body=payload,
                         headers=self._headers({"Content-Type": "application/json"}))
            resp = conn.getresponse()
            res.status = resp.status
            raw = resp.read()
            ctype = (resp.getheader("Content-Type") or "").split(";")[0].strip().lower()
            meta_header = resp.getheader("X-Clip-Meta")
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {raw[:MAX_ERROR_BODY].decode(errors='replace').strip()}"
            elif ctype.startswith("video/") or ctype == "application/octet-stream":
                res.video, res.via = raw, "body"
                # The engine picks the subtype and it becomes a file extension: sanitised like "format".
                res.format = _clip_format(ctype.split("/", 1)[1]) if ctype.startswith("video/") else "bin"
                meta = json.loads(meta_header) if meta_header else {}
                self._meta(res, meta)
                announced = _announced(meta)
            else:
                announced = self._json_answer(res, json.loads(raw))
            if res.video is not None and res.error is None:
                res.t_done_ns = time.perf_counter_ns()
                res.ok = len(res.video) > 0
                if not res.ok:
                    res.error = "empty clip"
                elif announced is not None:  # after t_done: the check is not timed, and a mismatch fails the clip
                    res.error = _mismatch(res.video, announced, res.via)
        except (OSError, http.client.HTTPException) as e:
            res.error = f"{type(e).__name__}: {e}"
        except (ValueError, binascii.Error) as e:  # JSON, base64 or the meta header did not parse
            res.error = f"bad answer: {type(e).__name__}: {e}"
        finally:
            conn.close()
            res.t_end_ns = time.perf_counter_ns()
        if res.error:
            res.ok, res.video = False, None
            res.error = scrub(res.error, self.ep)[:MAX_ERROR_BODY]
        return res

    def _json_answer(self, res: ClipResult, doc) -> tuple | None:
        """Take the clip from a JSON answer; return the announced (sha256, bytes) to check once it is in hand."""
        if not isinstance(doc, dict):
            raise ValueError("the answer is not a JSON object")
        announced = _announced(doc)
        if isinstance(doc.get("video_b64"), str):
            res.video, res.via = base64.b64decode(doc["video_b64"], validate=True), "video_b64"
        elif isinstance(doc.get("video_path"), str):
            if not self.ep.is_local():
                res.error = "video_path from a server on another host cannot be read; send video_b64"
                return None
            if announced is None or None in announced:
                # Without them the client cannot tell a finished file from one the server is still writing.
                res.error = "a video_path answer must carry video_sha256 and video_bytes of the finished file"
                return None
            try:
                res.video, res.via = Path(doc["video_path"]).read_bytes(), "video_path"
            except OSError as e:
                res.error = f"video_path not readable: {type(e).__name__}"
                return None
        else:
            res.error = "the answer has neither video_b64 nor video_path"
            return None
        res.format = _clip_format(doc.get("format"))
        self._meta(res, doc)
        return announced

    @staticmethod
    def _meta(res: ClipResult, doc: dict) -> None:
        if not isinstance(doc, dict):
            raise ValueError("X-Clip-Meta is not a JSON object")
        if doc.get("format"):
            res.format = _clip_format(doc["format"])
        res.stage_ms = doc.get("stage_ms") if isinstance(doc.get("stage_ms"), dict) else None
        res.effective = doc.get("effective") if isinstance(doc.get("effective"), dict) else None
        cache = doc.get("cache")
        if isinstance(cache, list):
            names = [str(c) for c in cache]
            res.cache = [c for c in names if c not in ALLOWED_CACHE]
            res.cache_allowed = [c for c in names if c in ALLOWED_CACHE]
        else:
            res.cache = None

    def describe(self) -> dict:
        return {"endpoint_kind": self.endpoint_kind, "paths": [HEALTH_PATH, CLIPS_PATH]}

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- backend registry

BackendFactory = Callable[..., ClipBackend]
_BACKENDS: dict[str, BackendFactory] = {"clip_http": ClipHTTP}
# Backends that live in their own module; it registers itself when imported.
_LAZY = {"comfyui": "inferencekit.engines.clip.comfyui"}


def register_backend(kind: str, factory: BackendFactory) -> None:
    """Make ``kind`` available to bench clip. ``factory(ep, timeout_s=..., workflow=...)`` returns a ClipBackend."""
    _BACKENDS[kind] = factory


def make_backend(kind: str, ep: Endpoint, timeout_s: float = DEFAULT_TIMEOUT_S,
                 workflow: str | None = None) -> ClipBackend:
    """The backend for an endpoint kind. Raises LookupError when it is not available in this build."""
    if kind not in _BACKENDS and kind in _LAZY:
        try:
            importlib.import_module(_LAZY[kind])
        except ImportError as e:
            raise LookupError(f"endpoint kind {kind!r} needs {_LAZY[kind]}, which is not in this build ({e})") from None
    if kind not in _BACKENDS:
        raise LookupError(f"unknown endpoint kind {kind!r}; known: {sorted(set(_BACKENDS) | set(_LAZY))}")
    return _BACKENDS[kind](ep, timeout_s=timeout_s, workflow=workflow)

