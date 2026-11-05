"""A ComfyUI client for timed video clips, and the ``comfyui`` backend bench clip drives. Standard library only.

Per-step dumps (stepdump.py, for quality clip-steps) cannot come from this client: the denoising loop runs inside the
ComfyUI process. A ComfyUI engine that must pass quality clip-steps writes them from a custom node that wraps its
sampler's step (x, the guided velocity (x - denoised) / sigma, the next latent), named by the sha256 of the clip it
saves; not provided here (ComfyUI's sampler code was not read for this).

ComfyUI's HTTP API as used here: ``GET /system_stats`` (ready check and version), ``POST /upload/image``
(the i2v image), ``POST /prompt`` {"prompt": <API-format workflow>, "client_id"} -> {"prompt_id"},
``GET /history/<prompt_id>`` (outputs and status messages once the prompt is done), ``GET /view``
(an output file's bytes), ``GET``/``POST /queue`` and ``POST /interrupt`` (after a timeout), and the
websocket ``/ws?clientId=...`` (execution_start, execution_cached, executing, progress, executed,
execution_success / execution_error). With an API key, every request carries ``Authorization: Bearer``,
the websocket upgrade included.

The workflow file (pinned: its sha256 goes into the record) wraps an API-format workflow with the places
the clip's settings go::

    {"format": "inferencekit-comfyui-workflow/1",
     "workflow": {"3": {"class_type": "KSampler", "inputs": {...}}, ...},     # Export (API) from ComfyUI
     "bindings": {"prompt": [["6", "text"]], "seed": [["3", "seed"]], "steps": [["3", "steps"]],
                  "width": [["5", "width"]], "height": [["5", "height"]], "num_frames": [["5", "length"]],
                  "fps": [["8", "frame_rate"]], "guidance_scale": [["3", "cfg"]],
                  "image": [["12", "image"]], "noise_seed": [...], "audio": [...]},   # the last three optional
     "output_node": "9",          # optional: the node whose file is the clip (else the first video output)
     "allow_cached": ["7"],       # optional: the fixed negative prompt's encoder may be reused (see Cache)
     "audio": true,               # optional: the graph always makes an audio track (true) or never (false)
     "sampler_nodes": ["3"]}      # optional: nodes whose progress counts as denoise steps (default: the
                                  # KSampler*, SamplerCustom* and *Sampler nodes; not KSamplerSelect)

Every field of a clip request must have a binding (``image`` only for i2v, ``noise_seed`` and ``audio`` only
when the request sets them): a setting the workflow would ignore is refused, never dropped. A graph whose
audio track cannot be switched (LTX-2's audio-and-video graphs always decode one) declares it with
``"audio": true`` (or false) instead of an audio binding: a request asking for that is served, the other
value is refused. With no
``noise_seed`` binding, the seed bindings get ``noise_seed`` when the request has one, since a ComfyUI
sampler draws its initial noise from its seed.

Timing (the same timer as the clip HTTP contract): ``t_send`` just before the image upload or POST
/prompt, ``t_done`` once the output file's bytes from /view are in hand. The websocket is connected
before ``t_send`` (it only observes). Without a websocket the client polls /history every ``poll_s``
(default 0.05 s, so up to that much is added to a clip).

Cache: ComfyUI skips a node whose inputs (and upstream nodes) equal an earlier run's and reports it in
``execution_cached`` (on the websocket and in the history status messages). Its default cache (RAM pressure,
ComfyUI 0.40) keeps the outputs of many earlier prompts, so a clip set's prompt, repeated in the next round,
finds its text encoding cached (observed on an RTX PRO 6000 with LTX-2: rounds 2 and 3 CLAIM INVALID). Time
ComfyUI launched with ``--cache-classic``, which keeps only the last prompt's outputs. What matters is a cached
node of a costly stage (COSTLY: text_encode, image_encode, denoise, upscale, vae_decode, audio_decode,
mux, from the class name, see STAGES): part of the clip was then not computed, so its id goes into
``ClipResult.cache`` and bench clip marks the claim invalid. Every other node may be cached: loaders (the
normal warm state, like weights staying in a server's memory) and constant helpers that cost nothing to
rebuild and are reused on every clip after the first (empty latents, KSamplerSelect, schedulers and
sigmas, guiders, noise of an unchanged seed). A negative-prompt encoder whose text never changes is
cached by ComfyUI after the first clip: list it in ``allow_cached`` to allow that one reuse, which every
engine may make (a clip HTTP engine reports it as cache ["negative_text_encode"]) and bench clip states as
a limitation. Only such a node may be listed there among the costly ones: a text encoder that no binding
reaches (its text is fixed) and that feeds an input named ``negative...``; anything else is refused. A
reused one is reported as ``ClipResult.cache_allowed`` = ["negative_text_encode"]. A prompt's own encoder
is cached when two clips in a row share the prompt: bench clip warms up with prompts of its own and orders
the clips so neighbours differ, gates clip gives each probe pair a prompt and image file of its own, and a
cached one makes the claim invalid. The record shows the allowed nodes (describe) and per clip what was
skipped (effective.cached_allowed). The class-name rules were written from ComfyUI's core nodes and the
LTX-Video ones as remembered, not checked against the LTX-2 template: check describe().allow_cached,
fixed_encoders and sampler_nodes on the GPU host.

Timeout: the client drops its prompt from the queue (``POST /queue {"delete": [id]}``) and interrupts
it only when ``GET /queue`` shows it running (``POST /interrupt {"prompt_id": id}``), so a prompt left
behind does not run during the next timed clip and another client's prompt is not interrupted.

What is reported (``effective``): steps_run = for each sampler node, the highest progress value seen on
the websocket, summed; absent when a sampler sent no progress or there was no websocket, and then
``steps_unchecked`` says why. ``websocket`` is "connected" or why the client polled /history instead
(e.g. the upgrade was refused). width, height, frames, fps as submitted (bench clip does not decode the
file: gates clip does). ``stage_ms`` (websocket only): queue_wait, the node times grouped by stage (the
costly ones, load and other), upload and fetch.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import http.client
import json
import os
import re
import socket
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlencode

FORMAT = "inferencekit-comfyui-workflow/1"
REQUIRED_BINDINGS = ("prompt", "seed", "steps", "width", "height", "num_frames", "fps", "guidance_scale")
OPTIONAL_BINDINGS = ("image", "noise_seed", "audio")
VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp")
LOADER = re.compile(r"Loader|^Load")
# the nodes that run a denoise loop (KSampler, KSamplerAdvanced, SamplerCustom, SamplerCustomAdvanced,
# LTXVBaseSampler...); not KSamplerSelect or the Sampler<Name> nodes that only build a sampler object
SAMPLER = re.compile(r"^KSampler(?!Select)|^SamplerCustom|Sampler$")
# class_type pattern -> stage name for node times (first match wins: a text-encoder loader is a loader,
# an audio VAE decoder is audio)
STAGES = ((LOADER, "load"), (re.compile(r"TextEncode|Conditioning"), "text_encode"),
          (re.compile(r"VAEEncode|ImgToVideo|ImageToVideo"), "image_encode"), (SAMPLER, "denoise"),
          (re.compile(r"Audio.*Decode|DecodeAudio|Vocoder"), "audio_decode"),
          (re.compile(r"VAEDecode|VideoDecode"), "vae_decode"),
          (re.compile(r"VideoCombine|SaveVideo|CreateVideo|SaveWEBM|SaveAnimated"), "mux"),
          (re.compile(r"Upscale|Upsampl"), "upscale"))
# a cached node of these stages means part of the clip was not computed; any other node may be cached
COSTLY = frozenset({"text_encode", "image_encode", "denoise", "upscale", "vae_decode", "audio_decode", "mux"})
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_ERROR = 2000


class WorkflowError(ValueError):
    """The workflow file or a request does not fit (bad bindings, a setting without a place to go)."""


@dataclass
class Workflow:
    """A pinned workflow file: the API-format graph, its bindings and the cache/sampler/output rules."""

    graph: dict
    bindings: dict
    sha256: str
    output_node: str | None = None
    allow_cached: frozenset = frozenset()
    sampler_nodes: frozenset = frozenset()
    fixed_encoders: frozenset = frozenset()  # allowed costly nodes: fixed negative-prompt encoders
    audio: bool | None = None  # the graph always (True) or never (False) makes audio; None: bound or unknown

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        raw = Path(path).read_bytes()
        return cls.parse(json.loads(raw), hashlib.sha256(raw).hexdigest())

    @classmethod
    def parse(cls, doc: dict, sha256: str = "") -> Workflow:
        if not isinstance(doc, dict) or doc.get("format") != FORMAT:
            raise WorkflowError(f"the workflow file must be a JSON object with format {FORMAT!r}")
        graph, bindings = doc.get("workflow"), doc.get("bindings")
        if not isinstance(graph, dict) or not graph or not all(
                isinstance(n, dict) and "class_type" in n and isinstance(n.get("inputs"), dict) for n in graph.values()):
            raise WorkflowError("workflow must be ComfyUI's API format: {node_id: {class_type, inputs}}")
        if not isinstance(bindings, dict):
            raise WorkflowError("bindings must map each clip setting to [[node_id, input_name], ...]")
        unknown = sorted(set(bindings) - set(REQUIRED_BINDINGS) - set(OPTIONAL_BINDINGS))
        missing = [k for k in REQUIRED_BINDINGS if not bindings.get(k)]
        if unknown or missing:
            raise WorkflowError(f"bindings: unknown {unknown}, missing {missing}")
        for key, targets in bindings.items():
            if not isinstance(targets, list) or not targets:
                raise WorkflowError(f"bindings.{key} must be a non-empty list of [node_id, input_name]")
            for t in targets:
                if not (isinstance(t, list) and len(t) == 2 and str(t[0]) in graph
                        and t[1] in graph[str(t[0])]["inputs"]):
                    raise WorkflowError(f"bindings.{key}: {t!r} is not an input of a node in the workflow")
        out = doc.get("output_node")
        if out is not None and str(out) not in graph:
            raise WorkflowError(f"output_node {out!r} is not in the workflow")

        def ids(key):
            v = doc.get(key)
            if v is None:
                return None
            if not isinstance(v, list) or any(str(i) not in graph for i in v):
                raise WorkflowError(f"{key} must list node ids of the workflow")
            return frozenset(str(i) for i in v)

        def where(rule):
            return frozenset(i for i, n in graph.items() if rule(str(n["class_type"])))

        cheap = where(lambda c: stage_of(c) not in COSTLY)
        extra = ids("allow_cached") or frozenset()
        encoders = frozenset(i for i in extra if i not in cheap)
        bound = {str(node) for targets in bindings.values() for node, _ in targets}
        for i in sorted(encoders):
            why = _not_fixed_negative(graph, bound, i)
            if why:
                raise WorkflowError(f"allow_cached: node {i} ({graph[i]['class_type']}) {why}; only the fixed "
                                    "negative prompt's encoder may be reused (a text encoder no binding reaches, "
                                    "feeding a 'negative' input)")
        samplers = ids("sampler_nodes")  # an explicit [] means none: steps_run is then not reported
        audio = doc.get("audio")
        if audio is not None and not isinstance(audio, bool):
            raise WorkflowError("audio must be true (the graph always makes an audio track) or false")
        if audio is not None and "audio" in bindings:
            raise WorkflowError("give an audio binding or a fixed audio value, not both")
        return cls(graph=graph, bindings={k: [[str(a), b] for a, b in v] for k, v in bindings.items()},
                   sha256=sha256, output_node=str(out) if out is not None else None,
                   allow_cached=cheap | extra, fixed_encoders=encoders, audio=audio,
                   sampler_nodes=samplers if samplers is not None else where(lambda c: bool(SAMPLER.search(c))))

    def build(self, values: dict) -> dict:
        """A copy of the graph with each bound input set; WorkflowError for a value with no binding."""
        g = copy.deepcopy(self.graph)
        for key, value in values.items():
            if value is None:
                continue
            if key not in self.bindings:
                raise WorkflowError(f"the request sets {key} but the workflow has no binding for it")
            for node, name in self.bindings[key]:
                g[node]["inputs"][name] = value
        return g

    def class_of(self, node: str) -> str:
        return str(self.graph.get(str(node), {}).get("class_type", ""))


def cache_mode(flags: list) -> str:
    """ComfyUI's node cache from its launch flags: classic keeps only the last prompt's outputs; the default
    (ram, ComfyUI 0.40) and lru keep many earlier prompts' outputs; none recomputes every node, loaders too."""
    for flag, mode in (("--cache-classic", "classic"), ("--cache-lru", "lru"), ("--cache-none", "none")):
        if flag in flags:
            return mode
    return "ram"


def stage_of(class_type: str) -> str:
    for rx, name in STAGES:
        if rx.search(class_type):
            return name
    return "other"


def _links(node: dict):
    """(input name, upstream node id) for each input of an API-format node that is a link [node_id, output]."""
    for name, value in node["inputs"].items():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) and not isinstance(value[0], list):
            yield name, str(value[0])


def _not_fixed_negative(graph: dict, bound: set, node: str) -> str | None:
    """Why ``node`` is not a fixed negative-prompt encoder (None when it is): a text_encode node whose own
    inputs and upstream nodes no binding sets (so its output is the same on every clip), feeding an input
    whose name starts with 'negative'."""
    if stage_of(str(graph[node]["class_type"])) != "text_encode":
        return f"is a {stage_of(str(graph[node]['class_type']))} node, not a text encoder"
    seen, todo = set(), [node]
    while todo:
        n = todo.pop()
        if n in seen or n not in graph:
            continue
        seen.add(n)
        if n in bound:
            return "depends on a clip setting" if n != node else "has a bound input (it changes per clip)"
        todo += [up for _, up in _links(graph[n])]
    feeds = [f"{i}.{name}" for i, n in graph.items() for name, up in _links(n) if up == node]
    if not any(f.split(".", 1)[1].lower().startswith("negative") for f in feeds):
        return f"feeds no 'negative' input ({feeds or 'nothing'})"
    return None


# --------------------------------------------------------------------------- websocket (RFC 6455, client side)


class WebSocket:
    """Just enough of a websocket client to read ComfyUI's messages: text and binary frames, ping/pong, close."""

    def __init__(self, host: str, port: int, path: str, timeout: float = 10.0, headers: dict | None = None):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""
        self.closed = False
        key = base64.b64encode(os.urandom(16)).decode()
        extra = "".join(f"{k}: {v}\r\n" for k, v in (headers or {}).items())
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                           f"{extra}\r\n").encode())
        deadline = time.monotonic() + timeout
        while b"\r\n\r\n" not in self.buf:
            self._fill(len(self.buf) + 1, deadline)
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
        if " 101 " not in lines[0] + " " or headers.get("sec-websocket-accept") != accept:
            self.sock.close()
            raise ConnectionError(f"websocket upgrade refused: {lines[0][:80]}")

    def _fill(self, n: int, deadline: float) -> None:
        while len(self.buf) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("websocket read timed out")
            self.sock.settimeout(left)
            chunk = self.sock.recv(65536)  # socket.timeout is TimeoutError
            if not chunk:
                self.closed = True
                raise ConnectionError("websocket closed by the server")
            self.buf += chunk

    def _frame(self, deadline: float) -> tuple[bool, int, bytes]:
        self._fill(2, deadline)
        b0, b1 = self.buf[0], self.buf[1]
        n, pos = b1 & 0x7F, 2
        if n == 126:
            self._fill(4, deadline)
            n, pos = struct.unpack(">H", self.buf[2:4])[0], 4
        elif n == 127:
            self._fill(10, deadline)
            n, pos = struct.unpack(">Q", self.buf[2:10])[0], 10
        mask = b""
        if b1 & 0x80:
            self._fill(pos + 4, deadline)
            mask, pos = self.buf[pos:pos + 4], pos + 4
        self._fill(pos + n, deadline)  # the whole frame is buffered before anything is consumed
        payload, self.buf = self.buf[pos:pos + n], self.buf[pos + n:]
        if mask:
            payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
        return bool(b0 & 0x80), b0 & 0x0F, payload

    def send(self, opcode: int, payload: bytes = b"") -> None:
        mask = os.urandom(4)
        n = len(payload)
        head = bytes([0x80 | opcode]) + (bytes([0x80 | n]) if n < 126 else
                                         bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else
                                         bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.sock.sendall(head + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(payload)))

    def recv(self, timeout: float) -> tuple[int, bytes]:
        """(opcode, payload) of the next whole message: 1 text, 2 binary, 8 close. TimeoutError when none came."""
        deadline = time.monotonic() + timeout
        parts, first = [], None
        while True:
            fin, op, payload = self._frame(deadline)
            if op == 9:  # ping
                self.send(10, payload)
                continue
            if op == 10:
                continue
            if op == 8:
                self.closed = True
                return 8, payload
            if op in (1, 2):
                first, parts = op, [payload]
            elif op == 0 and first is not None:
                parts.append(payload)
            if fin and first is not None:
                return first, b"".join(parts)

    def close(self) -> None:
        try:
            if not self.closed:
                self.send(8, struct.pack(">H", 1000))
        except OSError:
            pass
        self.sock.close()


# --------------------------------------------------------------------------- the client


@dataclass
class ComfyRun:
    """One clip through ComfyUI: client-clock times (perf_counter_ns) and what was observed."""

    ok: bool = False
    error: str | None = None
    status: int | None = None
    prompt_id: str | None = None
    t_send_ns: int = 0
    t_connected_ns: int | None = None
    t_done_ns: int | None = None
    t_end_ns: int = 0
    video: bytes | None = field(default=None, repr=False)
    format: str = "mp4"
    cached: list = field(default_factory=list)  # every node ComfyUI reported in execution_cached
    progress: dict = field(default_factory=dict)  # node -> highest progress value seen
    stage_ms: dict = field(default_factory=dict)
    websocket: bool = False  # True when the websocket watched the whole prompt
    ws_note: str = ""  # "connected", or why the client polled /history instead


class ComfyUIClient:
    def __init__(self, host: str, port: int, *, scheme: str = "http", prefix: str = "", api_key: str | None = None,
                 timeout_s: float = 1800.0, poll_s: float = 0.05, websocket: bool = True):
        self.ws_off = None if websocket else "turned off"
        if scheme != "http" and websocket:  # the websocket reader here is plain TCP; https polls /history
            websocket, self.ws_off = False, f"not used over {scheme} (the reader here is plain TCP)"
        self.host, self.port, self.scheme, self.prefix = host, port, scheme, prefix.rstrip("/")
        self.api_key, self.timeout_s, self.poll_s, self.use_ws = api_key, timeout_s, poll_s, websocket
        self.client_id = uuid.uuid4().hex

    def _conn(self, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)

    def _headers(self, extra: dict | None = None) -> dict:
        h = dict(extra or {})
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
                timeout: float = 30.0, conn: http.client.HTTPConnection | None = None) -> tuple[int, bytes]:
        own = conn is None
        conn = conn or self._conn(timeout)
        try:
            conn.request(method, self.prefix + path, body=body, headers=self._headers(headers))
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            if own:
                conn.close()

    def _json(self, method: str, path: str, doc=None, timeout: float = 30.0):
        body = json.dumps(doc).encode() if doc is not None else None
        status, raw = self.request(method, path, body, {"Content-Type": "application/json"} if body else None, timeout)
        if status != 200:
            raise RuntimeError(f"{method} {path}: HTTP {status}: {raw[:MAX_ERROR].decode(errors='replace').strip()}")
        return json.loads(raw) if raw.strip() else {}

    def system_stats(self) -> dict:
        return self._json("GET", "/system_stats", timeout=10.0)

    def history(self, prompt_id: str) -> dict:
        return self._json("GET", f"/history/{quote(prompt_id)}").get(prompt_id) or {}

    def _post_quiet(self, path: str, doc: dict) -> None:
        try:
            self.request("POST", path, json.dumps(doc).encode(), {"Content-Type": "application/json"}, timeout=5.0)
        except (OSError, http.client.HTTPException):
            pass

    def cancel(self, prompt_id: str | None) -> None:
        """After a timeout: stop our prompt, and only ours, so it does not run during the next timed clip.

        It is dropped from the queue if still waiting, and interrupted (by id: ComfyUI versions that take
        prompt_id interrupt only that prompt, older ones whatever runs) only when GET /queue shows it
        running or the queue cannot be read. Without a prompt_id (POST /prompt itself timed out) there is
        nothing to name, so nothing is sent.
        """
        if not prompt_id:
            return
        running = True
        try:
            q = self._json("GET", "/queue", timeout=5.0)
            running = any(isinstance(item, list) and len(item) > 1 and item[1] == prompt_id
                          for item in q.get("queue_running") or [])
        except (OSError, http.client.HTTPException, RuntimeError, ValueError, AttributeError):
            pass
        if running:
            self._post_quiet("/interrupt", {"prompt_id": prompt_id})
        self._post_quiet("/queue", {"delete": [prompt_id]})

    def upload_image(self, data: bytes, name: str) -> str:
        """POST /upload/image (multipart); returns the name to put in a LoadImage node."""
        boundary = uuid.uuid4().hex
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode(),
                 (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                  f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n",
                 f"--{boundary}--\r\n".encode()]
        status, raw = self.request("POST", "/upload/image", b"".join(parts),
                                   {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=60.0)
        if status != 200:
            raise RuntimeError(f"image upload: HTTP {status}: {raw[:MAX_ERROR].decode(errors='replace').strip()}")
        doc = json.loads(raw)
        return f"{doc['subfolder']}/{doc['name']}" if doc.get("subfolder") else doc["name"]

    def _websocket(self) -> tuple[WebSocket | None, str]:
        """(the websocket, "connected"), or (None, why there is none: the client then polls /history)."""
        if not self.use_ws:
            return None, self.ws_off or "turned off"
        try:
            return WebSocket(self.host, self.port, f"{self.prefix}/ws?clientId={self.client_id}", timeout=10.0,
                             headers=self._headers()), "connected"
        except (OSError, ConnectionError, TimeoutError) as e:
            return None, f"{type(e).__name__}: {e}"[:200]

    def run(self, wf: Workflow, values: dict, image: bytes | None = None) -> ComfyRun:
        """Queue one clip, wait for it, fetch its file. Never raises for server-side failures (ok=False)."""
        run = ComfyRun()
        graph = wf.build(values)  # before the timer: building the body is client-side preparation
        ws, run.ws_note = self._websocket()
        run.websocket = ws is not None
        run.t_send_ns = time.perf_counter_ns()
        deadline = time.monotonic() + self.timeout_s
        try:
            if image is not None:
                if "image" not in wf.bindings:
                    raise WorkflowError("task i2v needs an image binding in the workflow")
                t0 = time.perf_counter_ns()
                name = self.upload_image(image, f"ik-{hashlib.sha256(image).hexdigest()[:16]}.png")
                run.stage_ms["upload"] = (time.perf_counter_ns() - t0) / 1e6
                for node, inp in wf.bindings["image"]:
                    graph[node]["inputs"][inp] = name
            conn = self._conn(60.0)
            try:
                conn.connect()
                run.t_connected_ns = time.perf_counter_ns()
                status, raw = self.request("POST", "/prompt", json.dumps(
                    {"prompt": graph, "client_id": self.client_id}).encode(), {"Content-Type": "application/json"},
                    conn=conn)
            finally:
                conn.close()
            run.status = status
            if status != 200:
                raise RuntimeError(f"POST /prompt: HTTP {status}: {raw[:MAX_ERROR].decode(errors='replace').strip()}")
            run.prompt_id = json.loads(raw)["prompt_id"]
            t_queued = time.perf_counter_ns()
            if ws is not None:
                self._watch(ws, wf, run, t_queued, deadline)
            entry = self._wait_history(run.prompt_id, deadline)
            self._read_status(entry, run)
            if run.error is None:
                t0 = time.perf_counter_ns()
                run.video, run.format = self._fetch(wf, entry)
                run.t_done_ns = time.perf_counter_ns()
                run.stage_ms["fetch"] = (run.t_done_ns - t0) / 1e6
                run.ok = len(run.video) > 0
                if not run.ok:
                    run.error = "empty clip"
        except TimeoutError:
            run.error = f"no result within {self.timeout_s:g} s"
            self.cancel(run.prompt_id)
        except (OSError, http.client.HTTPException, RuntimeError, ValueError, KeyError) as e:
            run.error = f"{type(e).__name__}: {e}"
        finally:
            if ws is not None:
                ws.close()
            run.t_end_ns = time.perf_counter_ns()
        if run.error:
            run.ok, run.video = False, None
            run.error = run.error[:MAX_ERROR]
        return run

    def _watch(self, ws: WebSocket, wf: Workflow, run: ComfyRun, t_queued: int, deadline: float) -> None:
        """Read websocket messages for this prompt until it ends; record cache, progress and node times."""
        current, t_node = None, None
        nodes_ms: dict[str, float] = {}

        def close_node(t):
            if current is not None:
                stage = stage_of(wf.class_of(current))
                nodes_ms[stage] = nodes_ms.get(stage, 0.0) + (t - t_node) / 1e6

        while True:
            try:
                op, payload = ws.recv(max(0.0, deadline - time.monotonic()))
            except ConnectionError:
                op, payload = 8, b""
            if op == 8:  # closed early: the history poll still finds the result and the cached nodes
                run.websocket, run.ws_note = False, "closed by ComfyUI during the clip"
                return
            if op != 1:
                continue  # binary frames are previews
            t = time.perf_counter_ns()
            try:
                msg = json.loads(payload)
            except ValueError:
                continue
            kind, data = msg.get("type"), msg.get("data") or {}
            if data.get("prompt_id") not in (None, run.prompt_id):
                continue
            if kind == "execution_start":
                run.stage_ms["queue_wait"] = (t - t_queued) / 1e6
            elif kind == "execution_cached":
                run.cached.extend(str(n) for n in data.get("nodes") or [])
            elif kind == "progress" and data.get("node") is not None:
                node = str(data["node"])
                run.progress[node] = max(run.progress.get(node, 0), int(data.get("value") or 0))
            elif kind == "executing":
                close_node(t)
                current, t_node = (str(data["node"]), t) if data.get("node") is not None else (None, None)
                if data.get("node") is None and data.get("prompt_id") == run.prompt_id:
                    break
            elif kind in ("execution_success", "execution_error", "execution_interrupted"):
                close_node(t)
                break
        run.stage_ms.update({k: round(v, 3) for k, v in nodes_ms.items()})

    def _wait_history(self, prompt_id: str, deadline: float) -> dict:
        while True:
            entry = self.history(prompt_id)
            status = entry.get("status") or {}
            if entry and (status.get("completed") or status.get("status_str") in ("success", "error")):
                return entry
            if time.monotonic() >= deadline:
                raise TimeoutError
            time.sleep(self.poll_s)

    @staticmethod
    def _read_status(entry: dict, run: ComfyRun) -> None:
        status = entry.get("status") or {}
        for msg in status.get("messages") or []:
            if not (isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict)):
                continue
            kind, data = msg
            if kind == "execution_cached":
                run.cached.extend(str(n) for n in data.get("nodes") or [] if str(n) not in run.cached)
            elif kind == "execution_error":
                run.error = (f"ComfyUI error in node {data.get('node_id')} ({data.get('node_type')}): "
                             f"{data.get('exception_message', '').strip()}")
            elif kind == "execution_interrupted":
                run.error = "ComfyUI interrupted the prompt"
        if run.error is None and status.get("status_str") == "error":
            run.error = "ComfyUI reports status error"

    def _fetch(self, wf: Workflow, entry: dict) -> tuple[bytes, str]:
        files = []
        for node, out in (entry.get("outputs") or {}).items():
            if wf.output_node is not None and str(node) != wf.output_node:
                continue
            for value in out.values() if isinstance(out, dict) else []:
                for f in value if isinstance(value, list) else []:
                    if isinstance(f, dict) and f.get("filename", "").lower().endswith(VIDEO_EXT):
                        files.append(f)
        if not files:
            raise RuntimeError("no video file in the prompt's outputs" +
                               (f" of node {wf.output_node}" if wf.output_node else ""))
        f = files[0]
        q = urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
        status, raw = self.request("GET", f"/view?{q}", timeout=300.0)
        if status != 200:
            raise RuntimeError(f"GET /view: HTTP {status}")
        return raw, Path(f["filename"]).suffix.lstrip(".").lower() or "bin"


# --------------------------------------------------------------------------- the bench clip backend


def request_values(request, wf: Workflow) -> dict:
    """The bound values for a clip request (inferencekit.serve.clip_client.ClipRequest)."""
    seed = request.seed
    values = {"prompt": request.prompt, "steps": request.steps, "width": request.width, "height": request.height,
              "num_frames": request.num_frames, "fps": request.fps, "guidance_scale": request.guidance_scale}
    if request.noise_seed is not None and "noise_seed" in wf.bindings:
        values["noise_seed"] = request.noise_seed
    elif request.noise_seed is not None:
        seed = request.noise_seed  # the sampler's seed is its noise seed
    values["seed"] = seed
    if request.audio is not None:
        if "audio" not in wf.bindings and wf.audio is not None:
            if request.audio != wf.audio:
                raise WorkflowError(f"the request asks audio={str(request.audio).lower()} but this workflow "
                                    f"always makes audio={str(wf.audio).lower()}")
        else:
            values["audio"] = request.audio
    return values


def observed(run: ComfyRun, wf: Workflow) -> dict:
    """What the client saw of one run, for ``effective``: steps_run when every sampler node reported its
    progress (else ``steps_unchecked``: why not), the cached nodes that were allowed, and the websocket."""
    eff = {"cached_allowed": sorted({n for n in run.cached if n in wf.allow_cached}), "websocket": run.ws_note}
    silent = sorted(wf.sampler_nodes - set(run.progress))
    if not run.websocket:
        eff["steps_unchecked"] = f"no websocket ({run.ws_note})"
    elif not wf.sampler_nodes:
        eff["steps_unchecked"] = "the workflow has no sampler node (set sampler_nodes)"
    elif silent:  # a count from some samplers only would be a wrong count, not a check
        eff["steps_unchecked"] = f"no progress from sampler node(s) {silent} on the websocket"
    else:
        eff["steps_run"] = sum(run.progress[n] for n in wf.sampler_nodes)
    return eff


class ComfyUIBackend:
    """bench clip's ``comfyui`` endpoint kind (see inferencekit.serve.clip_client.ClipBackend)."""

    endpoint_kind = "comfyui"

    def __init__(self, ep, timeout_s: float = 1800.0, workflow: str | None = None, *, websocket: bool = True,
                 poll_s: float = 0.05):
        if not workflow:
            raise ValueError("the comfyui backend needs --workflow (an inferencekit ComfyUI workflow file)")
        self.ep = ep
        self.wf = Workflow.load(workflow)
        self.client = ComfyUIClient(ep.host, ep.port, scheme=ep.scheme, prefix=ep.prefix, api_key=ep.api_key,
                                    timeout_s=timeout_s, poll_s=poll_s, websocket=websocket)
        self._version: dict = {}

    def health(self) -> tuple[int | None, str]:
        from inferencekit.serve.client import scrub

        try:
            stats = self.client.system_stats()
        except RuntimeError as e:
            m = re.search(r"HTTP (\d+)", str(e))
            return (int(m.group(1)) if m else None), scrub(str(e), self.ep)[:200]
        except (OSError, http.client.HTTPException, ValueError) as e:
            return None, scrub(f"{type(e).__name__}: {e}", self.ep)
        system = stats.get("system") or {}
        self._version = {k: system[k] for k in ("comfyui_version", "pytorch_version") if system.get(k)}
        self._version["devices"] = [d.get("name") for d in stats.get("devices") or [] if isinstance(d, dict)]
        if isinstance(system.get("argv"), list):  # flag names only (values may be paths); the cache mode
            flags = [str(a).split("=", 1)[0] for a in system["argv"] if str(a).startswith("--")]
            self._version["launch_flags"] = flags
            self._version["cache_mode"] = cache_mode(flags)
        return 200, f"comfyui {system.get('comfyui_version', '?')}"

    def generate(self, request):
        from inferencekit.serve.client import scrub
        from inferencekit.serve.clip_client import NEGATIVE_PROMPT_CACHE, ClipResult

        res = ClipResult(id=request.id)
        try:
            values = request_values(request, self.wf)
            if request.task == "i2v" and request.image is None:
                raise WorkflowError("task i2v needs an image")
            run = self.client.run(self.wf, values, image=request.image if request.task == "i2v" else None)
        except WorkflowError as e:
            res.error, res.t_send_ns = f"workflow: {e}", time.perf_counter_ns()
            res.t_end_ns = res.t_send_ns
            return res
        res.ok, res.status, res.video, res.format = run.ok, run.status, run.video, run.format
        res.t_send_ns, res.t_connected_ns, res.t_done_ns, res.t_end_ns = (run.t_send_ns, run.t_connected_ns,
                                                                          run.t_done_ns, run.t_end_ns)
        res.error = scrub(run.error, self.ep) if run.error else None
        res.via = "view"
        res.cache = sorted({n for n in run.cached if n not in self.wf.allow_cached})
        res.cache_allowed = [NEGATIVE_PROMPT_CACHE] if self.wf.fixed_encoders & set(run.cached) else []
        eff = {"width": request.width, "height": request.height, "frames": request.num_frames, "fps": request.fps,
               "shape_source": "submitted", **observed(run, self.wf)}
        if self.wf.audio is not None:
            eff["audio"] = self.wf.audio  # as the workflow file declares it; gates clip decodes the track
        for key in ("websocket", "steps_unchecked"):  # error texts go into evidence
            if key in eff:
                eff[key] = scrub(eff[key], self.ep)
        res.effective = eff
        res.stage_ms = run.stage_ms or None
        return res

    def describe(self) -> dict:
        return {"endpoint_kind": self.endpoint_kind, "workflow_sha256": self.wf.sha256, **self._version,
                "bindings": self.wf.bindings, "allow_cached": sorted(self.wf.allow_cached),
                "fixed_encoders": sorted(self.wf.fixed_encoders),
                "sampler_nodes": sorted(self.wf.sampler_nodes), "output_node": self.wf.output_node,
                "audio": self.wf.audio,
                "websocket": self.client.use_ws}

    def close(self) -> None:
        return None


def register() -> None:
    """Make the ``comfyui`` endpoint kind available to bench clip (inferencekit.serve.clip_client)."""
    from inferencekit.serve.clip_client import register_backend

    register_backend("comfyui", ComfyUIBackend)


try:  # importing this module is enough for bench clip to find the endpoint kind
    register()
except ImportError:  # a build without bench clip's client
    pass
