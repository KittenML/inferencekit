"""A fake ComfyUI server (standard library), for testing the ComfyUI client without a GPU.

It implements the parts the client uses: GET /system_stats, POST /prompt, GET /history/<id>, GET /view,
POST /upload/image, GET /queue, POST /queue {"delete": [...]}, POST /interrupt (whatever runs, or only
{"prompt_id"} when that one runs) and the websocket /ws?clientId=... (text messages, one preview as a
binary frame, a message split over two frames). Prompts wait in a queue and run one at a time in a worker
thread. Knobs:

  node_s, step_s     time per ordinary node; time per sampler step (progress messages 1..steps)
  websocket          False: /ws is refused (the client then polls /history)
  progress           False: samplers send no progress messages
  skip_steps         a sampler that runs this many fewer steps than asked
  cache              ComfyUI's cache: a node whose inputs and upstream nodes equal the last run's is skipped
                     and reported in execution_cached
  history_delay_s    history appears this long after execution_success (ComfyUI writes it afterwards)
  fail_node          this node fails (execution_error)
  output_key         "gifs" (VideoHelperSuite style) or "images" (core SaveVideo style)
  hang               a prompt runs until it is interrupted (for timeouts)
  api_key            every request, the websocket upgrade included, needs "Authorization: Bearer <key>"

Use it in a with block: ``with FakeComfyUI(ComfyConfig()) as srv: srv.base_url``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import queue
import re
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OUTPUT_CLASSES = ("VHS_VideoCombine", "SaveVideo")
SAMPLER_CLASSES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")


@dataclass
class ComfyConfig:
    node_s: float = 0.002
    step_s: float = 0.002
    websocket: bool = True
    progress: bool = True
    skip_steps: int = 0
    cache: bool = True
    history_delay_s: float = 0.0
    fail_node: str | None = None
    output_key: str = "gifs"
    hang: bool = False
    api_key: str | None = None
    version: str = "0.3.99-fake"
    argv: list | None = None  # the server's launch argv as /system_stats reports it (None: not reported)


def _frame(opcode: int, payload: bytes, fin: bool = True) -> bytes:
    n = len(payload)
    head = bytes([(0x80 if fin else 0) | opcode])
    head += bytes([n]) if n < 126 else bytes([126]) + struct.pack(">H", n) if n < 65536 else \
        bytes([127]) + struct.pack(">Q", n)
    return head + payload


class FakeComfyUI:
    def __init__(self, cfg: ComfyConfig | None = None):
        self.cfg = cfg or ComfyConfig()
        self.history: dict = {}
        self.files: dict = {}
        self.uploads: dict = {}
        self.prompts: list = []  # every graph queued, in order
        self.requests: list = []  # (method, path)
        self.interrupts: list = []  # the prompt_id of each POST /interrupt (None: no id given)
        self.deleted: list = []  # prompt ids POST /queue asked to delete
        self.running: str | None = None
        self._last_sig: dict = {}
        self._clients: dict = {}
        self._pending: list = []  # (prompt_id, graph, client_id) waiting, in order
        self._cv = threading.Condition()
        self._interrupt = threading.Event()
        self._stop = threading.Event()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self.httpd.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._interrupt.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    # ------------------------------------------------------------------ execution

    def send(self, client_id, msg: dict) -> None:
        q = self._clients.get(client_id)
        if q is not None:
            q.put(msg)

    def _signatures(self, graph: dict) -> dict:
        """Each node's input signature including its upstream nodes (like ComfyUI's cache key)."""
        sig: dict = {}

        def of(nid):
            if nid not in sig:
                node = graph[nid]
                ups = [of(str(v[0])) for v in node["inputs"].values() if isinstance(v, list) and len(v) == 2
                       and str(v[0]) in graph]
                sig[nid] = hashlib.sha256(json.dumps([node["class_type"], node["inputs"], ups],
                                                     sort_keys=True).encode()).hexdigest()
            return sig[nid]

        for nid in graph:
            of(nid)
        return sig

    def queue_prompt(self, graph: dict, client: str | None) -> str:
        pid = uuid.uuid4().hex
        with self._cv:
            self.prompts.append(graph)
            self._pending.append((pid, graph, client))
            self._cv.notify_all()
        return pid

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop.is_set():
                    self._cv.wait(0.05)
                if self._stop.is_set():
                    return
                job = self._pending.pop(0)
                self.running = job[0]
                self._interrupt.clear()
            try:
                self._execute(*job)
            finally:
                with self._cv:
                    self.running = None

    @staticmethod
    def _steps(graph: dict, node: dict) -> int:
        """A sampler's steps: its own input, or (SamplerCustomAdvanced) its scheduler's, via the sigmas link."""
        steps = node["inputs"].get("steps")
        sigmas = node["inputs"].get("sigmas")
        if steps is None and isinstance(sigmas, list):
            steps = graph[str(sigmas[0])]["inputs"].get("steps")
        return int(steps or 1)

    def _execute(self, pid: str, graph: dict, client: str) -> None:
        cfg = self.cfg
        messages = [["execution_start", {"prompt_id": pid, "timestamp": int(time.time() * 1000)}]]
        self.send(client, {"type": "execution_start", "data": {"prompt_id": pid}})
        if cfg.hang:
            while not self._interrupt.wait(0.05) and not self._stop.is_set():
                pass
            messages.append(["execution_interrupted", {"prompt_id": pid}])
            self.history[pid] = {"prompt": [0, pid, graph, {}, []], "outputs": {},
                                 "status": {"status_str": "error", "completed": False, "messages": messages}}
            return
        sig = self._signatures(graph)
        cached = [n for n in graph if cfg.cache and self._last_sig.get(n) == sig[n]]
        messages.append(["execution_cached", {"nodes": cached, "prompt_id": pid}])
        self.send(client, {"type": "execution_cached", "data": {"nodes": cached, "prompt_id": pid}})
        outputs = {}
        for nid, node in graph.items():
            if node["class_type"] in OUTPUT_CLASSES:
                name = f"clip_{sig[nid][:12]}.mp4"
                self.files[name] = b"\x00\x00\x00\x18ftypmp42" + hashlib.sha256(sig[nid].encode()).digest() * 8
                outputs[nid] = {cfg.output_key: [{"filename": name, "subfolder": "", "type": "output",
                                                  "format": "video/h264-mp4"}]}
            if nid in cached:
                continue
            self.send(client, {"type": "executing", "data": {"node": nid, "prompt_id": pid}})
            if nid == cfg.fail_node:
                err = {"prompt_id": pid, "node_id": nid, "node_type": node["class_type"],
                       "exception_message": "fake failure", "exception_type": "RuntimeError"}
                messages.append(["execution_error", err])
                self.send(client, {"type": "execution_error", "data": err})
                self.history[pid] = {"prompt": [0, pid, graph, {}, []], "outputs": {},
                                     "status": {"status_str": "error", "completed": False, "messages": messages}}
                self._last_sig = {}
                return
            if node["class_type"] in SAMPLER_CLASSES:
                steps = self._steps(graph, node) - cfg.skip_steps
                for v in range(1, steps + 1):
                    time.sleep(cfg.step_s)
                    if cfg.progress:
                        self.send(client, {"type": "progress",
                                           "data": {"value": v, "max": steps, "prompt_id": pid, "node": nid}})
            else:
                time.sleep(cfg.node_s)
        self._last_sig = sig
        self.send(client, {"type": "executing", "data": {"node": None, "prompt_id": pid}})
        messages.append(["execution_success", {"prompt_id": pid}])
        self.send(client, {"type": "execution_success", "data": {"prompt_id": pid}})
        if cfg.history_delay_s:
            time.sleep(cfg.history_delay_s)
        self.history[pid] = {"prompt": [0, pid, graph, {}, []], "outputs": outputs,
                             "status": {"status_str": "success", "completed": True, "messages": messages}}


def _handler(fake: FakeComfyUI):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, status: int, doc) -> None:
            body = json.dumps(doc).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def _authorized(self) -> bool:
            key = fake.cfg.api_key
            if key and self.headers.get("Authorization") != f"Bearer {key}":
                self._json(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            u = urlsplit(self.path)
            fake.requests.append(("GET", u.path))
            if not self._authorized():
                return
            if u.path == "/system_stats":
                system = {"os": "posix", "comfyui_version": fake.cfg.version, "pytorch_version": "2.0-fake",
                          "python_version": "3.12"}
                if fake.cfg.argv is not None:
                    system["argv"] = list(fake.cfg.argv)
                self._json(200, {"system": system,
                                 "devices": [{"name": "cuda:0 Fake GPU", "type": "cuda", "vram_total": 1}]})
            elif u.path.startswith("/history/"):
                pid = u.path[len("/history/"):]
                self._json(200, {pid: fake.history[pid]} if pid in fake.history else {})
            elif u.path == "/view":
                q = parse_qs(u.query)
                name = (q.get("filename") or [""])[0]
                if name not in fake.files:
                    self._json(404, {"error": "not found"})
                    return
                data = fake.files[name]
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif u.path == "/queue":
                with fake._cv:
                    running = [[0, fake.running, {}, {}, []]] if fake.running else []
                    pending = [[i + 1, pid, {}, {}, []] for i, (pid, _, _) in enumerate(fake._pending)]
                self._json(200, {"queue_running": running, "queue_pending": pending})
            elif u.path == "/ws":
                self._websocket(parse_qs(u.query).get("clientId", [""])[0])
            else:
                self._json(404, {"error": "unknown"})

        def _websocket(self, client: str) -> None:
            key = self.headers.get("Sec-WebSocket-Key")
            if not fake.cfg.websocket or not key or self.headers.get("Upgrade", "").lower() != "websocket":
                self._json(404, {"error": "no websocket"})
                return
            accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
            q: queue.Queue = queue.Queue()
            fake._clients[client] = q  # before the 101: the client may queue a prompt as soon as it has it
            self.wfile.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                              f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
            status = json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}},
                                                            "sid": client}}).encode()
            try:
                # a status message split over two frames, a preview image as a binary frame, and a ping
                self.wfile.write(_frame(1, status[:10], fin=False) + _frame(0, status[10:]))
                self.wfile.write(_frame(2, b"\x00\x00\x00\x01fake-preview") + _frame(9, b"hi"))
                self.wfile.flush()
                while not fake._stop.is_set() and fake._clients.get(client) is q:
                    try:
                        msg = q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    self.wfile.write(_frame(1, json.dumps(msg).encode()))
                    self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True

        def do_POST(self):
            u = urlsplit(self.path)
            fake.requests.append(("POST", u.path))
            body = self._body()
            if not self._authorized():
                return
            if u.path == "/prompt":
                doc = json.loads(body)
                graph = doc.get("prompt")
                if not isinstance(graph, dict) or not all("class_type" in n for n in graph.values()):
                    self._json(400, {"error": {"type": "invalid_prompt", "message": "bad prompt"}, "node_errors": {}})
                    return
                pid = fake.queue_prompt(graph, doc.get("client_id"))
                self._json(200, {"prompt_id": pid, "number": len(fake.prompts), "node_errors": {}})
            elif u.path == "/upload/image":
                m = re.search(r"boundary=(\S+)", self.headers.get("Content-Type", ""))
                parts = body.split(b"--" + m.group(1).encode()) if m else []
                for part in parts:
                    head, _, data = part.partition(b"\r\n\r\n")
                    fn = re.search(rb'filename="([^"]+)"', head)
                    if fn:
                        fake.uploads[fn.group(1).decode()] = data[:-2]  # the part ends with \r\n
                        self._json(200, {"name": fn.group(1).decode(), "subfolder": "", "type": "input"})
                        return
                self._json(400, {"error": "no image"})
            elif u.path == "/interrupt":  # like newer ComfyUI: with a prompt_id, only when that one runs
                pid = (json.loads(body or b"{}") or {}).get("prompt_id")
                fake.interrupts.append(pid)
                with fake._cv:
                    if fake.running and pid in (None, fake.running):
                        fake._interrupt.set()
                self._json(200, {})
            elif u.path == "/queue":
                ids = (json.loads(body or b"{}") or {}).get("delete") or []
                with fake._cv:
                    fake.deleted.extend(ids)
                    fake._pending = [j for j in fake._pending if j[0] not in ids]
                self._json(200, {})
            else:
                self._json(404, {"error": "unknown"})

    return Handler


def ltx_workflow(i2v: bool = False, output_class: str = "VHS_VideoCombine", custom_sampler: bool = False) -> dict:
    """An inferencekit workflow file (as a dict) shaped like an LTX video graph in ComfyUI's API format.

    custom_sampler: SamplerCustomAdvanced with its helper nodes (KSamplerSelect, BasicScheduler,
    RandomNoise bound to the seed, CFGGuider, LTXVConditioning), as LTX templates are built, instead of
    one KSampler. The helpers are constants that ComfyUI reuses on every clip after the first.
    """
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx.safetensors"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "text_encoder.safetensors", "type": "ltxv"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality", "clip": ["2", 0]}},
        "5": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 768, "height": 512, "length": 97,
                                                               "batch_size": 1}},
    }
    if custom_sampler:
        g.update({
            "20": {"class_type": "LTXVConditioning", "inputs": {"positive": ["6", 0], "negative": ["7", 0],
                                                                "frame_rate": 24}},
            "21": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "22": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 30,
                                                              "denoise": 1.0}},
            "23": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
            "24": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["20", 0],
                                                         "negative": ["20", 1], "cfg": 3.0}},
            "3": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["23", 0], "guider": ["24", 0],
                                                                    "sampler": ["21", 0], "sigmas": ["22", 0],
                                                                    "latent_image": ["5", 0]}},
        })
        bindings = {"seed": [["23", "noise_seed"]], "steps": [["22", "steps"]], "guidance_scale": [["24", "cfg"]],
                    "fps": [["9", "frame_rate"], ["20", "frame_rate"]]}
    else:
        g["3"] = {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 30, "cfg": 3.0, "sampler_name": "euler",
                                                       "scheduler": "normal", "denoise": 1.0, "model": ["1", 0],
                                                       "positive": ["6", 0], "negative": ["7", 0],
                                                       "latent_image": ["5", 0]}}
        bindings = {"seed": [["3", "seed"]], "steps": [["3", "steps"]], "guidance_scale": [["3", "cfg"]],
                    "fps": [["9", "frame_rate"]]}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}}
    g["9"] = {"class_type": output_class, "inputs": {"images": ["8", 0], "frame_rate": 24, "format": "video/h264-mp4"}}
    bindings.update({"prompt": [["6", "text"]], "width": [["5", "width"]], "height": [["5", "height"]],
                     "num_frames": [["5", "length"]]})
    if i2v:
        g["12"] = {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}}
        g["5"]["inputs"]["image"] = ["12", 0]
        bindings["image"] = [["12", "image"]]
    return {"format": "inferencekit-comfyui-workflow/1", "workflow": g, "bindings": bindings, "output_node": "9"}
