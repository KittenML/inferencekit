"""A tiny fake OpenAI-compatible engine server for the round-runner tests (stdlib only).

Endpoints: GET /v1/models, GET /health, GET /metrics (Prometheus text), POST /v1/chat/completions
(non-streaming JSON, or SSE with "stream": true). Every chat response carries "x_fake" with this
server's label and its configured latency, so a stub client can prove which server it measured.

Flags simulate the failure modes the runner must survive: --startup-delay, --never-ready,
--exit-early, --ignore-sigterm, --spawn-child (a grandchild in the same process group).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--label", default="fake")
ap.add_argument("--ttft-ms", type=float, default=10.0)
ap.add_argument("--startup-delay", type=float, default=0.0)
ap.add_argument("--never-ready", action="store_true")
ap.add_argument("--exit-early", action="store_true")
ap.add_argument("--ignore-sigterm", action="store_true")
ap.add_argument("--spawn-child", help="write the pid of a sleeping grandchild here")
ap.add_argument("--log-fixture", help="print this file to stdout at startup (a canned engine log)")
ap.add_argument("--env-dump", help="write the names of the environment variables received here")
ap.add_argument("--no-metrics", action="store_true")
ap.add_argument("--request-log", help="append the first message text of every chat request here (JSONL)")
a = ap.parse_args()

if a.env_dump:
    with open(a.env_dump, "w") as f:
        json.dump(sorted(os.environ), f)
if a.log_fixture:
    with open(a.log_fixture) as f:
        sys.stdout.write(f.read())
print(f"fake server {a.label} starting on port {a.port}", flush=True)
if a.exit_early:
    print("ERROR fake failure during startup", flush=True)
    sys.exit(3)
if a.ignore_sigterm:
    signal.signal(signal.SIGTERM, lambda *_: print("ignoring SIGTERM", flush=True))
if a.spawn_child:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(a.spawn_child, "w") as f:
        f.write(str(child.pid))

START = time.monotonic()
REQUESTS = 0
LOCK = threading.Lock()


def ready() -> bool:
    return not a.never_ready and time.monotonic() - START >= a.startup_delay


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the log to what the tests look for
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/v1/models", "/health"):
            if not ready():
                return self._json(503, {"error": "loading"})
            return self._json(200, {"object": "list", "data": [{"id": f"fake-model-{a.label}", "object": "model"}]})
        if self.path == "/metrics" and not a.no_metrics:
            text = ("# HELP fake_requests_total Requests served.\n# TYPE fake_requests_total counter\n"
                    f'fake_requests_total{{model="fake-model-{a.label}"}} {REQUESTS}\n'
                    "# TYPE fake_running gauge\nfake_running 0\n")
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        global REQUESTS
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path != "/v1/chat/completions" or not ready():
            return self._json(404 if ready() else 503, {"error": "no"})
        with LOCK:
            REQUESTS += 1
            if a.request_log:
                first = (req.get("messages") or [{}])[0].get("content")
                with open(a.request_log, "a") as f:
                    f.write(json.dumps({"first": first, "stream": bool(req.get("stream"))}) + "\n")
        time.sleep(a.ttft_ms / 1000.0)
        fake = {"label": a.label, "ttft_ms": a.ttft_ms, "model": req.get("model")}
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for tok in ("Hello", " world"):
                chunk = {"choices": [{"index": 0, "delta": {"content": tok}}], "x_fake": fake}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(200, {"id": "c1", "object": "chat.completion", "x_fake": fake,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"},
                                      "finish_reason": "length"}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})


# The socket opens right away; readiness (--startup-delay, --never-ready) is what /v1/models reports.
ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
