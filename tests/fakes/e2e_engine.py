#!/usr/bin/env python3
"""A standalone fake OpenAI-compatible engine process for the end-to-end test (tests/test_e2e_fake.py).

It is launched like a real engine (from a run manifest's ``launch`` argv, by ``bench rounds`` or by the
test itself) and serves on ``--port`` until it is killed. Two existing fakes do the work:

  * streamed chat (``"stream": true``: ``bench serve``, ``gates run``) goes to the planted-cheat fake
    (``tests/cheats/fake_engine.py``, cheat ``honest``): one character per token, ids = code points,
    honours ``ignore_eos``, ``return_token_ids``, ``logprobs`` and T>0 sampling;
  * non-streamed chat (``quality trajectory``), ``/v1/completions`` with ``prompt_logprobs`` and
    ``/tokenize`` (``quality score``) go to the quality fake (``inferencekit.quality.fake.FakeLM``), a toy
    model whose logprobs are known in closed form; ``--perturb`` > 0 plays a lower-precision engine.

``--per-token-ms`` / ``--ttft-ms`` set the streamed timing, so one process can play a slower engine.
``--request-log FILE`` appends one JSON line per POST (path, stream flag, ``chat_template_kwargs``) so a
test can check that every engine received the same chat-template settings from every command.
On start it prints the custom-engine resolved-config line that ``bench rounds`` parses.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ (for cheats.fake_engine)

from cheats import fake_engine as fe  # noqa: E402

from inferencekit.quality.fake import FakeLM  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--model", default="fake-model")
    p.add_argument("--ttft-ms", type=float, default=15.0)
    p.add_argument("--per-token-ms", type=float, default=1.5)
    p.add_argument("--perturb", type=float, default=0.0)
    p.add_argument("--request-log")
    a = p.parse_args(argv)

    fe.TTFT_S, fe.PER_TOKEN_S = a.ttft_ms / 1000.0, a.per_token_ms / 1000.0
    streamer = fe.FakeEngine("honest")
    streamer.server.server_close()  # only its request handling is used; this process owns the socket
    lm = FakeLM(perturb=a.perturb, name=a.model, max_model_len=1 << 20)
    log_lock = threading.Lock()

    def log(path: str, body: dict) -> None:
        if not a.request_log:
            return
        row = {"port": a.port, "path": path, "stream": bool(body.get("stream")),
               "chat_template_kwargs": body.get("chat_template_kwargs")}
        with log_lock, open(a.request_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):  # quiet
            pass

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path.endswith("/v1/models"):
                self._json(200, {"object": "list", "data": [{"id": a.model, "object": "model",
                                                             "max_model_len": 1 << 20}]})
            elif path in ("/health", ""):
                self._json(200, {"status": "ok"})
            elif path == "/metrics":
                data = b"# fake engine\nfake_requests_total 0\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "bad json"}})
            path = self.path.split("?")[0].rstrip("/")
            log(path, body)
            if path.endswith("/v1/chat/completions"):
                if body.get("stream"):
                    return streamer.handle(self, body)
                status, obj = lm.chat(body)
            elif path.endswith("/v1/completions"):
                status, obj = lm.completions(body)
            elif path.endswith("/tokenize"):
                status, obj = lm.tokenize(body)
            else:
                status, obj = 404, {"error": {"message": "not found"}}
            self._json(status, obj)

    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.daemon_threads = True
    print(json.dumps({"inferencekit_resolved_config": {"cuda_graphs": False, "max_num_seqs": 1,
                                                       "per_token_ms": a.per_token_ms, "model": a.model}}),
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
