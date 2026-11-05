"""A fake OpenAI-compatible streaming server (stdlib asyncio), for testing bench serve without a GPU.

It streams ``max_tokens`` tokens (or stops early at a fake EOS unless ``ignore_eos`` is sent) with a
configured time to first token and per-token delay, so a test can check that the client measures what
the server did. Delays are scheduled against the loop clock (no drift). Knobs:

  ttft_s, itl_s          time to first token; time per token after it (an event of k tokens takes k*itl_s)
  jitter_s               uniform extra delay in [0, jitter_s) per event (seeded)
  tokens_per_event       tokens packed into one SSE event (speculative decoding / coalescing)
  ids                    where token ids go: "choice" (vLLM choices[].token_ids), "logprobs_id" (llama.cpp
                         logprobs.content[].id), "token_id_str" (vLLM "token_id:N"), "none"
  ids_on_request_only    send ids only when the request asks (return_token_ids / logprobs)
  usage                  honour stream_options.include_usage with a final usage chunk
  stop_at                fake EOS after this many tokens unless the request sends ignore_eos
  honour_ignore_eos      False: a server that silently ignores ignore_eos
  fail_rate, fail_mode   "http500" (error response) or "midstream" (connection dropped after 2 tokens)
  max_concurrency        requests beyond this wait in a queue (their TTFT grows)
  ttft_per_active_s      extra TTFT per other request being served (latency rising with load)
  chunked                Transfer-Encoding: chunked (else Connection: close streaming)
  host                   bind address (127.0.0.2 is loopback on Linux but not "local" to the client)
  refuse_after_models    stop listening after answering /v1/models (later connects are refused,
                         like an engine that crashed after the readiness check)
  slow_first_n, slow_first_s   extra TTFT for the first n generation requests (lazy init / JIT)
  pad_ids                extra ids streamed with the last token (ids disagree with usage)

Run it in a background thread: ``with FakeServer(FakeConfig(...)) as srv: srv.base_url``.
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
from dataclasses import dataclass, field


@dataclass
class FakeConfig:
    ttft_s: float = 0.03
    itl_s: float = 0.005
    jitter_s: float = 0.0
    tokens_per_event: int = 1
    ids: str = "choice"
    ids_on_request_only: bool = True
    usage: bool = True
    stop_at: int | None = None
    honour_ignore_eos: bool = True
    fail_rate: float = 0.0
    fail_mode: str = "http500"
    max_concurrency: int = 0
    ttft_per_active_s: float = 0.0
    chunked: bool = True
    role_chunk: bool = True
    host: str = "127.0.0.1"
    refuse_after_models: bool = False
    slow_first_n: int = 0
    slow_first_s: float = 0.0
    pad_ids: int = 0
    seed: int = 0
    models: list = field(default_factory=lambda: ["fake-model"])


class FakeServer:
    def __init__(self, config: FakeConfig | None = None):
        self.config = config or FakeConfig()
        self.requests: list[dict] = []
        self.max_active_seen = 0
        self._active = 0
        self._rng = random.Random(self.config.seed)
        self._loop = None
        self._server = None
        self._thread = None
        self._ready = threading.Event()
        self.port = None

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("fake server did not start")
        return self

    def __exit__(self, *exc):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.port}"

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._sem = asyncio.Semaphore(self.config.max_concurrency) if self.config.max_concurrency else None
        self._server = self._loop.run_until_complete(asyncio.start_server(self._handle, self.config.host, 0))
        self.port = self._server.sockets[0].getsockname()[1]
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._server.close()
            self._loop.close()

    # ------------------------------------------------------------------ HTTP
    async def _handle(self, reader, writer):
        try:
            line = await reader.readline()
            if not line:
                return
            method, path, _ = line.decode("latin-1").split(" ", 2)
            headers = {}
            while True:
                h = await reader.readline()
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            body = await reader.readexactly(int(headers.get("content-length", "0") or 0))
            if method == "GET" and path.endswith("/v1/models"):
                if self.config.refuse_after_models:
                    # Stop listening BEFORE answering: closing after the reply raced with the client's next
                    # connect on a loaded machine, so a request could still be served.
                    self._server.close()
                await self._json(writer, 200, {"object": "list", "data": [
                    {"id": m, "object": "model", "max_model_len": 4096} for m in self.config.models]})
            elif method == "POST" and path.endswith(("/v1/chat/completions", "/v1/completions")):
                req = json.loads(body)
                self.requests.append(req)
                await self._generate(writer, req, chat=path.endswith("/chat/completions"))
            else:
                await self._json(writer, 404, {"error": {"message": f"no route {method} {path}"}})
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _json(self, writer, status, obj):
        data = json.dumps(obj).encode()
        writer.write(f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
                     f"Connection: close\r\n\r\n".encode() + data)
        await writer.drain()

    async def _send_event(self, writer, obj):
        data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
        if self.config.chunked:
            writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        else:
            writer.write(data)
        await writer.drain()

    # ------------------------------------------------------------------ generation
    async def _generate(self, writer, req, chat):
        cfg = self.config
        loop = asyncio.get_running_loop()
        t_arrive = loop.time()
        if cfg.fail_rate and self._rng.random() < cfg.fail_rate and cfg.fail_mode == "http500":
            await self._json(writer, 500, {"error": {"message": "fake overload", "type": "server_error"}})
            return
        fail_midstream = bool(cfg.fail_rate) and cfg.fail_mode == "midstream" and self._rng.random() < cfg.fail_rate
        if self._sem is not None:
            await self._sem.acquire()
        self._active += 1
        self.max_active_seen = max(self.max_active_seen, self._active)
        try:
            await self._stream(writer, req, chat, loop, fail_midstream, t_arrive)
        finally:
            self._active -= 1
            if self._sem is not None:
                self._sem.release()

    def _wants_ids(self, req) -> bool:
        if not self.config.ids_on_request_only:
            return True
        return bool(req.get("return_token_ids") or req.get("logprobs") not in (None, False)
                    or req.get("return_tokens_as_token_ids"))

    async def _stream(self, writer, req, chat, loop, fail_midstream, t_arrive):
        cfg = self.config
        n = int(req.get("max_tokens") or 16)
        if cfg.stop_at and not (req.get("ignore_eos") and cfg.honour_ignore_eos):
            n = min(n, cfg.stop_at)
        hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
        hdr += "Transfer-Encoding: chunked\r\n\r\n" if cfg.chunked else "Connection: close\r\n\r\n"
        writer.write(hdr.encode())
        await writer.drain()
        obj_kind = "chat.completion.chunk" if chat else "text_completion"
        base = {"id": "cmpl-fake", "object": obj_kind, "created": 0, "model": req.get("model")}
        wants_ids = self._wants_ids(req) and cfg.ids != "none"
        # first token: TTFT from arrival (queue wait included) plus load-dependent prefill time
        t_next = loop.time() + cfg.ttft_s + cfg.ttft_per_active_s * max(0, self._active - 1)
        if len(self.requests) <= cfg.slow_first_n:
            t_next += cfg.slow_first_s
        if cfg.role_chunk and chat:
            await self._send_event(writer, {**base, "choices": [{"index": 0, "delta": {"role": "assistant",
                                                                                     "content": ""},
                                                                 "finish_reason": None}]})
        i = 0
        first = True
        while i < n:
            k = 1 if first else min(cfg.tokens_per_event, n - i)
            if first:
                k = min(cfg.tokens_per_event, n)
            if not first:
                t_next += cfg.itl_s * k
            if cfg.jitter_s:
                t_next += self._rng.random() * cfg.jitter_s
            await asyncio.sleep(max(0.0, t_next - loop.time()))
            ids = list(range(1000 + i, 1000 + i + k))
            text = "".join(f" t{x}" for x in ids)
            if i + k >= n and cfg.pad_ids:
                ids += [ids[-1]] * cfg.pad_ids  # a server inflating its own token count
            choice = {"index": 0, "finish_reason": None}
            if chat:
                choice["delta"] = {"content": text}
            else:
                choice["text"] = text
            if wants_ids:
                if cfg.ids == "choice":
                    choice["token_ids"] = ids
                elif cfg.ids == "logprobs_id":
                    choice["logprobs"] = {"content": [{"id": x, "token": f" t{x}", "logprob": -0.1, "bytes": [],
                                                       "top_logprobs": []} for x in ids]}
                elif cfg.ids == "token_id_str":
                    if chat:
                        choice["logprobs"] = {"content": [{"token": f"token_id:{x}", "logprob": -0.1}
                                                          for x in ids]}
                    else:
                        choice["logprobs"] = {"tokens": [f"token_id:{x}" for x in ids], "token_logprobs": [-0.1] * k}
            await self._send_event(writer, {**base, "choices": [choice]})
            i += k
            first = False
            if fail_midstream and i >= 2:
                writer.transport.abort()  # drop the connection mid-stream
                return
        finish = {"index": 0, "finish_reason": "length" if n == int(req.get("max_tokens") or 16) else "stop"}
        finish["delta" if chat else "text"] = {} if chat else ""
        await self._send_event(writer, {**base, "choices": [finish]})
        if cfg.usage and (req.get("stream_options") or {}).get("include_usage"):
            await self._send_event(writer, {**base, "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n}})
        await self._send_event(writer, "[DONE]")
        if cfg.chunked:
            writer.write(b"0\r\n\r\n")
            await writer.drain()
