"""A deterministic fake OpenAI-compatible server, for tests and dry runs of the quality pipeline.

It serves a toy "model" whose next-token distribution is a fixed function of the last
two tokens, so everything (greedy text, top-k logprobs, teacher-forced prompt logprobs,
exact KL between two fakes) is reproducible and computable in closed form.

Endpoints: ``GET /v1/models``, ``POST /tokenize`` (vLLM style, ``messages`` or
``prompt``), ``POST /v1/chat/completions`` (``logprobs``/``top_logprobs``, vLLM's
``return_tokens_as_token_ids`` and ``return_token_ids``), ``POST /v1/completions`` with
a token-id prompt and vLLM's ``prompt_logprobs``.

``perturb`` shifts the logits by a fixed pattern: a fake with ``perturb=0`` plays the
reference; one with ``perturb > 0`` plays a lower-precision engine. ``sharpness`` scales
the logits (> 1 gives the peaked next-token distributions of a real LLM). ``max_logprobs``
mimics vLLM's cap (HTTP 400 above it); ``has_tokenize=False`` drops ``/tokenize`` (404), like
servers without vLLM's tokenizer API. ``answers`` maps a substring of the user prompt
to a canned reply (for GSM8K tests). The toy tokenizer reads ``<tN>`` as token N, so the
text of a generation re-tokenizes to its own ids.

``logprob_mode``: "ok"; "ignore" answers without logprobs (an engine with no logprob
support); "refuse" answers HTTP 400 when logprobs are asked for.

Sampling: a chat request with ``temperature > 0`` samples, with ``top_k``, ``top_p`` and
``seed`` applied the usual way (temperature, then top-k, then top-p on the renormalised
rest). ``seeds``: "honour" (a draw is a function of seed, prompt and position), "ignore"
(a fresh draw per request, not reproducible), "fixed" (every seed gives the same draw).
``rng_salt`` changes the random stream (a second engine with a correct sampler but other
random numbers); ``wrong_top_p`` / ``wrong_temperature`` replace the requested top_p / temperature
when sampling (sampler bugs; the logprobs stay right). ``top_p_order="joint"`` samples as SGLang
does: top-p over the whole distribution, then the first top_k tokens. ``batch_shift`` moves the
logits by ``batch_shift * (requests in flight - 1)`` times a fixed pattern, like an engine whose
numerics change with the batch; ``max_inflight`` is the most requests seen in flight at once.

Server facts for concurrency probes: ``server_info`` is served at ``GET /get_server_info``
(SGLang) and ``props`` at ``GET /props`` (llama.cpp); ``slots`` caps how many generation
requests run at once (the rest wait) and ``delay_s`` is how long each one takes.

Usage::

    with serve(FakeLM()) as base_url:
        ...
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLM:
    BOS, ASSISTANT = 1, 2
    _MARK = re.compile(r"<t(\d+)>")

    def __init__(self, vocab: int = 16, perturb: float = 0.0, max_logprobs: int = 20, ids: str = "vllm",
                 answers: dict | None = None, name: str = "fake-model", max_model_len: int = 4096,
                 has_tokenize: bool = True, sharpness: float = 1.0, logprob_mode: str = "ok",
                 seeds: str = "honour", rng_salt: str = "", wrong_top_p: float | None = None,
                 server_info: dict | None = None, props: dict | None = None, slots: int | None = None,
                 delay_s: float = 0.0, wrong_temperature: float | None = None, batch_shift: float = 0.0,
                 top_p_order: str = "top_k_first"):
        self.vocab, self.perturb, self.max_logprobs, self.ids = vocab, perturb, max_logprobs, ids
        self.sharpness = sharpness
        self.has_tokenize = has_tokenize
        self.answers = answers or {}
        self.name, self.max_model_len = name, max_model_len
        self.logprob_mode, self.seeds, self.rng_salt, self.wrong_top_p = logprob_mode, seeds, rng_salt, wrong_top_p
        self.wrong_temperature, self.batch_shift, self.top_p_order = wrong_temperature, batch_shift, top_p_order
        self._inflight = self.max_inflight = 0
        self._tl = threading.local()  # requests in flight when this thread's request arrived (batch_shift)
        self.server_info, self.props, self.delay_s = server_info, props, delay_s
        self._slots = threading.Semaphore(slots) if slots else None
        self._draws = 0
        self._lock = threading.Lock()
        self.requests: list[dict] = []

    # ----------------------------------------------------------------- the toy model
    def logprobs(self, context: list[int]) -> list[float]:
        a = context[-1] if context else 0
        b = context[-2] if len(context) > 1 else 0
        logits = [3.0 * math.sin(1.7 * a + 0.37 * b * j + 0.91 * j + 0.13 * j * j) for j in range(self.vocab)]
        if self.perturb:
            logits = [x + self.perturb * math.cos(2.3 * j + 0.7 * a) for j, x in enumerate(logits)]
        batch = getattr(self._tl, "inflight", 1)
        if self.batch_shift and batch > 1:
            logits = [x + self.batch_shift * (batch - 1) * math.cos(1.3 * j + 0.4 * a) for j, x in enumerate(logits)]
        if self.sharpness != 1.0:
            logits = [self.sharpness * x for x in logits]
        m = max(logits)
        lse = m + math.log(sum(math.exp(x - m) for x in logits))
        return [x - lse for x in logits]

    def tokenize_text(self, text: str) -> list[int]:
        def chars(part: str) -> list[int]:
            return [3 + (ord(c) % (self.vocab - 3)) for c in part]

        ids, pos = [], 0
        for m in self._MARK.finditer(text):
            if int(m.group(1)) < self.vocab:
                ids += chars(text[pos:m.start()]) + [int(m.group(1))]
                pos = m.end()
        return ids + chars(text[pos:])

    def template(self, messages: list[dict]) -> list[int]:
        ids = [self.BOS]
        for m in messages:
            ids += self.tokenize_text(f"{m.get('role')}:{m.get('content')}")
        return ids + [self.ASSISTANT]

    def greedy(self, prompt: list[int], n: int) -> list[int]:
        ctx, out = list(prompt), []
        for _ in range(n):
            lp = self.logprobs(ctx)
            t = max(range(self.vocab), key=lambda j: lp[j])
            out.append(t)
            ctx.append(t)
        return out

    def top(self, lp: list[float], k: int) -> list[int]:
        return sorted(range(self.vocab), key=lambda j: -lp[j])[:k]

    def _rng(self, body: dict, prompt: list[int], step: int) -> random.Random:
        where = f"{','.join(map(str, prompt))}|{step}"
        if self.seeds == "ignore":
            with self._lock:
                self._draws += 1
                return random.Random(f"{self.rng_salt}|draw|{self._draws}")
        if self.seeds == "fixed":
            return random.Random(f"{self.rng_salt}|fixed|{where}")
        return random.Random(f"{self.rng_salt}|{body.get('seed')}|{where}")

    def sample(self, prompt: list[int], n: int, body: dict) -> list[int]:
        """Seeded sampling: temperature, then top-k, then top-p on the renormalised remainder (``joint``: top-p
        over the whole distribution, then the first top_k)."""
        temp = float(body["temperature"] if self.wrong_temperature is None else self.wrong_temperature)
        top_k = int(body.get("top_k") or 0)
        top_p = self.wrong_top_p if self.wrong_top_p is not None else body.get("top_p")
        ctx, out = list(prompt), []
        for step in range(n):
            lp = self.logprobs(ctx)
            m = max(lp)
            w = {j: math.exp((x - m) / temp) for j, x in enumerate(lp)}
            order = sorted(w, key=lambda j: (-w[j], j))
            joint = self.top_p_order == "joint"
            if top_k > 0 and not joint:
                order = order[:top_k]
            if top_p is not None and float(top_p) < 1.0:
                total, kept, acc = sum(w[j] for j in order), [], 0.0
                for j in order:
                    kept.append(j)
                    acc += w[j] / total
                    if acc >= float(top_p):
                        break
                order = kept
            if top_k > 0 and joint:
                order = order[:top_k]
            u = self._rng(body, prompt, step).random() * sum(w[j] for j in order)
            t = order[-1]
            for j in order:
                u -= w[j]
                if u < 0:
                    t = j
                    break
            out.append(t)
            ctx.append(t)
        return out

    # ----------------------------------------------------------------- endpoints
    def _tokstr(self, t: int, as_ids: bool) -> str:
        return f"token_id:{t}" if as_ids else f"<t{t}>"

    def chat(self, body: dict) -> tuple[int, dict]:
        messages = body.get("messages") or []
        text = "\n".join(str(m.get("content")) for m in messages)
        for key, reply in self.answers.items():
            if key in text:
                return 200, {"id": "fake", "object": "chat.completion", "model": self.name,
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": reply}}],
                             "usage": {"prompt_tokens": len(text), "completion_tokens": len(reply.split())}}
        k = int(body.get("top_logprobs") or 0)
        if body.get("logprobs") and self.logprob_mode == "refuse":
            return 400, {"error": {"message": "logprobs are not supported by this server", "code": 400}}
        if body.get("logprobs") and k > self.max_logprobs:
            return 400, {"error": {"message": f"Requested sample logprobs of {k}, which is greater than max "
                                              f"allowed: {self.max_logprobs}", "code": 400}}
        prompt = self.template(messages)
        n = int(body.get("max_tokens") or 16)
        gen = self.sample(prompt, n, body) if float(body.get("temperature") or 0) > 0 else self.greedy(prompt, n)
        as_ids = bool(body.get("return_tokens_as_token_ids")) and self.ids == "vllm"
        choice: dict = {"index": 0, "finish_reason": "length",
                        "message": {"role": "assistant", "content": "".join(f"<t{t}>" for t in gen)}}
        if body.get("logprobs") and self.logprob_mode == "ok":
            content, ctx = [], list(prompt)
            for t in gen:
                lp = self.logprobs(ctx)
                entry = {"token": self._tokstr(t, as_ids), "logprob": lp[t], "bytes": None,
                         "top_logprobs": [{"token": self._tokstr(j, as_ids), "logprob": lp[j], "bytes": None}
                                          for j in self.top(lp, k)]}
                if self.ids == "llamacpp":
                    entry["id"] = t
                    for alt, j in zip(entry["top_logprobs"], self.top(lp, k)):
                        alt["id"] = j
                content.append(entry)
                ctx.append(t)
            choice["logprobs"] = {"content": content}
        resp = {"id": "fake", "object": "chat.completion", "model": self.name, "choices": [choice],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(gen),
                          "total_tokens": len(prompt) + len(gen)}}
        if body.get("return_token_ids") and self.ids == "vllm":
            choice["token_ids"] = gen
            resp["prompt_token_ids"] = prompt
        return 200, resp

    def completions(self, body: dict) -> tuple[int, dict]:
        prompt = body.get("prompt")
        if not (isinstance(prompt, list) and all(isinstance(t, int) for t in prompt)):
            prompt = self.tokenize_text(str(prompt))
        if len(prompt) + int(body.get("max_tokens") or 1) > self.max_model_len:
            return 400, {"error": {"message": "This model's maximum context length is exceeded", "code": 400}}
        k = body.get("prompt_logprobs")
        if k is not None and int(k) > self.max_logprobs:
            return 400, {"error": {"message": f"Requested prompt logprobs of {k}, which is greater than max "
                                              f"allowed: {self.max_logprobs}", "code": 400}}
        choice: dict = {"index": 0, "text": "", "finish_reason": "length"}
        if k is not None:
            plp: list = [None]
            for i in range(1, len(prompt)):
                lp = self.logprobs(prompt[:i])
                ranks = {j: r + 1 for r, j in enumerate(self.top(lp, self.vocab))}
                keep = self.top(lp, int(k)) + [prompt[i]]
                plp.append({str(j): {"logprob": lp[j], "rank": ranks[j], "decoded_token": f"<t{j}>"}
                            for j in keep})
            choice["prompt_logprobs"] = plp
        gen = self.greedy(prompt, int(body.get("max_tokens") or 1))
        choice["text"] = "".join(f"<t{t}>" for t in gen)
        return 200, {"id": "fake", "object": "text_completion", "model": self.name, "choices": [choice],
                     "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(gen)}}

    def generate(self, body: dict) -> tuple[int, dict]:
        """A chat request as the server runs it: wait for a slot, take ``delay_s``, answer."""
        with self._lock:
            self._inflight += 1
            self._tl.inflight = self._inflight
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            return self._generate(body)
        finally:
            with self._lock:
                self._inflight -= 1

    def _generate(self, body: dict) -> tuple[int, dict]:
        if self._slots is None:
            if self.delay_s:
                time.sleep(self.delay_s)
            return self.chat(body)
        with self._slots:
            if self.delay_s:
                time.sleep(self.delay_s)
            return self.chat(body)

    def tokenize(self, body: dict) -> tuple[int, dict]:
        if "messages" in body:
            ids = self.template(body["messages"])
        else:
            ids = self.tokenize_text(str(body.get("prompt", "")))
        return 200, {"tokens": ids, "count": len(ids), "max_model_len": self.max_model_len}


def _handler(lm: FakeLM):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def _send(self, status: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": lm.name, "object": "model"}]})
            elif path == "/get_server_info" and lm.server_info is not None:
                self._send(200, lm.server_info)
            elif path == "/props" and lm.props is not None:
                self._send(200, lm.props)
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            lm.requests.append({"path": self.path, "body": body})
            routes = {"/v1/chat/completions": lm.generate, "/v1/completions": lm.completions}
            if lm.has_tokenize:
                routes["/tokenize"] = lm.tokenize
            fn = routes.get(self.path)
            if fn is None:
                self._send(404, {"error": {"message": "not found"}})
                return
            status, obj = fn(body)
            self._send(status, obj)

    return Handler


class _Server(ThreadingHTTPServer):
    request_queue_size = 128  # many short concurrent connections (sampling histograms)


@contextlib.contextmanager
def serve(lm: FakeLM, host: str = "127.0.0.1"):
    """Run ``lm`` on a free port in a background thread; yields the base URL."""
    server = _Server((host, 0), _handler(lm))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
