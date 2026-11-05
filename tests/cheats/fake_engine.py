"""Planted-cheat fake engines: tiny stdlib OpenAI-compatible streaming servers.

The fake "model" works on characters: every character is one token (id = code point).
Greedy output is a deterministic function of the prompt:
  * an echo prompt (the last block between one of the gates' delimiter pairs, e.g. <<< >>>,
    [[ ]], { { } }, « », ⟦ ⟧, BEGIN: :END) is answered with that block's text;
  * anything else gets pseudo-random words seeded by the prompt's hash, with a natural
    end-of-sequence after 24-71 tokens;
  * ignore_eos keeps generating filler until max_tokens.
T>0 sampling changes ~30% of positions, seeded by (seed, prompt) when a seed is given.
Logprobs: the greedy token has p=0.6, the alternatives p=0.3/2^i.
Timing: TTFT ~ 15 ms + prompt length, ~1.5 ms per token, all with real sleeps.

Cheats (``FakeEngine(cheat=...)``):
  honest       none
  cache        exact repeats replayed instantly from a cache; near-duplicate prompts
               (character-trigram Jaccard >= 0.8) answered with the cached text
  truncate     stops at half the requested length but reports the full length in usage
  pruned       vocabulary pruned to ASCII: every non-ASCII character becomes "?"
  notemp       ignores temperature and seed (always greedy)
  spec_diverge an unverified draft: on ~half of the prompts the output changes from token 3
  spec_late    a draft that drifts late: every output changes from token 40 on
  refuser      a pruned vocabulary that hides: HTTP 500 for any non-ASCII prompt, HTTP 400 for
               ignore_eos or top_logprobs requests (everything else honest)
  no_eos       honest, but ignores ignore_eos (stops at EOS with finish_reason "stop")

Honest quirks the gates must not fail:
  confident    a confident model: on a share of the sampled prompts (``confident=``, spread
               evenly over the prompts in the order they arrive) T=1 returns the greedy text
  sglang_like  special tokens are counted in usage but never streamed: the EOS that ends a
               reply, and under ignore_eos 1-4 "<eos>" tokens generated right after the
               natural EOS (no chunk, no logprob entry). With ``skip_special=True`` a request
               with skip_special_tokens: false gets those "<eos>" tokens streamed as text.
  nonce_slip   copies an echoed string that holds non-ASCII text but drops its last
               character (the model slips on the random ASCII nonce at the end)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = ("alpha", "bravo", "cargo", "delta", "ember", "fable", "gamma", "harbor", "ivory", "jungle", "kettle",
         "lumen", "meadow", "nectar", "orbit", "pepper", "quartz", "river", "saddle", "timber", "umber", "velvet")
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
TTFT_S = 0.015
PER_TOKEN_S = 0.0015


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def _prompt(body: dict) -> str:
    return "\n".join(str(m.get("content", "")) for m in body.get("messages", []))


DELIMS = (("<<<", ">>>"), ("[[", "]]"), ("{{", "}}"), ("«", "»"), ("⟦", "⟧"), ("BEGIN:", ":END"))


def echo_block(prompt: str) -> str | None:
    """Text of the last delimited block in the prompt (the payload of an echo probe)."""
    best = None
    for o, c in DELIMS:
        for m in re.finditer(re.escape(o) + "(.*?)" + re.escape(c), prompt, re.S):
            if best is None or m.start() > best[0]:
                best = (m.start(), m.group(1))
    return best[1] if best else None


def greedy_text(prompt: str, length: int) -> tuple[str, int]:
    """(text of at least ``length`` chars, natural EOS position)."""
    block = echo_block(prompt)
    rng = random.Random(_seed("greedy", prompt))
    if block is not None:
        head = block
        eos = len(head)
    else:
        head = ""
        eos = 24 + _seed("eos", prompt) % 48
    text = head
    while len(text) < max(length, eos):
        text += (" " if text else "") + rng.choice(WORDS)
    return text, eos


def _trigrams(s: str) -> set:
    return {s[i:i + 3] for i in range(max(1, len(s) - 2))}


class FakeEngine:
    def __init__(self, cheat: str = "honest", token_ids: bool = True, logprobs: bool = True,
                 confident: float = 0.5, skip_special: bool = False):
        self.cheat, self.token_ids, self.logprobs = cheat, token_ids, logprobs
        self.share, self.skip_special = confident, skip_special
        self.cache: dict[str, list[str]] = {}
        self.sure: dict[str, bool] = {}  # confident mode: sampled prompt -> answers greedily
        self.lock = threading.Lock()
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                if self.path.rstrip("/").endswith("/v1/models"):
                    self._json(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self._json(400, {"error": "bad json"})
                if not self.path.rstrip("/").endswith("/v1/chat/completions"):
                    return self._json(404, {"error": "not found"})
                engine.handle(self, body)

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    # ------------------------------------------------------------------ the "model"
    def _confident(self, prompt: str) -> bool:
        """Confident mode: the i-th new sampled prompt answers greedily when floor((i+1)*share) steps up,
        so exactly floor(n * share) of the first n prompts do, whatever order they arrive in."""
        with self.lock:
            if prompt not in self.sure:
                i = len(self.sure)
                self.sure[prompt] = math.floor((i + 1) * self.share) > math.floor(i * self.share)
            return self.sure[prompt]

    def generate(self, body: dict) -> tuple[list[str], list[list[tuple[str, float]]], str, bool, set, int]:
        """(tokens, top-k per position, finish_reason, instant, special positions, uncounted EOS)."""
        prompt = _prompt(body)
        max_tokens = int(body.get("max_tokens") or 16)
        ignore_eos = bool(body.get("ignore_eos"))
        temp = float(body.get("temperature") or 0.0)
        text, eos = greedy_text(prompt, max_tokens)
        block = echo_block(prompt)
        if self.cheat == "nonce_slip" and block and any(ord(c) > 127 for c in block):
            eos -= 1  # stops one character short of the copied string
        if self.cheat == "no_eos":
            ignore_eos = False
        n = max_tokens if ignore_eos else min(eos, max_tokens)
        finish = "length" if (ignore_eos or eos >= max_tokens) else "stop"
        toks = list(text[:n])
        special: set = set()
        eos_tokens = 0  # an EOS that ends the reply: counted in usage, never streamed
        if self.cheat == "sglang_like":
            if ignore_eos and eos < n:
                k = min(1 + _seed("special", prompt) % 4, n - eos)
                toks[eos:eos + k] = ["<eos>"] * k
                special = set(range(eos, eos + k))
            elif finish == "stop":
                eos_tokens = 1
        greedy = list(toks)
        if self.cheat == "spec_diverge" and _seed("spec", prompt) % 2 == 0 and len(toks) > 4:
            toks[3:] = ["#" if c != "#" else "%" for c in toks[3:]]
            greedy = list(toks)
        if self.cheat == "spec_late" and len(toks) > 40:
            toks[40:] = ["#" if c != "#" else "%" for c in toks[40:]]
            greedy = list(toks)
        sampled = temp > 0 and self.cheat != "notemp" and not (self.cheat == "confident" and self._confident(prompt))
        if sampled:
            seed = body.get("seed")
            rng = random.Random(_seed("sample", seed, prompt)) if seed is not None else random.Random(os.urandom(8))
            toks = [rng.choice(ALPHABET) if rng.random() < 0.3 else c for c in toks]
        if self.cheat == "pruned":
            toks = [c if ord(c) < 128 else "?" for c in toks]
            greedy = [c if ord(c) < 128 else "?" for c in greedy]
        tops = []
        for g in greedy:
            alts = [c for c in ALPHABET if c != g][:4]
            tops.append([(g, math.log(0.6))] + [(c, math.log(0.3 / 2 ** (i + 1))) for i, c in enumerate(alts)])
        instant = False
        if self.cheat == "cache":
            with self.lock:
                if prompt in self.cache:
                    toks, instant = list(self.cache[prompt]), True
                else:
                    tri = _trigrams(prompt)
                    near = [k for k in self.cache if len(tri & _trigrams(k)) / max(1, len(tri | _trigrams(k))) >= 0.8]
                    if near:
                        toks = list(self.cache[near[0]])
                    self.cache[prompt] = list(toks)
        return toks, tops, finish, instant, special, eos_tokens

    def refuse(self, h: BaseHTTPRequestHandler, body: dict) -> bool:
        if self.cheat != "refuser":
            return False
        code = None
        if any(ord(ch) > 127 for ch in _prompt(body)):
            code, msg = 500, "internal error"
        elif body.get("ignore_eos") or body.get("top_logprobs"):
            code, msg = 400, "unsupported parameter"
        if code is None:
            return False
        data = json.dumps({"error": {"message": msg}}).encode()
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)
        return True

    def handle(self, h: BaseHTTPRequestHandler, body: dict) -> None:
        if self.refuse(h, body):
            return
        toks, tops, finish, instant, special, eos_tokens = self.generate(body)
        want_ids = bool(body.get("return_token_ids")) and self.token_ids
        want_lp = bool(body.get("logprobs")) and self.logprobs
        k = int(body.get("top_logprobs") or 0)
        reported = len(toks) + eos_tokens
        if self.skip_special and body.get("skip_special_tokens") is False:
            special = set()  # asked to keep special tokens: stream them like any other
        if self.cheat == "truncate":
            reported = len(toks)
            toks = toks[: max(1, len(toks) // 2)]
            finish = "length" if body.get("ignore_eos") else finish
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()

        def send(obj):
            h.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            h.wfile.flush()

        if not instant:
            time.sleep(TTFT_S + len(_prompt(body)) * 2e-7)
        for i, c in enumerate(toks):
            if i in special:
                continue  # generated and counted, but no chunk is sent for it
            ch = {"index": 0, "delta": {"content": c}, "finish_reason": None}
            if want_ids:
                ch["token_ids"] = [ord(c) if len(c) == 1 else 1]
            if want_lp:
                top = tops[i] if i < len(tops) else [(c, math.log(0.6))]
                lp = next((v for t, v in top if t == c), math.log(0.01))
                ch["logprobs"] = {"content": [{"token": c, "logprob": lp,
                                               "top_logprobs": [{"token": t, "logprob": v} for t, v in top[:k]]}]}
            send({"object": "chat.completion.chunk", "model": "fake-model", "choices": [ch]})
            if not instant:
                time.sleep(PER_TOKEN_S)
        send({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        if (body.get("stream_options") or {}).get("include_usage"):
            send({"object": "chat.completion.chunk", "choices": [],
                  "usage": {"prompt_tokens": len(_prompt(body)), "completion_tokens": reported,
                            "total_tokens": len(_prompt(body)) + reported}})
        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()
