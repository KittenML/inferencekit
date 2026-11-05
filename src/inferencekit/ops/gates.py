"""gates run / gates spec-exact: anti-cheat gates for a speed claim.

A speed number is only worth publishing if the engine did the work it claims. These
gates probe a live OpenAI-compatible endpoint (``/v1/chat/completions``, streamed) and
check a ``bench serve`` run directory. Each gate ends PASS, FAIL, UNPROVEN or NOT RUN
with the requests and numbers that decided it:

  caching          replaying prompts must not be >10% faster (TTFT or decode) than the
                   first time (median over pairs, and in >= 2/3 of the pairs), and a
                   prompt changed by one character must not get the old answer back
                   (output / "semantic" caching).
  token_count      usage.completion_tokens == streamed token ids (or logprob entries)
                   == re-tokenized text (when a tokenizer is given). A reply cut at
                   max_tokens whose logprob stream is shorter than usage is UNPROVEN, not
                   FAIL: special tokens (end of turn, padding) can be counted but not streamed;
                   re-tokenized text that covers usage settles it. Copied strings (rare,
                   echo) and text after EOS are not held to the re-tokenized count.
  output_length    with ignore_eos the output is exactly max_tokens long; in a run dir,
                   per-prompt lengths are within +/-10% of the reference engine's.
  unique_prompts   (run dir) the load schedule never repeats a prompt inside a level.
  spec_exact       greedy outputs with speculation (A) match the engine without it (B);
                   early divergences may not exceed the A/A floor of B run twice.
  hidden_probes    T=1 requests (top_p 1, top_k off) must not equal the greedy output on
                   every prompt (temperature honoured; seeds reported); top-k logprobs must
                   be finite, normalised, sorted, and the greedy token must be the argmax.
  rare_tokens      CJK, emoji, code identifiers, long numbers and uncommon Unicode must
                   be reproduced verbatim (catches pruned vocabularies / logit masks); a
                   slip in the random ASCII nonces around a string is recorded, not a miss.

The live probes run as one randomly interleaved stream: each repeat (and each perturbed
prompt) follows its first-seen request a few requests later, mixed with every other probe,
so an engine cannot tell a probe by its position and slow drift over the run affects
both members of a pair alike. Nothing about a probe is predictable from this code: the
seed is drawn fresh for every run (and recorded), every probe prompt carries a random
tag, the echo instruction, its delimiters and padding are drawn from pools, and the rare
strings are sampled from a pool and generated at random around a random nonce. For an
evaluation, run the probes on the sealed prompt set (``--prompts sealed.jsonl``) so not
even the underlying prompts were public.

Refusing a probe never helps: a probe that gets an HTTP error or a broken stream is
evidence. A refused rare string counts as a miss (unless the reference engine refuses it
too); a refused cache, count, ignore_eos, sampling or logprob probe makes its gate at
least UNPROVEN, with the status and error listed. Only when the endpoint answers no
probe at all are the live gates NOT RUN. A gate that the inputs asked for (or allowed)
and that ends NOT RUN makes the record's overall verdict UNPROVEN, never PASS. The
optional extensions the probes add (``Client.OPTIONAL``: token ids, special tokens in the
stream, top_p / top_k on the samples) are the exception: a server that rejects one with a
400 gets the request again without it, and the reply records the dropped field.

Limitation: the hidden probes run in their own sequential stream (concurrency 1), not
inside the ``bench serve`` load, so an engine that behaves only when idle is not caught
by them; ``probe_bodies()`` returns the probe requests for a load runner to mix in.

The SSE client here is deliberately small and private to the gates (the ``bench serve``
client is the timing reference). Times are ``time.perf_counter_ns`` on the client.

Run directory layout read by the static checks (as bench serve writes it):
  outputs/<engine>/<bucket>/<id>.json   {"text", "token_ids", "finish_reason", "usage", "error"?, ...}
  inputs/<engine>/<bucket>/<id>.json    the request body (max_tokens, ignore_eos / min_tokens);
  inputs/<bucket>/<id>.json             else the engine-independent copy
  load mode (bench serve): inputs/<engine>/load-<level>/<bucket>/<id>[~k].json request bodies and
      outputs/<engine>/load-<level>/<bucket>/<id>[~k].json ({"id", "salt", "level", ...});
  or any ``*schedule*.jsonl`` / ``*requests*.jsonl`` with one load request per line:
      {"level": <concurrency or rate>, "prompt_id"|"id": "...", "salt"?: "...", "prompt_sha256"?: "...",
       "engine"?: "..."}
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import secrets
import statistics
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit import prompts as ps
from inferencekit.ops.correctness import first_divergence
from inferencekit.ops.intake import apply_pin
from inferencekit.registry import Arg, OpError, operation
from inferencekit.stats import combine_claims

GATES = ("caching", "token_count", "output_length", "unique_prompts", "spec_exact", "hidden_probes", "rare_tokens")
LIVE = {"caching", "token_count", "output_length", "spec_exact", "hidden_probes", "rare_tokens"}
STATIC = {"token_count", "output_length", "unique_prompts"}
# "gates" is a v2 evidence kind; on a v1-only store the record falls back to "correctness".
KIND = "gates" if "gates" in ev.KINDS else "correctness"

# Strings a model must copy verbatim, by the part of the vocabulary a pruned vocabulary or a
# logit mask would drop first. Each run samples one string per class and wraps it in random
# ASCII nonces; "code" and "number" are generated fresh every run.
RARE_POOL = {
    "cjk": ("東京特許許可局許可局長", "鬱蒼とした森の奥で", "魑魅魍魎が跋扈する", "齟齬を来す麒麟と鸚鵡", "饕餮纹的青铜鼎",
            "薔薇の棘と檸檬の皮"),
    "hangul": ("한국어 맞춤법 검사기", "훈민정음 해례본", "닭볶음탕과 떡볶이"),
    "emoji": ("🦀🧬🪐🫧", "🦩🪼🫎🪿", "🧿🪬🫀🫁", "🛸🌌🔭🧪", "🦔🪵🍄🌰"),
    "emoji_zwj": ("👩🏽‍🔬 👨‍👩‍👧‍👦", "🧑🏿‍🚀 👩‍🌾", "🧑‍🤝‍🧑 👨🏻‍💻"),
    "unicode": ("ꙮ ᚠᚢᚦᚨᚱᚲ Ⱎ ꓘ", "ᚠᛇᚻ ᛒᛦᚦ ᚹᛁᚾ", "Ꭰ ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ", "ꝏ ȸ ȹ ʬ ʭ"),
    "devanagari": ("नमस्ते दुनिया", "संस्कृतम् भाषा", "क्षत्रिय ज्ञान"),
    "arabic": ("مرحبا بالعالم", "اللغة العربية الفصحى", "خوارزمية الترتيب"),
    "math": ("Ελληνικά ∮ E⋅da = Q, ∀x∈ℝ", "∑ᵢ xᵢ² ≤ ∏ⱼ ‖vⱼ‖", "ℵ₀ < 𝔠 ⊆ ℘(ℕ)"),
}
RARE_STRINGS = tuple((cls, s) for cls, xs in RARE_POOL.items() for s in xs)  # the fixed part of the pool
# Echo prompts: instruction phrasings x delimiter pairs x padding, all drawn per probe pair.
ECHO_TEMPLATES = (
    "Repeat the text between {o} and {c} exactly, character for character. Output only that text.\n{o}{s}{c}",
    "Copy the string inside {o} {c} verbatim and reply with nothing else.\n{o}{s}{c}",
    "Please echo back exactly what appears between {o} and {c}, keeping every character unchanged.\n\n{o}{s}{c}",
    "Echo task: return the characters enclosed by {o} and {c}, unchanged, with no commentary. {o}{s}{c}",
    "Here is a string: {o}{s}{c}\nWrite it out again exactly as given and output only that string.",
)
ECHO_DELIMS = (("<<<", ">>>"), ("[[", "]]"), ("{{", "}}"), ("«", "»"), ("⟦", "⟧"), ("BEGIN:", ":END"))
PADDING = ("", "Reference {n}.", "(ticket {n})", "Request id {n}. Thanks.", "Batch {n}: one item.", "Note {n}.")
INSTANT = 1e6  # "speedup" recorded when a repeat streams all its tokens in one instant
NONCE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"


# --------------------------------------------------------------------------- SSE client

@dataclass
class Reply:
    """One streamed completion, as the client saw it."""

    status: int = 0
    error: str | None = None
    text: str = ""
    reasoning: str = ""  # delta.reasoning_content of reasoning models (counted as output tokens)
    token_ids: list | None = None
    lp: list | None = None  # logprobs.content entries, one per token
    finish_reason: str | None = None
    usage: dict | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    t_first_ns: int | None = None
    t_last_ns: int | None = None
    chunks: int = 0
    dropped: list = field(default_factory=list)  # request fields the server rejected

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def full_text(self) -> str:
        """Everything generated: reasoning (if streamed separately) then the answer."""
        return self.reasoning + self.text

    def streamed_count(self) -> tuple[int | None, str | None]:
        """Tokens counted from the stream itself: token ids, else logprob entries."""
        if self.token_ids:
            return len(self.token_ids), "token_ids"
        if self.lp:
            return len(self.lp), "logprobs"
        return None, None

    def usage_count(self) -> int | None:
        u = self.usage or {}
        v = u.get("completion_tokens")
        return v if isinstance(v, int) else None

    def decode_tok_s(self) -> float | None:
        n, _ = self.streamed_count()
        n = n if n is not None else self.usage_count()
        if not n or n < 2 or self.t_first_ns is None or self.t_last_ns is None or self.t_last_ns <= self.t_first_ns:
            return None
        return (n - 1) / ((self.t_last_ns - self.t_first_ns) / 1e9)

    def summary(self) -> dict:
        n, how = self.streamed_count()
        return {"status": self.status, "error": self.error, "text": self.text[:2000],
                "reasoning": self.reasoning[:2000] or None, "finish_reason": self.finish_reason,
                "usage_completion_tokens": self.usage_count(), "streamed_tokens": n, "streamed_method": how,
                "ttft_ms": self.ttft_ms, "total_ms": self.total_ms, "decode_tok_s": self.decode_tok_s(),
                "chunks": self.chunks, "dropped_fields": self.dropped}


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")


def models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    for suffix in ("/chat/completions",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base + ("/models" if base.endswith("/v1") else "/v1/models")


class Client:
    """Minimal streaming client for an OpenAI-compatible chat endpoint."""

    # Extensions the probes add. A 400 on a request carrying ones the server has not yet accepted is
    # retried without them (top_k alone first, see SUSPECT_FIRST); a field refused that way is left out
    # of later requests too.
    OPTIONAL = ("return_token_ids", "skip_special_tokens", "top_p", "top_k")

    def __init__(self, base_url: str, model: str | None, api_key: str | None = None, timeout: float = 120.0,
                 token_ids: bool = True, extra: dict | None = None):
        self.base_url, self.api_key, self.timeout = base_url, api_key, timeout
        self.ask_ids = token_ids
        self.extra = dict(extra or {})
        self.accepted: set[str] = set()  # optional fields seen in a request that got a 200
        self.rejected: set[str] = set()  # optional fields the server refused with a 400
        self.model = model or self._first_model()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _first_model(self) -> str:
        req = urllib.request.Request(models_url(self.base_url), headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - user-given endpoint
                data = json.loads(r.read().decode("utf-8"))
            return data["data"][0]["id"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
            raise OpError(f"cannot list models at {models_url(self.base_url)} ({e}); pass --model") from None

    def body(self, messages, max_tokens, temperature=0.0, seed=None, ignore_eos=False, logprobs=False,
             top_logprobs=0, full_vocab=False, keep_special=False) -> dict:
        """One request. ``full_vocab``: sample from the whole distribution (top_p 1, top_k off), not
        the model's default truncation. ``keep_special``: stream special tokens too, so every counted
        token shows up in the stream (vLLM and SGLang ``skip_special_tokens``)."""
        b = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
             "stream": True, "stream_options": {"include_usage": True}}
        if seed is not None:
            b["seed"] = seed
        if ignore_eos:
            b["ignore_eos"] = True
        if logprobs:
            b["logprobs"] = True
            if top_logprobs:
                b["top_logprobs"] = top_logprobs
        if full_vocab:
            b["top_p"], b["top_k"] = 1.0, -1
        if keep_special:
            b["skip_special_tokens"] = False
        if self.ask_ids:
            b["return_token_ids"] = True
        for k, v in self.extra.items():
            b.setdefault(k, v)
        return b

    def _ours(self, body: dict) -> list[str]:
        """Optional fields in ``body`` that the probes set (a value from --extra-body is the user's)."""
        return [k for k in self.OPTIONAL if k in body and (k not in self.extra or self.extra[k] != body[k])]

    def _without(self, body: dict, keys: list[str]) -> dict:
        out = {k: v for k, v in body.items() if k not in keys}
        out.update({k: self.extra[k] for k in keys if k in self.extra})  # the user's own value, if any
        return out

    # Dropped alone first on a 400, so that refusing it does not take the other fields with it: top_k is
    # the one servers refuse most (the OpenAI API has no top_k; some servers want top_k > 0).
    SUSPECT_FIRST = ("top_k",)

    def send(self, body: dict) -> Reply:
        known = [k for k in self._ours(body) if k in self.rejected]  # refused before: do not ask again
        if known:
            body = self._without(body, known)
        reply = self._send(body)
        # Only fields this server has not already taken can be the reason for a 400.
        unsure = [k for k in self._ours(body) if k not in self.accepted] if reply.status == 400 else []
        tries = [[k] for k in self.SUSPECT_FIRST if k in unsure and len(unsure) > 1] + ([unsure] if unsure else [])
        drop: list[str] = []
        for drop in tries:
            reply = self._send(self._without(body, drop))
            if reply.status != 400:
                break
        if drop and reply.ok:
            body = self._without(body, drop)
            self.rejected.update(drop)
            if "return_token_ids" in drop:
                self.ask_ids = False
        if reply.ok:
            self.accepted.update(self._ours(body))
        reply.dropped = known + drop
        return reply

    def _send(self, body: dict) -> Reply:
        rep = Reply()
        req = urllib.request.Request(chat_url(self.base_url), data=json.dumps(body).encode("utf-8"),
                                     headers=self._headers(), method="POST")
        t0 = time.perf_counter_ns()
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310 - user-given endpoint
        except urllib.error.HTTPError as e:
            rep.status = e.code
            rep.error = e.read()[:500].decode("utf-8", "replace")
            return rep
        except (urllib.error.URLError, OSError) as e:
            rep.error = f"connection failed: {e}"
            return rep
        rep.status = resp.status
        ids, lps = [], []
        try:
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    now = time.perf_counter_ns()
                    rep.chunks += 1
                    if chunk.get("usage"):
                        rep.usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        piece = delta.get("content") or ""
                        think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        tids = ch.get("token_ids") or delta.get("token_ids") or []
                        lpc = ((ch.get("logprobs") or {}).get("content")) or []
                        if piece or think or tids or lpc:
                            if rep.t_first_ns is None:
                                rep.t_first_ns = now
                                rep.ttft_ms = (now - t0) / 1e6
                            rep.t_last_ns = now
                        rep.text += piece
                        rep.reasoning += think
                        ids.extend(tids)
                        lps.extend(lpc)
                        if ch.get("finish_reason"):
                            rep.finish_reason = ch["finish_reason"]
        except (OSError, ValueError) as e:
            rep.error = f"stream broke: {e}"
        rep.total_ms = (time.perf_counter_ns() - t0) / 1e6
        rep.token_ids = ids or None
        rep.lp = lps or None
        return rep


# --------------------------------------------------------------------------- helpers

def load_tokenizer(name: str | None):
    """A ``text -> token count`` function from a Hugging Face tokenizer, or None."""
    if not name:
        return None
    try:
        from transformers import AutoTokenizer  # optional; not a dependency
    except ImportError:
        return None
    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def nonce(rng: random.Random, n: int = 16) -> str:
    return "".join(rng.choice(NONCE_ALPHABET) for _ in range(n))


def perturb(s: str, rng: random.Random) -> str:
    """The same string with exactly one character changed."""
    i = rng.randrange(len(s))
    choices = [c for c in NONCE_ALPHABET if c != s[i]]
    return s[:i] + rng.choice(choices) + s[i + 1:]


def echo_style(rng: random.Random) -> tuple[str, tuple[str, str], str, bool]:
    """(template, delimiters, padding, padding first?) for one echo probe or pair."""
    pad = rng.choice(PADDING).format(n=nonce(rng, 6))
    return rng.choice(ECHO_TEMPLATES), rng.choice(ECHO_DELIMS), pad, rng.random() < 0.5


def echo_prompt(s: str, style) -> str:
    tmpl, (o, c), pad, first = style
    body = tmpl.format(o=o, c=c, s=s)
    if not pad:
        return body
    return f"{pad}\n{body}" if first else f"{body}\n{pad}"


def rare_targets(rng: random.Random) -> list[tuple[str, str]]:
    """One string per class from the pool plus fresh code / number strings, each between random nonces."""
    out = [(cls, rng.choice(xs)) for cls, xs in RARE_POOL.items()]
    ident = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
    out.append(("code", f"__{ident}_{nonce(rng, 4)}__ fn_{nonce(rng, 7)} {ident.capitalize()}Handler"
                        f"V{rng.randrange(2, 99)}"))
    out.append(("number", "".join(rng.choice("0123456789") for _ in range(rng.randrange(30, 46)))))
    out.append(("number", f"0x{rng.getrandbits(64):016x} {rng.randrange(1, 10)}.{rng.randrange(10**8):08d}e-"
                          f"{rng.randrange(10, 40)}"))
    return [(cls, f"{nonce(rng, 4)} {t} {nonce(rng, 4)}") for cls, t in out]


def rare_core(target: str) -> str:
    """The rare string itself, without the random ASCII nonce on each side ("<nonce> <string> <nonce>").

    The nonces only make a probe unpredictable. A pruned vocabulary or a logit mask loses the rare
    characters, never plain ASCII letters, so a slip in a nonce says nothing about the vocabulary."""
    head, sep, rest = target.partition(" ")
    core, sep2, _ = rest.rpartition(" ")
    return core if sep and sep2 and core else target


def tag_prompt(text: str, rng: random.Random) -> str:
    """The prompt with a random tag, so a probe is never a prompt seen before (dev set or not)."""
    t = nonce(rng, 8)
    forms = (f"[{t}] {text}", f"{text}\n\n(ref {t})", f"Request {t}:\n{text}", f"{text} #{t}")
    return rng.choice(forms)


def reproduced(target: str, text: str) -> bool:
    """Verbatim (or Unicode-NFC-equivalent) occurrence of target in text."""
    return target in text or unicodedata.normalize("NFC", target) in unicodedata.normalize("NFC", text)


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _gate(name, verdict, reason, metrics=None, details=None, limitations=None) -> dict:
    return {"gate": name, "verdict": verdict, "reason": reason, "metrics": metrics or {},
            "details": details or [], "limitations": limitations or []}


def _failed(jobs: list) -> list[dict]:
    """Probes that got no usable reply (HTTP error, broken stream, never sent): evidence, not noise."""
    out = []
    for j in jobs:
        r = j.reply
        if r is None or not r.ok:
            out.append({"kind": j.kind, "key": j.key, "status": r.status if r else None,
                        "error": ((r.error if r else "not sent") or "")[:300]})
    return out


def _fail_note(failed: list[dict], what: str = "probe") -> str:
    by: dict[str, int] = {}
    for f in failed:
        k = f"HTTP {f['status']}" if f["status"] else "no response"
        by[k] = by.get(k, 0) + 1
    first = next((f["error"] for f in failed if f["error"]), "")
    return (f"{len(failed)} {what}(s) failed ({', '.join(f'{k} x{v}' for k, v in sorted(by.items()))}"
            + (f"; first error: {first[:160]}" if first else "") + ")")


def alive(jobs: dict) -> tuple[bool, str]:
    """(the endpoint answered at least one probe, first error otherwise)."""
    allj = [j for js in jobs.values() for j in js if not j.kind.endswith("reference")]
    if any(j.reply is not None and j.reply.ok for j in allj):
        return True, ""
    failed = _failed(allj)
    return False, _fail_note(failed) if failed else "no probe was sent"


def _sequence(rep: Reply) -> tuple[list, str]:
    """What to compare token by token: ids, else logprob token strings, else characters."""
    if rep.token_ids:
        return list(rep.token_ids), "token_ids"
    if rep.lp:
        return [e.get("token") for e in rep.lp], "logprob_tokens"
    return list(rep.full_text), "characters"


# --------------------------------------------------------------------------- live probe plan

@dataclass
class Job:
    kind: str
    key: str
    body: dict
    meta: dict = field(default_factory=dict)
    reply: Reply | None = None


class Prober:
    def __init__(self, client: Client, max_tokens: int, seed: int, log: list):
        self.c, self.max_tokens, self.rng, self.log = client, max_tokens, random.Random(seed), log

    def job(self, kind, key, text, **kw) -> Job:
        max_tokens = kw.pop("max_tokens", self.max_tokens)
        meta = kw.pop("meta", {})
        return Job(kind, key, self.c.body(_msgs(text), max_tokens, **kw), meta)

    def run(self, jobs: list[Job], shuffle: bool = True) -> None:
        order = list(jobs)
        if shuffle:
            self.rng.shuffle(order)
        for j in order:
            j.reply = self.c.send(j.body)
            self.log.append({"kind": j.kind, "key": j.key, "meta": j.meta,
                             "request": {k: v for k, v in j.body.items() if k != "messages"},
                             "prompt_sha256": hashlib.sha256(json.dumps(j.body["messages"], ensure_ascii=False)
                                                             .encode("utf-8")).hexdigest(),
                             "reply": j.reply.summary()})


def stagger(first: list, then: list) -> list:
    """[f0, f1, t0, f2, t1, ...]: each follow-up shortly after its first request, so slow drift
    (clocks, other tenants) hits a first-seen request and its repeat alike."""
    out = []
    for i, f in enumerate(first):
        out.append(f)
        if i >= 1:
            out.append(then[i - 1])
    out += then[len(first) - 1:] if first else []
    return out


def interleave(chains: list[list], rng: random.Random) -> list:
    """A uniformly random merge of several sequences that keeps each sequence's own order."""
    chains = [list(c) for c in chains if c]
    out = []
    while chains:
        r = rng.randrange(sum(len(c) for c in chains))
        for c in chains:
            if r < len(c):
                out.append(c.pop(0))
                break
            r -= len(c)
        chains = [c for c in chains if c]
    return out


N_COUNT, N_IGNORE_EOS, N_LOGPROBS = 4, 3, 3  # count, ignore_eos and logprob probes per run
SAMPLING_DEFAULT = 8  # hidden_probes sampling prompts when --sampling-prompts is not given


def probe_pool(items: list[dict]) -> list[dict]:
    return [it for it in items if it["bucket"] in ("short", "medium")] or list(items)


def probes_needed(gates: set, n_cache: int, n_sampling: int) -> int:
    """Distinct short/medium prompts the live probes draw: 2 warm-ups plus each wanted gate's share."""
    return (2 + (n_cache if "caching" in gates else 0) + (N_COUNT if "token_count" in gates else 0)
            + (N_IGNORE_EOS if "output_length" in gates else 0)
            + (n_sampling + N_LOGPROBS if "hidden_probes" in gates else 0))


def sampling_count(pool_size: int, gates: set, n_cache: int, asked: int | None) -> tuple[int, str | None]:
    """(sampling prompts, limitation or None). ``asked`` is used as given. Unset, it is SAMPLING_DEFAULT, or
    what a smaller prompt set leaves room for (never below SAMPLING_MIN_FAIL), and the record says so."""
    if asked is not None or "hidden_probes" not in gates:
        return (SAMPLING_DEFAULT if asked is None else asked), None
    n = min(SAMPLING_DEFAULT, max(SAMPLING_MIN_FAIL, pool_size - probes_needed(gates, n_cache, 0)))
    if n == SAMPLING_DEFAULT:
        return n, None
    return n, (f"the prompt set has {pool_size} short/medium prompts, room for {n} sampling prompts, not "
               f"{SAMPLING_DEFAULT}: an honest engine whose samples match greedy on a prompt with chance 1/3 "
               f"FAILs with chance 1 in {3 ** n}, not 1 in {3 ** SAMPLING_DEFAULT} (derived; use a larger set, or "
               "fewer --cache-prompts)")


def plan_and_run(client: Client, items: list[dict], *, gates: set, max_tokens: int, seed: int, n_cache: int,
                 log: list, reference: Client | None, echo_tokens: int = 64,
                 n_sampling: int = SAMPLING_DEFAULT) -> dict[str, list[Job]]:
    """Build the probe jobs, run them as one randomly interleaved stream, return them grouped by kind.

    Ordered pairs (first-seen -> repeat, original -> perturbed) stay in order but are
    staggered and mixed with every other probe, so nothing marks a request as a probe and
    drift over the run affects both members of a pair alike.
    """
    rng = random.Random(seed)
    pr = Prober(client, max_tokens, seed, log)
    pool = probe_pool(items)
    need = probes_needed(gates, n_cache, n_sampling)
    if len(pool) < need:
        raise OpError(f"the prompt set has {len(pool)} short/medium prompts; the probes need {need} distinct ones "
                      "(use a larger set, or lower --sampling-prompts / --cache-prompts)")
    rng.shuffle(pool)
    take = iter(pool)

    def nxt():
        return tag_prompt(ps.prompt_text(next(take)), rng)

    echo_max = max(max_tokens, echo_tokens)
    warm = [pr.job("warmup", f"w{i}", nxt()) for i in range(2)]
    pr.run(warm, shuffle=False)
    chains, free, out = [], [], {"warmup": warm}
    if "caching" in gates:
        texts = [nxt() for _ in range(n_cache)]
        first = [pr.job("cache_first", f"c{i}", t) for i, t in enumerate(texts)]
        again = [pr.job("cache_repeat", f"c{i}", t) for i, t in enumerate(texts)]
        nonces = [nonce(rng) for _ in range(4)]
        styles = [echo_style(rng) for _ in nonces]  # one style per pair: the pair differs in one character only
        echo1 = [pr.job("echo_orig", f"e{i}", echo_prompt(s, st), meta={"nonce": s}, max_tokens=echo_max)
                 for i, (s, st) in enumerate(zip(nonces, styles))]
        perturbed = [perturb(s, rng) for s in nonces]
        echo2 = [pr.job("echo_perturbed", f"e{i}", echo_prompt(p, st), meta={"nonce": p, "orig": s},
                        max_tokens=echo_max) for i, (s, p, st) in enumerate(zip(nonces, perturbed, styles))]
        chains += [stagger(first, again), stagger(echo1, echo2)]
        out.update(cache_first=first, cache_repeat=again, echo_orig=echo1, echo_perturbed=echo2)
    # Count and exact-length probes ask for special tokens in the stream: an engine that counts an
    # end-of-turn token in usage but streams nothing for it would otherwise look like it lies.
    if "token_count" in gates:
        out["count"] = [pr.job("count", f"n{i}", nxt(), logprobs=True, keep_special=True) for i in range(N_COUNT)]
        free += out["count"]
    if "output_length" in gates:
        out["ignore_eos"] = [pr.job("ignore_eos", f"l{i}", nxt(), ignore_eos=True, logprobs=True, keep_special=True)
                             for i in range(N_IGNORE_EOS)]
        free += out["ignore_eos"]
    if "hidden_probes" in gates:
        samp = []
        for i in range(n_sampling):
            t = nxt()
            sa, sb = rng.randrange(1, 2**31), rng.randrange(1, 2**31)
            samp += [pr.job("greedy", f"s{i}", t),
                     pr.job("sample_a", f"s{i}", t, temperature=1.0, seed=sa, full_vocab=True),
                     pr.job("sample_a2", f"s{i}", t, temperature=1.0, seed=sa, full_vocab=True),
                     pr.job("sample_b", f"s{i}", t, temperature=1.0, seed=sb, full_vocab=True)]
        lps = [pr.job("logprobs", f"p{i}", nxt(), logprobs=True, top_logprobs=5) for i in range(N_LOGPROBS)]
        free += samp + lps
        out.update(sampling=samp, logprobs=lps)
    if "rare_tokens" in gates:
        rare = [pr.job("rare", f"r{i}", echo_prompt(t, echo_style(rng)),
                       meta={"class": cls, "target": t, "core": rare_core(t)}, max_tokens=echo_max)
                for i, (cls, t) in enumerate(rare_targets(rng))]
        free += rare
        out["rare"] = rare
        if reference is not None:
            ref_jobs = [Job("rare_reference", j.key, {**reference.body(j.body["messages"], j.body["max_tokens"])},
                            j.meta) for j in rare]
            Prober(reference, max_tokens, seed, log).run(ref_jobs, shuffle=False)
            out["rare_reference"] = ref_jobs
    pr.run(interleave(chains + [[j] for j in free], rng), shuffle=False)
    return out


def probe_bodies(seed: int | None = None, model: str = "", max_tokens: int = 64, n_sampling: int = SAMPLING_DEFAULT,
                 n_logprobs: int = N_LOGPROBS, prompts: str = "dev") -> list[dict]:
    """Hidden-probe requests for a load runner to mix into its schedule (``{"id", "kind", "body"}``).

    The sampling probes come in groups that share a prompt (greedy, two samples with one seed, one
    with another); the logprob probes ask for top-5 logprobs. Run them inside the load, write their
    replies as ``outputs/<engine>/probes/<id>.json`` and ``gates run --run-dir`` checks them with the
    same rules as the live hidden_probes gate. ``optional`` lists body fields a runner may drop and
    resend if the server answers 400 (as the live client does).
    """
    rng = random.Random(secrets.randbits(63) if seed is None else seed)
    pool = [it for it in ps.load(prompts) if it["bucket"] in ("short", "medium")]
    if len(pool) < n_sampling + n_logprobs:
        raise OpError(f"the prompt set has {len(pool)} short/medium prompts; the probes need "
                      f"{n_sampling + n_logprobs} distinct ones")
    rng.shuffle(pool)
    take = iter(pool)
    c = Client.__new__(Client)  # body() only: no endpoint is contacted
    c.model, c.ask_ids, c.extra = model, False, {}
    out = []
    sample = {"temperature": 1.0, "full_vocab": True}
    for i in range(n_sampling):
        t = tag_prompt(ps.prompt_text(next(take)), rng)
        sa, sb = rng.randrange(1, 2**31), rng.randrange(1, 2**31)
        for kind, kw in (("greedy", {}), ("sample_a", {**sample, "seed": sa}), ("sample_a2", {**sample, "seed": sa}),
                         ("sample_b", {**sample, "seed": sb})):
            body = c.body(_msgs(t), max_tokens, **kw)
            out.append({"id": f"probe-s{i}-{kind}", "kind": kind, "group": f"s{i}", "body": body,
                        "optional": [k for k in Client.OPTIONAL if k in body]})
    for i in range(n_logprobs):
        out.append({"id": f"probe-p{i}", "kind": "logprobs", "group": f"p{i}",
                    "body": c.body(_msgs(tag_prompt(ps.prompt_text(next(take)), rng)), max_tokens, logprobs=True,
                                   top_logprobs=5), "optional": []})
    return out


# --------------------------------------------------------------------------- live gates

def gate_caching(jobs: dict, threshold: float) -> dict:
    first = {j.key: j.reply for j in jobs.get("cache_first", [])}
    again = {j.key: j.reply for j in jobs.get("cache_repeat", [])}
    failed = _failed([j for k in ("cache_first", "cache_repeat", "echo_orig", "echo_perturbed")
                      for j in jobs.get(k, [])])
    rows, ttft_sp, dec_sp = [], [], []
    for k, a in first.items():
        b = again.get(k)
        if not (a and b and a.ok and b.ok):
            continue
        t = a.ttft_ms / b.ttft_ms if a.ttft_ms and b.ttft_ms else None
        d = (b.decode_tok_s() / a.decode_tok_s()) if a.decode_tok_s() and b.decode_tok_s() else None
        if d is None and a.decode_tok_s() and b.t_first_ns is not None and b.t_first_ns == b.t_last_ns \
                and (b.streamed_count()[0] or b.usage_count() or 0) >= 2:
            d = INSTANT  # every token arrived at once: nothing was decoded
        ttft_sp.append(t)
        dec_sp.append(d)
        rows.append({"key": k, "first_ttft_ms": a.ttft_ms, "repeat_ttft_ms": b.ttft_ms, "ttft_speedup": t,
                     "first_decode_tok_s": a.decode_tok_s(), "repeat_decode_tok_s": b.decode_tok_s(),
                     "decode_speedup": d, "same_text": a.full_text == b.full_text})
    mt, md = _median(ttft_sp), _median(dec_sp)
    cached, unclear = [], 0
    origs = {j.key: j for j in jobs.get("echo_orig", [])}
    for j in jobs.get("echo_perturbed", []):
        o = origs.get(j.key)
        if not (o and o.reply and o.reply.ok and j.reply and j.reply.ok):
            continue  # counted in `failed`
        new, old = j.meta["nonce"], j.meta["orig"]
        # The old string coming back for the new prompt is a cache hit; a model that echoes
        # neither string tells us nothing.
        hit = old in j.reply.full_text and new not in j.reply.full_text
        if hit:
            cached.append({"key": j.key, "sent": new, "got": j.reply.full_text[:200], "first_reply_to": old})
        elif new not in j.reply.full_text and old not in j.reply.full_text:
            unclear += 1
    judged = sum(1 for j in jobs.get("echo_perturbed", []) if j.reply and j.reply.ok
                 and origs.get(j.key) and origs[j.key].reply and origs[j.key].reply.ok) - unclear
    metrics = {"ttft_speedup_median": mt, "decode_speedup_median": md, "pairs": len(rows),
               "perturbed_cache_hits": len(cached), "perturbed_pairs": judged, "failed_probes": len(failed)}
    lim = []
    problems = []

    def most(xs):  # a speedup must be typical, not one lucky pair: >= 2/3 of pairs beyond the limit
        xs = [x for x in xs if x is not None]
        return bool(xs) and 3 * sum(1 for x in xs if x > 1 + threshold) >= 2 * len(xs)

    if md is not None and md > 1 + threshold and most(dec_sp):
        problems.append(f"repeated prompts decode {md:.2f}x faster than first-seen (output caching?)")
    if mt is not None and mt > 1 + threshold and most(ttft_sp):
        problems.append(f"repeated prompts reach the first token {mt:.2f}x faster than first-seen: a prefix or output "
                        "cache is on; benchmark rounds that replay prompts would be inflated (disable prefix "
                        "caching for timing, or use unique prompts per round)")
    if cached:
        problems.append(f"{len(cached)} prompt(s) changed by one character got the previous answer back "
                        "(similarity/semantic output cache)")
    if unclear:
        lim.append(f"{unclear} perturbed echo probe(s) were inconclusive (the model did not echo either string)")
    if mt is None or md is None:
        lim.append("timing pairs incomplete; the timing half of the gate is weaker")
    if problems:
        return _gate("caching", "FAIL", "; ".join(problems), metrics, rows + cached + failed, lim)
    if failed:
        return _gate("caching", "UNPROVEN", f"{_fail_note(failed)}: a refused probe cannot be judged, so the cache "
                     "check is incomplete", metrics, rows + failed, lim)
    if not rows and not judged:
        return _gate("caching", "UNPROVEN", "no first/repeat pair and no perturbed pair could be judged", metrics,
                     rows, lim)
    if mt is None or md is None:
        return _gate("caching", "UNPROVEN", "no cache hits on perturbed prompts, but the repeat timing could not be "
                     "measured", metrics, rows, lim)
    return _gate("caching", "PASS", f"repeat speedup median TTFT {mt:.2f}x, decode {md:.2f}x (limit "
                 f"{1 + threshold:.2f}x); 0 perturbed prompts answered from cache", metrics, rows, lim)


def count_rows(replies: list[tuple], count_tokens) -> list[dict]:
    """One row per usable reply; ``replies`` holds (key, reply) or (key, reply, max_tokens)."""
    rows = []
    for key, r, *mt in replies:
        if not r or not r.ok:
            continue
        n, how = r.streamed_count()
        retok = count_tokens(r.full_text) if count_tokens else None
        rows.append({"key": key, "usage": r.usage_count(), "streamed": n, "streamed_method": how,
                     "retokenized": retok, "finish_reason": r.finish_reason, "max_tokens": mt[0] if mt else None})
    return rows


# Replies whose text is not expected to re-tokenize to the generated count: after EOS (ignore_eos) an
# engine may generate special tokens it strips from the text, and a copied rare or echo string can be
# generated as a non-canonical split (17 generated tokens re-tokenize to 22, say).
RETOK_EXEMPT = ("ignore_eos:", "rare:", "echo_orig:", "echo_perturbed:")
# The copy probes (rare, echo) ask for no logprobs, so on an engine without token ids usage and the text
# are their only counts, and the gap is set aside the same way: the count probes check usage. (13 Oct:
# rare 'ArrF क्षत्रिय ज्ञान 6bzC' copied verbatim, usage 17, re-tokenized 14, nothing streamed; observed.)
# An ignore_eos reply asks for logprobs; without a stream its text stays the check on usage.
COPY_PROBES = ("rare:", "echo_orig:", "echo_perturbed:")
UNSTREAMED_EOS = "tokens not streamed: special tokens after EOS?"
UNSTREAMED_CUT = "tokens counted but not streamed (a held-back partial character at the cut?)"


def unstreamed(u, s, how, finish, max_tokens) -> bool:
    """A logprob stream shorter than usage on a reply cut at max_tokens: the gap may be special tokens
    (end of turn, padding) that were generated and counted but never streamed. The stream cannot tell
    that from an engine that stops early and over-reports, so such a row is unproven, not failed."""
    return (u is not None and s is not None and s < u and how == "logprobs" and finish == "length"
            and max_tokens is not None and u == max_tokens)


def text_covers(t, u, retok_tol: float) -> bool:
    """The re-tokenized text holds at least as many tokens as usage claims (and not wildly more): it
    refutes over-reporting, the only cheat a short stream could hide."""
    return t is not None and u is not None and u <= t <= u + max(2, retok_tol * u)


def judge_counts(rows: list[dict], retok_tol: float) -> dict:
    """Sort count rows into {"bad": mismatches (FAIL), "unsettled": logprob streams short of usage at the
    max_tokens limit (UNPROVEN), "by_text": such streams whose re-tokenized text covers usage (a
    limitation), "retok_only": re-tokenize-only gaps on RETOK_EXEMPT rows (a limitation), "checked": rows
    with >= 2 independent counts}."""
    bad, unsettled, by_text, retok_only, checked = [], [], [], [], 0
    for row in rows:
        u, s, t = row["usage"], row["streamed"], row["retokenized"]
        key = str(row.get("key", ""))
        have = [x for x in (u, s, t) if x is not None]
        if len(have) < 2:
            continue
        why = []
        # An engine may count the end-of-sequence token in usage without streaming it.
        slack = 1 if row.get("finish_reason") == "stop" else 0
        stream_agrees = u is not None and s is not None and abs(u - s) <= slack
        settled = False
        if u is not None and s is not None and not stream_agrees:
            if unstreamed(u, s, row.get("streamed_method"), row.get("finish_reason"), row.get("max_tokens")):
                what = UNSTREAMED_EOS if key.startswith("ignore_eos:") else UNSTREAMED_CUT
                if text_covers(t, u, retok_tol):  # the text proves the count the stream could not
                    settled = True
                    by_text.append({**row, "problem": f"{u - s} {what}; the re-tokenized text ({t}) covers usage"})
                else:
                    unsettled.append({**row, "problem": f"{u - s} {what}"})
            else:
                why.append(f"usage {u} != streamed {s}")
        ref = s if s is not None else u
        set_aside = False
        if t is not None and ref is not None and abs(t - ref) > max(2, retok_tol * ref) and not settled:
            if u is not None and key.startswith(RETOK_EXEMPT) and (s is not None or key.startswith(COPY_PROBES)):
                # Stream and usage already judged this reply (agreeing, or caught above), or it is a copy
                # probe with usage alone; the text is not a fair count here.
                set_aside = s is None
                retok_only.append({**row, "problem": f"re-tokenized {t} vs {ref}"})
            else:
                why.append(f"re-tokenized {t} vs {ref}")
        checked += not set_aside  # usage against the text alone, set aside: nothing was checked
        if why:
            bad.append({**row, "problem": "; ".join(why)})
    return {"bad": bad, "unsettled": unsettled, "by_text": by_text, "retok_only": retok_only, "checked": checked}


def unstreamed_note(rows: list[dict], after_eos: bool | None = None) -> str:
    """'a to b tokens not streamed: ...' for each kind of row: ignore_eos replies (``after_eos``, or by
    key) and replies cut at max_tokens without it."""
    out = []
    for eos, what in ((True, UNSTREAMED_EOS), (False, UNSTREAMED_CUT)):
        gaps = sorted(r["usage"] - r["streamed"] for r in rows
                      if (str(r.get("key", "")).startswith("ignore_eos:") if after_eos is None else after_eos) == eos)
        if gaps:
            out.append(f"{gaps[0]} to {gaps[-1]} {what}" if gaps[0] != gaps[-1] else f"{gaps[0]} {what}")
    return "; ".join(out)


def gate_token_count(jobs: dict, count_tokens, retok_tol: float, static_rows: list[dict] | None = None) -> dict:
    replies = [(f"{k}:{j.key}", j.reply, j.body.get("max_tokens")) for k, js in jobs.items()
               if not k.endswith("reference") for j in js]
    rows = count_rows(replies, count_tokens) + list(static_rows or [])
    judged = judge_counts(rows, retok_tol)
    bad, unsettled, checked = judged["bad"], judged["unsettled"], judged["checked"]
    failed = _failed(jobs.get("count", []))
    lim = []
    if count_tokens is None:
        lim.append("no tokenizer given: re-tokenized counts not checked (pass --tokenizer)")
    if judged["retok_only"]:
        lim.append(f"{len(judged['retok_only'])} ignore_eos / echo / rare repl(ies) re-tokenize to a different count "
                   "than they streamed or usage says (special tokens stripped from the text, or a copied string "
                   "generated as a non-canonical split); the re-tokenized gap is not counted as a mismatch (their "
                   "stream, where there is one, was judged against usage, and the count probes check usage)")
    if judged["by_text"]:
        lim.append(f"{len(judged['by_text'])} repl(ies) cut at max_tokens streamed fewer logprob entries than usage "
                   f"counts ({unstreamed_note(judged['by_text'])}), but their re-tokenized text covers usage: the "
                   "count is proven by the text, not the stream")
    methods = sorted({r["streamed_method"] for r in rows if r.get("streamed_method")})
    metrics = {"replies_checked": checked, "mismatches": len(bad), "unstreamed_at_limit": len(unsettled),
               "unstreamed_settled_by_text": len(judged["by_text"]),
               "retokenize_only_gaps": len(judged["retok_only"]), "streamed_methods": methods,
               "failed_probes": len(failed)}
    if bad:
        return _gate("token_count", "FAIL", f"{len(bad)}/{checked} replies report a token count that the stream "
                     "does not support", metrics, bad[:50] + unsettled[:20] + failed, lim)
    notes = []
    if failed:
        notes.append(_fail_note(failed, "count probe"))
    if unsettled:
        notes.append(f"{len(unsettled)}/{checked} replies cut at max_tokens stream fewer logprob entries than usage "
                     f"counts ({unstreamed_note(unsettled)}): the stream cannot prove the count (the engine returned "
                     "no token ids, and skip_special_tokens: false on the count and ignore_eos probes did not bring "
                     "the missing tokens into the stream" + ("; pass --tokenizer so the text can settle it)"
                                                              if count_tokens is None else ")"))
    if notes:
        return _gate("token_count", "UNPROVEN", "; ".join(notes) + f"; {checked - len(unsettled)} other replies agree",
                     metrics, unsettled[:50] + failed, lim)
    if checked == 0:
        verdict = "UNPROVEN" if rows else "NOT RUN"
        return _gate("token_count", verdict, "no reply carried two independent token counts (no usage, token ids, "
                     "logprobs or tokenizer)", metrics, judged["retok_only"][:20], lim)
    return _gate("token_count", "PASS", f"{checked} replies: usage, streamed and re-tokenized counts agree",
                 metrics, judged["by_text"][:20] + judged["retok_only"][:20], lim)


def gate_output_length_live(jobs: dict, count_tokens) -> dict:
    rows, bad, stop_short, short_stream, by_text, usage_only = [], [], [], [], [], 0
    failed = _failed(jobs.get("ignore_eos", []))
    for j in jobs.get("ignore_eos", []):
        r = j.reply
        want = j.body["max_tokens"]
        if not r or not r.ok:
            continue
        n, how = r.streamed_count()
        if n is None and count_tokens is not None:
            n, how = count_tokens(r.full_text), "retokenized"
        if n is None:
            n, how = r.usage_count(), "usage"
            usage_only += 1
        u, st = r.usage_count(), r.streamed_count()[0]
        slack = 1 if r.finish_reason == "stop" else 0  # the EOS token may be counted on one side only
        disagree = u is not None and st is not None and abs(u - st) > slack
        row = {"key": j.key, "max_tokens": want, "length": n, "method": how, "finish_reason": r.finish_reason,
               "usage": u, "streamed": st}
        rows.append(row)
        if how == "retokenized":  # detokenize -> retokenize is not exact; allow a small slack
            wrong = n is None or abs(n - want) > max(2, 0.02 * want)
        else:
            wrong = n != want
        problems = []
        if u is not None and u > want:
            problems.append(f"usage {u} > max_tokens {want}")
        if u is not None and st is not None and st > u + slack:
            problems.append(f"streamed {st} > usage {u}")
        if r.finish_reason == "length" and u is not None and u < want:
            problems.append(f"finish_reason 'length' with usage {u} < max_tokens {want}")
        if not wrong and not problems:
            continue
        if not problems and unstreamed(u, st, how, r.finish_reason, want):
            # usage says max_tokens and the reply was cut there, but some counted tokens never reached the
            # logprob stream (an engine that skips special tokens after EOS); the stream cannot settle it,
            # re-tokenized text that covers max_tokens can.
            retok = count_tokens(r.full_text) if count_tokens is not None else None
            (by_text if text_covers(retok, u, 0.02) else short_stream).append(
                {**row, "retokenized": retok, "problem": f"{u - st} {UNSTREAMED_EOS}"})
        elif not problems and r.finish_reason == "stop" and not disagree and n is not None and n < want:
            stop_short.append(row)  # ended at EOS with consistent counts: ignore_eos not supported
        else:
            bad.append({**row, "problem": "; ".join(problems) or f"length {n} != max_tokens {want}"})
    metrics = {"ignore_eos_requests": len(rows), "exact": len(rows) - len(bad) - len(stop_short) - len(short_stream),
               "unstreamed_at_limit": len(short_stream), "unstreamed_settled_by_text": len(by_text),
               "failed_probes": len(failed)}
    lim = [f"{usage_only} length(s) known only from usage (no token ids, logprobs or tokenizer)"] if usage_only else []
    if by_text:
        lim.append(f"{len(by_text)} ignore_eos repl(ies) streamed fewer logprob entries than max_tokens "
                   f"({unstreamed_note(by_text, after_eos=True)}), but their re-tokenized text covers max_tokens: the "
                   "length is proven by the text, not the stream")
    if bad:
        return _gate("output_length", "FAIL", f"{len(bad)}/{len(rows)} ignore_eos replies are not max_tokens long "
                     "(finish_reason 'length' below max_tokens, usage above it, or usage disagrees with the stream)",
                     metrics, bad + short_stream + failed, lim)
    if failed:
        return _gate("output_length", "UNPROVEN", f"{_fail_note(failed, 'ignore_eos request')}: the engine refused "
                     "exact-length requests, so fixed-length runs cannot be checked live (compare lengths against a "
                     "reference engine: gates run --run-dir --reference-engine)", metrics, failed + rows, lim)
    unsure = []
    if stop_short:
        unsure.append(f"{len(stop_short)}/{len(rows)} ignore_eos replies stopped at EOS (finish_reason 'stop', counts "
                      "consistent): the engine does not honour ignore_eos, so fixed-length runs are not fixed length; "
                      "compare lengths against a reference engine instead (gates run --run-dir --reference-engine)")
    if short_stream:
        unsure.append(f"{len(short_stream)}/{len(rows)} ignore_eos replies report usage = max_tokens and finish "
                      f"'length' but stream fewer logprob entries ({unstreamed_note(short_stream, after_eos=True)}): "
                      "the length is not proven by the stream (the engine returned no token ids and did not stream "
                      "the missing tokens with skip_special_tokens: false; compare lengths against a reference "
                      "engine: gates run --run-dir --reference-engine)")
    if unsure:
        return _gate("output_length", "UNPROVEN", "; ".join(unsure), metrics, stop_short + short_stream, lim)
    if not rows:
        return _gate("output_length", "NOT RUN", "no ignore_eos probe was sent", metrics)
    verdict = "UNPROVEN" if usage_only == len(rows) else "PASS"
    return _gate("output_length", verdict, f"{len(rows)} ignore_eos replies are exactly max_tokens long"
                 + (" (by usage only)" if verdict == "UNPROVEN" else ""), metrics, rows, lim)


def check_logprob_entry(e: dict, greedy: bool) -> list[str]:
    out = []
    lp = e.get("logprob")
    top = e.get("top_logprobs") or []
    vals = [t.get("logprob") for t in top]
    if not isinstance(lp, (int, float)) or not math.isfinite(lp) or lp > 1e-6:
        out.append(f"chosen logprob {lp!r} not finite and <= 0")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v > 1e-6 for v in vals):
        out.append("a top logprob is not finite and <= 0")
        return out
    if vals:
        if any(vals[i] < vals[i + 1] - 1e-6 for i in range(len(vals) - 1)):
            out.append("top_logprobs not sorted")
        mass = sum(math.exp(v) for v in vals)
        if mass > 1 + 1e-3:
            out.append(f"top-k probability mass {mass:.4f} > 1 (not normalised)")
        if greedy and isinstance(lp, (int, float)) and lp < max(vals) - 1e-3:
            out.append(f"greedy token logprob {lp:.4f} < top logprob {max(vals):.4f} (greedy token is not the argmax)")
        if greedy and top and top[0].get("token") != e.get("token") and isinstance(lp, (int, float)) \
                and abs(lp - vals[0]) > 1e-3:
            out.append(f"greedy token {e.get('token')!r} != top-1 {top[0].get('token')!r}")
    return out


HIDDEN_LIMIT = ("probes ran in their own sequential stream (concurrency 1), not inside the benchmark load: an engine "
                "that behaves only when idle is not caught here (mix probe_bodies() into the load and pass --run-dir "
                "to check their replies)")


# hidden_probes sampling rule. A confident model can return its greedy answer at T=1 by chance: on
# 14 Oct, short extraction answers came back byte for byte equal to greedy from stock engines, on 2/3,
# 0/3 and 1/3 prompts on three hosts (observed), so take p ~ 1/3 as the chance that both samples of a
# prompt equal greedy (inferred). The old rule (FAIL when >= half of 3 prompts matched) then failed an
# honest engine with P = 3p^2(1-p) + p^3 = 7/27, about 26% of runs (derived). FAIL now needs every one
# of at least SAMPLING_MIN_FAIL prompts to match: with n = 8, P = p^8 = 1/6561, about 1.5e-4 (derived).
# >= 3/4 matching is UNPROVEN (6-7 of 8: about 2% at p = 1/3, derived). An engine that ignores
# temperature matches on every prompt, so it still FAILs.
SAMPLING_MIN_FAIL = 4


def sampling_verdict(equal: int, n: int) -> tuple[str, str]:
    """(verdict, reason) for ``equal`` of ``n`` prompts whose T=1 samples both equal the greedy output."""
    if n >= SAMPLING_MIN_FAIL and equal == n:
        return "FAIL", f"T=1 samples equal the greedy output on all {n} prompts (temperature ignored?)"
    if 4 * equal >= 3 * n:
        why = (f"with fewer than {SAMPLING_MIN_FAIL} prompts that can be chance" if n < SAMPLING_MIN_FAIL
               else "more than chance usually gives on a confident model, but not all, so temperature is not shown "
                    "to be ignored")
        return "UNPROVEN", (f"T=1 samples equal the greedy output on {equal}/{n} prompts: {why} (rerun with more "
                            "--sampling-prompts)")
    return "PASS", f"T=1 samples differ from greedy on {n - equal}/{n} prompts"


def gate_hidden_probes(jobs: dict, where: str = "live") -> dict:
    by_key: dict[str, dict] = {}
    for j in jobs.get("sampling", []):
        by_key.setdefault(j.key, {})[j.kind] = j.reply
    samp_failed = _failed(jobs.get("sampling", []))
    rows, ignored, seed_ok, seed_checked = [], 0, 0, 0
    for k, d in by_key.items():
        g, a, a2, b = (d.get(x) for x in ("greedy", "sample_a", "sample_a2", "sample_b"))
        if not all(r and r.ok for r in (g, a, b)):
            continue
        same_as_greedy = a.full_text == g.full_text and b.full_text == g.full_text
        ignored += same_as_greedy
        seeded = None
        if a2 and a2.ok:
            seeded = a.full_text == a2.full_text
            seed_checked += 1
            seed_ok += bool(seeded)
        rows.append({"key": k, "samples_equal_greedy": same_as_greedy, "seed_reproduced": seeded,
                     "seeds_differ": a.full_text != b.full_text})
    sub, details = [], list(rows)
    lim = [HIDDEN_LIMIT] if where == "live" else []
    metrics = {"sampling_prompts": len(rows), "samples_equal_greedy": ignored,
               "seed_reproduced": f"{seed_ok}/{seed_checked}", "failed_probes": len(samp_failed)}
    if rows:
        sub.append(sampling_verdict(ignored, len(rows)))
        if len(rows) < SAMPLING_MIN_FAIL:
            lim.append(f"only {len(rows)} sampling prompt(s) came back complete: too few to FAIL an engine that "
                       f"ignores temperature (at least {SAMPLING_MIN_FAIL} are needed; the default is "
                       f"{SAMPLING_DEFAULT})")
        if seed_checked and seed_ok < seed_checked:
            lim.append(f"the same seed reproduced the sample on {seed_ok}/{seed_checked} prompts: the engine ignores "
                       "`seed` or samples nondeterministically (seeded comparisons are not possible)")
    refused = sorted({k for j in jobs.get("sampling", []) if j.reply
                      for k in j.reply.dropped if k in ("top_p", "top_k")})
    if refused:
        lim.append(f"the server refused {' and '.join(refused)} on the T=1 samples (top_p 1 and top_k -1 ask for the "
                   f"whole distribution), so they used its default {' and '.join(refused)}: a default that cuts the "
                   "distribution makes samples equal to greedy more likely")
    if samp_failed:
        sub.append(("UNPROVEN", _fail_note(samp_failed, "sampling probe")))
        details += samp_failed
    elif not rows:
        sub.append(("UNPROVEN", "no sampling probe group came back complete"))
    lp_jobs = jobs.get("logprobs", [])
    lp_failed = _failed(lp_jobs)
    lp_missing = [j.key for j in lp_jobs if j.reply and j.reply.ok and not j.reply.lp]
    lp_bad, lp_entries, lp_replies = [], 0, 0
    for j in lp_jobs:
        r = j.reply
        if not r or not r.ok or not r.lp:
            continue
        lp_replies += 1
        for pos, e in enumerate(r.lp):
            lp_entries += 1
            for prob in check_logprob_entry(e, greedy=True):
                lp_bad.append({"key": j.key, "position": pos, "problem": prob})
    metrics.update(logprob_replies=lp_replies, logprob_positions=lp_entries, logprob_violations=len(lp_bad),
                   logprob_failed=len(lp_failed), logprob_missing=len(lp_missing))
    if lp_bad:
        sub.append(("FAIL", f"{len(lp_bad)} logprob violations in {lp_entries} positions"))
        details += lp_bad[:50]
    elif lp_replies:
        sub.append(("PASS", f"{lp_entries} positions: logprobs finite, sorted, normalised, greedy == argmax"))
    if lp_failed:
        sub.append(("UNPROVEN", _fail_note(lp_failed, "logprob probe")))
        details += lp_failed
    if lp_missing:
        sub.append(("UNPROVEN", f"{len(lp_missing)}/{len(lp_jobs)} logprob probes came back without logprobs "
                                "(logprobs were requested): the logprob check could not run"))
    if not lp_jobs:
        sub.append(("UNPROVEN", "no logprob probe"))
    verdict = combine_claims(v for v, _ in sub)
    return _gate("hidden_probes", verdict, "; ".join(f"{v}: {why}" for v, why in sub), metrics, details, lim)


def gate_rare(jobs: dict) -> dict:
    # A probe is judged on the rare string itself (meta "core"); the random ASCII nonces around it only
    # make it unpredictable. In the 13-14 Oct records all 3 rare_tokens FAILs (two stock vLLM configs and
    # one custom engine, none with a reference engine; observed) were one miss each: the model cut the
    # trailing nonce ("... 72Ry" -> "... 72R", or dropped it) and ended at EOS well under max_tokens, with
    # the CJK / emoji / Hangul string intact. A pruned vocabulary or logit mask cannot cause that slip.
    ref = {j.key: j.reply for j in jobs.get("rare_reference", [])}
    rows, misses, ok_n, refused, slips = [], [], 0, 0, []
    for j in jobs.get("rare", []):
        r = j.reply
        rr = ref.get(j.key)
        core = j.meta.get("core") or j.meta["target"]
        ref_errored = rr is not None and not rr.ok
        ref_hit = reproduced(core, rr.full_text) if rr is not None and rr.ok else None
        row = {"key": j.key, "class": j.meta["class"], "target": j.meta["target"], "reference_reproduced": ref_hit}
        if not r or not r.ok:
            # A refused rare string is a miss (it is exactly what a pruned vocabulary would want to
            # hide), unless the reference engine refuses it too.
            refused += 1
            row.update(reproduced=False, refused=True, status=r.status if r else None,
                       error=((r.error if r else "not sent") or "")[:200], reference_refused=ref_errored)
            rows.append(row)
            if not ref_errored:
                misses.append(row)
            continue
        ok_n += 1
        verbatim = reproduced(j.meta["target"], r.full_text)
        hit = verbatim or reproduced(core, r.full_text)
        row.update(reproduced=hit, nonces_intact=verbatim, got=r.full_text[:200])
        rows.append(row)
        if hit and not verbatim:
            slips.append(row)
        if not hit and ref_hit is not False:
            misses.append(row)
    if not rows:
        return _gate("rare_tokens", "NOT RUN", "no rare-string probe was sent")
    lim = [] if ref else ["no reference engine: a miss may be the model's own weakness (pass --reference-url)"]
    if slips:
        lim.append(f"{len(slips)} rare string(s) came back intact inside a slipped random ASCII nonce (a model copy "
                   "error; an engine that drops rare tokens keeps plain ASCII), not counted as misses")
    metrics = {"probes": len(rows), "answered": ok_n, "refused": refused,
               "reproduced": sum(1 for r in rows if r["reproduced"]), "nonce_slips": len(slips),
               "counted_misses": len(misses)}
    if misses:
        classes = sorted({m["class"] for m in misses})
        n_ref = sum(1 for m in misses if m.get("refused"))
        return _gate("rare_tokens", "FAIL", f"{len(misses)}/{len(rows)} rare strings not reproduced verbatim "
                     f"({', '.join(classes)})" + (f"; {n_ref} of them refused with an error" if n_ref else ""),
                     metrics, misses, lim)
    n_rep = metrics["reproduced"]
    said = f"{n_rep} rare strings reproduced verbatim" + (f" ({len(slips)} inside a slipped ASCII nonce)" if slips
                                                          else "")
    shared = ok_n - n_rep
    said += f"; {shared} missed by the reference engine too (not counted)" if shared else ""
    if refused:
        return _gate("rare_tokens", "UNPROVEN", f"{said}; {refused} refused by both engines (not judged)", metrics,
                     [r for r in rows if r.get("refused")] + slips, lim)
    return _gate("rare_tokens", "PASS", said, metrics, slips, lim)


# --------------------------------------------------------------------------- spec exactness

def spec_exact(a: Client, b: Client, items: list[dict], max_tokens: int, early_tokens: int, tolerance: int,
               log: list) -> dict:
    """A (speculative) vs B (plain), and B vs B again for the A/A floor, greedy, same prompts."""
    rows = []
    for it in items:
        msgs = it["messages"]
        rb = b.send(b.body(msgs, max_tokens))
        ra = a.send(a.body(msgs, max_tokens))
        rb2 = b.send(b.body(msgs, max_tokens))
        for tag, r in (("B", rb), ("A", ra), ("B2", rb2)):
            log.append({"kind": f"spec_{tag}", "key": it["id"], "reply": r.summary()})
        if not (ra.ok and rb.ok and rb2.ok):
            rows.append({"id": it["id"], "error": ra.error or rb.error or rb2.error})
            continue
        sa, ma = _sequence(ra)
        sb, mb = _sequence(rb)
        sb2, _ = _sequence(rb2)
        if ma != mb:  # compare like with like
            sa, sb, sb2, ma = list(ra.full_text), list(rb.full_text), list(rb2.full_text), "characters"
        rows.append({"id": it["id"], "bucket": it.get("bucket"), "method": ma,
                     "ab_first_divergence": first_divergence(sa, sb),
                     "aa_first_divergence": first_divergence(sb, sb2), "len_a": len(sa), "len_b": len(sb)})
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    if not ok:
        return _gate("spec_exact", "UNPROVEN" if rows else "NOT RUN", "no prompt got three successful replies"
                     + (f" ({len(errs)} failed; first error: {str(errs[0]['error'])[:160]})" if errs else ""), {}, rows)

    def early(key):
        return sum(1 for r in ok if r[key] is not None and r[key] < early_tokens)

    ab_div = [r["ab_first_divergence"] for r in ok if r["ab_first_divergence"] is not None]
    aa_div = [r["aa_first_divergence"] for r in ok if r["aa_first_divergence"] is not None]
    e_ab, e_aa = early("ab_first_divergence"), early("aa_first_divergence")
    metrics = {"prompts": len(ok), "ab_identical": len(ok) - len(ab_div), "aa_identical": len(ok) - len(aa_div),
               "ab_early_divergences": e_ab, "aa_early_divergences": e_aa, "early_tokens": early_tokens,
               "ab_first_divergence_median": _median(ab_div), "aa_first_divergence_median": _median(aa_div),
               "methods": sorted({r["method"] for r in ok})}
    lim = []
    if "characters" in metrics["methods"]:
        lim.append("no token ids or logprobs: compared characters (divergence positions are in characters)")
    diverged = [r for r in ok if r["ab_first_divergence"] is not None][:50]
    if e_ab > e_aa + tolerance:
        return _gate("spec_exact", "FAIL", f"{e_ab}/{len(ok)} prompts diverge from the non-speculative engine before "
                     f"token {early_tokens}; the A/A floor is {e_aa} (+{tolerance} allowed)", metrics,
                     diverged + errs, lim)
    # Late divergences: one prompt in eight above the A/A floor is allowed for reordered float
    # sums in the verify pass; more is not proven exact (the quality gates must decide).
    late_allow = tolerance + max(1, len(ok) // 8)
    metrics["late_allowed"] = late_allow
    if len(ab_div) > len(aa_div) + late_allow:
        return _gate("spec_exact", "UNPROVEN", f"{len(ab_div)}/{len(ok)} prompts diverge from the non-speculative "
                     f"engine (all at or after token {early_tokens}) vs an A/A floor of {len(aa_div)} (+{late_allow} "
                     "allowed): speculation is not shown to be exact; compare quality before publishing", metrics,
                     diverged + errs, lim)
    if errs:
        return _gate("spec_exact", "UNPROVEN", f"{len(errs)} prompt(s) failed (first error: "
                     f"{str(errs[0]['error'])[:160]}); on the other {len(ok)}: {len(ok) - len(ab_div)} identical",
                     metrics, diverged + errs, lim)
    return _gate("spec_exact", "PASS", f"{len(ok) - len(ab_div)}/{len(ok)} identical; early divergences {e_ab} <= "
                 f"A/A floor {e_aa} + {tolerance}; any divergence {len(ab_div)} <= A/A {len(aa_div)} + {late_allow}",
                 metrics, diverged, lim)


# --------------------------------------------------------------------------- static run-dir checks

def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _out_len(o: dict) -> tuple[int | None, str | None]:
    """Generated length of a saved output, or (None, None) when it is unreadable or the request failed."""
    if o.get("error"):
        return None, None
    if isinstance(o.get("token_ids"), list):
        return len(o["token_ids"]), "token_ids"
    u = (o.get("usage") or {}).get("completion_tokens")
    return (u, "usage") if isinstance(u, int) else (None, None)


def outputs_of(root: Path) -> dict[str, dict]:
    """{id: output} for outputs/<engine>/<bucket>/<id>.json (root = outputs/<engine>; load-* and probes skipped)."""
    out = {}
    if root.is_dir():
        for p in sorted(root.glob("*/*.json")):
            if p.parent.name.startswith("load-") or p.parent.name == "probes":
                continue
            o = _read_json(p)
            out[p.stem] = {**o, "_bucket": p.parent.name} if isinstance(o, dict) else {"_bucket": p.parent.name,
                                                                                         "_unreadable": True}
    return out


def static_counts(run_dir: Path, engine: str) -> list[dict]:
    rows = []
    for k, o in outputs_of(run_dir / "outputs" / engine).items():
        if o.get("_unreadable") or o.get("error"):
            continue
        ids = o.get("token_ids")
        u = (o.get("usage") or {}).get("completion_tokens")
        rows.append({"key": f"run:{k}", "usage": u if isinstance(u, int) else None,
                     "streamed": len(ids) if isinstance(ids, list) else None,
                     "streamed_method": "token_ids" if isinstance(ids, list) else None, "retokenized": None,
                     "finish_reason": o.get("finish_reason")})
    return rows


def _inputs_of(run_dir: Path, engine: str) -> dict[str, dict]:
    """{id: request body}: inputs/<engine>/<bucket>/ (bench serve) else the shared inputs/<bucket>/."""
    out: dict[str, dict] = {}
    for root in (run_dir / "inputs", run_dir / "inputs" / engine):  # the engine's own copy wins
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*/*.json")):
            if p.parent.name.startswith("load-"):
                continue
            body = _read_json(p)
            if isinstance(body, dict) and ("max_tokens" in body or "messages" in body or "prompt" in body):
                out[p.stem] = body
    return out


def _exact_length(req: dict) -> int | None:
    """max_tokens when the request forces an exact length (ignore_eos, or min_tokens == max_tokens)."""
    mt = req.get("max_tokens") if isinstance(req.get("max_tokens"), int) else req.get("max_completion_tokens")
    if not isinstance(mt, int):
        return None
    if req.get("ignore_eos") or req.get("min_tokens") == mt:
        return mt
    return None


def gate_output_length_static(run_dir: Path, engine: str, reference: Path | None, tol: float,
                              max_outliers: float) -> dict:
    cand = outputs_of(run_dir / "outputs" / engine)
    if not cand:
        return _gate("output_length", "NOT RUN", f"no outputs under outputs/{engine}/")
    sub, details, metrics = [], [], {}
    # Exact-length requests must come back exactly max_tokens long; a missing or failed output is unproven.
    eos_bad, eos_missing, eos_n = [], [], 0
    for k, req in _inputs_of(run_dir, engine).items():
        want = _exact_length(req)
        if want is None:
            continue
        eos_n += 1
        o = cand.get(k)
        n, how = _out_len(o) if o and not o.get("_unreadable") else (None, None)
        if n is None:
            eos_missing.append({"id": k, "max_tokens": want, "problem": "no readable output" if not o or
                                o.get("_unreadable") else f"request failed: {str(o.get('error'))[:120]}"})
        elif n != want:
            eos_bad.append({"id": k, "max_tokens": want, "length": n, "method": how})
    if eos_n:
        v = "FAIL" if eos_bad else ("UNPROVEN" if eos_missing else "PASS")
        sub.append((v, f"exact length: {eos_n - len(eos_bad) - len(eos_missing)}/{eos_n} outputs exactly max_tokens "
                       f"long, {len(eos_bad)} wrong, {len(eos_missing)} missing or failed"))
        details += eos_bad[:50] + eos_missing[:50]
    metrics.update(exact_length_outputs=eos_n, exact_length_wrong=len(eos_bad), exact_length_missing=len(eos_missing))
    if reference is not None:
        ref = outputs_of(reference)
        ratios, outl, missing, ref_n = [], [], [], 0
        for k, r_o in ref.items():
            m, _ = _out_len(r_o) if not r_o.get("_unreadable") else (None, None)
            if m is None:
                continue  # the reference itself has no length here: nothing to compare against
            ref_n += 1
            o = cand.get(k)
            n, _ = _out_len(o) if o and not o.get("_unreadable") else (None, None)
            if n is None:
                # A prompt the candidate did not answer is a failure, not a skipped row.
                missing.append({"id": k, "reference_length": m, "problem": "no readable candidate output" if not o or
                                o.get("_unreadable") else f"candidate request failed: {str(o.get('error'))[:120]}"})
                continue
            ratio = n / m if m else (1.0 if n == 0 else None)
            ratios.append(ratio if ratio is not None else 0.0)
            if ratio is None or abs(ratio - 1) > tol:
                outl.append({"id": k, "length": n, "reference_length": m,
                             "ratio": round(ratio, 4) if ratio is not None else None})
        coverage = (ref_n - len(missing)) / ref_n if ref_n else None
        metrics.update(reference_outputs=ref_n, reference_pairs=len(ratios), outside_tolerance=len(outl),
                       missing_outputs=len(missing), coverage=coverage, length_ratio_median=_median(ratios))
        if ref_n:
            off = len(outl) + len(missing)  # missing outputs count as outliers
            frac = off / ref_n
            total_ratio = sum(ratios) / ref_n  # missing outputs count as length 0
            bad = frac > max_outliers or abs(total_ratio - 1) > tol
            v = "FAIL" if bad else ("UNPROVEN" if missing else "PASS")
            sub.append((v, f"{len(outl)}/{ref_n} prompts outside +/-{tol:.0%} of the reference length and "
                           f"{len(missing)} without a candidate output (together {frac:.0%}, allowed "
                           f"{max_outliers:.0%}); coverage {coverage:.0%}; mean ratio {total_ratio:.3f}"))
            details += outl[:50] + missing[:50]
        else:
            sub.append(("NOT RUN", "the reference outputs carry no lengths"))
    if not sub:
        return _gate("output_length", "NOT RUN", "no exact-length requests and no reference outputs given", metrics)
    return _gate("output_length", combine_claims(v for v, _ in sub), "; ".join(f"{v}: {w}" for v, w in sub),
                 metrics, details)


def _load_requests(run_dir: Path) -> list[dict]:
    """One row per load request: {"source", "engine", "level", "key", "salt", "prompt"}.

    bench serve layout first (inputs/<engine>/load-<level>/<bucket>/<id>[~k].json request bodies,
    else outputs/<engine>/load-<level>/...), then any *schedule*.jsonl / *requests*.jsonl."""
    rows = []
    for kind in ("inputs", "outputs"):
        base = run_dir / kind
        found = False
        for d in sorted(base.glob("*/load-*")) if base.is_dir() else []:
            if not d.is_dir():
                continue
            found = True
            engine, level = d.parent.name, d.name[len("load-"):]
            for p in sorted(d.glob("*/*.json")):
                o = _read_json(p)
                if not isinstance(o, dict):
                    continue
                pid = o.get("id") or p.stem.split("~")[0]
                if kind == "inputs" and (o.get("messages") is not None or o.get("prompt") is not None):
                    key = hashlib.sha256(json.dumps(o.get("messages", o.get("prompt")), ensure_ascii=False,
                                                    sort_keys=True).encode("utf-8")).hexdigest()
                else:
                    key = f"{pid}|{o.get('salt') or ''}"
                salt = o.get("salt") if kind == "outputs" else _salt_of(o)
                rows.append({"source": str(p.relative_to(run_dir)), "engine": engine, "level": level, "key": key,
                             "salt": salt, "prompt": pid})
        if found:
            return rows
    for f in sorted(p for p in run_dir.rglob("*.jsonl") if "schedule" in p.name or "requests" in p.name):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line) if line.strip() else None
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            level = r.get("level", r.get("concurrency", r.get("rate_rps")))
            if level is None:
                continue
            if r.get("prompt_sha256"):
                key = r["prompt_sha256"]
            elif r.get("messages") is not None or r.get("prompt") is not None:
                key = hashlib.sha256(json.dumps(r.get("messages", r.get("prompt")), ensure_ascii=False,
                                                sort_keys=True).encode("utf-8")).hexdigest()
            elif r.get("prompt_id", r.get("id")) is not None:
                key = f"{r.get('prompt_id', r.get('id'))}|{r.get('salt') or ''}"
            else:
                continue
            rows.append({"source": str(f.relative_to(run_dir)), "engine": r.get("engine") or str(
                f.relative_to(run_dir).parent), "level": level, "key": key, "salt": r.get("salt"),
                "prompt": r.get("prompt_id", r.get("id", key))})
    return rows


def _salt_of(body: dict) -> str | None:
    """The bench serve reuse tag ("[ik-...] ") at the start of the first user message, if any."""
    for m in body.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            if c.startswith("[ik-") and "] " in c[:40]:
                return c[1:c.index("] ")]
            return None
    return None


def has_load_data(run_dir: Path) -> bool:
    return bool(_load_requests(run_dir))


def gate_unique_prompts(run_dir: Path) -> dict:
    reqs = _load_requests(run_dir)
    if not reqs:
        return _gate("unique_prompts", "NOT RUN", "no load requests in the run dir (bench serve load-<level>/ "
                     "inputs or outputs, or a *schedule*.jsonl / *requests*.jsonl)")
    seen: dict[tuple, dict] = {}
    repeats, salted = [], 0
    for r in reqs:
        scope = (r["engine"], r["level"])
        counts = seen.setdefault(scope, {})
        counts[r["key"]] = counts.get(r["key"], 0) + 1
        salted += bool(r["salt"])
        if counts[r["key"]] == 2:
            repeats.append({"source": r["source"], "engine": r["engine"], "level": r["level"], "prompt": r["prompt"]})
    metrics = {"requests": len(reqs), "levels": len(seen), "repeated_prompts": len(repeats), "salted_reuses": salted}
    lim = []
    if salted:
        lim.append(f"{salted} load request(s) reuse a prompt behind a unique tag (the prompt pool ran out): exact and "
                   "prefix caches cannot serve them, a near-duplicate cache could")
    if repeats:
        return _gate("unique_prompts", "FAIL", f"{len(repeats)} prompt(s) repeat inside a load level (a cache would "
                     "serve them)", metrics, repeats[:50], lim)
    return _gate("unique_prompts", "PASS", f"{len(reqs)} load requests over {len(seen)} level(s): no prompt repeats "
                 "inside a level", metrics, [], lim)


def static_probe_jobs(run_dir: Path, engine: str) -> dict:
    """Hidden-probe replies a load runner saved under outputs/<engine>/probes/<id>.json (see probe_bodies)."""
    d = run_dir / "outputs" / engine / "probes"
    jobs: dict[str, list] = {}
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        o = _read_json(p)
        if not isinstance(o, dict) or not o.get("kind"):
            continue
        lp = o.get("logprobs")
        lp = lp.get("content") if isinstance(lp, dict) else lp
        rep = Reply(status=200 if not o.get("error") else int(o.get("status") or 500), error=o.get("error"),
                    text=o.get("text") or "", reasoning=o.get("reasoning") or "", token_ids=o.get("token_ids"),
                    lp=lp if isinstance(lp, list) and lp else None, finish_reason=o.get("finish_reason"),
                    usage=o.get("usage"))
        kind = o["kind"]
        job = Job(kind, o.get("group") or p.stem, {}, {}, rep)
        jobs.setdefault("logprobs" if kind == "logprobs" else "sampling", []).append(job)
    return jobs


# --------------------------------------------------------------------------- record

def _metrics_of(gates: list[dict]) -> list[dict]:
    out = []
    for g in gates:
        for k, v in g["metrics"].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            unit = "x" if "speedup" in k or "ratio" in k else ("tokens" if "divergence" in k or "early" in k
                                                                else "fraction" if "coverage" in k else "count")
            stat = "median" if "median" in k else "count"
            out.append(ev.metric(f"{g['gate']}.{k}", v, unit, stat))
    return out


def overall(gates: list[dict]) -> str:
    """The record's verdict. A gate the inputs asked for (or allowed) that ended NOT RUN counts as
    UNPROVEN: missing evidence is never a pass."""
    return combine_claims("UNPROVEN" if g["verdict"] == "NOT RUN" and g.get("applicable") else g["verdict"]
                          for g in gates)


def _record(call, gates: list[dict], *, live: bool, engine: str | None, model: str | None, prompt_set: dict | None,
            log: list, extra: dict) -> dict:
    verdict = overall(gates)
    store = ev.Store()
    with store.staging_dir() as staging:
        artifacts = []
        if log:
            p = staging / "probes.jsonl"
            p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in log), encoding="utf-8")
            artifacts.append(ev.artifact(p, staging, "probe requests and replies"))
        lim = [f"{g['gate']}: {x}" for g in gates for x in g["limitations"]]
        missing = [g["gate"] for g in gates if g["verdict"] == "NOT RUN"]
        summary = f"gates {verdict}: " + ", ".join(f"{g['gate']} {g['verdict']}" for g in gates)
        rec = ev.build(call, kind=KIND, claim_class="observed" if live else "derived", summary=summary,
                       identity_=ev.identity(engine={"name": engine}, model={"id": model}),
                       metrics=_metrics_of(gates), artifacts=artifacts, missing=missing, limitations=lim,
                       data={"verdict": verdict, "gates": {g["gate"]: g for g in gates}, "prompt_set": prompt_set,
                             "not_run_but_expected": [g["gate"] for g in gates
                                                      if g["verdict"] == "NOT RUN" and g.get("applicable")],
                             "gates_format": "inferencekit.gates.v2", **extra})
        return store.save(rec, staging)


def _load_prompts(name_or_path: str) -> list[dict]:
    try:
        items = ps.load(name_or_path)
    except (OSError, ValueError) as e:
        raise OpError(f"cannot read prompt set {name_or_path!r}: {e}") from None
    if not items:
        raise OpError(f"prompt set {name_or_path!r} is empty")
    return items


def _extra(text: str | None) -> dict:
    if not text:
        return {}
    try:
        val = json.loads(text)
    except ValueError:
        raise OpError("extra_body must be a JSON object") from None
    if not isinstance(val, dict):
        raise OpError("extra_body must be a JSON object")
    return val


def _template_kwargs(extra: dict, flag: str | None, items: list[dict]) -> dict:
    """chat_template_kwargs for every probe, merged per key: --extra-body's, --chat-template-kwargs, then the
    prompt records' own field when every record carries the same one (probes are built from the set's text,
    so a per-record value that differs between records cannot follow a probe)."""
    try:
        own = json.loads(flag) if flag else {}
    except ValueError:
        raise OpError("chat_template_kwargs must be a JSON object") from None
    if not isinstance(own, dict):
        raise OpError("chat_template_kwargs must be a JSON object")
    recs = [it.get("chat_template_kwargs") for it in items if isinstance(it, dict)]
    shared = recs[0] if recs and isinstance(recs[0], dict) and all(r == recs[0] for r in recs) else None
    if shared is None and any(isinstance(r, dict) for r in recs):
        raise OpError("prompt records carry different chat_template_kwargs; gates probes need one setting for "
                      "the whole set (pass --chat-template-kwargs)")
    out: dict = {}
    for layer in (extra.get("chat_template_kwargs"), own, shared):
        if isinstance(layer, dict):
            out.update(layer)
    return out


def _api_key(env_name: str | None) -> str | None:
    if not env_name:
        return None
    v = os.environ.get(env_name)
    if not v:
        raise OpError(f"environment variable {env_name} is empty")
    return v


@operation("run_gates", ("gates", "run"), "Anti-cheat gates: probe a live OpenAI-compatible endpoint (caching, "
           "token counts, exact lengths, sampling/logprob probes, rare-token pack) and check a bench serve run dir "
           "(unique prompts per load level, lengths vs the reference). Each gate: PASS/FAIL/UNPROVEN/NOT RUN; a "
           "refused probe is evidence, and an expected gate that did not run makes the verdict UNPROVEN.",
           [Arg("base_url", "string", "Endpoint base URL, e.g. http://127.0.0.1:8000 (live gates need it)."),
            Arg("model", "string", "Model name to send (default: the first id from /v1/models)."),
            Arg("engine", "string", "Engine label: names outputs/<engine>/ in the run dir and the record."),
            Arg("run_dir", "string", "bench serve run directory for the static checks."),
            Arg("reference_engine", "string", "Reference engine label under <run_dir>/outputs/ for length checks."),
            Arg("reference_outputs", "string", "Or: a reference outputs directory (<bucket>/<id>.json)."),
            Arg("reference_url", "string", "Reference endpoint for the rare-token pack (misses it shares are "
                "not counted)."),
            Arg("reference_api_key_env", "string", "Environment variable with the reference endpoint's API key "
                "(default: none)."),
            Arg("spec_off_url", "string", "Same model without speculative decoding: runs the spec_exact gate with "
                "base_url as the speculative engine."),
            Arg("prompts", "string", "Prompt set for the probes: 'dev' or a .jsonl path (use the sealed set for an "
                "evaluation).", default="dev"),
            Arg("gates", "array", "Gates to run (default: every gate the inputs allow).", choices=GATES),
            Arg("tokenizer", "string", "Hugging Face tokenizer id for re-tokenized counts (needs transformers)."),
            Arg("max_tokens", "integer", "Generation length of the probes.", default=64),
            Arg("cache_prompts", "integer", "Prompts replayed by the caching gate.", default=6),
            Arg("sampling_prompts", "integer", "hidden_probes: prompts sampled at T=1 (each sent greedy, twice with "
                "one seed, once with another). FAIL needs the samples to equal greedy on every prompt and at "
                "least 4 prompts; >= 3/4 equal is UNPROVEN. Default 8, or fewer (not below 4) when the prompt "
                "set is too small, recorded as a limitation."),
            Arg("threshold", "number", "Caching gate: largest allowed repeat speedup - 1.", default=0.10),
            Arg("length_tolerance", "number", "Per-prompt length tolerance vs the reference.", default=0.10),
            Arg("max_length_outliers", "number", "Share of prompts allowed outside the length tolerance (missing "
                "outputs count).", default=0.10),
            Arg("spec_limit", "integer", "spec_exact: prompts used (0 = all).", default=16),
            Arg("early_tokens", "integer", "spec_exact: a divergence before this token is 'early'.", default=32),
            Arg("spec_tolerance", "integer", "spec_exact: early divergences allowed above the A/A floor.", default=0),
            Arg("seed", "integer", "Probe seed (default: drawn at random for every run and recorded; pass it only to "
                "replay a run)."),
            Arg("echo_tokens", "integer", "max_tokens of the echo / rare-string probes (room for a reasoning "
                "model's preamble).", default=64),
            Arg("extra_body", "string", "JSON object merged into every request, e.g. "
                "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}'."),
            Arg("api_key_env", "string", "Name of the environment variable holding the API key (value never stored)."),
            Arg("timeout", "number", "Per-request timeout in seconds.", default=120.0),
            Arg("chat_template_kwargs", "string", "JSON object sent as chat_template_kwargs on every request, e.g. "
                "'{\"enable_thinking\": false}' (merged over --extra-body's; a chat_template_kwargs field shared by "
                "every record of --prompts is merged over this, as bench serve does per request).")])
def run_gates(call, base_url, model, engine, run_dir, reference_engine, reference_outputs, reference_url,
              reference_api_key_env, spec_off_url, prompts, gates, tokenizer, max_tokens, cache_prompts, threshold,
              length_tolerance, max_length_outliers, spec_limit, early_tokens, spec_tolerance, seed, echo_tokens,
              extra_body, api_key_env, timeout, chat_template_kwargs=None, sampling_prompts=None):
    wanted = set(gates or GATES)
    if not base_url and not run_dir:
        raise OpError("give base_url (live gates) and/or run_dir (static checks)")
    if sampling_prompts is not None and sampling_prompts < 1:
        raise OpError(f"sampling_prompts must be at least 1 (FAIL needs at least {SAMPLING_MIN_FAIL}; the default "
                      f"is {SAMPLING_DEFAULT})")
    if "spec_exact" in wanted and not spec_off_url and gates:
        raise OpError("spec_exact needs spec_off_url (the same model without speculation)")
    key = _api_key(api_key_env)
    ref_key = _api_key(reference_api_key_env)
    extra = _extra(extra_body)
    count_tokens = load_tokenizer(tokenizer)
    seed = secrets.randbits(63) if seed is None else int(seed)
    results, log = [], []
    items = _load_prompts(prompts)
    ctk = apply_pin(_template_kwargs(extra, chat_template_kwargs, items), run_dir, "gates run")
    if ctk:
        extra["chat_template_kwargs"] = ctk
    rd = Path(run_dir) if run_dir else None
    if rd is not None and not rd.is_dir():
        raise OpError(f"run dir {rd} not found")
    # Gates the inputs ask for: named in --gates, or (by default) every gate the inputs make possible.
    if gates:
        expected = set(gates)
    else:
        expected = set()
        if base_url:
            expected |= LIVE - {"spec_exact"}
            if spec_off_url:
                expected.add("spec_exact")
        if rd is not None and engine:
            expected |= {"token_count", "output_length"}
        if rd is not None and has_load_data(rd):
            expected.add("unique_prompts")
    jobs: dict = {}
    client = None
    up, why_down = True, ""
    sampling_note = None
    if base_url:
        client = Client(base_url, model, key, timeout, extra=extra)
        ref_client = (Client(reference_url, None, ref_key, timeout, extra=extra)
                      if reference_url and "rare_tokens" in wanted else None)
        n_sampling, sampling_note = sampling_count(len(probe_pool(items)), wanted & LIVE, cache_prompts,
                                                   sampling_prompts)
        jobs = plan_and_run(client, items, gates=wanted & LIVE, max_tokens=max_tokens, seed=seed,
                            n_cache=cache_prompts, log=log, reference=ref_client, echo_tokens=echo_tokens,
                            n_sampling=n_sampling)
        up, why_down = alive(jobs)
    down = _gate("", "NOT RUN", f"the endpoint answered no probe: {why_down}")
    static_rows = static_counts(rd, engine) if rd is not None and engine else []
    static_probes = static_probe_jobs(rd, engine) if rd is not None and engine else {}

    def live(name, fn):
        if not client:
            return _gate(name, "NOT RUN", "no base_url")
        return fn() if up else {**down, "gate": name}

    for g in GATES:
        if g not in wanted:
            continue
        if g == "caching":
            res = live(g, lambda: gate_caching(jobs, threshold))
        elif g == "token_count":
            if (client and up) or static_rows:
                res = gate_token_count(jobs if up else {}, count_tokens, 0.02, static_rows)
            elif client:
                res = {**down, "gate": g}
            else:
                res = _gate(g, "NOT RUN", "no base_url and no run-dir outputs (pass --engine)")
        elif g == "output_length":
            parts = []
            if client:
                parts.append(live(g, lambda: gate_output_length_live(jobs, count_tokens)))
            if rd is not None and engine:
                ref = Path(reference_outputs) if reference_outputs else (
                    rd / "outputs" / reference_engine if reference_engine else None)
                parts.append(gate_output_length_static(rd, engine, ref, length_tolerance, max_length_outliers))
            res = _merge(g, parts) if parts else _gate(g, "NOT RUN", "no base_url and no run dir + engine")
        elif g == "unique_prompts":
            res = gate_unique_prompts(rd) if rd is not None else _gate(g, "NOT RUN", "no run_dir")
        elif g == "spec_exact":
            if client and spec_off_url:
                off = Client(spec_off_url, None, key, timeout, extra=extra)
                sel = items[:spec_limit] if spec_limit else items
                res = spec_exact(client, off, sel, max_tokens, early_tokens, spec_tolerance, log)
            else:
                res = _gate(g, "NOT RUN", "needs base_url and spec_off_url (or use gates spec-exact)")
        elif g == "hidden_probes":
            parts = []
            if client:
                part = live(g, lambda: gate_hidden_probes(jobs))
                if sampling_note:
                    part["limitations"] = [*part.get("limitations", []), sampling_note]
                parts.append(part)
            if static_probes:
                parts.append(gate_hidden_probes(static_probes, where="run dir"))
            res = _merge(g, parts) if parts else _gate(g, "NOT RUN", "no base_url and no probe replies in the run dir")
        else:  # rare_tokens
            res = live(g, lambda: gate_rare(jobs))
        res["applicable"] = g in expected
        results.append(res)
    return _record(call, results, live=client is not None, engine=engine, model=client.model if client else model,
                   prompt_set=ps.identity(prompts), log=log,
                   extra={"endpoint_kind": "openai-chat" if client else None, "max_tokens": max_tokens,
                          "seed": seed, "expected_gates": sorted(expected)})


def _merge(name: str, parts: list[dict]) -> dict:
    """One gate from its live and run-dir halves."""
    if len(parts) == 1:
        return parts[0]
    tags = ("live", "run dir")
    return _gate(name, combine_claims(p["verdict"] for p in parts),
                 " | ".join(f"{t}: {p['verdict']}: {p['reason']}" for t, p in zip(tags, parts)),
                 {f"{t.replace(' ', '_')}_{k}": v for t, p in zip(tags, parts) for k, v in p["metrics"].items()},
                 [d for p in parts for d in p["details"]], [x for p in parts for x in p["limitations"]])


@operation("gates_spec_exact", ("gates", "spec-exact"), "Speculative-decoding exactness: greedy outputs of engine A "
           "(speculation on) vs B (off) on the prompt set; FAIL if early divergences exceed B's own A/A floor.",
           [Arg("a", "string", "Base URL of the engine with speculation / MTP.", required=True),
            Arg("b", "string", "Base URL of the same model without speculation.", required=True),
            Arg("model", "string", "Model name for A (default: first id from A's /v1/models)."),
            Arg("model_b", "string", "Model name for B (default: first id from B's /v1/models)."),
            Arg("prompts", "string", "Prompt set: 'dev' or a .jsonl path.", default="dev"),
            Arg("limit", "integer", "Use the first N prompts (0 = all).", default=0),
            Arg("max_tokens", "integer", "Greedy generation length.", default=128),
            Arg("early_tokens", "integer", "A divergence before this token counts as early.", default=32),
            Arg("tolerance", "integer", "Early divergences allowed above the A/A floor.", default=0),
            Arg("extra_body", "string", "JSON object merged into every request."),
            Arg("api_key_env", "string", "Name of the environment variable holding the API key."),
            Arg("timeout", "number", "Per-request timeout in seconds.", default=120.0)])
def gates_spec_exact(call, a, b, model, model_b, prompts, limit, max_tokens, early_tokens, tolerance, extra_body,
                     api_key_env, timeout):
    key = _api_key(api_key_env)
    extra = _extra(extra_body)
    ca, cb = Client(a, model, key, timeout, extra=extra), Client(b, model_b, key, timeout, extra=extra)
    items = _load_prompts(prompts)
    items = items[:limit] if limit else items
    log: list = []
    res = spec_exact(ca, cb, items, max_tokens, early_tokens, tolerance, log)
    res["applicable"] = True
    return _record(call, [res], live=True, engine=None, model=ca.model, prompt_set=ps.identity(prompts), log=log,
                   extra={"endpoint_kind": "openai-chat", "max_tokens": max_tokens})
