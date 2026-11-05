"""Fake tokenizers and engines for the skeleton server tests.

Tokens are characters: id = Unicode code point, and ids 0-2 are special tokens (<bos>, <im>, <eos>).
The engines implement ikengine.engine_api.Engine (stream -> StepOut, vocab_size, max_seq,
logprobs_supported) and record every call, so a test can check what the server asked for.

  ScriptEngine  streams a fixed token list (cycled); logprobs and alternatives are simple functions of
                the position, so the response format can be checked exactly.
  CharLM        an honest toy language model for the inferencekit checks: greedy output echoes the last
                delimited block of the prompt (the gates' echo probes) or writes pseudo-random words with
                an EOS after 24-71 tokens; every step has a real distribution over 30 candidates, sampled
                with temperature / top-k / top-p / min-p and a per-request seed exactly as a sampler
                should; logprobs are the log-softmax of the raw logits.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

SKELETON = Path(__file__).resolve().parents[2] / "plugin" / "skills" / "optimizing-kernels" / "skeleton"
if str(SKELETON) not in sys.path:
    sys.path.insert(0, str(SKELETON))

from ikengine.engine_api import SamplingParams, StepOut  # noqa: E402

BOS, IM, EOS = 0, 1, 2
SPECIAL = {BOS: "<bos>", IM: "<im>", EOS: "<eos>"}


def ids(text):
    return [ord(c) for c in text]


class FakeTokenizer:
    """Character-level. The chat template is "[role]content..." plus "[assistant]" for the generation
    prompt, and a "~" when the template kwargs say enable_thinking: false. add_special_tokens adds <bos>."""

    eos_token_id = EOS
    version = "fake"

    def __init__(self, path=None, chat_template=None):
        self.path, self.chat_template = path, chat_template
        self.template_calls = []

    def encode(self, text, add_special_tokens=True):
        return ([BOS] if add_special_tokens else []) + ids(text)

    def apply_chat_template(self, messages, add_generation_prompt=True, add_special_tokens=False, **kwargs):
        self.template_calls.append(dict(kwargs))
        text = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "[assistant]"
        if kwargs.get("enable_thinking") is False:
            text += "~"
        return self.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, toks, skip_special_tokens=True):
        return "".join(("" if skip_special_tokens else SPECIAL[i]) if i in SPECIAL else chr(i) for i in toks)


class ByteTokenizer:
    """Byte-level: id = one UTF-8 byte, so a character can span several tokens."""

    def decode(self, toks, skip_special_tokens=True):
        return bytes(toks).decode("utf-8", errors="replace")


class _Recorder:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def _record(self, prompt_ids, max_tokens, sampling, top_logprobs):
        call = {"prompt_ids": list(prompt_ids), "max_tokens": max_tokens, "sampling": sampling,
                "top_logprobs": top_logprobs, "t_start": time.perf_counter(), "t_end": None, "yielded": 0,
                "closed_early": False}
        with self._lock:
            self.calls.append(call)
        return call


class ScriptEngine(_Recorder):
    """Streams a fixed token list (cycled), one token per `delay` seconds. Position i has logprob
    -0.25 * (i + 1); its alternatives are (token + j, logprob - j) for j = 0..k-1 (the chosen token first)."""

    def __init__(self, tokens, delay=0.0, max_seq=64, logprobs_supported=True, vocab_size=1 << 20, **attrs):
        super().__init__()
        self.tokens, self.delay, self.max_seq = list(tokens), delay, max_seq
        self.logprobs_supported, self.vocab_size = logprobs_supported, vocab_size
        self.__dict__.update(attrs)

    def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
        call = self._record(prompt_ids, max_tokens, sampling, top_logprobs)
        info["prefill"] = "fake"
        try:
            for i in range(max_tokens):
                if self.delay:
                    time.sleep(self.delay)
                call["yielded"] += 1
                tok = self.tokens[i % len(self.tokens)]
                if top_logprobs is None:
                    yield StepOut(tok)
                else:
                    lp = -0.25 * (i + 1)
                    yield StepOut(tok, lp, [(tok + j, lp - j) for j in range(top_logprobs)])
        except GeneratorExit:
            call["closed_early"] = call["yielded"] < max_tokens
            raise
        finally:
            call["t_end"] = time.perf_counter()


# --------------------------------------------------------------------------------------------------
# CharLM
# --------------------------------------------------------------------------------------------------
WORDS = ("alpha", "bravo", "cargo", "delta", "ember", "fable", "gamma", "harbor", "ivory", "jungle", "kettle",
         "lumen", "meadow", "nectar", "orbit", "pepper", "quartz", "river", "saddle", "timber", "umber", "velvet")
ALPHABET = "abcdefghijklmnopqrstuvwxyz 0123456789"
DELIMS = (("<<<", ">>>"), ("[[", "]]"), ("{{", "}}"), ("«", "»"), ("⟦", "⟧"), ("BEGIN:", ":END"))
N_CANDIDATES = 30


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def echo_block(prompt):
    """Text of the last delimited block in the prompt (the payload of an echo probe), or None."""
    best = None
    for o, c in DELIMS:
        for m in re.finditer(re.escape(o) + "(.*?)" + re.escape(c), prompt, re.S):
            if best is None or m.start() > best[0]:
                best = (m.start(), m.group(1))
    return best[1] if best else None


def greedy_ids(prompt, n):
    """The greedy token at each of n positions: the echo block (then EOS) or words with an EOS at 24..71;
    after the EOS (ignore_eos) more words follow."""
    block = echo_block(prompt)
    rng = random.Random(_seed("greedy", prompt))
    text = block if block is not None else ""
    eos = len(text) if block is not None else 24 + _seed("eos", prompt) % 48
    while len(text) < max(n, eos) + 1:
        text += (" " if text else "") + rng.choice(WORDS)
    out = ids(text[:max(n, eos) + 1])
    out[eos] = EOS
    return out[:n]


def candidates(g):
    """(token ids, raw logits) of one step whose argmax is g."""
    alts = [ord(c) for c in ALPHABET if ord(c) != g][:N_CANDIDATES - 1]
    return [g] + alts, [3.0] + [1.5 - 0.1 * i for i in range(len(alts))]


def log_softmax(logits):
    m = max(logits)
    z = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - z for x in logits]


def sample(toks, logits, sp: SamplingParams, rng):
    """Temperature, then top-k, then top-p, then min-p (the vLLM / Hugging Face order); greedy at T 0."""
    if sp.greedy:
        return max(zip(toks, logits), key=lambda t: (t[1], -t[0]))[0]
    order = sorted(zip(toks, [x / sp.temperature for x in logits]), key=lambda t: (-t[1], t[0]))
    if sp.top_k and sp.top_k > 0:
        order = order[:sp.top_k]
    probs = [math.exp(p) for p in log_softmax([x for _, x in order])]
    if sp.top_p < 1.0:
        keep, cum = 0, 0.0
        for p in probs:
            keep += 1
            cum += p
            if cum >= sp.top_p:
                break
        order, probs = order[:keep], probs[:keep]
    if sp.min_p > 0:
        floor = sp.min_p * max(probs)
        kept = [(t, p) for t, p in zip(order, probs) if p >= floor]
        order, probs = [t for t, _ in kept], [p for _, p in kept]
    r = rng.random() * sum(probs)
    for (tok, _), p in zip(order, probs):
        r -= p
        if r <= 0:
            return tok
    return order[-1][0]


class CharLM(_Recorder):
    vocab_size = 0x110000
    logprobs_supported = True
    eos_token_ids = frozenset({EOS})

    def __init__(self, max_seq=8192, ttft_s=0.0, per_token_s=0.0, **attrs):
        super().__init__()
        self.max_seq, self.ttft_s, self.per_token_s = max_seq, ttft_s, per_token_s
        self.numerics = {"weights": "toy logits", "kv": "none"}
        self.__dict__.update(attrs)

    def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
        call = self._record(prompt_ids, max_tokens, sampling, top_logprobs)
        prompt = "".join(chr(t) for t in prompt_ids if t not in SPECIAL)
        greedy = greedy_ids(prompt, max_tokens)
        rng = random.Random(_seed("sample", sampling.seed, prompt) if sampling.seed is not None else os.urandom(8))
        if self.ttft_s:
            time.sleep(self.ttft_s)
        info["prefill"] = "fake"
        try:
            for g in greedy:
                toks, logits = candidates(g)
                tok = sample(toks, logits, sampling, rng)
                call["yielded"] += 1
                if top_logprobs is None:
                    yield StepOut(tok)
                else:
                    lps = dict(zip(toks, log_softmax(logits)))
                    top = sorted(lps.items(), key=lambda t: (-t[1], t[0]))[:top_logprobs]
                    yield StepOut(tok, lps[tok], top)
                if self.per_token_s:
                    time.sleep(self.per_token_s)
        except GeneratorExit:
            call["closed_early"] = call["yielded"] < max_tokens
            raise
        finally:
            call["t_end"] = time.perf_counter()


class Model:
    """A model object as an engine would hold it (only `numerics` and `eos_token_ids` are read)."""

    def __init__(self, numerics):
        self.numerics = numerics
        self.eos_token_ids = frozenset({EOS})


def builder(engine, record=None):
    """A build_engine stand-in that returns `engine` and records its arguments."""
    def build_engine(model_dir, device="cuda", max_seq=4096, act="as_shipped", kv="as_shipped"):
        if record is not None:
            record.update(model_dir=model_dir, device=device, max_seq=max_seq, act=act, kv=kv)
        engine.max_seq = max_seq
        return engine
    return build_engine
