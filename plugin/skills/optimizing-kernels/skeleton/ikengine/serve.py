"""OpenAI-compatible streaming HTTP server for an ikengine Engine (batch 1), so a client-side timer
(`inferencekit bench serve`) and the checks (`inferencekit gates run`, `inferencekit quality ...`) can
measure it like any other engine.

  python -m ikengine.serve --model DIR --port P [--served-model-name N] [--chat-template FILE]
                           [--default-chat-template-kwargs JSON] [--max-seq N] [--act as_shipped|a16]
                           [--kv as_shipped|bf16] [--max-queue N] [--device cuda] [--host 127.0.0.1]
                           [--tokenizer DIR] [--engine-json PATH] [--access-log]

Standard library only, plus transformers for the tokenizer and chat template. The engine comes from
ikengine.engine.build_engine(model_dir, device, max_seq, act, kv) and is used only through the contract
in engine_api.py: Engine.stream(prompt_ids, max_tokens, SamplingParams, top_logprobs, info) -> StepOut,
plus vocab_size, max_seq and logprobs_supported. This file owns the API, the chat template, tokenization,
the queue, the stop rules and the response format; make the engine fast and leave this file alone.

Scheduling: batch 1. One worker thread owns the GPU and runs one request at a time; other requests
wait in a FIFO queue (`--max-queue N` answers 503 beyond N waiting requests; 0 = unbounded). A client
that disconnects (streaming or not) cancels its request, also while it is still waiting.

Endpoints
  GET  /health                 {"status": "ok", "waiting": n, "running": bool, "completed": n}
  GET  /v1/models              the served model, with max_model_len
  GET  /engine/config          the resolved configuration (same JSON as the startup line)
  GET  /get_server_info        {"max_running_requests": 1, ...}: requests served at once (SGLang's
                               endpoint, which `inferencekit quality gsm8k --concurrency 0` reads)
  POST /v1/chat/completions    messages -> chat template -> tokens
  POST /v1/completions         prompt: a string, a list of token ids, or a 1-element list of either
  POST /tokenize               {"prompt": str} or {"messages": [...]} -> {"tokens", "count", "max_model_len"}
  POST /detokenize             {"tokens": [...]} -> {"prompt": str}

Sampling (as vLLM): temperature, top_k, top_p, min_p and seed. A field the request leaves out (or sends
as null) takes the model's default from generation_config.json (SamplingParams.from_generation_config),
so a request with no sampling fields samples exactly as the baseline does. temperature 0 (or below 1e-5)
is greedy. The same seed gives the same tokens. Penalties and logit_bias are not implemented: only their
neutral values are accepted. n and best_of must be 1.

Logprobs are the log-softmax of the raw logits over the full vocabulary (before temperature, top-k and
top-p). Chat: logprobs: true and top_logprobs 0..20; completions: logprobs 0..20. A logprob request
always takes the engine's full output-layer path. Response shapes and order are vLLM's: chat lists k
entries, the chosen token first and then the most likely others in rank order (decoding greedily that is
exactly the k most likely, descending); completions gives one {token: logprob} object per position with
the chosen token first and then the k most likely. return_tokens_as_token_ids writes "token_id:<id>" for
every token string. -inf is sent as -9999.0, as vLLM does.

Stop rules (as vLLM). EOS ids are generation_config.json's plus the tokenizer's. `ignore_eos: true`
ignores them; `stop_token_ids` apply even then. A stop token is generated: it counts in token_ids and
usage, but its text is not output unless include_stop_str_in_output is true. `stop` strings end the
output where the string starts (or after it with include_stop_str_in_output); while streaming, the last
len(longest stop string) - 1 characters are held back until they cannot start a stop string. Stop
strings are checked once more than min_tokens tokens exist. `min_tokens` is accepted with ignore_eos:
true and no stop_token_ids (the body `bench serve --engine-kind vllm` sends): nothing can end the output
early then, so the token count is exactly vLLM's. The one difference: vLLM also masks the EOS ids until
min_tokens tokens exist, so an EOS the model picks here is generated as an ordinary token where vLLM
would have sampled another one. Anywhere else min_tokens would need the engine to mask logits, which
Engine.stream has no input for: 400. `max_tokens` (or `max_completion_tokens`) defaults to
max_seq - prompt_len. Unknown fields are 400.

Chat template: the tokenizer's own, or `--chat-template FILE`. `--default-chat-template-kwargs JSON`
(e.g. '{"enable_thinking": false}') applies to every request; a request's chat_template_kwargs win per
key. Chat prompts are tokenized without adding special tokens (the template holds them), completions
with them; both follow the request's add_special_tokens when it is sent.

Streaming (stream: true): Server-Sent Events, `data: <json>\\n\\n`, terminated by `data: [DONE]`.
  - Exactly one chunk per generated token, sent as soon as that token is on the host. The first
    token's chunk also carries delta.role = "assistant" (chat): no role-only chunk is sent first, so
    time to first token = time to first event.
  - finish_reason ("stop" for a stop token or string, "length" for max_tokens) is set on the last chunk.
  - vLLM's return_token_ids extension is on unless the request sends return_token_ids: false:
    choices[0].token_ids = [the token's id] in every chunk, and prompt_token_ids (top level for chat,
    in the choice for completions) in the first chunk.
  - stream_options.include_usage: every chunk carries "usage": null and a final chunk with
    "choices": [] carries usage {prompt_tokens, completion_tokens, total_tokens}.
  - The last chunk with content in it (the usage chunk if requested, else the last token chunk) also
    carries "inferencekit": {"queue_ms", "prompt_tokens", ...whatever the engine wrote into `info`}.

Startup: tokenizer, engine, a short warm-up generation (so the first request does not pay for graph
capture), then the resolved configuration on stderr as
  [serve] resolved_config {...json...}
  {"inferencekit_resolved_config": {...json...}}
(the second line is the format inferencekit's managed server runs read from a custom engine's log; its
top-level "cuda_graphs" feeds the CUDA-graph check). The server binds its port, then fills ENGINE.json
(`--engine-json PATH`; default: ENGINE.json at the top of the git repository it runs in, else in the
current directory; "" skips it) with
  "launch"            this process's command line, made to run from the repository root (a fresh copy of
                      the repository, no .git needed):
                      started in a subdirectory it becomes `sh -c 'cd <dir> && exec ...'`, and an ikengine
                      found through PYTHONPATH gets `env PYTHONPATH=<dir>`
  "port"              the port it listens on
  "numerics"          the model module's `numerics` dict (engine_api.describe_numerics), plus "requested"
                      (the --act / --kv flags) and "changed" (what differs from the checkpoint as shipped:
                      the module's own list when it declares one, else the flags that are not as_shipped)
  "numerics_summary"  the same in one line, ending "(numerics changed: ...)" when something changed
  "resolved_config"   the resolved configuration, rewritten on every start.
"_written_by_ikengine" records a digest of each value the server wrote. On the next start those fields
are refreshed (a restart with other flags updates launch and numerics); a field the server did not write,
or one changed by hand since, is kept as it is (delete it to hand it back). Every other field ("install",
"notes", ...) is yours. Declare numerics in the model module (its `numerics` dict), not by editing
ENGINE.json by hand.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import queue
import select
import shlex
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine_api import MAX_TOP_LOGPROBS, SamplingParams, StepOut, describe_numerics

ENGINE_NAME = "ikengine"
CONFIG_PREFIX = "[serve] resolved_config "
MAX_BODY_BYTES = 32 << 20
SAMPLING_EPS = 1e-5          # vLLM: a temperature below this is greedy
LOGPROB_FLOOR = -9999.0      # vLLM sends -inf as this (JSON has no infinity)
SEED_RANGE = (-(1 << 63), (1 << 64) - 1)   # what torch.Generator.manual_seed accepts
WARMUP_TOKENS = 2
PEER_CHECK_S = 0.1           # how often a handler checks that its client is still connected
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the directory that holds ikengine/
WRITTEN_KEY = "_written_by_ikengine"   # ENGINE.json: field -> digest of the value the server wrote


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# =================================================================================================
# Request validation (pure Python; no engine, no torch)
# =================================================================================================
class RequestError(Exception):
    def __init__(self, message, param=None, status=400, type_="invalid_request_error", code=None):
        super().__init__(message)
        self.message, self.param, self.status, self.type, self.code = message, param, status, type_, code

    def body(self):
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


_COMMON_FIELDS = {"model", "stream", "stream_options", "max_tokens", "temperature", "top_p", "top_k", "min_p",
                  "seed", "n", "ignore_eos", "stop", "stop_token_ids", "include_stop_str_in_output", "logprobs",
                  "return_token_ids", "return_tokens_as_token_ids", "skip_special_tokens", "add_special_tokens",
                  "user", "presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias", "min_tokens"}
CHAT_FIELDS = _COMMON_FIELDS | {"messages", "max_completion_tokens", "top_logprobs", "chat_template_kwargs",
                                "add_generation_prompt"}
COMPLETION_FIELDS = _COMMON_FIELDS | {"prompt", "echo", "best_of", "suffix"}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _bool(body, name, default):
    v = body.get(name)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise RequestError(f"{name} must be a boolean", name)
    return v


def _check_fields(body, allowed):
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise RequestError(f"unsupported field(s) {unknown}: this server rejects parameters it does not "
                           f"implement instead of ignoring them", unknown[0])


def parse_sampling(body, defaults, vocab_size=None):
    """The request's SamplingParams: its own fields over the model's defaults. Values a sampler cannot
    honour exactly are 400; nothing is silently ignored."""
    t = body.get("temperature")
    if t is not None and not (_is_num(t) and t >= 0):
        raise RequestError(f"temperature must be a number >= 0, got {t!r}", "temperature")
    p = body.get("top_p")
    if p is not None and not (_is_num(p) and 0 < p <= 1):
        raise RequestError("top_p must be in (0, 1]", "top_p")
    k = body.get("top_k")
    if k is not None and not (_is_int(k) and k >= -1):
        raise RequestError("top_k must be an integer >= 1, or 0 / -1 (off)", "top_k")
    mp = body.get("min_p")
    if mp is not None and not (_is_num(mp) and 0 <= mp <= 1):
        raise RequestError("min_p must be in [0, 1]", "min_p")
    seed = body.get("seed")
    if seed is not None and not (_is_int(seed) and SEED_RANGE[0] <= seed <= SEED_RANGE[1]):
        raise RequestError("seed must be an integer in the 64-bit range", "seed")
    for name, neutral in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1)):
        v = body.get(name)
        if v is not None and (not _is_num(v) or v != neutral):
            raise RequestError(f"{name} is not implemented (only {neutral}, which changes nothing)", name)
    if body.get("logit_bias"):
        raise RequestError("logit_bias is not implemented", "logit_bias")
    if body.get("n") not in (None, 1):
        raise RequestError("n must be 1 (one sequence per request)", "n")
    sp = defaults.merged(body)
    top_k = sp.top_k
    if top_k < 0 or (vocab_size and top_k >= vocab_size):
        top_k = 0                                   # -1 and k >= vocabulary both mean "off"
    if sp.temperature < SAMPLING_EPS:               # greedy: the truncations cannot change the argmax
        return dataclasses.replace(sp, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0)
    return dataclasses.replace(sp, top_k=top_k)


def _parse_stop(body):
    stop = body.get("stop")
    if stop is None:
        return ()
    if isinstance(stop, str):
        stop = [stop]
    if not (isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
        raise RequestError("stop must be a string or a list of strings", "stop")
    return tuple(dict.fromkeys(s for s in stop if s))


def _check_min_tokens(min_tokens, max_tokens, stop_ids):
    """Accept min_tokens only where it cannot change the token count; 400 otherwise.

    vLLM's min_tokens masks stop_token_ids and the EOS ids (even under ignore_eos) until min_tokens tokens
    exist. With no stop ids in force here (ignore_eos: true and no stop_token_ids, stop_ids empty) nothing
    can end the output before max_tokens, so the count matches vLLM's; only an EOS the model picks is kept
    as an ordinary token instead of being masked. This is the body `bench serve --engine-kind vllm|sglang|
    trtllm` sends (ignore_eos + min_tokens = max_tokens). With a non-empty stop set the engine would have to
    mask those logits, which Engine.stream has no input for."""
    if min_tokens is None:
        return 0
    if not _is_int(min_tokens) or min_tokens < 0:
        raise RequestError("min_tokens must be an integer >= 0", "min_tokens")
    if min_tokens > max_tokens:
        raise RequestError(f"min_tokens ({min_tokens}) must be <= max_tokens ({max_tokens})", "min_tokens")
    if min_tokens and stop_ids:
        raise RequestError("min_tokens > 0 is only accepted with ignore_eos: true and no stop_token_ids (then it "
                           "masks nothing); otherwise it would mask stop tokens in the logits, which this server "
                           "does not implement", "min_tokens")
    return min_tokens


@dataclasses.dataclass
class GenRequest:
    """A validated generation request."""
    kind: str                      # "chat" | "completion"
    prompt_ids: list
    max_tokens: int
    sampling: SamplingParams
    stop_ids: frozenset
    stop: tuple = ()               # stop strings
    min_tokens: int = 0
    include_stop_str: bool = False
    stream: bool = False
    include_usage: bool = False
    logprobs: bool = False         # the chosen token's logprob is returned
    top_logprobs: int = 0          # and this many alternatives
    return_token_ids: bool = True
    tokens_as_ids: bool = False
    skip_special_tokens: bool = True
    model: str | None = None

    @property
    def engine_top_logprobs(self):
        """Engine.stream's top_logprobs: None = no logprobs (fast path), 0 = chosen only, k = also top-k."""
        return self.top_logprobs if self.logprobs else None


def _message(m, i):
    if not isinstance(m, dict):
        raise RequestError(f"messages[{i}] must be an object", "messages")
    extra = sorted(set(m) - {"role", "content", "name"})
    if extra:
        raise RequestError(f"messages[{i}] has unsupported field(s) {extra}", "messages")
    if not isinstance(m.get("role"), str):
        raise RequestError(f"messages[{i}].role must be a string", "messages")
    c = m.get("content")
    if c is None or isinstance(c, str):
        return {**m, "content": c or ""}
    if isinstance(c, list):
        texts = []
        for part in c:
            if not (isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)):
                raise RequestError(f"messages[{i}].content: only text parts are supported", "messages")
            texts.append(part["text"])
        return {**m, "content": "".join(texts)}
    raise RequestError(f"messages[{i}].content must be a string or a list of text parts", "messages")


def parse_messages(body):
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise RequestError("messages must be a non-empty list", "messages")
    return [_message(m, i) for i, m in enumerate(msgs)]


def template_kwargs(body, defaults):
    """The server's default chat_template_kwargs with the request's on top (the request wins per key)."""
    own = body.get("chat_template_kwargs")
    if own is None:
        own = {}
    if not isinstance(own, dict):
        raise RequestError("chat_template_kwargs must be an object", "chat_template_kwargs")
    return {**(defaults or {}), **own}


def chat_prompt_ids(body, tokenizer, default_kwargs=None):
    msgs = parse_messages(body)
    kwargs = template_kwargs(body, default_kwargs)
    agp = _bool(body, "add_generation_prompt", True)
    ast = _bool(body, "add_special_tokens", False)
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=agp, add_special_tokens=ast, **kwargs)


def completion_prompt_ids(prompt, tokenizer, add_special_tokens=True):
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
        prompt = prompt[0]
    if isinstance(prompt, str):
        if not prompt:
            raise RequestError("prompt is empty", "prompt")
        return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if isinstance(prompt, list) and prompt and all(_is_int(t) for t in prompt):
        return list(prompt)
    if isinstance(prompt, list) and len(prompt) > 1:
        raise RequestError("one prompt per request (batch 1)", "prompt")
    raise RequestError("prompt must be a non-empty string or a non-empty list of token ids", "prompt")


def _top_k_arg(v, name, what):
    if v is not None and not (_is_int(v) and 0 <= v <= MAX_TOP_LOGPROBS):
        raise RequestError(f"{name} must be {what} in [0, {MAX_TOP_LOGPROBS}]", name)
    return v


def parse_request(kind, body, tokenizer, ctx):
    """kind: "chat" | "completion". ctx: model_names, max_seq, eos_ids, vocab_size, logprobs_supported,
    sampling_defaults, chat_template_kwargs."""
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    _check_fields(body, CHAT_FIELDS if kind == "chat" else COMPLETION_FIELDS)
    model = body.get("model")
    if model is not None and model not in ctx["model_names"]:
        raise RequestError(f"The model `{model}` does not exist.", "model", status=404, type_="NotFoundError",
                           code="model_not_found")
    sampling = parse_sampling(body, ctx["sampling_defaults"], ctx.get("vocab_size"))
    stop = _parse_stop(body)
    stream = _bool(body, "stream", False)
    so = body.get("stream_options")
    include_usage = False
    if so is not None:
        if not stream:
            raise RequestError("stream_options is only allowed with stream: true", "stream_options")
        if not isinstance(so, dict) or set(so) - {"include_usage"}:
            raise RequestError("stream_options supports only include_usage", "stream_options")
        include_usage = _bool(so, "include_usage", False)

    if kind == "chat":
        lp = body.get("logprobs")
        if lp is not None and not isinstance(lp, bool):
            raise RequestError("logprobs must be a boolean", "logprobs")
        logprobs = bool(lp)
        top = _top_k_arg(body.get("top_logprobs"), "top_logprobs", "an integer")
        if top and not logprobs:
            raise RequestError("top_logprobs requires logprobs: true", "top_logprobs")
        top_logprobs = top or 0
        mt, mct = body.get("max_tokens"), body.get("max_completion_tokens")
        if mt is not None and mct is not None and mt != mct:
            raise RequestError("max_tokens and max_completion_tokens differ", "max_tokens")
        max_tokens = mt if mt is not None else mct
    else:
        if body.get("echo"):
            raise RequestError("echo is not implemented", "echo")
        if body.get("best_of") not in (None, 1):
            raise RequestError("best_of must be 1", "best_of")
        if body.get("suffix") is not None:
            raise RequestError("suffix is not implemented", "suffix")
        lp = _top_k_arg(body.get("logprobs"), "logprobs", "null or an integer")
        logprobs = lp is not None
        top_logprobs = lp or 0
        max_tokens = body.get("max_tokens")
        if "prompt" not in body:
            raise RequestError("prompt is required", "prompt")
    if logprobs and not ctx["logprobs_supported"]:
        raise RequestError("this engine does not return logprobs (Engine.logprobs_supported is false)", "logprobs")

    stop_tok = body.get("stop_token_ids")
    if stop_tok is not None and not (isinstance(stop_tok, list) and all(_is_int(t) for t in stop_tok)):
        raise RequestError("stop_token_ids must be a list of integers", "stop_token_ids")
    ignore_eos = _bool(body, "ignore_eos", False)
    flags = {name: _bool(body, name, default) for name, default in (
        ("return_token_ids", True), ("return_tokens_as_token_ids", False), ("skip_special_tokens", True),
        ("include_stop_str_in_output", False))}
    if body.get("user") is not None and not isinstance(body["user"], str):
        raise RequestError("user must be a string", "user")

    if kind == "chat":
        prompt_ids = chat_prompt_ids(body, tokenizer, ctx.get("chat_template_kwargs"))
    else:
        prompt_ids = completion_prompt_ids(body["prompt"], tokenizer, _bool(body, "add_special_tokens", True))
    vocab = ctx.get("vocab_size")
    if vocab and any(not (0 <= t < vocab) for t in prompt_ids):
        raise RequestError(f"prompt token ids must be in [0, {vocab})", "prompt")
    P, max_seq = len(prompt_ids), ctx["max_seq"]
    if max_tokens is None:
        max_tokens = max(1, max_seq - P)          # a prompt that fills the context gets the length error below
    if not _is_int(max_tokens) or max_tokens < 1:
        raise RequestError("max_tokens must be an integer >= 1", "max_tokens")
    if P + max_tokens > max_seq:
        raise RequestError(f"This model's maximum context length is {max_seq} tokens. However, you requested "
                           f"{P + max_tokens} tokens ({P} in the messages, {max_tokens} in the completion).",
                           "max_tokens")
    stop_ids = frozenset(stop_tok or ()) | (frozenset() if ignore_eos else frozenset(ctx["eos_ids"]))
    min_tokens = _check_min_tokens(body.get("min_tokens"), max_tokens, stop_ids)
    return GenRequest(kind, list(prompt_ids), max_tokens, sampling, stop_ids, stop=stop, min_tokens=min_tokens,
                      include_stop_str=flags["include_stop_str_in_output"], stream=stream,
                      include_usage=include_usage, logprobs=logprobs, top_logprobs=top_logprobs,
                      return_token_ids=flags["return_token_ids"], tokens_as_ids=flags["return_tokens_as_token_ids"],
                      skip_special_tokens=flags["skip_special_tokens"], model=model)


# =================================================================================================
# Detokenization
# =================================================================================================
class IncrementalDetokenizer:
    """Text deltas from a growing id list (the prefix/read-offset method of HF TextStreamer and vLLM).

    A token that ends inside a multi-byte UTF-8 character decodes to U+FFFD; its text is held back until
    a later token completes the character, so the deltas never contain a broken character and their
    concatenation equals decode(all ids) (flush() releases what is left at the end)."""

    def __init__(self, tokenizer, skip_special_tokens=True):
        self.tok, self.skip = tokenizer, skip_special_tokens
        self.ids, self.prefix_offset, self.read_offset = [], 0, 0

    def add(self, token_id):
        self.ids.append(token_id)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=self.skip)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=self.skip)
        if len(full) > len(prefix) and not full.endswith("�"):
            self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
            return full[len(prefix):]
        return ""

    def flush(self):
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=self.skip)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=self.skip)
        self.prefix_offset = self.read_offset = len(self.ids)
        return full[len(prefix):] if len(full) > len(prefix) else ""


def find_stop(text, new_chars, stops, include):
    """vLLM's stop-string rule: search only where a match can end in the newest `new_chars` characters;
    the first stop string in request order that matches wins. Returns the length to cut the text to,
    or None (no match)."""
    if not new_chars:
        return None
    for s in stops:
        i = text.find(s, max(0, len(text) - new_chars - len(s) + 1))
        if i >= 0:
            return min(len(text), i + len(s)) if include else i
    return None


class TokenStrings:
    """Single-token strings for logprob entries, cached: a token decodes the same way every time."""

    def __init__(self, tokenizer):
        self.tok, self._cache = tokenizer, {}

    def __call__(self, tid, as_id=False):
        if as_id:
            return f"token_id:{tid}"
        s = self._cache.get(tid)
        if s is None:
            s = self._cache[tid] = self.tok.decode([tid], skip_special_tokens=False)
        return s


# =================================================================================================
# Scheduler: one worker thread owns the engine; FIFO queue; batch 1
# =================================================================================================
class QueueFull(Exception):
    pass


class Job:
    def __init__(self, req, gone=None):
        self.req = req
        self.gone = gone            # callable: True once the client has disconnected (checked before it runs)
        self.out = queue.Queue()
        self.cancelled = threading.Event()
        self.t_enqueued = time.perf_counter()
        self.info = {}


def as_step(x):
    """A StepOut from what Engine.stream yielded (a (token_id, logprob[, top]) tuple is accepted too)."""
    if isinstance(x, StepOut):
        return x
    if isinstance(x, tuple) and 1 <= len(x) <= 3:
        return StepOut(int(x[0]), x[1] if len(x) > 1 else None, list(x[2]) if len(x) > 2 and x[2] else [])
    raise TypeError(f"Engine.stream must yield engine_api.StepOut, got {type(x).__name__}")


class Scheduler:
    """Runs engine.stream(prompt_ids, max_tokens, sampling, top_logprobs, info) for one job at a time, in
    arrival order.

    Items put on job.out: ("token", StepOut, finish_reason_or_None) per token (finish_reason only on the
    last one), then ("end",); or ("error", message). The handler may finish earlier (a stop string) and
    sets job.cancelled, which stops the worker after the token in hand."""

    def __init__(self, engine, max_queue=0):
        self.engine, self.max_queue = engine, max_queue
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self.waiting = 0
        self.running = False
        self.completed = 0
        self._thread = threading.Thread(target=self._loop, name="engine-worker", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._q.put(None)
        self._thread.join(timeout=10)

    def submit(self, req, gone=None):
        job = Job(req, gone)
        with self._lock:
            if self.max_queue and self.waiting >= self.max_queue:
                raise QueueFull(f"{self.waiting} requests waiting (--max-queue {self.max_queue})")
            self.waiting += 1
        self._q.put(job)
        return job

    def status(self):
        with self._lock:
            return {"waiting": self.waiting, "running": self.running, "completed": self.completed}

    def _loop(self):
        while True:
            job = self._q.get()
            if job is None:
                return
            with self._lock:
                self.waiting -= 1
                self.running = True
            try:
                if job.gone is not None and not job.cancelled.is_set() and job.gone():
                    job.cancelled.set()     # its client left while it waited: never start it
                if not job.cancelled.is_set():
                    self._run(job)
                else:
                    job.out.put(("end",))
            except Exception as e:  # noqa: BLE001 - e.g. a failed drain in gen.close(): keep serving
                log(f"[serve] worker error: {e!r}\n{traceback.format_exc()}")
                job.out.put(("error", f"engine error: {e!r}"))
            finally:
                with self._lock:
                    self.running = False
                    self.completed += 1

    def _run(self, job):
        req = job.req
        info = job.info
        info.update(queue_ms=round((time.perf_counter() - job.t_enqueued) * 1e3, 3), prompt_tokens=len(req.prompt_ids))
        gen = self.engine.stream(req.prompt_ids, req.max_tokens, req.sampling, req.engine_top_logprobs, info)
        n = 0
        try:
            for x in gen:
                step = as_step(x)
                n += 1
                if req.logprobs and step.logprob is None:
                    raise RuntimeError("the engine returned no logprob although the request asked for one")
                finish = "stop" if step.token_id in req.stop_ids else ("length" if n >= req.max_tokens else None)
                job.out.put(("token", step, finish))
                if finish or job.cancelled.is_set():
                    break
            else:
                if n < req.max_tokens:
                    raise RuntimeError(f"the engine stopped after {n} of {req.max_tokens} tokens")
        except Exception as e:  # noqa: BLE001 - reported to the client, logged here
            log(f"[serve] engine error: {e!r}\n{traceback.format_exc()}")
            job.out.put(("error", f"engine error: {e!r}"))
            return
        finally:
            gen.close()   # the engine drains in-flight GPU work before the next job
        job.out.put(("end",))


# =================================================================================================
# Response chunks
# =================================================================================================
def _lp(v):
    return LOGPROB_FLOOR if v is None or v != v or v < LOGPROB_FLOOR else float(v)


def _ranked(top, k):
    """The k most likely alternatives, descending (ties: lower id first), whatever order they came in."""
    return sorted(top, key=lambda t: (-t[1], t[0]))[:k] if k else []


class ResponseBuilder:
    """Builds OpenAI chat / completion chunks and final bodies for one request."""

    def __init__(self, req, tokenizer, model_name, token_str=None):
        self.req, self.tok, self.model_name = req, tokenizer, model_name
        self.tstr = token_str or TokenStrings(tokenizer)
        self.id = ("chatcmpl-" if req.kind == "chat" else "cmpl-") + uuid.uuid4().hex
        self.created = int(time.time())
        self.detok = IncrementalDetokenizer(tokenizer, req.skip_special_tokens)
        self.ids, self.lp_entries, self.deltas, self.n = [], [], [], 0
        self.text, self.sent = "", 0      # output text so far (cut at a stop string) and how much was sent
        self.hold = max(map(len, req.stop)) - 1 if req.stop and not req.include_stop_str else 0
        self.finish = None

    def _base(self, obj):
        return {"id": self.id, "object": obj, "created": self.created, "model": self.model_name}

    def _text_delta(self, step, finish):
        """This token's text delta and the finish reason after the stop-string check."""
        req = self.req
        if finish == "stop" and not req.include_stop_str:
            new = self.detok.flush()          # a stop token's own text is not output (vLLM)
        else:
            new = self.detok.add(step.token_id)
            if finish:
                new += self.detok.flush()
        if new:
            self.text += new
            if req.stop and self.n > req.min_tokens:
                cut = find_stop(self.text, len(new), req.stop, req.include_stop_str)
                if cut is not None:
                    self.text, finish = self.text[:cut], "stop"
        end = len(self.text) if finish else max(self.sent, len(self.text) - self.hold)
        delta, self.sent = self.text[self.sent:end], end
        return delta, finish

    def _chat_lp(self, tid, lp, top):
        """vLLM's chat entry: top_logprobs is the chosen token, then the most likely others in rank order,
        k entries in all (decoding greedily that is exactly the k most likely, descending)."""
        k, as_id = self.req.top_logprobs, self.req.tokens_as_ids
        s = self.tstr(tid, as_id)
        e = {"token": s, "logprob": _lp(lp), "bytes": list(s.encode("utf-8", "replace")), "top_logprobs": []}
        if k:
            for aid, alp in ([(tid, lp)] + [(a, x) for a, x in _ranked(top, k) if a != tid])[:k]:
                a = self.tstr(aid, as_id)
                e["top_logprobs"].append({"token": a, "logprob": _lp(alp),
                                          "bytes": list(a.encode("utf-8", "replace"))})
        return e

    def _completion_lp(self, tid, lp, top):
        s = self.tstr(tid, self.req.tokens_as_ids)
        d = {s: _lp(lp)}                       # the chosen token first, then the k most likely (vLLM)
        for aid, alp in _ranked(top, self.req.top_logprobs):
            d.setdefault(self.tstr(aid, self.req.tokens_as_ids), _lp(alp))
        return s, d

    def add(self, step, finish):
        """Account one generated token. Returns (its SSE chunk, the finish reason, which a stop string can
        set). Exactly one chunk per token; its text may be empty (held back or a stop token)."""
        req = self.req
        self.n += 1
        delta, finish = self._text_delta(step, finish)
        tid = step.token_id
        self.ids.append(tid)
        self.deltas.append(delta)
        first = self.n == 1
        if req.kind == "chat":
            c = self._base("chat.completion.chunk")
            d = {"role": "assistant", "content": delta} if first else {"content": delta}
            ch = {"index": 0, "delta": d, "logprobs": None, "finish_reason": finish}
            if req.logprobs:
                e = self._chat_lp(tid, step.logprob, step.top)
                self.lp_entries.append(e)
                ch["logprobs"] = {"content": [e]}
            if req.return_token_ids and first:
                c["prompt_token_ids"] = list(req.prompt_ids)
        else:
            c = self._base("text_completion")
            ch = {"index": 0, "text": delta, "logprobs": None, "finish_reason": finish}
            if req.logprobs:
                s, d = self._completion_lp(tid, step.logprob, step.top)
                self.lp_entries.append((s, _lp(step.logprob), d))
                ch["logprobs"] = {"tokens": [s], "token_logprobs": [_lp(step.logprob)], "top_logprobs": [d],
                                  "text_offset": [self.sent - len(delta)]}
            if req.return_token_ids and first:
                ch["prompt_token_ids"] = list(req.prompt_ids)
        if req.return_token_ids:
            ch["token_ids"] = [tid]
        c["choices"] = [ch]
        if req.include_usage:
            c["usage"] = None
        self.finish = finish
        return c, finish

    def usage(self):
        P = len(self.req.prompt_ids)
        return {"prompt_tokens": P, "completion_tokens": self.n, "total_tokens": P + self.n}

    def usage_chunk(self):
        c = self._base("chat.completion.chunk" if self.req.kind == "chat" else "text_completion")
        c["choices"] = []
        c["usage"] = self.usage()
        return c

    def final_body(self, info):
        """The non-streaming response."""
        req = self.req
        text = "".join(self.deltas)
        if req.kind == "chat":
            body = self._base("chat.completion")
            ch = {"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": None,
                  "finish_reason": self.finish}
            if req.logprobs:
                ch["logprobs"] = {"content": list(self.lp_entries)}
            if req.return_token_ids:
                body["prompt_token_ids"] = list(req.prompt_ids)
        else:
            body = self._base("text_completion")
            ch = {"index": 0, "text": text, "logprobs": None, "finish_reason": self.finish}
            if req.logprobs:
                offs, o = [], 0
                for piece in self.deltas:
                    offs.append(o)
                    o += len(piece)
                ch["logprobs"] = {"tokens": [s for s, _, _ in self.lp_entries],
                                  "token_logprobs": [lp for _, lp, _ in self.lp_entries],
                                  "top_logprobs": [d for _, _, d in self.lp_entries], "text_offset": offs}
            if req.return_token_ids:
                ch["prompt_token_ids"] = list(req.prompt_ids)
        if req.return_token_ids:
            ch["token_ids"] = list(self.ids)
        body["choices"] = [ch]
        body["usage"] = self.usage()
        if info:
            body["inferencekit"] = info
        return body


# =================================================================================================
# HTTP
# =================================================================================================
class _ClientGone(Exception):
    pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ikengine-serve"
    sys_version = ""

    def setup(self):
        super().setup()
        self._peer_check_at = 0.0
        try:  # small per-token writes must not wait for Nagle / delayed ACK
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, fmt, *args):
        if self.server.access_log:
            log(f"[serve] {self.address_string()} {fmt % args}")

    # -- plumbing ----------------------------------------------------------------------------------
    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, err):
        self._send_json(err.status, err.body())

    def _chunk(self, data: bytes):
        try:
            self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            raise _ClientGone() from e

    def _event(self, obj):
        self._chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")

    def _peer_closed(self):
        """True once the client has closed its side (EOF) or the connection failed. Bytes of a next
        keep-alive request are not EOF. poll, not select: select fails on descriptors >= 1024, which a busy
        server with a high open-file limit hands out."""
        try:
            p = select.poll()
            p.register(self.connection, select.POLLIN)
            events = p.poll(0)
            if not events:
                return False
            if events[0][1] & select.POLLNVAL:
                return True
            return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    def _next(self, job):
        """The worker's next item; raises _ClientGone once the client has left. The connection is checked
        every PEER_CHECK_S, also while tokens keep coming: a non-streaming request writes nothing until it
        ends, so a closed connection would otherwise go unnoticed for the whole generation."""
        while True:
            now = time.monotonic()
            if now >= self._peer_check_at:
                if self._peer_closed():
                    raise _ClientGone()
                self._peer_check_at = now + PEER_CHECK_S
            try:
                return job.out.get(timeout=PEER_CHECK_S)
            except queue.Empty:
                pass

    def _read_json(self):
        n = self.headers.get("Content-Length")
        if n is None:
            self.close_connection = True
            raise RequestError("Content-Length is required", status=411)
        try:
            n = int(n)
        except ValueError:
            self.close_connection = True
            raise RequestError("bad Content-Length")
        if n > MAX_BODY_BYTES:
            self.close_connection = True
            raise RequestError(f"body larger than {MAX_BODY_BYTES} bytes", status=413)
        raw = self.rfile.read(n)
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError) as e:
            raise RequestError(f"invalid JSON body: {e}")

    # -- routes ------------------------------------------------------------------------------------
    def do_GET(self):
        srv = self.server
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, {"status": "ok", **srv.scheduler.status()})
        elif path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": srv.served_model_name, "object": "model", "created": srv.started, "owned_by": ENGINE_NAME,
                 "root": srv.ctx.get("model_path"), "max_model_len": srv.ctx["max_seq"]}]})
        elif path == "/engine/config":
            self._send_json(200, srv.resolved_config)
        elif path == "/get_server_info":
            self._send_json(200, {"engine": ENGINE_NAME, "max_running_requests": 1, "batch": 1,
                                  "max_queue": srv.scheduler.max_queue, "max_model_len": srv.ctx["max_seq"],
                                  **srv.scheduler.status()})
        elif path in ("/v1/chat/completions", "/v1/completions", "/tokenize", "/detokenize"):
            self._error(RequestError(f"use POST for {path}", status=405))
        else:
            self._error(RequestError(f"not found: {path}", status=404, type_="NotFoundError"))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        routes = {"/v1/chat/completions": "chat", "/v1/completions": "completion"}
        try:
            if path not in routes and path not in ("/tokenize", "/detokenize"):
                if self.headers.get("Content-Length"):
                    self.rfile.read(int(self.headers["Content-Length"]))
                raise RequestError(f"not found: {path}", status=404, type_="NotFoundError")
            body = self._read_json()
            if path == "/tokenize":
                return self._tokenize(body)
            if path == "/detokenize":
                return self._detokenize(body)
            req = parse_request(routes[path], body, self.server.tokenizer, self.server.ctx)
        except RequestError as e:
            return self._error(e)
        except Exception as e:  # noqa: BLE001 - e.g. a chat template that raises on these messages
            return self._error(RequestError(f"could not build the prompt: {e!r}"))
        try:
            job = self.server.scheduler.submit(req, gone=self._peer_closed)
        except QueueFull as e:
            return self._error(RequestError(f"server busy: {e}", status=503, type_="ServiceUnavailableError"))
        try:
            if req.stream:
                self._stream(job)
            else:
                self._complete(job)
        except _ClientGone:
            self.close_connection = True
        finally:
            job.cancelled.set()   # stops the worker if this handler left early; a no-op after the last token

    def _tokenize(self, body):
        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        srv = self.server
        if "messages" in body:
            _check_fields(body, {"model", "messages", "add_generation_prompt", "chat_template_kwargs",
                                 "add_special_tokens"})
            ids = chat_prompt_ids(body, srv.tokenizer, srv.ctx.get("chat_template_kwargs"))
        elif "prompt" in body:
            _check_fields(body, {"model", "prompt", "add_special_tokens"})
            if not isinstance(body["prompt"], str):
                raise RequestError("prompt must be a string", "prompt")
            ids = srv.tokenizer.encode(body["prompt"], add_special_tokens=_bool(body, "add_special_tokens", True))
        else:
            raise RequestError("send prompt or messages")
        self._send_json(200, {"tokens": ids, "count": len(ids), "max_model_len": srv.ctx["max_seq"]})

    def _detokenize(self, body):
        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        _check_fields(body, {"model", "tokens"})
        toks = body.get("tokens")
        if not (isinstance(toks, list) and all(_is_int(t) for t in toks)):
            raise RequestError("tokens must be a list of integers", "tokens")
        self._send_json(200, {"prompt": self.server.tokenizer.decode(toks, skip_special_tokens=False)})

    def _builder(self, job):
        srv = self.server
        return ResponseBuilder(job.req, srv.tokenizer, srv.served_model_name, srv.token_strings)

    def _stream(self, job):
        req = job.req
        rb = self._builder(job)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            item = self._next(job)
            if item[0] == "token":
                c, finish = rb.add(item[1], item[2])
                if finish:
                    job.cancelled.set()          # a stop string can end the request before the worker does
                    info = dict(job.info)
                    if not req.include_usage and info:
                        c["inferencekit"] = info
                    self._event(c)
                    if req.include_usage:
                        u = rb.usage_chunk()
                        if info:
                            u["inferencekit"] = info
                        self._event(u)
                    break
                self._event(c)
            elif item[0] == "error":
                self._event(RequestError(item[1], status=500, type_="InternalServerError").body())
                break
            else:   # "end" without a finishing token (cancelled)
                break
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"")   # zero-length chunk: end of the chunked body

    def _complete(self, job):
        rb = self._builder(job)
        while True:
            item = self._next(job)
            if item[0] == "token":
                if rb.add(item[1], item[2])[1]:
                    job.cancelled.set()
                    break
            elif item[0] == "error":
                return self._error(RequestError(item[1], status=500, type_="InternalServerError"))
            else:
                break
        self._send_json(200, rb.final_body(dict(job.info)))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 1024   # listen backlog: load tests open many connections at once

    def handle_error(self, request, client_address):
        # a client closing its connection (e.g. after reading what it needed) is not a server error
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def make_server(engine, tokenizer, *, host="127.0.0.1", port=0, served_model_name=ENGINE_NAME, model_path=None,
                eos_ids=(), sampling_defaults=None, chat_template_kwargs=None, resolved_config=None, max_queue=0,
                access_log=False):
    """Build (not start) the HTTP server and its scheduler. Tests inject a fake engine and tokenizer."""
    srv = Server((host, port), Handler)
    srv.tokenizer = tokenizer
    srv.token_strings = TokenStrings(tokenizer)
    srv.served_model_name = served_model_name
    srv.started = int(time.time())
    srv.access_log = access_log
    names = {served_model_name}
    if model_path:
        names.add(model_path)
    srv.ctx = {"model_names": names, "model_path": model_path, "max_seq": engine.max_seq, "eos_ids": tuple(eos_ids),
               "vocab_size": getattr(engine, "vocab_size", None),
               "logprobs_supported": bool(getattr(engine, "logprobs_supported", False)),
               "sampling_defaults": sampling_defaults or SamplingParams(),
               "chat_template_kwargs": dict(chat_template_kwargs or {})}
    srv.resolved_config = resolved_config or {}
    srv.scheduler = Scheduler(engine, max_queue=max_queue).start()
    return srv


def format_config_line(cfg: dict) -> str:
    return CONFIG_PREFIX + json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def format_shared_config_line(cfg: dict) -> str:
    """The same config in the line format that inferencekit's server runner reads from any custom engine's
    log (`{"inferencekit_resolved_config": {...}}`), so a managed run records it as resolved."""
    return json.dumps({"inferencekit_resolved_config": cfg}, sort_keys=True, separators=(",", ":"))


def parse_config_line(line: str):
    """The resolved-config JSON from a log line, or None if the line is not one."""
    i = line.find(CONFIG_PREFIX)
    return json.loads(line[i + len(CONFIG_PREFIX):]) if i >= 0 else None


# =================================================================================================
# Startup: tokenizer, model defaults, engine, ENGINE.json (transformers and torch imported lazily)
# =================================================================================================
class HFTokenizer:
    """transformers AutoTokenizer: encode/decode and the chat template (the tokenizer's own, or the one
    given). Calls are serialized: fast tokenizers are not guaranteed thread-safe and requests are tokenized
    in their handler threads."""

    def __init__(self, path, chat_template=None):
        # A directory named `kernels` on sys.path (yours, next to ikengine/) would shadow the Hub `kernels`
        # package that transformers probes for; keep the current directory off sys.path while it loads.
        here = os.path.abspath(os.getcwd())
        saved = list(sys.path)
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
        try:
            import transformers
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path)
            self.version = transformers.__version__
        finally:
            sys.path[:] = saved
        self.chat_template = chat_template
        self._lock = threading.Lock()

    @property
    def eos_token_id(self):
        return getattr(self.tok, "eos_token_id", None)

    def encode(self, text, add_special_tokens=True):
        with self._lock:
            return list(self.tok.encode(text, add_special_tokens=add_special_tokens))

    def apply_chat_template(self, messages, add_generation_prompt=True, add_special_tokens=False, **kwargs):
        if self.chat_template is not None:
            kwargs["chat_template"] = self.chat_template
        with self._lock:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                                **kwargs)
            # The template already holds the special tokens (vLLM's chat path does the same).
            return list(self.tok.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, ids, skip_special_tokens=True):
        # No clean-up of spaces (as vLLM): it can rewrite text already sent (" ." -> "."), and the
        # incremental detokenizer slices new text off the end of what it decoded before.
        with self._lock:
            return self.tok.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)


def read_json_file(path):
    try:
        with open(path) as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def load_chat_template(arg):
    """(template text or None, description) for --chat-template: a file path, or the template itself."""
    if not arg:
        return None, {"source": "tokenizer"}
    if os.path.isfile(arg):
        with open(arg, encoding="utf-8") as f:
            text = f.read()
        src = {"source": "file", "path": os.path.abspath(arg)}
    elif "{{" in arg or "{%" in arg:
        text, src = arg, {"source": "inline"}
    else:
        raise SystemExit(f"--chat-template {arg!r}: no such file (and not a Jinja template)")
    return text, {**src, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def eos_token_ids(generation_config, tokenizer):
    """EOS ids as vLLM takes them: generation_config.json's eos_token_id (an id or a list) plus the
    tokenizer's eos_token_id."""
    ids = set()
    e = (generation_config or {}).get("eos_token_id")
    ids.update(e if isinstance(e, list) else [e] if e is not None else [])
    t = getattr(tokenizer, "eos_token_id", None)
    if t is not None:
        ids.add(t)
    return sorted(i for i in ids if _is_int(i))


# generation_config.json fields vLLM applies by default that this server has no sampler for.
UNSUPPORTED_DEFAULTS = (("repetition_penalty", 1.0), ("presence_penalty", 0.0), ("frequency_penalty", 0.0))


def _text(v):
    return isinstance(v, (str, int, float)) and not isinstance(v, bool) and v != ""


def numerics_changed(numerics, act="as_shipped", kv="as_shipped"):
    """What this run computes differently from the checkpoint as shipped. The model module decides when it
    says (a "changed" list, or "as_shipped"): it knows whether a flag asks for a format the checkpoint
    already ships (--kv bf16 on a BF16-KV checkpoint changes nothing). Only when it says neither do the
    --act / --kv flags that are not as_shipped count as changes."""
    if "changed" not in numerics and "as_shipped" not in numerics:
        return [f"--{k} {v}" for k, v in (("act", act), ("kv", kv)) if v != "as_shipped"]
    own = numerics.get("changed")
    changed = [str(c) for c in own if c] if isinstance(own, (list, tuple)) else [str(own)] if _text(own) else []
    if not changed and numerics.get("as_shipped") is False:
        changed = ["the model module reports as_shipped: false without a changed list"]
    return changed


def numerics_summary(numerics, changed=()):
    """ENGINE.json's one-line "numerics_summary": every text value of the numerics dict, nested ones by
    dotted key ("activations.fp8_block: ..."), then "(numerics changed: ...)" when `changed` is not empty.
    Flags (true/false) and the bookkeeping keys (changed, as_shipped, requested) are left to "numerics"."""
    parts = []

    def walk(key, v):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{key}.{k}", x)
        elif isinstance(v, (list, tuple)):
            text = ", ".join(str(x) for x in v if x is not None and x != "" and not isinstance(x, bool))
            if text:
                parts.append(f"{key}: {text}")
        elif _text(v):
            parts.append(f"{key}: {v}")

    for k, v in numerics.items():
        if k not in ("changed", "as_shipped", "requested"):
            walk(str(k), v)
    line = "; ".join(parts) or "not declared by the model module"
    return line + (f" (numerics changed: {'; '.join(changed)})" if changed else "")


def engine_json_numerics(model_numerics, act, kv):
    """ENGINE.json's (and the resolved config's) "numerics": the model's dict, the flags asked for, and
    the changed list (numerics_changed)."""
    return {**model_numerics, "requested": {"act": act, "kv": kv},
            "changed": numerics_changed(model_numerics, act, kv)}


def model_eos_ids(engine):
    """The EOS ids the model module declares (engine_api.Model.eos_token_ids), for the record only: the
    server stops on generation_config.json's and the tokenizer's, as vLLM does."""
    ids = getattr(engine, "eos_token_ids", None) or getattr(getattr(engine, "model", None), "eos_token_ids", None)
    return sorted(int(t) for t in ids or ())


def engine_numerics(engine):
    model = getattr(engine, "model", None)
    return describe_numerics(model if getattr(model, "numerics", None) else engine)


def engine_config(engine):
    """What the engine reports about itself: the first of resolved_config / config that is a dict or a
    method returning one (ikengine.engine.Engine has config())."""
    for name in ("resolved_config", "config"):
        rc = getattr(engine, name, None)
        try:
            rc = rc() if callable(rc) else rc
        except Exception as e:  # noqa: BLE001 - a broken self-report must not stop the server
            return {"error": f"{name}() raised {e!r}"}
        if isinstance(rc, dict):
            return dict(rc)
    return {}


def git_toplevel(path):
    """The nearest directory at or above `path` that holds a .git entry (a directory, or a file for a
    worktree), or None."""
    p = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def engine_json_path(arg, cwd):
    """--engine-json resolved: None (not given) is ENGINE.json at the git top level of `cwd` (the run
    directory, also when the server runs from a subdirectory), else in `cwd`; "" means no file; a relative
    path is taken from `cwd`."""
    if arg is not None:
        return os.path.join(cwd, arg) if arg else ""
    return os.path.join(git_toplevel(cwd) or os.path.abspath(cwd), "ENGINE.json")


def portable_launch(argv, cwd, top, pkg_root=PKG_ROOT):
    """ENGINE.json's "launch": the command line `argv`, started in `cwd`, made to run from the repository
    root `top` (where a fresh copy of the repository, no .git needed, is launched). From a subdirectory it becomes
    sh -c 'cd <subdir> && exec <argv>', so relative paths in it still resolve; when ikengine/ was found
    through PYTHONPATH rather than the current directory (and lives in the repository), env
    PYTHONPATH=<its parent, relative> goes first. Outside a repository it is kept as it is."""
    argv = [str(a) for a in argv]
    if top is None:
        return argv
    top, cwd, pkg_root = os.path.abspath(top), os.path.abspath(cwd), os.path.abspath(pkg_root)

    def inside(p):
        return p == top or p.startswith(top.rstrip(os.sep) + os.sep)

    if not inside(cwd):
        return argv
    env = ["env", f"PYTHONPATH={os.path.relpath(pkg_root, cwd)}"] if pkg_root != cwd and inside(pkg_root) else []
    sub = os.path.relpath(cwd, top)
    if sub == ".":
        return env + argv
    return ["sh", "-c", f"cd {shlex.quote(sub)} && exec {shlex.join(env + argv)}"]


def _digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:16]


def write_engine_json(path, fields, resolved_config):
    """Fill ENGINE.json (see the module docstring) with `fields` ({"launch", "port", "numerics",
    "numerics_summary"}) and "resolved_config". A field is written when the file lacks it or still holds
    what the server wrote last time (its digest in WRITTEN_KEY matches); anything else is kept. Never
    rewrites an unchanged file, writes atomically, and leaves a file that is not a JSON object alone.
    Returns the document or None."""
    doc = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            log(f"[serve] warning: {path} is not readable JSON ({e}); left as it is")
            return None
        if not isinstance(doc, dict):
            log(f"[serve] warning: {path} is not a JSON object; left as it is")
            return None
    before = json.dumps(doc, indent=2) + "\n"
    previous = doc.get(WRITTEN_KEY) if isinstance(doc.get(WRITTEN_KEY), dict) else {}
    written, kept = {}, []
    for k, v in fields.items():
        if k not in doc or previous.get(k) == _digest(doc[k]):
            doc[k] = v
            written[k] = _digest(v)
        else:
            kept.append(k)
    doc["resolved_config"] = resolved_config
    doc.pop(WRITTEN_KEY, None)
    if written:
        doc[WRITTEN_KEY] = written
    if kept:
        log(f"[serve] {path}: kept your {', '.join(kept)} (not written by this server, or edited since); "
            f"delete a field to have the server fill it again")
    text = json.dumps(doc, indent=2) + "\n"
    if text != before or not os.path.exists(path):
        tmp = f"{path}.tmp{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except OSError as e:
            log(f"[serve] warning: could not write {path} ({e})")
            return None
    return doc


def resolved_config(args, engine, tokenizer, *, served, eos_ids, defaults, gen_config, template_src, numerics, argv,
                    cwd):
    eng = engine_config(engine)
    cfg = {"engine": ENGINE_NAME, "served_model_name": served, "model_path": args.model}
    cg = eng.get("cuda_graphs", getattr(engine, "cuda_graphs", None))
    if isinstance(cg, bool):
        cfg["cuda_graphs"] = cg     # top-level key read by inferencekit's CUDA-graph check for custom engines
    ignored = {k: gen_config[k] for k, neutral in UNSUPPORTED_DEFAULTS
               if _is_num(gen_config.get(k)) and gen_config[k] != neutral}
    cfg.update({
        "numerics": numerics,
        "settings": {"device": args.device, "max_seq": engine.max_seq, "act": args.act, "kv": args.kv,
                     "vocab_size": getattr(engine, "vocab_size", None)},
        "engine_config": eng,
        "server": {
            "batch": 1, "scheduling": "FIFO, one request at a time", "max_queue": args.max_queue,
            "logprobs": bool(getattr(engine, "logprobs_supported", False)), "max_top_logprobs": MAX_TOP_LOGPROBS,
            "sampling_defaults": {k: v for k, v in dataclasses.asdict(defaults).items() if k != "seed"},
            "sampling_defaults_source": ("generation_config.json" if gen_config
                                         else "built-in (no generation_config.json)"),
            "generation_config_not_applied": ignored,
            "eos_token_ids": list(eos_ids), "model_eos_token_ids": model_eos_ids(engine),
            "chat_template": template_src,
            "default_chat_template_kwargs": args.default_chat_template_kwargs,
            "host": args.host, "port": args.port, "argv": list(argv), "cwd": cwd},
        "versions": {"python": platform.python_version(), "transformers": getattr(tokenizer, "version", None),
                     "torch": getattr(sys.modules.get("torch"), "__version__", None)},
    })
    return cfg


def warm_up(engine, tokenizer, defaults, chat_template_kwargs):
    """A few tokens through every path a request can take (greedy; sampled with logprobs), so the first
    timed request pays for no graph capture or compilation."""
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True,
                                        add_special_tokens=False, **(chat_template_kwargs or {}))
    ids = list(ids)[: max(1, engine.max_seq - WARMUP_TOKENS)]
    runs = [(SamplingParams(temperature=0.0), None)]
    if not defaults.greedy or getattr(engine, "logprobs_supported", False):
        runs.append((dataclasses.replace(defaults, seed=0),
                     MAX_TOP_LOGPROBS if getattr(engine, "logprobs_supported", False) else None))
    for sp, top in runs:
        gen = engine.stream(ids, WARMUP_TOKENS, sp, top, {})
        try:
            for _ in gen:
                pass
        finally:
            gen.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ikengine.serve", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="local checkpoint directory (config.json, *.safetensors)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--served-model-name", help="model name in the API (default: --model as given)")
    ap.add_argument("--tokenizer", help="tokenizer directory (default: --model)")
    ap.add_argument("--chat-template", help="chat template file (default: the tokenizer's own)")
    ap.add_argument("--default-chat-template-kwargs", default="{}",
                    help="JSON object applied to every chat request, e.g. '{\"enable_thinking\": false}'; "
                         "a request's chat_template_kwargs win per key")
    ap.add_argument("--max-seq", type=int, default=4096, help="max context (prompt + generated)")
    ap.add_argument("--act", choices=("as_shipped", "a16"), default="as_shipped",
                    help="activation numerics: the checkpoint's scheme, or 16-bit activations")
    ap.add_argument("--kv", choices=("as_shipped", "bf16"), default="as_shipped",
                    help="KV-cache dtype: the checkpoint's, or bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-queue", type=int, default=0, help="503 beyond this many waiting requests (0: no limit)")
    ap.add_argument("--engine-json", default=None,
                    help="ENGINE.json to fill at startup (default: at the top of the git repository the server "
                         "runs in, else in the current directory; '' skips it)")
    ap.add_argument("--access-log", action="store_true")
    args = ap.parse_args(argv)
    try:
        kw = json.loads(args.default_chat_template_kwargs or "{}")
    except ValueError as e:
        ap.error(f"--default-chat-template-kwargs is not JSON: {e}")
    if not isinstance(kw, dict):
        ap.error("--default-chat-template-kwargs must be a JSON object")
    args.default_chat_template_kwargs = kw
    if args.max_seq < 2:
        ap.error("--max-seq must be >= 2")
    return args


def build(args, *, builder=None, tokenizer_factory=HFTokenizer, argv=None, cwd=None):
    """Tokenizer, engine, warm-up, resolved config, the (not yet started) server bound to its port, then
    ENGINE.json. `argv` / `cwd`: the command line and directory this process was started with (default:
    the real ones). Tests pass a fake builder and tokenizer factory."""
    argv = [str(a) for a in (argv if argv is not None else getattr(sys, "orig_argv", sys.argv))]
    cwd = os.path.abspath(cwd or os.getcwd())
    served = args.served_model_name or args.model
    template, template_src = load_chat_template(args.chat_template)
    t = time.perf_counter()
    tokenizer = tokenizer_factory(args.tokenizer or args.model, template)
    gen_config = read_json_file(os.path.join(args.model, "generation_config.json"))
    eos_ids = eos_token_ids(gen_config, tokenizer)
    defaults = SamplingParams.from_generation_config(gen_config)
    log(f"[serve] tokenizer loaded in {time.perf_counter() - t:.1f}s; eos ids {eos_ids}; sampling defaults "
        f"{dataclasses.asdict(defaults)}")
    if builder is None:
        from .engine import build_engine as builder   # torch and the model module load here
    t = time.perf_counter()
    engine = builder(args.model, device=args.device, max_seq=args.max_seq, act=args.act, kv=args.kv)
    log(f"[serve] engine built in {time.perf_counter() - t:.1f}s")
    t = time.perf_counter()
    warm_up(engine, tokenizer, defaults, args.default_chat_template_kwargs)
    log(f"[serve] warm-up done in {time.perf_counter() - t:.1f}s")
    model_numerics = engine_numerics(engine)
    numerics = engine_json_numerics(model_numerics, args.act, args.kv)
    cfg = resolved_config(args, engine, tokenizer, served=served, eos_ids=eos_ids, defaults=defaults,
                          gen_config=gen_config, template_src=template_src, numerics=numerics, argv=argv, cwd=cwd)
    for k, v in cfg["server"]["generation_config_not_applied"].items():
        log(f"[serve] warning: generation_config.json sets {k}={v}, which this server does not apply "
            f"(vLLM would, by default)")
    unlisted = sorted(set(cfg["server"]["model_eos_token_ids"]) - set(eos_ids))
    if unlisted:
        log(f"[serve] warning: the model module declares EOS ids {unlisted} that neither generation_config.json "
            f"nor the tokenizer lists; requests do not stop on them (vLLM would not either)")
    # Bind before ENGINE.json: a port in use fails here and leaves the file alone, and --port 0 records
    # the port actually taken.
    srv = make_server(engine, tokenizer, host=args.host, port=args.port, served_model_name=served,
                      model_path=args.model, eos_ids=eos_ids, sampling_defaults=defaults,
                      chat_template_kwargs=args.default_chat_template_kwargs, resolved_config=cfg,
                      max_queue=args.max_queue, access_log=args.access_log)
    port = srv.server_address[1]
    cfg["server"]["port"] = port
    log(format_config_line(cfg))
    log(format_shared_config_line(cfg))
    path = engine_json_path(args.engine_json, cwd)
    if path:
        fields = {"launch": portable_launch(argv, cwd, git_toplevel(cwd)), "port": port, "numerics": numerics,
                  "numerics_summary": numerics_summary(model_numerics, numerics["changed"])}
        doc = write_engine_json(path, fields, cfg)
        if doc is not None:
            log(f"[serve] wrote {path}")
            if doc.get("numerics") != numerics and numerics["changed"]:
                log(f"[serve] warning: {path} keeps your own numerics; this run's numerics changed: "
                    f"{'; '.join(numerics['changed'])}. Make sure yours say so")
    return srv, cfg


def main(argv=None):
    args = parse_args(argv)
    srv, _ = build(args)
    log(f"[serve] listening on http://{args.host}:{srv.server_address[1]} (model {srv.served_model_name!r}, "
        f"batch 1)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        srv.scheduler.stop()


if __name__ == "__main__":
    main()
