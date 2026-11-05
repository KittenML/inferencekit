"""The engine skeleton's OpenAI server (plugin/skills/optimizing-kernels/skeleton/ikengine/serve.py) with fake
engines and tokenizers: no GPU, no torch, no transformers.

Ported from an earlier engine's server tests (its test_serve.py and
test_serve_engine_adapter.py) onto the engine_api contract, plus sampling defaults and overrides, seeds,
top-20 logprobs, chat-template defaults, stop strings, ENGINE.json, and the requests the inferencekit
checks themselves send (gates run, bench serve, quality trajectory) run against the server.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import types

import pytest

from fakes_skeleton.engines import (EOS, SKELETON, ByteTokenizer, CharLM, FakeTokenizer, Model, ScriptEngine, builder,
                                    ids)

assert str(SKELETON) in sys.path
from ikengine import serve  # noqa: E402
from ikengine.engine_api import MAX_TOP_LOGPROBS, SamplingParams, StepOut  # noqa: E402

TOKENS = ids("Hello") + [EOS] + ids(" world")   # EOS at index 5
DEFAULTS = SamplingParams(temperature=1.0, top_k=20, top_p=0.95)   # a generation_config.json's defaults
GREEDY = SamplingParams(temperature=0.0)


@pytest.fixture
def server():
    started = []

    def start(engine=None, tokenizer=None, **kw):
        engine = engine or ScriptEngine(TOKENS)
        kw.setdefault("sampling_defaults", DEFAULTS)
        kw.setdefault("eos_ids", (EOS,))
        srv = serve.make_server(engine, tokenizer or FakeTokenizer(), host="127.0.0.1", port=0,
                                served_model_name="fake-model", resolved_config={"engine": "fake"}, **kw)
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        th.start()
        started.append(srv)
        return srv.server_address[1], engine

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()
        srv.scheduler.stop()


# --------------------------------------------------------------------------------------------------
# client helpers
# --------------------------------------------------------------------------------------------------
def request(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, dict(r.getheaders()), json.loads(raw)
    except ValueError:
        return r.status, dict(r.getheaders()), raw


def sse(port, path, body, stop_after=None):
    """POST and read the SSE stream. Returns (status, headers, [(t_arrival, event_json_or_DONE)])."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    headers = dict(r.getheaders())
    if r.status != 200:
        out = json.loads(r.read())
        c.close()
        return r.status, headers, out
    events = []
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        assert line.startswith(b"data: "), line
        payload = line[len(b"data: "):]
        ev = "[DONE]" if payload == b"[DONE]" else json.loads(payload)
        events.append((time.perf_counter(), ev))
        if ev == "[DONE]" or (stop_after and len(events) >= stop_after):
            break
    if stop_after and events[-1][1] != "[DONE]":
        c.sock.shutdown(socket.SHUT_RDWR)
    c.close()
    return 200, headers, events


def chunks_of(events):
    return [e for _, e in events if e != "[DONE]"]


def text_of(chunks, kind="chat"):
    key = (lambda ch: ch["delta"]["content"]) if kind == "chat" else (lambda ch: ch["text"])
    return "".join(key(c["choices"][0]) for c in chunks if c["choices"])


def chat(**kw):
    body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "temperature": 0,
            "stream": True}
    body.update(kw)
    return body


def completion(**kw):
    body = {"model": "fake-model", "prompt": "hi", "temperature": 0, "stream": True}
    body.update(kw)
    return body


def wait_for(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not cond():
        time.sleep(0.01)
    return cond()


# --------------------------------------------------------------------------------------------------
# streaming format (ported)
# --------------------------------------------------------------------------------------------------
def test_chat_stream_sse_format_and_usage(server):
    port, engine = server()
    status, headers, events = sse(port, "/v1/chat/completions",
                                  chat(max_tokens=20, stream_options={"include_usage": True}))
    assert status == 200
    assert headers["Content-Type"] == "text/event-stream"
    assert events[-1][1] == "[DONE]"
    chunks = chunks_of(events)
    token_chunks, usage_chunk = chunks[:-1], chunks[-1]
    # one chunk per generated token, up to and including EOS
    assert len(token_chunks) == 6
    assert [c["choices"][0]["token_ids"] for c in token_chunks] == [[t] for t in TOKENS[:6]]
    ids_ = {c["id"] for c in chunks}
    assert len(ids_) == 1 and next(iter(ids_)).startswith("chatcmpl-")
    for c in token_chunks:
        assert c["object"] == "chat.completion.chunk" and c["model"] == "fake-model"
        assert c["usage"] is None and len(c["choices"]) == 1 and c["choices"][0]["index"] == 0
    first = token_chunks[0]
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "H"}
    assert first["prompt_token_ids"] == FakeTokenizer().apply_chat_template([{"role": "user", "content": "hi"}])
    assert all("role" not in c["choices"][0]["delta"] for c in token_chunks[1:])
    assert text_of(token_chunks) == "Hello"   # the stop token's text is not output
    assert [c["choices"][0]["finish_reason"] for c in token_chunks] == [None] * 5 + ["stop"]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {"prompt_tokens": len(first["prompt_token_ids"]), "completion_tokens": 6,
                                    "total_tokens": len(first["prompt_token_ids"]) + 6}
    assert usage_chunk["inferencekit"]["prefill"] == "fake"
    assert usage_chunk["inferencekit"]["prompt_tokens"] == len(first["prompt_token_ids"])
    call = engine.calls[0]
    assert call["max_tokens"] == 20 and call["closed_early"]
    assert call["sampling"] == GREEDY and call["top_logprobs"] is None


def test_stream_without_usage_has_no_usage_fields(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=3))
    chunks = chunks_of(events)
    assert len(chunks) == 3 and all("usage" not in c for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert "inferencekit" in chunks[-1]


def test_ignore_eos_generates_exactly_max_tokens(server):
    port, engine = server()
    _, _, events = sse(port, "/v1/chat/completions",
                       chat(max_tokens=15, ignore_eos=True, stream_options={"include_usage": True}))
    chunks = chunks_of(events)
    toks = [c["choices"][0]["token_ids"][0] for c in chunks[:-1]]
    assert toks == (TOKENS * 2)[:15] and EOS in toks
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 15
    assert not engine.calls[0]["closed_early"]


def test_skip_special_tokens_false_shows_eos_text_mid_stream(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions",
                       chat(max_tokens=8, ignore_eos=True, skip_special_tokens=False))
    assert text_of(chunks_of(events)) == "Hello<eos> w"


def test_max_tokens_before_eos_finishes_with_length(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions", chat(max_completion_tokens=3))
    chunks = chunks_of(events)
    assert text_of(chunks) == "Hel" and chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_stop_token_ids_apply_even_with_ignore_eos(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions",
                       chat(max_tokens=30, ignore_eos=True, stop_token_ids=[ord("w")]))
    chunks = chunks_of(events)
    # the stop token is generated (and counted) but its text is not output, as in vLLM
    assert text_of(chunks) == "Hello " and chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["token_ids"] == [ord("w")]
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=30, ignore_eos=True, stop_token_ids=[ord("w")],
                                                          include_stop_str_in_output=True))
    assert text_of(chunks_of(events)) == "Hello w"


def test_return_token_ids_false_omits_extension(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=2, return_token_ids=False))
    chunks = chunks_of(events)
    assert all("token_ids" not in c["choices"][0] and "prompt_token_ids" not in c for c in chunks)


def test_completions_stream_with_token_id_prompt_and_logprobs(server):
    port, engine = server()
    prompt = [72, 105, 33]
    _, _, events = sse(port, "/v1/completions", completion(prompt=prompt, max_tokens=4, logprobs=1))
    chunks = chunks_of(events)
    assert len(chunks) == 4
    assert engine.calls[0]["prompt_ids"] == prompt and engine.calls[0]["top_logprobs"] == 1
    c0 = chunks[0]
    assert c0["object"] == "text_completion" and c0["id"].startswith("cmpl-")
    assert c0["choices"][0]["prompt_token_ids"] == prompt
    assert [c["choices"][0]["text"] for c in chunks] == ["H", "e", "l", "l"]
    lp = [c["choices"][0]["logprobs"] for c in chunks]
    assert [x["token_logprobs"] for x in lp] == [[-0.25], [-0.5], [-0.75], [-1.0]]
    assert [x["tokens"] for x in lp] == [["H"], ["e"], ["l"], ["l"]]
    assert [x["text_offset"] for x in lp] == [[0], [1], [2], [3]]
    assert lp[0]["top_logprobs"] == [{"H": -0.25}]


def test_chat_logprobs_format(server):
    port, engine = server()
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=2, logprobs=True))
    c = events[0][1]["choices"][0]["logprobs"]["content"][0]
    assert c == {"token": "H", "logprob": -0.25, "bytes": [72], "top_logprobs": []}
    assert engine.calls[0]["top_logprobs"] == 0   # the chosen token's logprob only


def test_chat_template_kwargs_reach_the_template(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions",
                       chat(max_tokens=1, chat_template_kwargs={"enable_thinking": False}))
    assert events[0][1]["prompt_token_ids"][-1] == ord("~")


def test_non_streaming_chat_and_completion(server):
    port, _ = server()
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=20))
    assert status == 200 and body["object"] == "chat.completion"
    ch = body["choices"][0]
    assert ch["message"] == {"role": "assistant", "content": "Hello"} and ch["finish_reason"] == "stop"
    assert ch["token_ids"] == TOKENS[:6] and body["usage"]["completion_tokens"] == 6
    assert body["inferencekit"]["prefill"] == "fake"
    status, _, body = request(port, "POST", "/v1/completions", completion(stream=False, max_tokens=8, logprobs=0))
    ch = body["choices"][0]
    assert status == 200 and ch["text"] == "Hello" and ch["finish_reason"] == "stop"
    assert ch["logprobs"]["token_logprobs"] == [-0.25 * (i + 1) for i in range(6)]
    # vLLM: logprobs 0 still lists the chosen token in top_logprobs
    assert ch["logprobs"]["top_logprobs"][0] == {"H": -0.25} and len(ch["logprobs"]["top_logprobs"]) == 6
    assert ch["logprobs"]["text_offset"] == [0, 1, 2, 3, 4, 5]


# --------------------------------------------------------------------------------------------------
# rejection: unsupported parameters are 400, never silently ignored
# --------------------------------------------------------------------------------------------------
REJECT = [
    ("chat", {"temperature": -0.5}, "temperature"),
    ("chat", {"temperature": True}, "temperature"),
    ("completion", {"temperature": "1"}, "temperature"),
    ("chat", {"top_k": -2}, "top_k"),
    ("chat", {"top_k": 2.5}, "top_k"),
    ("chat", {"top_p": 0}, "top_p"),
    ("chat", {"top_p": 1.5}, "top_p"),
    ("chat", {"min_p": 1.5}, "min_p"),
    ("chat", {"seed": "7"}, "seed"),
    ("chat", {"seed": 1.5}, "seed"),
    ("chat", {"seed": 1 << 70}, "seed"),
    ("chat", {"n": 2}, "n"),
    ("chat", {"stop": [1]}, "stop"),
    ("chat", {"stop": 5}, "stop"),
    ("chat", {"presence_penalty": 0.5}, "presence_penalty"),
    ("chat", {"frequency_penalty": 0.1}, "frequency_penalty"),
    ("chat", {"repetition_penalty": 1.1}, "repetition_penalty"),
    ("chat", {"logit_bias": {"5": 10}}, "logit_bias"),
    ("chat", {"min_tokens": 4}, "min_tokens"),                                   # would mask EOS
    ("chat", {"min_tokens": 1}, "min_tokens"),
    ("chat", {"min_tokens": 4, "ignore_eos": True, "stop_token_ids": [7]}, "min_tokens"),  # would mask id 7
    ("chat", {"min_tokens": 5, "ignore_eos": True}, "min_tokens"),               # > max_tokens (4)
    ("chat", {"min_tokens": "4", "ignore_eos": True}, "min_tokens"),
    ("chat", {"min_tokens": -1, "ignore_eos": True}, "min_tokens"),
    ("completion", {"min_tokens": 4}, "min_tokens"),
    ("chat", {"top_logprobs": 5}, "top_logprobs"),                               # needs logprobs: true
    ("chat", {"top_logprobs": 21, "logprobs": True}, "top_logprobs"),
    ("chat", {"top_logprobs": -1, "logprobs": True}, "top_logprobs"),
    ("chat", {"logprobs": 1}, "logprobs"),
    ("completion", {"logprobs": 21}, "logprobs"),
    ("completion", {"logprobs": True}, "logprobs"),
    ("chat", {"response_format": {"type": "json_object"}}, "response_format"),
    ("chat", {"tools": []}, "tools"),
    ("chat", {"prompt_logprobs": 1}, "prompt_logprobs"),
    ("chat", {"max_tokens": 0}, "max_tokens"),
    ("chat", {"max_tokens": 1000}, "max_tokens"),           # prompt + 1000 > max_seq 64
    ("completion", {"max_tokens": None, "prompt": "x" * 70}, "max_tokens"),   # the prompt alone fills max_seq
    ("chat", {"max_tokens": 5, "max_completion_tokens": 6}, "max_tokens"),
    ("chat", {"stream": False, "stream_options": {"include_usage": True}}, "stream_options"),
    ("chat", {"stream_options": {"continuous_usage_stats": True}}, "stream_options"),
    ("chat", {"messages": []}, "messages"),
    ("chat", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}, "messages"),
    ("chat", {"chat_template_kwargs": ["x"]}, "chat_template_kwargs"),
    ("chat", {"ignore_eos": "yes"}, "ignore_eos"),
    ("chat", {"return_tokens_as_token_ids": 1}, "return_tokens_as_token_ids"),
    ("completion", {"echo": True}, "echo"),
    ("completion", {"best_of": 3}, "best_of"),
    ("completion", {"suffix": "x"}, "suffix"),
    ("completion", {"prompt": ["a", "b"]}, "prompt"),
    ("completion", {"prompt": ""}, "prompt"),
    ("completion", {"prompt": [1, -5]}, "prompt"),
]


@pytest.mark.parametrize("kind,patch,param", REJECT, ids=[f"{k}-{p}-{i}" for i, (k, _, p) in enumerate(REJECT)])
def test_unsupported_parameters_are_rejected(server, kind, patch, param):
    port, engine = server()
    body = chat(max_tokens=4) if kind == "chat" else completion(max_tokens=4)
    body.update(patch)
    path = "/v1/chat/completions" if kind == "chat" else "/v1/completions"
    status, _, out = request(port, "POST", path, body)
    assert status == 400, out
    assert out["error"]["param"] == param and out["error"]["message"]
    assert engine.calls == []


def test_neutral_values_are_accepted(server):
    port, _ = server()
    status, _, body = request(port, "POST", "/v1/chat/completions",
                              chat(stream=False, max_tokens=2, top_p=0.9, top_k=20, min_p=0.0, seed=7,
                                   presence_penalty=0, repetition_penalty=1, frequency_penalty=0, n=1, stop=[],
                                   logit_bias={}, user="u", min_tokens=0, top_logprobs=0))
    assert status == 200 and body["choices"][0]["message"]["content"] == "He"


# --------------------------------------------------------------------------------------------------
# sampling: defaults from generation_config, the request wins, seeds
# --------------------------------------------------------------------------------------------------
def _sampling_of(port, engine, **kw):
    body = chat(stream=False, max_tokens=1, **kw)
    if kw.get("temperature", "absent") is None or "temperature" not in kw:
        body.pop("temperature")
        if kw.get("temperature", "absent") is None:
            body["temperature"] = None
    status, _, out = request(port, "POST", "/v1/chat/completions", body)
    assert status == 200, out
    return engine.calls[-1]["sampling"]


def test_sampling_defaults_come_from_generation_config_and_the_request_wins(server):
    port, engine = server()
    assert _sampling_of(port, engine) == DEFAULTS                       # no sampling fields: the model's defaults
    assert _sampling_of(port, engine, temperature=None, top_k=None) == DEFAULTS   # null means "not sent"
    assert _sampling_of(port, engine, temperature=0.7) == SamplingParams(0.7, 20, 0.95)
    assert _sampling_of(port, engine, temperature=1.0, top_k=5, top_p=0.5, min_p=0.1, seed=9) == \
        SamplingParams(1.0, 5, 0.5, 0.1, 9)
    assert _sampling_of(port, engine, seed=3) == SamplingParams(1.0, 20, 0.95, 0.0, 3)


def test_greedy_and_off_values_are_normalised(server):
    port, engine = server(ScriptEngine(TOKENS, vocab_size=1000))
    assert _sampling_of(port, engine, temperature=0, top_k=5, seed=4) == SamplingParams(0.0, 0, 1.0, 0.0, 4)
    assert _sampling_of(port, engine, temperature=1e-6) == GREEDY       # below 1e-5 is greedy, as in vLLM
    assert _sampling_of(port, engine, temperature=1.0, top_k=-1).top_k == 0
    assert _sampling_of(port, engine, temperature=1.0, top_k=1000).top_k == 0   # >= vocabulary: off
    assert _sampling_of(port, engine, temperature=1.0, top_k=999).top_k == 999


def test_no_generation_config_means_vllm_defaults(server):
    port, engine = server(sampling_defaults=None)
    assert _sampling_of(port, engine) == SamplingParams(1.0, 0, 1.0, 0.0, None)


def _sampled_text(port, **kw):
    status, _, body = request(port, "POST", "/v1/chat/completions",
                              chat(stream=False, max_tokens=24, ignore_eos=True,
                                   messages=[{"role": "user", "content": "tell me a story"}], **kw))
    assert status == 200, body
    return body["choices"][0]["message"]["content"]


def test_seeded_sampling_is_reproducible_and_differs_from_greedy(server):
    port, _ = server(CharLM(max_seq=512))
    greedy = _sampled_text(port, temperature=0)
    a, a2, b = (_sampled_text(port, temperature=1.0, seed=s) for s in (11, 11, 12))
    assert a == a2 and a != b and a != greedy and b != greedy
    assert _sampled_text(port, temperature=1.0, top_k=1, seed=11) == greedy      # top_k 1 is the argmax
    unseeded = {_sampled_text(port, temperature=1.0) for _ in range(3)}
    assert len(unseeded) > 1


# --------------------------------------------------------------------------------------------------
# logprobs: up to 20 alternatives, vLLM shapes
# --------------------------------------------------------------------------------------------------
def test_chat_top20_logprobs_are_the_full_distribution_ranked(server):
    port, engine = server(CharLM(max_seq=512))
    status, _, body = request(port, "POST", "/v1/chat/completions",
                              chat(stream=False, max_tokens=6, logprobs=True, top_logprobs=MAX_TOP_LOGPROBS))
    assert status == 200 and engine.calls[-1]["top_logprobs"] == MAX_TOP_LOGPROBS
    for e in body["choices"][0]["logprobs"]["content"]:
        top = e["top_logprobs"]
        vals = [t["logprob"] for t in top]
        assert len(top) == 20 and vals == sorted(vals, reverse=True)
        assert top[0]["token"] == e["token"] and top[0]["logprob"] == e["logprob"]   # greedy: chosen = argmax
        assert sum(math.exp(v) for v in vals) <= 1 + 1e-9
        assert all(t["bytes"] == list(t["token"].encode()) for t in top)


def test_alternatives_are_ranked_whatever_order_the_engine_uses(server):
    class Unsorted(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            for s in super().stream(prompt_ids, max_tokens, sampling, top_logprobs, info):
                yield StepOut(s.token_id, s.logprob, list(reversed(s.top)))

    port, _ = server(Unsorted(TOKENS))
    _, _, body = request(port, "POST", "/v1/chat/completions",
                         chat(stream=False, max_tokens=1, logprobs=True, top_logprobs=3))
    top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    assert [t["token"] for t in top] == ["H", "I", "J"] and [t["logprob"] for t in top] == [-0.25, -1.25, -2.25]


def test_chat_top_logprobs_list_the_chosen_token_first_as_vllm(server):
    """vLLM's order: the chosen token, then the most likely others; k entries in all."""
    class Sampled(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            for s in super().stream(prompt_ids, max_tokens, sampling, top_logprobs, info):
                if s.token_id == ord("H"):      # chosen, not among the three most likely
                    yield StepOut(s.token_id, -3.0, [(ord("a"), -0.1), (ord("b"), -0.5), (ord("c"), -2.0)])
                else:                           # chosen, ranked second
                    yield StepOut(s.token_id, -0.5, [(ord("a"), -0.1), (s.token_id, -0.5), (ord("c"), -2.0)])

    port, _ = server(Sampled(TOKENS))
    _, _, body = request(port, "POST", "/v1/chat/completions",
                         chat(stream=False, max_tokens=2, logprobs=True, top_logprobs=3, temperature=1.0))
    first, second = body["choices"][0]["logprobs"]["content"]
    assert [(t["token"], t["logprob"]) for t in first["top_logprobs"]] == [("H", -3.0), ("a", -0.1), ("b", -0.5)]
    assert [(t["token"], t["logprob"]) for t in second["top_logprobs"]] == [("e", -0.5), ("a", -0.1), ("c", -2.0)]


def test_completion_top_logprobs_hold_the_chosen_token_first(server):
    class Sampled(ScriptEngine):   # the chosen token is not the most likely one
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            for s in super().stream(prompt_ids, max_tokens, sampling, top_logprobs, info):
                yield StepOut(s.token_id, -3.0, [(ord("a"), -0.1), (ord("b"), -0.5)])

    port, _ = server(Sampled(TOKENS))
    _, _, body = request(port, "POST", "/v1/completions", completion(stream=False, max_tokens=1, logprobs=2,
                                                                     temperature=1.0))
    d = body["choices"][0]["logprobs"]["top_logprobs"][0]
    assert list(d.items()) == [("H", -3.0), ("a", -0.1), ("b", -0.5)]


def test_tokens_as_token_ids_and_logprob_floor(server):
    class Inf(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            for s in super().stream(prompt_ids, max_tokens, sampling, top_logprobs, info):
                yield StepOut(s.token_id, s.logprob, [(s.token_id, s.logprob), (5, float("-inf"))])

    port, _ = server(Inf(TOKENS))
    _, _, body = request(port, "POST", "/v1/chat/completions",
                         chat(stream=False, max_tokens=1, logprobs=True, top_logprobs=2,
                              return_tokens_as_token_ids=True))
    e = body["choices"][0]["logprobs"]["content"][0]
    assert e["token"] == "token_id:72" and e["bytes"] == list(b"token_id:72")
    assert e["top_logprobs"][1] == {"token": "token_id:5", "logprob": -9999.0, "bytes": list(b"token_id:5")}


def test_logprobs_need_engine_support(server):
    port, engine = server(ScriptEngine(TOKENS, logprobs_supported=False))
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(logprobs=True, max_tokens=2))
    assert status == 400 and "logprobs_supported" in out["error"]["message"] and engine.calls == []


# --------------------------------------------------------------------------------------------------
# chat template defaults, stop strings
# --------------------------------------------------------------------------------------------------
def test_default_template_kwargs_apply_and_the_request_wins_per_key(server):
    tok = FakeTokenizer()
    port, _ = server(tokenizer=tok, chat_template_kwargs={"enable_thinking": False, "x": 1})
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=1))
    assert events[0][1]["prompt_token_ids"][-1] == ord("~")
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=1, chat_template_kwargs={"enable_thinking": True}))
    assert events[0][1]["prompt_token_ids"][-1] == ord("]")
    assert tok.template_calls[-1] == {"enable_thinking": True, "x": 1}
    _, _, t = request(port, "POST", "/tokenize", {"messages": [{"role": "user", "content": "hi"}]})
    assert t["tokens"] == ids("[user]hi[assistant]~")
    _, _, t = request(port, "POST", "/tokenize", {"messages": [{"role": "user", "content": "hi"}],
                                                  "chat_template_kwargs": {"enable_thinking": True},
                                                  "add_special_tokens": False, "add_generation_prompt": True})
    assert t["tokens"] == ids("[user]hi[assistant]")


@pytest.mark.parametrize("stream", [True, False])
def test_stop_strings_cut_the_text_and_count_the_tokens(server, stream):
    port, engine = server(ScriptEngine(TOKENS, delay=0.005))   # slow enough to see the worker stop early

    def run(**kw):   # every stop below comes by token 13; the margin to 40 keeps a slow handler from racing
        body = chat(max_tokens=40, ignore_eos=True, stream=stream, **kw)
        if stream:
            body["stream_options"] = {"include_usage": True}
            _, _, events = sse(port, "/v1/chat/completions", body)
            ch = chunks_of(events)
            return text_of(ch), ch[-2]["choices"][0]["finish_reason"], ch[-1]["usage"]["completion_tokens"], ch
        _, _, b = request(port, "POST", "/v1/chat/completions", body)
        return b["choices"][0]["message"]["content"], b["choices"][0]["finish_reason"], \
            b["usage"]["completion_tokens"], b

    text, finish, n, _ = run(stop=["lo"])
    assert (text, finish, n) == ("Hel", "stop", 5)                         # the token that completed "lo" counts
    assert run(stop="lo", include_stop_str_in_output=True)[:3] == ("Hello", "stop", 5)
    assert run(stop=["zz", " w"])[:3] == ("Hello", "stop", 8)               # EOS text is empty; " w" ends at 8
    # no stop check before min_tokens: the first "H" is never checked, the next one (token 13) stops it
    assert run(stop=["H"], min_tokens=3)[:3] == ("Hello world", "stop", 13)
    assert run(stop=["l"], min_tokens=3)[:3] == ("Hel", "stop", 4)         # first check once 4 tokens exist
    if stream:   # held back while it could still start a stop string, never sent and then taken back
        *_, ch = run(stop=["lo w"])
        assert [c["choices"][0]["delta"]["content"] for c in ch[:-1]][:4] == ["", "", "", "H"]
    assert wait_for(lambda: all(c["t_end"] is not None for c in engine.calls))
    # each request ended on a stop string, and the worker stopped its generation there, not at max_tokens
    assert len(engine.calls) == (6 if stream else 5)
    assert all(c["closed_early"] and c["yielded"] < 40 for c in engine.calls), \
        [(c["yielded"], c["closed_early"]) for c in engine.calls]


# --------------------------------------------------------------------------------------------------
# the requests the inferencekit tools send
# --------------------------------------------------------------------------------------------------
# The bodies `inferencekit bench serve` builds per --engine-kind (serve/profiles.py build_body with the
# default token_ids=auto and exact_length=true), written out literally (the profile code is checked
# against the live server below): vllm/sglang/trtllm force the length with ignore_eos + min_tokens.
BENCH_SERVE_EXACT = {
    "vllm": {"ignore_eos": True, "min_tokens": "{n}"},
    "sglang": {"ignore_eos": True, "min_tokens": "{n}"},
    "trtllm": {"ignore_eos": True, "min_tokens": "{n}"},
    "llamacpp": {"ignore_eos": True},
}
BENCH_SERVE_IDS = {"vllm": {"return_token_ids": True}}


def bench_serve_body(api, kind, n):
    body = {"model": "fake-model"}
    msgs = [{"role": "user", "content": "hi"}]
    if api == "chat":
        body["messages"] = msgs
    else:
        body["prompt"] = "\n\n".join(m["content"] for m in msgs)
    body.update({"max_tokens": n, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}})
    body.update({k: (n if v == "{n}" else v) for k, v in BENCH_SERVE_EXACT.get(kind, {}).items()})
    body.update(BENCH_SERVE_IDS.get(kind, {}))
    return body


@pytest.mark.parametrize("api", ["chat", "completions"])
@pytest.mark.parametrize("kind", ["vllm", "sglang", "trtllm", "llamacpp"])
def test_bench_serve_exact_length_bodies_get_exactly_max_tokens(server, api, kind):
    """ignore_eos + min_tokens <= max_tokens with no stop_token_ids masks nothing, so it is accepted."""
    port, engine = server()
    n = 20
    path = "/v1/chat/completions" if api == "chat" else "/v1/completions"
    status, _, events = sse(port, path, bench_serve_body(api, kind, n))
    assert status == 200, events
    chunks = chunks_of(events)
    token_chunks = [c for c in chunks if c["choices"]]
    assert [c["choices"][0]["token_ids"][0] for c in token_chunks] == (TOKENS * 3)[:n]   # EOS (id 2) included
    assert token_chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == n
    assert engine.calls[0]["max_tokens"] == n and not engine.calls[0]["closed_early"]


def test_min_tokens_zero_is_always_a_noop(server):
    port, _ = server()
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=20, min_tokens=0))
    assert status == 200 and body["choices"][0]["finish_reason"] == "stop"


def test_profile_and_gate_bodies_are_accepted(server):
    """Every body shape the harness builds (bench serve profiles with a sampling extra body, the gates'
    probes, quality's trajectory / GSM8K / tokenize requests) gets a 200, not a 400."""
    profiles = pytest.importorskip("inferencekit.serve.profiles")
    gates = pytest.importorskip("inferencekit.ops.gates")
    port, engine = server(ScriptEngine(TOKENS, max_seq=4096))
    msgs = [{"role": "user", "content": "hi"}]
    sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 1234}
    bodies = []
    for kind in profiles.ENGINE_KINDS:
        for api in profiles.APIS:
            for token_ids in profiles.TOKEN_ID_MODES:
                for extra in (None, sampling, {"chat_template_kwargs": {"enable_thinking": False}}):
                    if api == "completions" and extra and "chat_template_kwargs" in extra:
                        continue
                    body, _ = profiles.build_body(api, kind, "fake-model", msgs, 16, token_ids=token_ids, extra=extra)
                    bodies.append(("/v1/chat/completions" if api == "chat" else "/v1/completions", body))
    bodies += [("/v1/chat/completions", p["body"]) for p in gates.probe_bodies(seed=1, model="fake-model")]
    client = gates.Client.__new__(gates.Client)
    client.model, client.ask_ids, client.extra = "fake-model", True, {}
    for kw in ({}, {"temperature": 1.0, "seed": 5}, {"ignore_eos": True, "logprobs": True},
               {"logprobs": True, "top_logprobs": 5}):
        bodies.append(("/v1/chat/completions", client.body(msgs, 16, **kw)))
    bodies.append(("/v1/chat/completions", {**client.body(msgs, 16, ignore_eos=True), "skip_special_tokens": False}))
    # the gates' probes: full-vocabulary samples (top_p 1, top_k -1) and special tokens kept in the stream
    bodies.append(("/v1/chat/completions", {**client.body(msgs, 16, temperature=1.0, seed=3), "top_p": 1.0,
                                            "top_k": -1}))
    bodies.append(("/v1/chat/completions", {**client.body(msgs, 16, ignore_eos=True, logprobs=True),
                                            "skip_special_tokens": False}))
    # quality sampling: first-token draws with the workload's sampling settings, and the top-20 logprob request
    bodies.append(("/v1/chat/completions", {**sampling, "model": "fake-model", "messages": msgs, "max_tokens": 1,
                                            "seed": 9, "stream": False, "return_token_ids": True,
                                            "chat_template_kwargs": {"enable_thinking": False}}))
    bodies.append(("/v1/chat/completions", {**sampling, "model": "fake-model", "messages": msgs, "max_tokens": 1,
                                            "seed": 0, "stream": False, "logprobs": True, "top_logprobs": 20,
                                            "return_tokens_as_token_ids": True, "return_token_ids": True,
                                            "chat_template_kwargs": {"enable_thinking": False}}))
    bodies.append(("/v1/chat/completions", {"model": "fake-model", "messages": msgs, "max_tokens": 16, "temperature": 0,
                                            "seed": 0, "logprobs": True, "stream": False, "top_logprobs": 20,
                                            "return_tokens_as_token_ids": True, "return_token_ids": True,
                                            "chat_template_kwargs": {"enable_thinking": False}}))
    bodies.append(("/tokenize", {"model": "fake-model", "messages": msgs, "add_generation_prompt": True,
                                 "add_special_tokens": False}))
    bodies.append(("/tokenize", {"model": "fake-model", "prompt": "x y", "add_special_tokens": False}))
    for path, body in bodies:
        if body.get("stream"):
            status, _, out = sse(port, path, body)
        else:
            status, _, out = request(port, "POST", path, body)
        assert status == 200, (path, body, out)


def test_unknown_model_is_404_and_bad_json_is_400(server):
    port, _ = server()
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(model="other"))
    assert status == 404 and out["error"]["code"] == "model_not_found"
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", body=b"{not json", headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 400 and "invalid JSON" in json.loads(r.read())["error"]["message"]
    status, _, _ = request(port, "POST", "/v1/nothing", {})
    assert status == 404
    status, _, _ = request(port, "GET", "/v1/chat/completions")
    assert status == 405


# --------------------------------------------------------------------------------------------------
# batch 1: queueing and cancellation (ported)
# --------------------------------------------------------------------------------------------------
def test_requests_queue_fifo_one_at_a_time(server):
    port, engine = server(ScriptEngine(TOKENS, delay=0.02))
    results, health = {}, []

    def client(i):
        body = chat(max_tokens=5, messages=[{"role": "user", "content": f"q{i}"}],
                    stream_options={"include_usage": True})
        results[i] = sse(port, "/v1/chat/completions", body)

    threads = []
    for i in range(3):
        t = threading.Thread(target=client, args=(i,))
        t.start()
        threads.append(t)
        time.sleep(0.03)   # fixes the arrival order
    health.append(request(port, "GET", "/health")[2])
    for t in threads:
        t.join(10)
    calls = engine.calls
    assert len(calls) == 3
    assert [c["prompt_ids"][-len("q0[assistant]"):] for c in calls] == [ids(f"q{i}[assistant]") for i in range(3)]
    for a, b in zip(calls, calls[1:]):
        assert a["t_end"] <= b["t_start"]
    spans = []
    for i in range(3):
        status, _, events = results[i]
        assert status == 200
        tok_times = [t for t, e in events if e != "[DONE]" and e["choices"]]
        assert len(tok_times) == 5
        spans.append((tok_times[0], tok_times[-1]))
        usage = [e for _, e in events if e != "[DONE]" and not e["choices"]][0]
        if i > 0:
            assert usage["inferencekit"]["queue_ms"] > 20   # it waited for the request ahead of it
    for (_, last), (first, _) in zip(spans, spans[1:]):
        assert last <= first
    assert health[0]["status"] == "ok" and health[0]["running"] and health[0]["waiting"] >= 1


def test_max_queue_answers_503(server):
    port, _ = server(ScriptEngine(TOKENS, delay=0.05), max_queue=1)
    done = []
    t1 = threading.Thread(target=lambda: done.append(sse(port, "/v1/chat/completions", chat(max_tokens=4))))
    t1.start()
    time.sleep(0.05)   # request 1 is running
    t2 = threading.Thread(target=lambda: done.append(sse(port, "/v1/chat/completions", chat(max_tokens=4))))
    t2.start()
    time.sleep(0.05)   # request 2 is waiting
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(max_tokens=4))
    assert status == 503 and "busy" in out["error"]["message"]
    t1.join(10)
    t2.join(10)
    assert [d[0] for d in done] == [200, 200]


def test_client_disconnect_stops_generation_and_frees_the_engine(server):
    port, engine = server(ScriptEngine(ids("abcdefghij"), delay=0.01, max_seq=256))
    status, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=60), stop_after=2)
    assert status == 200 and len(events) == 2
    assert wait_for(lambda: engine.calls and engine.calls[0]["t_end"] is not None)
    assert engine.calls[0]["closed_early"] and engine.calls[0]["yielded"] < 60
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=3))
    assert status == 200 and body["choices"][0]["message"]["content"] == "abc"


def test_non_streaming_disconnect_cancels_running_and_waiting_requests(server):
    port, engine = server(ScriptEngine(ids("abcdefghij"), delay=0.01, max_seq=256))

    def leave(body):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
        time.sleep(0.05)
        c.sock.shutdown(socket.SHUT_RDWR)
        c.close()

    t1 = threading.Thread(target=leave, args=(chat(stream=False, max_tokens=200),))
    t1.start()
    time.sleep(0.02)
    leave(chat(stream=False, max_tokens=200, messages=[{"role": "user", "content": "second"}]))   # still waiting
    t1.join(10)
    assert wait_for(lambda: engine.calls and engine.calls[0]["t_end"] is not None)
    assert engine.calls[0]["closed_early"] and engine.calls[0]["yielded"] < 200
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=3))
    assert status == 200 and body["choices"][0]["message"]["content"] == "abc"
    assert len(engine.calls) == 2   # the waiting request never ran


def test_engine_error_mid_stream_is_reported(server):
    class Broken(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            yield StepOut(72)
            raise RuntimeError("boom")

    port, _ = server(Broken(TOKENS))
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=5))
    assert events[0][1]["choices"][0]["delta"]["content"] == "H"
    assert "boom" in events[1][1]["error"]["message"] and events[-1][1] == "[DONE]"
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=5))
    assert status == 500 and "boom" in out["error"]["message"]


def test_worker_survives_a_failing_drain(server):
    class BadClose(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            try:
                yield from super().stream(prompt_ids, max_tokens, sampling, top_logprobs, info)
            finally:
                if len(self.calls) == 1:
                    raise RuntimeError("drain failed")

    port, _ = server(BadClose(TOKENS))
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=3))
    assert events[-1][1] == "[DONE]" and len(events) == 4   # the response was complete before the drain
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=2))
    assert status == 200 and body["choices"][0]["message"]["content"] == "He"


# --------------------------------------------------------------------------------------------------
# the engine contract as the server uses it (port of the earlier adapter tests)
# --------------------------------------------------------------------------------------------------
def test_engine_gets_the_contract_arguments_and_its_info_is_reported(server):
    class Timed(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            assert isinstance(prompt_ids, list) and isinstance(sampling, SamplingParams)
            assert set(info) == {"queue_ms", "prompt_tokens"}
            info["prefill_ms"] = 1.5
            yield from super().stream(prompt_ids, max_tokens, sampling, top_logprobs, info)

    port, engine = server(Timed(TOKENS))
    _, _, body = request(port, "POST", "/v1/completions", completion(stream=False, max_tokens=2, logprobs=3,
                                                                     temperature=0.5, seed=1))
    call = engine.calls[0]
    assert call["prompt_ids"] == [0] + ids("hi") and call["max_tokens"] == 2 and call["top_logprobs"] == 3
    assert call["sampling"] == SamplingParams(0.5, 20, 0.95, 0.0, 1)
    assert body["inferencekit"]["prefill_ms"] == 1.5 and body["inferencekit"]["prompt_tokens"] == 3


def test_tuple_steps_are_accepted_as_step_out(server):
    class Old(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            for s in super().stream(prompt_ids, max_tokens, sampling, top_logprobs, info):
                yield (s.token_id, s.logprob)

    port, _ = server(Old(TOKENS))
    _, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=2, logprobs=True))
    assert [e["logprob"] for e in body["choices"][0]["logprobs"]["content"]] == [-0.25, -0.5]
    assert serve.as_step((5,)) == StepOut(5) and serve.as_step(StepOut(1, -1.0)) == StepOut(1, -1.0)
    with pytest.raises(TypeError):
        serve.as_step(5)


def test_missing_logprob_and_a_short_stream_are_errors(server):
    class NoLp(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            for s in super().stream(prompt_ids, max_tokens, sampling, None, info):
                yield s

    class Short(ScriptEngine):
        def stream(self, prompt_ids, max_tokens, sampling, top_logprobs, info):
            yield from super().stream(prompt_ids, 2, sampling, top_logprobs, info)

    port, _ = server(NoLp(TOKENS))
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=2, logprobs=True))
    assert status == 500 and "no logprob" in out["error"]["message"]
    port, _ = server(Short(TOKENS))
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=4, ignore_eos=True))
    assert status == 500 and "stopped after 2 of 4" in out["error"]["message"]


# --------------------------------------------------------------------------------------------------
# other endpoints and helpers (ported)
# --------------------------------------------------------------------------------------------------
def test_health_models_config_tokenize(server):
    port, _ = server()
    status, _, h = request(port, "GET", "/health")
    assert status == 200 and h["status"] == "ok" and h["waiting"] == 0
    status, _, m = request(port, "GET", "/v1/models")
    assert m["data"][0]["id"] == "fake-model" and m["data"][0]["max_model_len"] == 64
    assert m["data"][0]["owned_by"] == "ikengine"
    assert request(port, "GET", "/engine/config")[2] == {"engine": "fake"}
    status, _, info = request(port, "GET", "/get_server_info")
    assert status == 200 and info["max_running_requests"] == 1 and info["max_model_len"] == 64
    status, _, t = request(port, "POST", "/tokenize", {"prompt": "hey"})
    assert status == 200 and t == {"tokens": [0] + ids("hey"), "count": 4, "max_model_len": 64}
    status, _, t = request(port, "POST", "/tokenize", {"prompt": "hey", "add_special_tokens": False})
    assert t["tokens"] == ids("hey")
    status, _, t = request(port, "POST", "/tokenize", {"messages": [{"role": "user", "content": "hi"}]})
    assert t["tokens"] == ids("[user]hi[assistant]")
    status, _, d = request(port, "POST", "/detokenize", {"tokens": ids("ok") + [EOS]})
    assert d == {"prompt": "ok<eos>"}
    status, _, out = request(port, "POST", "/tokenize", {"prompt": "x", "bogus": 1})
    assert status == 400


def test_keep_alive_serves_several_requests_on_one_connection(server):
    port, _ = server()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    for stream in (True, False, True):
        c.request("POST", "/v1/chat/completions", body=json.dumps(chat(max_tokens=2, stream=stream)).encode(),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read()
        assert r.status == 200 and (raw.endswith(b"data: [DONE]\n\n") if stream else json.loads(raw)["choices"])
    c.close()


def test_incremental_detokenizer_holds_back_partial_utf8():
    text = "héllo € 😀!"
    d = serve.IncrementalDetokenizer(ByteTokenizer())
    deltas = [d.add(b) for b in text.encode("utf-8")]
    deltas.append(d.flush())
    assert "".join(deltas) == text
    assert not any("�" in x for x in deltas)


def test_find_stop_searches_only_where_a_new_match_can_end():
    assert serve.find_stop("Hello", 1, ("lo",), False) == 3
    assert serve.find_stop("Hello", 1, ("lo",), True) == 5
    assert serve.find_stop("lo Hel", 1, ("lo",), False) is None      # an old match is not found again
    assert serve.find_stop("abcd", 2, ("zz", "bc", "cd"), False) == 1  # request order wins, not position
    assert serve.find_stop("abc", 0, ("c",), False) is None


def test_config_line_round_trip():
    cfg = {"settings": {"act": "as_shipped"}, "cuda_graphs": True}
    line = serve.format_config_line(cfg)
    assert "\n" not in line and line.startswith("[serve] resolved_config ")
    assert serve.parse_config_line("2026-10-13 " + line) == cfg
    assert serve.parse_config_line("[serve] listening") is None
    shared = serve.format_shared_config_line(cfg)
    assert "\n" not in shared
    i = shared.find('{"inferencekit_resolved_config"')
    assert i == 0 and json.loads(shared[i:])["inferencekit_resolved_config"] == cfg


# --------------------------------------------------------------------------------------------------
# startup: CLI, generation_config, chat template file, warm-up, resolved config, ENGINE.json
# --------------------------------------------------------------------------------------------------
def make_model_dir(tmp_path, gen=None):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["FakeForCausalLM"]}))
    if gen is not None:
        (d / "generation_config.json").write_text(json.dumps(gen))
    return d


def start_built(tmp_path, argv, engine, gen=None, record=None, factory=FakeTokenizer, cwd=None, build=None):
    """serve.build with a fake engine, started in `cwd` (default tmp_path). tmp_path holds a .git so it is
    the repository root whatever lies above it (ENGINE.json goes there by default)."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    model = make_model_dir(tmp_path, gen)
    args = serve.parse_args(["--model", str(model), "--port", "0"] + argv)
    srv, cfg = serve.build(args, builder=build or builder(engine, record), tokenizer_factory=factory,
                           argv=["python", "-m", "ikengine.serve", "--model", str(model)] + argv,
                           cwd=str(cwd or tmp_path))
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    th.start()
    return srv, cfg


def stop_built(srv):
    srv.shutdown()
    srv.server_close()
    srv.scheduler.stop()


def test_cli_flags_and_errors(capsys):
    a = serve.parse_args(["--model", "m"])
    assert (a.port, a.host, a.max_seq, a.act, a.kv, a.device, a.max_queue, a.engine_json) == \
        (8000, "127.0.0.1", 4096, "as_shipped", "as_shipped", "cuda", 0, None)
    a = serve.parse_args(["--model", "m", "--default-chat-template-kwargs", '{"enable_thinking": false}',
                          "--act", "a16", "--kv", "bf16", "--max-queue", "3", "--served-model-name", "x"])
    assert a.default_chat_template_kwargs == {"enable_thinking": False} and (a.act, a.kv) == ("a16", "bf16")
    for bad in (["--act", "fp4"], ["--kv", "fp8"], ["--default-chat-template-kwargs", "{nope"],
                ["--default-chat-template-kwargs", "[1]"]):
        with pytest.raises(SystemExit):
            serve.parse_args(["--model", "m"] + bad)


def test_module_entry_point_runs_without_torch(tmp_path):
    """`python -m ikengine.serve` as an agent runs it: --help and argument errors need neither torch nor
    transformers (both are blocked here)."""
    env = {**os.environ, "PYTHONPATH": str(SKELETON)}
    blocked = ("import runpy, sys; sys.modules['torch'] = None; sys.modules['transformers'] = None; "
               "sys.argv = ['ikengine.serve'] + sys.argv[1:]; runpy.run_module('ikengine.serve', run_name='__main__')")
    for cmd in ([sys.executable, "-m", "ikengine.serve", "--help"], [sys.executable, "-c", blocked, "--help"]):
        r = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert all(f in r.stdout for f in ("--model", "--act", "--kv", "--engine-json", "--max-queue"))
    r = subprocess.run([sys.executable, "-c", blocked, "--act", "fp4"], cwd=tmp_path, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 2 and "--model" in r.stderr and "invalid choice" in r.stderr


def test_build_imports_build_engine_lazily(tmp_path, monkeypatch):
    """With no builder, build() takes ikengine.engine.build_engine (imported only then: torch loads there)."""
    record = {}
    fake = types.ModuleType("ikengine.engine")
    fake.build_engine = builder(CharLM(), record)
    monkeypatch.setitem(sys.modules, "ikengine.engine", fake)
    (tmp_path / ".git").mkdir()
    model = make_model_dir(tmp_path)
    args = serve.parse_args(["--model", str(model), "--port", "0", "--kv", "bf16", "--engine-json", ""])
    srv, cfg = serve.build(args, tokenizer_factory=FakeTokenizer, argv=["x"], cwd=str(tmp_path))
    srv.server_close()   # never served: shutdown() would wait for serve_forever
    srv.scheduler.stop()
    assert record == {"model_dir": str(model), "device": "cuda", "max_seq": 4096, "act": "as_shipped", "kv": "bf16"}
    assert cfg["settings"]["kv"] == "bf16"


def test_build_reads_generation_config_template_and_writes_engine_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    tpl = tmp_path / "chat_template.jinja"
    tpl.write_text("{{ messages }}")
    made, record = [], {}

    def factory(path, chat_template):
        made.append((path, chat_template))
        return FakeTokenizer(path, chat_template)

    model_numerics = {"weights": "nvfp4 as stored", "activations": "nvfp4 (input_scale)", "kv": "fp8 e4m3"}
    engine = CharLM(model=Model(model_numerics), resolved_config=lambda: {"cuda_graphs": True, "decode": "x"})
    gen = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "eos_token_id": [1, 106], "repetition_penalty": 1.05}
    srv, cfg = start_built(tmp_path, ["--chat-template", str(tpl), "--default-chat-template-kwargs",
                                      '{"enable_thinking": false}', "--max-seq", "1024", "--device", "cuda:1"],
                           engine, gen=gen, record=record, factory=factory)
    try:
        port = srv.server_address[1]
        assert record == {"model_dir": str(tmp_path / "model"), "device": "cuda:1", "max_seq": 1024,
                          "act": "as_shipped", "kv": "as_shipped"}
        assert made == [(str(tmp_path / "model"), "{{ messages }}")]
        # warm-up: greedy, then the default sampling with top-20 logprobs, a few tokens each
        warm = [(c["sampling"], c["top_logprobs"], c["max_tokens"]) for c in engine.calls]
        assert warm == [(GREEDY, None, 2), (SamplingParams(1.0, 64, 0.95, 0.0, 0), 20, 2)]
        assert engine.calls[0]["prompt_ids"] == ids("[user]Hello[assistant]~")
        # requests: model defaults, EOS ids from generation_config + tokenizer, template defaults
        _, _, body = request(port, "POST", "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
        assert engine.calls[-1]["sampling"] == SamplingParams(1.0, 64, 0.95)
        assert engine.calls[-1]["prompt_ids"][-1] == ord("~")
        assert srv.ctx["eos_ids"] == (1, 2, 106)
        s = cfg["server"]
        assert cfg["cuda_graphs"] is True and cfg["engine_config"] == {"cuda_graphs": True, "decode": "x"}
        assert s["sampling_defaults"] == {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0}
        assert s["eos_token_ids"] == [1, 2, 106] and s["generation_config_not_applied"] == {"repetition_penalty": 1.05}
        assert s["chat_template"]["source"] == "file" and len(s["chat_template"]["sha256"]) == 64
        assert s["default_chat_template_kwargs"] == {"enable_thinking": False}
        assert s["port"] == port and s["cwd"] == str(tmp_path) and s["argv"][:3] == ["python", "-m", "ikengine.serve"]
        assert cfg["numerics"] == {**model_numerics, "requested": {"act": "as_shipped", "kv": "as_shipped"},
                                   "changed": []}
        assert request(port, "GET", "/engine/config")[2] == cfg
    finally:
        stop_built(srv)
    err = capsys.readouterr().err
    assert serve.parse_config_line(next(ln for ln in err.splitlines() if "resolved_config " in ln)) == cfg
    assert any(ln.startswith('{"inferencekit_resolved_config"') for ln in err.splitlines())
    assert "repetition_penalty=1.05" in err
    doc = json.loads((tmp_path / "ENGINE.json").read_text())   # the repository root (tmp_path holds .git)
    assert doc["launch"] == cfg["server"]["argv"]                # started at the root: the command as run
    assert doc["port"] == port != 0                              # bound first: --port 0 records the real port
    assert doc["numerics"] == cfg["numerics"]
    assert doc["numerics_summary"] == "weights: nvfp4 as stored; activations: nvfp4 (input_scale); kv: fp8 e4m3"
    assert doc["resolved_config"] == cfg
    assert sorted(doc[serve.WRITTEN_KEY]) == ["launch", "numerics", "numerics_summary", "port"]


def test_engine_json_keeps_what_is_there_and_marks_changed_numerics(tmp_path, capsys):
    path = tmp_path / "run" / "ENGINE.json"
    path.parent.mkdir()
    mine = {"install": ["pip install -e ."], "launch": ["bash", "LAUNCH.sh"], "port": 8000, "notes": "mine"}
    path.write_text(json.dumps(mine))
    engine = CharLM(numerics={"weights": "fp8 block-128 as stored", "activations": "bf16"})
    srv, cfg = start_built(tmp_path, ["--engine-json", str(path), "--act", "a16"], engine)
    stop_built(srv)
    doc = json.loads(path.read_text())
    assert {k: doc[k] for k in mine} == mine
    assert "kept your launch, port" in capsys.readouterr().err
    # the module declares no changed list, so the flag that is not as_shipped is the change
    assert doc["numerics"]["changed"] == ["--act a16"] and doc["numerics"]["requested"] == {"act": "a16",
                                                                                            "kv": "as_shipped"}
    assert doc["numerics_summary"] == \
        "weights: fp8 block-128 as stored; activations: bf16 (numerics changed: --act a16)"
    assert sorted(doc[serve.WRITTEN_KEY]) == ["numerics", "numerics_summary"]
    # unchanged content is not rewritten; a file that is not a JSON object is left alone
    mtime = os.stat(path).st_mtime_ns
    fields = {k: doc[k] for k in ("numerics", "numerics_summary")}
    assert serve.write_engine_json(str(path), {"launch": ["x"], "port": 1, **fields}, doc["resolved_config"]) == doc
    assert os.stat(path).st_mtime_ns == mtime
    path.write_text("[1, 2]")
    assert serve.write_engine_json(str(path), {"launch": ["x"]}, {}) is None
    assert path.read_text() == "[1, 2]"
    assert "not a JSON object" in capsys.readouterr().err


def _numerics_by_flags(act, kv):
    """A model module's numerics as the Gemma 4 / Qwen3.8 modules write them: NVFP4 weights and activations
    as shipped, BF16 KV as shipped (so --kv bf16 changes nothing), per-kind activations."""
    a16 = act == "a16"
    return {"weights": "as stored: nvfp4 x4, bf16 x2",
            "activations": {"nvfp4": "bf16 (a16)" if a16 else "nvfp4 e2m1, static input_scale", "bf16": "bf16"},
            "kv": "bf16", "as_shipped": not a16,
            "changed": ["activations kept 16-bit (a16) for the nvfp4 layers"] if a16 else []}


def test_engine_json_restart_refreshes_what_the_server_wrote(tmp_path, capsys):
    """Started as shipped, then again with --act a16: launch, port and numerics follow the second run;
    fields the agent added are kept; a field the agent edits by hand is kept from then on."""
    run, engine_dir = tmp_path, tmp_path / "engine"
    engine_dir.mkdir()

    def build_for(record):
        def build_engine(model_dir, device="cuda", max_seq=4096, act="as_shipped", kv="as_shipped"):
            record.update(act=act, kv=kv)
            return CharLM(max_seq=max_seq, numerics=_numerics_by_flags(act, kv))
        return build_engine

    # 1: as shipped, started in /work/run/engine as the README says: ENGINE.json lands at the repo root
    srv, cfg1 = start_built(tmp_path, ["--kv", "bf16"], None, cwd=engine_dir, build=build_for({}))
    stop_built(srv)
    path = run / "ENGINE.json"
    doc = json.loads(path.read_text())
    assert not (engine_dir / "ENGINE.json").exists()
    assert doc["launch"][:2] == ["sh", "-c"]
    assert doc["launch"][2].startswith("cd engine && exec python -m ikengine.serve --model ")
    assert doc["numerics"]["changed"] == [] and "numerics changed" not in doc["numerics_summary"]
    assert doc["numerics_summary"] == ("weights: as stored: nvfp4 x4, bf16 x2; activations.nvfp4: nvfp4 e2m1, static "
                                       "input_scale; activations.bf16: bf16; kv: bf16")
    first_port = doc["port"]
    doc.update(install=["pip install -e ."], notes="mine")
    path.write_text(json.dumps(doc))
    # 2: --act a16, started from the root this time
    srv, cfg2 = start_built(tmp_path, ["--act", "a16"], None, build=build_for({}))
    stop_built(srv)
    doc = json.loads(path.read_text())
    assert doc["launch"] == cfg2["server"]["argv"] and "--act" in doc["launch"]
    assert doc["port"] == srv.server_address[1] and doc["resolved_config"] == cfg2
    assert first_port == cfg1["server"]["port"]
    assert doc["numerics"] == cfg2["numerics"] and doc["numerics"]["requested"]["act"] == "a16"
    assert doc["numerics"]["changed"] == ["activations kept 16-bit (a16) for the nvfp4 layers"]
    assert doc["numerics_summary"].endswith("(numerics changed: activations kept 16-bit (a16) for the nvfp4 layers)")
    assert "activations.nvfp4: bf16 (a16)" in doc["numerics_summary"]
    assert (doc["install"], doc["notes"]) == (["pip install -e ."], "mine")
    capsys.readouterr()
    # 3: the agent takes "launch" over by editing it; a restart keeps it and still refreshes the rest
    doc["launch"] = ["bash", "LAUNCH.sh"]
    path.write_text(json.dumps(doc))
    srv, cfg3 = start_built(tmp_path, [], None, build=build_for({}))
    stop_built(srv)
    doc = json.loads(path.read_text())
    assert doc["launch"] == ["bash", "LAUNCH.sh"] and doc["numerics"]["changed"] == []
    assert "kept your launch" in capsys.readouterr().err
    assert sorted(doc[serve.WRITTEN_KEY]) == ["numerics", "numerics_summary", "port"]


def test_a_port_in_use_fails_before_engine_json_is_written(tmp_path):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    try:
        (tmp_path / ".git").mkdir()
        args = serve.parse_args(["--model", str(make_model_dir(tmp_path)), "--port", str(blocker.getsockname()[1])])
        with pytest.raises(OSError):
            serve.build(args, builder=builder(CharLM()), tokenizer_factory=FakeTokenizer, argv=["x"],
                        cwd=str(tmp_path))
        assert not (tmp_path / "ENGINE.json").exists()
    finally:
        blocker.close()


def test_numerics_changed_trusts_the_model_module():
    # the module says nothing changed (the checkpoint ships BF16 KV): --kv bf16 is no change
    assert serve.numerics_changed({"kv": "bf16", "changed": []}, kv="bf16") == []
    assert serve.numerics_changed({"weights": "bf16", "as_shipped": True}, act="a16") == []
    # it lists its changes: those, and not the flags
    assert serve.numerics_changed({"changed": ["kv bf16; as shipped fp8"]}, kv="bf16") == ["kv bf16; as shipped fp8"]
    assert serve.numerics_changed({"changed": "kv bf16"}) == ["kv bf16"]
    # it declares neither: the flags that are not as_shipped
    assert serve.numerics_changed({"kv": "bf16"}, act="a16", kv="bf16") == ["--act a16", "--kv bf16"]
    assert serve.numerics_changed({}) == []
    assert serve.numerics_changed({"as_shipped": False}) == \
        ["the model module reports as_shipped: false without a changed list"]


def test_numerics_summary_flattens_nested_values():
    """A model module's numerics dict (the Qwen3.8 module's shape: per-kind activations) -> one line."""
    n = {"weights": "as stored: fp8_block x10, bf16 x3",
         "activations": {"bf16": "bf16", "fp8_block": "fp8 e4m3, dynamic per token in groups of 128"},
         "kv": "bf16", "state": "fp32 (Gated DeltaNet recurrent state)", "group": 16, "tied": True,
         "formats": ["fp8", "bf16"], "as_shipped": True, "changed": [], "requested": {"act": "x"}, "empty": ""}
    assert serve.numerics_summary(n) == (
        "weights: as stored: fp8_block x10, bf16 x3; activations.bf16: bf16; activations.fp8_block: fp8 e4m3, "
        "dynamic per token in groups of 128; kv: bf16; state: fp32 (Gated DeltaNet recurrent state); group: 16; "
        "formats: fp8, bf16")
    assert serve.numerics_summary({"weights": "bf16"}, ["--act a16"]) == "weights: bf16 (numerics changed: --act a16)"
    assert serve.numerics_summary({}) == "not declared by the model module"


def test_engine_json_path_and_portable_launch(tmp_path):
    run = tmp_path / "run"
    (run / ".git").mkdir(parents=True)
    (run / "engine" / "sub dir").mkdir(parents=True)
    eng = str(run / "engine")
    assert serve.git_toplevel(eng) == str(run) and serve.git_toplevel(str(run)) == str(run)
    assert serve.engine_json_path(None, eng) == str(run / "ENGINE.json")
    assert serve.engine_json_path("", eng) == ""
    assert serve.engine_json_path("x.json", eng) == os.path.join(eng, "x.json")
    assert serve.engine_json_path("/abs/E.json", eng) == "/abs/E.json"
    argv = ["python", "-m", "ikengine.serve", "--model", "../ckpt dir"]
    top = str(run)
    assert serve.portable_launch(argv, eng, None, pkg_root=eng) == argv              # not in a repository
    assert serve.portable_launch(argv, top, top, pkg_root=top) == argv               # at the root, found via cwd
    assert serve.portable_launch(argv, top, top, pkg_root=eng) == ["env", "PYTHONPATH=engine"] + argv
    assert serve.portable_launch(argv, top, top, pkg_root=str(tmp_path / "site")) == argv   # installed elsewhere
    assert serve.portable_launch(argv, eng, top, pkg_root=eng) == \
        ["sh", "-c", "cd engine && exec python -m ikengine.serve --model '../ckpt dir'"]
    assert serve.portable_launch(argv, str(run / "engine" / "sub dir"), top, pkg_root=eng) == \
        ["sh", "-c", "cd 'engine/sub dir' && exec env PYTHONPATH=.. python -m ikengine.serve --model '../ckpt dir'"]


def test_portable_launch_runs_from_the_repository_root(tmp_path):
    """The rewritten command, run from the root as a fresh copy of the repository is, starts where the original
    started and finds the package the original found."""
    run = tmp_path / "run"
    (run / ".git").mkdir(parents=True)
    (run / "engine").mkdir()
    (run / "tools").mkdir()
    (run / "engine" / "probe_mod.py").write_text("X = 'found'\n")
    probe = [sys.executable, "-c", "import os, probe_mod; print(os.getcwd(), probe_mod.X)"]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for cwd in (run / "engine", run, run / "tools"):
        launch = serve.portable_launch(probe, str(cwd), str(run), pkg_root=str(run / "engine"))
        r = subprocess.run(launch, cwd=run, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (launch, r.stderr)
        assert r.stdout.strip().rsplit(" ", 1) == [os.path.realpath(cwd), "found"], launch


def test_no_engine_json_flag_and_no_numerics_declared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = ScriptEngine(TOKENS, max_seq=64)
    srv, cfg = start_built(tmp_path, ["--engine-json", ""], engine)
    stop_built(srv)
    assert not (tmp_path / "ENGINE.json").exists()
    assert "cuda_graphs" not in cfg
    assert cfg["numerics"] == {"requested": {"act": "as_shipped", "kv": "as_shipped"}, "changed": []}
    assert cfg["server"]["sampling_defaults_source"].startswith("built-in")


def test_engine_config_method_and_model_eos_ids_are_recorded(tmp_path, monkeypatch, capsys):
    """ikengine.engine.Engine reports itself through config(); EOS ids only the model module declares are
    recorded and warned about, not stopped on (the stop set is generation_config's plus the tokenizer's)."""
    monkeypatch.chdir(tmp_path)
    engine = CharLM(config=lambda: {"cuda_graphs": False, "lookahead": 2}, eos_token_ids=frozenset({EOS, 106}))
    srv, cfg = start_built(tmp_path, [], engine, gen={"eos_token_id": EOS})
    stop_built(srv)
    assert cfg["engine_config"] == {"cuda_graphs": False, "lookahead": 2} and cfg["cuda_graphs"] is False
    assert cfg["server"]["eos_token_ids"] == [EOS] and cfg["server"]["model_eos_token_ids"] == [EOS, 106]
    assert "declares EOS ids [106]" in capsys.readouterr().err
    assert json.loads((tmp_path / "ENGINE.json").read_text())["resolved_config"] == cfg


def test_peer_check_works_on_descriptors_above_1024():
    """select() cannot watch a descriptor >= 1024; the check must still tell a live client from a gone one."""
    resource = pytest.importorskip("resource")
    high = 1500
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard <= high:
        pytest.skip(f"open-file limit {hard} is too low for descriptor {high}")
    if soft != resource.RLIM_INFINITY and soft <= high:
        resource.setrlimit(resource.RLIMIT_NOFILE, (high + 1, hard))
    socks = []

    def free_fd(n):
        while True:
            try:
                os.fstat(n)
            except OSError:
                return n
            n -= 1

    def handler_on_high_fd():
        a, b = socket.socketpair()
        hi = socket.socket(fileno=os.dup2(a.fileno(), free_fd(high)))
        a.close()
        socks.extend([hi, b])
        return types.SimpleNamespace(connection=hi), b

    try:
        h, peer = handler_on_high_fd()
        assert h.connection.fileno() >= 1024
        assert serve.Handler._peer_closed(h) is False          # connected, nothing sent
        peer.sendall(b"GET / HTTP/1.1\r\n")
        assert serve.Handler._peer_closed(h) is False          # a next request's bytes are not EOF
        h2, peer2 = handler_on_high_fd()
        assert h2.connection.fileno() >= 1024
        peer2.close()
        assert serve.Handler._peer_closed(h2) is True          # closed
    finally:
        for s_ in socks:
            s_.close()
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_hf_tokenizer_decode_keeps_spaces_as_generated():
    """No clean_up_tokenization_spaces (as vLLM): the incremental detokenizer relies on text it already
    sent staying the same when more tokens are decoded."""
    class Tok:
        def decode(self, ids, **kw):
            self.kw = kw
            return "x"

    t = object.__new__(serve.HFTokenizer)
    t.tok, t._lock = Tok(), threading.Lock()
    assert t.decode([1, 2]) == "x"
    assert t.tok.kw == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}


def test_chat_template_argument_forms(tmp_path):
    assert serve.load_chat_template(None) == (None, {"source": "tokenizer"})
    text, src = serve.load_chat_template("{% for m in messages %}{{ m.content }}{% endfor %}")
    assert src["source"] == "inline" and text.startswith("{%")
    with pytest.raises(SystemExit):
        serve.load_chat_template(str(tmp_path / "missing.jinja"))


def test_hf_tokenizer_with_a_template_file(tmp_path):
    """The real tokenizer wrapper (needs transformers): template file, kwargs, special tokens."""
    pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast
    vocab = {"<bos>": 0, "<eos>": 1, "[UNK]": 2}
    for w in ("user", "assistant", "hi", "think", ":", "|"):
        vocab[w] = len(vocab)
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tk.post_processor = tokenizers.processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 0)])
    PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", unk_token="[UNK]") \
        .save_pretrained(tmp_path / "tok")
    tpl = "{% for m in messages %}{{ m.role }} : {{ m.content }} | {% endfor %}{% if enable_thinking %}think{% endif %}"
    t = serve.HFTokenizer(str(tmp_path / "tok"), tpl)
    msgs = [{"role": "user", "content": "hi"}]
    assert t.apply_chat_template(msgs, enable_thinking=True) == [vocab[w] for w in ("user", ":", "hi", "|", "think")]
    assert t.apply_chat_template(msgs, add_special_tokens=True)[0] == 0
    assert t.encode("hi") == [0, vocab["hi"]] and t.decode([0, vocab["hi"], 1]) == "hi"
    assert t.eos_token_id == 1 and serve.eos_token_ids({"eos_token_id": 7}, t) == [1, 7]


# --------------------------------------------------------------------------------------------------
# the inferencekit tools against the server (the honest toy model must pass every check)
# --------------------------------------------------------------------------------------------------
LIVE_GATES = ["caching", "token_count", "output_length", "hidden_probes", "rare_tokens"]


@pytest.fixture
def char_counts(monkeypatch, store):
    gates = pytest.importorskip("inferencekit.ops.gates")
    monkeypatch.setattr(gates, "load_tokenizer", lambda name: (lambda text: len(text)) if name else None)
    return gates


def test_gates_run_passes_on_an_honest_engine(server, char_counts):
    from inferencekit import registry
    port, engine = server(CharLM(ttft_s=0.015, per_token_s=0.0015))
    rec = registry.call("run_gates", {"base_url": f"http://127.0.0.1:{port}", "max_tokens": 24, "gates": LIVE_GATES,
                                      "tokenizer": "chars", "seed": 7})
    verdicts = {name: g["verdict"] for name, g in rec["data"]["gates"].items()}
    assert verdicts == {g: "PASS" for g in LIVE_GATES}, json.dumps(rec["data"]["gates"], indent=1)[:4000]
    # the sampling probes ran seeded, with the model's top_k / top_p underneath their temperature, or with the
    # full vocabulary when a probe asks for it (top_p 1, top_k -1 = off: the gates do)
    sampled = [c["sampling"] for c in engine.calls if c["sampling"].temperature == 1.0]
    assert sampled and all((s.top_k, s.top_p) in {(20, 0.95), (0, 1.0)} and s.seed is not None for s in sampled)


def test_bench_serve_times_default_sampling_with_exact_lengths(server, store, tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))
    from inferencekit import registry
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"short-{i}", "bucket": "short", "source": "test", "kind": "chat",
                                           "messages": [{"role": "user", "content": f"say something {i}"}],
                                           "max_tokens": 32}) + "\n" for i in range(2)))
    port, engine = server(CharLM(per_token_s=0.001))
    rec = registry.call("bench_serve", {"base_url": f"http://127.0.0.1:{port}", "model": "fake-model",
                                        "prompts": str(prompts), "engine_kind": "vllm", "rounds": 1, "warmup": 1,
                                        "resamples": 100, "extra_body": json.dumps({"temperature": 1.0, "seed": 1234})})
    assert rec["data"]["token_count_methods"] == {"stream_token_ids:choice_token_ids": 2}
    timed = [c for c in engine.calls if c["max_tokens"] == 32]
    assert len(timed) >= 2 and all(c["yielded"] == 32 and not c["closed_early"] for c in timed)
    assert all(c["sampling"] == SamplingParams(1.0, 20, 0.95, 0.0, 1234) for c in timed)


def test_quality_trajectory_gets_top20_with_exact_ids(server, store, tmp_path):
    from inferencekit import registry
    prompts = tmp_path / "q.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"q{i}", "bucket": "short", "kind": "chat", "source": "test",
                                           "messages": [{"role": "user", "content": f"question {i}"}],
                                           "max_tokens": 12}) + "\n" for i in range(2)))
    port, _ = server(CharLM())
    rec = registry.call("quality_trajectory", {"base_url": f"http://127.0.0.1:{port}", "model": "fake-model",
                                               "prompts": str(prompts), "label": "skel", "run_dir": str(tmp_path / "run"),
                                               "engine_kind": "vllm",
                                               "chat_template_kwargs": json.dumps({"enable_thinking": False})})
    m = {x["name"]: x["value"] for x in rec["metrics"]}
    assert m["prompts_ok"] == 2 and m["top_logprobs_returned"] == 20
    doc = json.loads((tmp_path / "run" / "quality" / "skel" / "q0.json").read_text())
    assert doc["token_id_method"] != "strings" and not doc["id_mismatch_positions"]
    assert doc["prompt_token_ids"][-1] == ord("~") and doc["prompt_token_ids_source"] == "response.prompt_token_ids"
