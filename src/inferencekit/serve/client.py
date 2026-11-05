"""A streaming client for OpenAI-compatible servers, on plain asyncio sockets (HTTP/1.1, SSE).

Why raw sockets: the client is the only timer for published numbers, so it should add as little as
possible between "bytes arrived" and "timestamp taken", and it should not pull in an HTTP stack whose
buffering we cannot see. Every timestamp is ``time.perf_counter_ns()`` (monotonic, nanoseconds):

  t_send     just before the TCP connection is opened (TTFT and end-to-end start here; the connect
             time is recorded separately as connect_ms, about 0.1 ms on localhost)
  t_event    when the bytes holding an SSE event were read from the socket
  t_first    the first event that carries generated tokens (token ids, or non-empty text)
  t_last     the last event that carries generated tokens
  t_done     the "[DONE]" sentinel, or end of stream

Token ids are collected from every place a server is known to put them (see profiles.py): the
choice's ``token_ids`` (vLLM, TRT-LLM completions), logprob entries with an ``id`` (llama.cpp) or a
``token_id:N`` token (vLLM return_tokens_as_token_ids), and ``sglext.output_ids`` (SGLang). An event
may carry more than one token (speculative decoding, server-side coalescing), so tokens are counted
from ids, never from the number of events.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import ssl
import time
from dataclasses import asdict, dataclass, field
from urllib.parse import urlsplit

_TOKEN_ID = re.compile(r"^token_id:(-?\d+)$")
MAX_ERROR_BODY = 2000
# Addresses in error text: socket errors name the peer ("Connect call failed ('203.0.113.7', 8000)"), and
# a proxy's error page can too. Evidence is published, so they are replaced before anything is stored.
_IPV4 = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?:%\w+)?(?![\w:])")
_PORT_AFTER = re.compile(r"(<server>|<addr>)(['\]]?)(\s*[,:]\s*)\d{1,5}\b")


@dataclass(frozen=True)
class Endpoint:
    """Where requests go. ``base_url`` is the server root or its /v1 root."""

    scheme: str
    host: str
    port: int
    prefix: str  # path prefix before /v1/...
    api_key: str | None = None

    @classmethod
    def parse(cls, base_url: str, api_key: str | None = None) -> "Endpoint":
        u = urlsplit(base_url if "://" in base_url else "http://" + base_url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"base_url must be http(s)://host[:port][/prefix], got {base_url!r}")
        prefix = u.path.rstrip("/")
        if prefix.endswith("/v1"):
            prefix = prefix[:-3]
        port = u.port or (443 if u.scheme == "https" else 80)
        return cls(u.scheme, u.hostname, port, prefix, api_key)

    def path(self, api: str) -> str:
        return f"{self.prefix}/v1/chat/completions" if api == "chat" else f"{self.prefix}/v1/completions"

    def is_local(self) -> bool:
        return self.host in ("localhost", "127.0.0.1", "::1")


def scrub(text: str | None, ep: Endpoint) -> str | None:
    """``text`` without the server's host name, IP addresses or port (error strings go into evidence)."""
    if not text:
        return text
    out = re.sub(rf"(?<![A-Za-z0-9.-]){re.escape(ep.host)}(?![A-Za-z0-9-])", "<server>", text, flags=re.IGNORECASE)
    out = _IPV4.sub("<addr>", out)
    out = _IPV6.sub("<addr>", out)
    return _PORT_AFTER.sub(r"\1\2\3<port>", out)


@dataclass
class Event:
    t_ns: int
    n_ids: int | None  # token ids in this event (None: the server sent no ids)
    n_chars: int


@dataclass
class RequestResult:
    id: str
    bucket: str
    salt: str | None = None  # unique prefix tag added to the prompt, if any
    ok: bool = False
    error: str | None = None
    status: int | None = None
    t_scheduled_ns: int | None = None
    t_send_ns: int = 0
    t_connected_ns: int | None = None
    t_headers_ns: int | None = None
    t_first_ns: int | None = None
    t_last_ns: int | None = None
    t_done_ns: int | None = None
    t_end_ns: int = 0
    events: list = field(default_factory=list)  # token-bearing Events only
    text: str = ""
    reasoning: str = ""
    token_ids: list | None = None
    ids_source: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    n_sse_events: int = 0
    # filled by count_tokens()
    n_tokens: int | None = None
    token_count_method: str | None = None
    count_mismatch: str | None = None
    n_ids: int | None = None  # streamed ids, when the server sent any
    n_usage: int | None = None  # usage.completion_tokens, when the server sent it
    # the exact JSON body sent (set by the caller; written to inputs/ for the gates)
    request_body: dict | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d["events"] = [[e.t_ns, e.n_ids, e.n_chars] for e in self.events]
        d.pop("request_body", None)
        return d


# --------------------------------------------------------------------------- SSE parsing

class _Stream:
    """Accumulates one response's SSE events into a RequestResult."""

    def __init__(self, res: RequestResult):
        self.res = res
        self.buf = b""
        self.event_name = None
        self.data_lines: list[str] = []
        self.done = False
        self._ids: list[int] = []
        self._ids_seen = False
        self._ids_via = None
        self._sglext_ids = None

    def feed(self, data: bytes, t_ns: int) -> None:
        self.buf += data
        while not self.done:
            i = self.buf.find(b"\n")
            if i < 0:
                return
            line, self.buf = self.buf[:i].rstrip(b"\r").decode("utf-8", "replace"), self.buf[i + 1:]
            if line == "":
                self._dispatch(t_ns)
            elif line.startswith(":"):
                continue  # SSE comment / keep-alive
            elif line.startswith("data:"):
                value = line[5:]
                self.data_lines.append(value[1:] if value.startswith(" ") else value)  # SSE: drop one space
            elif line.startswith("event:"):
                self.event_name = line[6:].strip()

    def close(self, t_ns: int) -> None:
        if self.buf.strip():  # a last event without its blank line
            self.feed(b"\n\n", t_ns)
        elif self.data_lines:
            self._dispatch(t_ns)

    def _dispatch(self, t_ns: int) -> None:
        if not self.data_lines:
            self.event_name = None
            return
        payload = "\n".join(self.data_lines)
        name, self.data_lines, self.event_name = self.event_name, [], None
        res = self.res
        res.n_sse_events += 1
        if payload.strip() == "[DONE]":
            self.done = True
            res.t_done_ns = t_ns
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            res.error = res.error or f"unparseable SSE data: {payload[:200]!r}"
            return
        if not isinstance(obj, dict):
            return
        if name == "sglext_ids" or isinstance(obj.get("sglext"), dict):
            ext = obj if name == "sglext_ids" else obj["sglext"]
            out = ext.get("output_ids")
            if isinstance(out, list) and out and isinstance(out[0], list):
                self._sglext_ids = [int(x) for x in out[0]]
        if obj.get("error") and not obj.get("choices"):
            err = obj["error"]
            res.error = res.error or ("server error in stream: " + (err.get("message") if isinstance(err, dict)
                                                                     else str(err))[:500])
            return
        if isinstance(obj.get("usage"), dict):
            res.usage = obj["usage"]
        for c in obj.get("choices") or []:
            if not isinstance(c, dict) or c.get("index", 0) != 0:
                continue
            text, reasoning = "", ""
            delta = c.get("delta")
            if isinstance(delta, dict):  # chat
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if delta.get("tool_calls"):
                    text = text or json.dumps(delta["tool_calls"])
            elif isinstance(c.get("text"), str):  # completions
                text = c["text"]
            ids, via = _ids_in_choice(c)
            if ids is not None:
                self._ids_seen = True
                self._ids_via = self._ids_via or via
                self._ids.extend(ids)
            res.text += text
            res.reasoning += reasoning
            n_chars = len(text) + len(reasoning)
            if (ids is not None and len(ids) > 0) or (ids is None and n_chars > 0):
                ev = Event(t_ns, len(ids) if ids is not None else None, n_chars)
                res.events.append(ev)
                if res.t_first_ns is None:
                    res.t_first_ns = t_ns
                res.t_last_ns = t_ns
            if c.get("finish_reason"):
                res.finish_reason = c["finish_reason"]

    def finish(self) -> None:
        res = self.res
        if self._ids_seen:
            res.token_ids, res.ids_source = list(self._ids), self._ids_via
        elif self._sglext_ids is not None:
            res.token_ids, res.ids_source = self._sglext_ids, "sglext_output_ids"


def _ids_in_choice(c: dict) -> tuple[list[int] | None, str | None]:
    """Generated token ids carried by one streamed choice, and where they were found."""
    tid = c.get("token_ids")
    if isinstance(tid, list):
        return [int(x) for x in tid], "choice_token_ids"
    lp = c.get("logprobs")
    if isinstance(lp, dict):
        content = lp.get("content")
        if isinstance(content, list):  # chat logprobs
            out = []
            for e in content:
                if not isinstance(e, dict):
                    return None, None
                if isinstance(e.get("id"), int):
                    out.append(e["id"])
                elif isinstance(e.get("token"), str) and _TOKEN_ID.match(e["token"]):
                    out.append(int(_TOKEN_ID.match(e["token"]).group(1)))
                else:
                    return None, None
            return out, "logprobs_ids"
        tokens = lp.get("tokens")
        if isinstance(tokens, list) and all(isinstance(t, str) and _TOKEN_ID.match(t) for t in tokens):
            return [int(_TOKEN_ID.match(t).group(1)) for t in tokens], "logprobs_ids"
    return None, None


# --------------------------------------------------------------------------- HTTP

def _request_bytes(ep: Endpoint, path: str, body: bytes, method: str = "POST") -> bytes:
    host = ep.host if ":" not in ep.host else f"[{ep.host}]"
    lines = [f"{method} {path} HTTP/1.1", f"Host: {host}:{ep.port}", "User-Agent: inferencekit-bench-serve",
             "Accept: text/event-stream", "Connection: close"]
    if method == "POST":
        lines += ["Content-Type: application/json", f"Content-Length: {len(body)}"]
    if ep.api_key:
        lines.append(f"Authorization: Bearer {ep.api_key}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


async def _open(ep: Endpoint):
    ctx = ssl.create_default_context() if ep.scheme == "https" else None
    return await asyncio.open_connection(ep.host, ep.port, ssl=ctx, limit=1 << 20)


async def _read_head(reader) -> tuple[int, dict]:
    status_line = await reader.readline()
    if not status_line:
        raise ConnectionError("server closed the connection before responding")
    parts = status_line.decode("latin-1").split(" ", 2)
    if len(parts) < 2 or not parts[0].startswith("HTTP/"):
        raise ConnectionError(f"not an HTTP response: {status_line[:80]!r}")
    status = int(parts[1])
    headers = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("latin-1").partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers


async def _body_chunks(reader, headers: dict):
    """Yield (bytes, perf_counter_ns at arrival) for a chunked, sized or close-delimited body."""
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size_line = await reader.readline()
            if not size_line:
                return
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                return
            data = await reader.readexactly(size + 2)
            yield data[:-2], time.perf_counter_ns()
    elif "content-length" in headers:
        left = int(headers["content-length"])
        while left > 0:
            data = await reader.read(min(left, 1 << 16))
            if not data:
                return
            left -= len(data)
            yield data, time.perf_counter_ns()
    else:
        while True:
            data = await reader.read(1 << 16)
            if not data:
                return
            yield data, time.perf_counter_ns()


async def _close(writer) -> None:
    writer.close()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(writer.wait_closed(), 1.0)


async def stream(ep: Endpoint, api: str, body: dict, *, rid: str, bucket: str, timeout_s: float = 600.0,
                 t_scheduled_ns: int | None = None) -> RequestResult:
    """Send one streamed request and time it. Never raises: failures are in ``result.error``."""
    res = RequestResult(id=rid, bucket=bucket, t_scheduled_ns=t_scheduled_ns)
    payload = _request_bytes(ep, ep.path(api), json.dumps(body, ensure_ascii=False).encode("utf-8"))
    st = _Stream(res)
    writer = None

    async def run():
        nonlocal writer
        reader, writer = await _open(ep)
        res.t_connected_ns = time.perf_counter_ns()
        writer.write(payload)
        await writer.drain()
        res.status, headers = await _read_head(reader)
        res.t_headers_ns = time.perf_counter_ns()
        if res.status != 200:
            raw = b""
            async for data, _ in _body_chunks(reader, headers):
                raw += data
                if len(raw) > MAX_ERROR_BODY:
                    break
            res.error = f"HTTP {res.status}: {raw[:MAX_ERROR_BODY].decode('utf-8', 'replace')}"
            return
        async for data, t in _body_chunks(reader, headers):
            st.feed(data, t)
            if st.done:
                break
        st.close(time.perf_counter_ns())

    res.t_send_ns = time.perf_counter_ns()
    try:
        await asyncio.wait_for(run(), timeout_s)
    except asyncio.TimeoutError:
        res.error = f"timeout after {timeout_s:g} s"
    except (OSError, ConnectionError, asyncio.IncompleteReadError, ValueError, UnicodeError) as e:
        res.error = f"{type(e).__name__}: {e}"[:500]
    finally:
        res.t_end_ns = time.perf_counter_ns()
        if writer is not None:
            await _close(writer)
    st.finish()
    if res.error is None and not st.done and res.finish_reason is None:
        # The connection closed before the server said the generation was over: a dropped stream.
        res.error = "stream ended early (no finish_reason and no [DONE])"
    if res.t_done_ns is None and res.error is None:
        res.t_done_ns = res.t_end_ns
    if res.error is None and res.t_first_ns is None:
        res.error = "stream ended without any generated token"
    res.ok = res.error is None
    res.error = scrub(res.error, ep)
    return res


async def get_json(ep: Endpoint, path: str, timeout_s: float = 10.0) -> tuple[int, object]:
    """GET a small JSON document (e.g. /v1/models). Returns (status, parsed body or text)."""

    async def run():
        reader, writer = await _open(ep)
        try:
            writer.write(_request_bytes(ep, ep.prefix + path, b"", method="GET"))
            await writer.drain()
            status, headers = await _read_head(reader)
            raw = b""
            async for data, _ in _body_chunks(reader, headers):
                raw += data
            try:
                return status, json.loads(raw)
            except ValueError:
                return status, raw[:MAX_ERROR_BODY].decode("utf-8", "replace")
        finally:
            await _close(writer)

    return await asyncio.wait_for(run(), timeout_s)


# --------------------------------------------------------------------------- token counting

def count_tokens(res: RequestResult, tokenizer=None) -> None:
    """Fill n_tokens and token_count_method, in the contract's order of preference:
    streamed token ids > usage.completion_tokens > re-tokenizing the text > counting SSE events.

    When the server sends both ids and usage and they disagree, the smaller count is used
    (method ``min(stream_token_ids:...,usage)``) and both are kept (n_ids, n_usage, count_mismatch):
    a server that streams padded or duplicated ids must not be able to raise its own decode speed.

    ``tokenizer`` is a callable text -> list of ids (an HF tokenizer's ``encode`` without special
    tokens). The last resort, counting token-bearing events, is only right when a server sends exactly
    one token per event; the method name says so.
    """
    usage_n = None
    if isinstance(res.usage, dict) and isinstance(res.usage.get("completion_tokens"), int):
        usage_n = res.usage["completion_tokens"]
    res.n_usage = usage_n
    res.n_ids = len(res.token_ids) if res.token_ids is not None else None
    if res.token_ids is not None and (res.token_ids or usage_n in (None, 0)):
        res.n_tokens, res.token_count_method = len(res.token_ids), f"stream_token_ids:{res.ids_source}"
        if usage_n is not None and usage_n != len(res.token_ids):
            res.count_mismatch = f"{len(res.token_ids)} streamed ids vs usage.completion_tokens={usage_n}"
            res.n_tokens = min(len(res.token_ids), usage_n)
            res.token_count_method = f"min(stream_token_ids:{res.ids_source},usage)"
    elif usage_n is not None:
        res.n_tokens, res.token_count_method = usage_n, "usage"
    elif tokenizer is not None and res.text + res.reasoning:
        res.n_tokens, res.token_count_method = len(tokenizer(res.reasoning + res.text)), "retokenize"
    elif res.events:
        res.n_tokens, res.token_count_method = len(res.events), "sse_events"


def api_key_from_env(name: str | None) -> str | None:
    """The API key is read from an environment variable and never stored or printed."""
    if not name:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"environment variable {name} is not set")
    return value
