"""Client-timed PCM speech over HTTP NDJSON, using only the standard library.

POST /v1/audio/speech accepts AudioRequest.body(). The response is UTF-8 NDJSON:
metadata {type, sample_rate, channels, encoding: "pcm_s16le"}, then one or more
audio {type, pcm_b64}, then final {type, sample_count, effective}. sample_count
counts PCM frames PER CHANNEL, not interleaved int16 values. Metadata must match
the request. effective must report model, seed and every requested setting.

First audio means decoded PCM bytes, including silence; signal onset is reported
separately. Neither signal detection nor these structural checks proves speech
intelligibility. The complete timer ends only after consuming the entire HTTP
body, including final and EOF. Engine-provided durations are never trusted.
"""

from __future__ import annotations

import base64
import binascii
import http.client
import ipaddress
import json
import math
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

from inferencekit import evidence as ev
from inferencekit.canonical import canonical_json
from inferencekit.serve.client import Endpoint, scrub

AUDIO_PATH = "/v1/audio/speech"
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_PCM_BYTES = 256 * 1024 * 1024
ENCODING = "pcm_s16le"


def _connect_bounded(address, timeout, source_address=None):
    """Bound DNS plus TCP connect; an abandoned resolver cannot open a socket."""
    host, port = address
    deadline = time.monotonic() + timeout
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        addresses = [(family, socket.SOCK_STREAM, 0, "", (host, port))]
    else:
        resolved = queue.Queue(maxsize=1)

        def resolve():
            try:
                resolved.put((socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM), None))
            except OSError as exc:
                resolved.put((None, exc))

        threading.Thread(target=resolve, daemon=True).start()
        try:
            addresses, error = resolved.get(timeout=max(0.001, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError("audio DNS deadline exceeded") from None
        if error:
            raise error
    error = OSError("audio endpoint resolved no addresses")
    for family, kind, protocol, _, destination in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("audio connection deadline exceeded")
        sock = socket.socket(family, kind, protocol)
        try:
            sock.settimeout(remaining)
            if source_address:
                sock.bind(source_address)
            sock.connect(destination)
            return sock
        except OSError as exc:
            sock.close()
            error = exc
    raise error


@dataclass(frozen=True)
class AudioRequest:
    id: str
    text: str
    model: str
    seed: int = 0
    sample_rate: int = 44100
    channels: int = 1
    settings: dict = field(default_factory=dict)
    reference_audio: bytes | None = field(default=None, repr=False)
    reference_text: str | None = None

    def body(self) -> dict:
        body = {"input": self.text, "model": self.model, "seed": self.seed,
                "sample_rate": self.sample_rate, "channels": self.channels,
                "encoding": ENCODING, "stream": True, "settings": self.settings}
        if self.reference_audio is not None:
            body["reference_audio_b64"] = base64.b64encode(self.reference_audio).decode("ascii")
            body["reference_text"] = self.reference_text
        return body


@dataclass
class AudioResult:
    id: str
    ok: bool = False
    error: str | None = None
    status: int | None = None
    t_send_ns: int = 0
    t_headers_ns: int | None = None
    t_first_ns: int | None = None
    t_signal_ns: int | None = None
    t_audible_ns: int | None = None
    t_done_ns: int | None = None
    t_end_ns: int = 0
    pcm: bytes = field(default=b"", repr=False)
    sample_rate: int | None = None
    channels: int | None = None
    sample_count: int = 0
    leading_silent_frames: int = 0
    effective: dict | None = None
    chunks: list[tuple[int, int]] = field(default_factory=list)  # arrival ns, PCM frames

    def milliseconds(self, timestamp: int | None) -> float | None:
        return (timestamp - self.t_send_ns) / 1e6 if self.ok and timestamp is not None else None

    @property
    def first_audio_ms(self):
        return self.milliseconds(self.t_first_ns)

    @property
    def first_signal_ms(self):
        return self.milliseconds(self.t_signal_ns)

    @property
    def first_audible_ms(self):
        return self.milliseconds(self.t_audible_ns)

    @property
    def complete_audio_ms(self):
        return self.milliseconds(self.t_done_ns)

    @property
    def audio_duration_s(self):
        return self.sample_count / self.sample_rate if self.ok and self.sample_rate else None


def playback_metrics(chunks: list[tuple[int, int]], sample_rate: int) -> dict:
    """Simulate playback starting immediately at the first chunk; no hidden prebuffer.

    Stall intervals count once per late arrival. Stream closure after the last
    sample cannot cause an underrun. This is a derived PCM buffer simulation,
    not a claim about an operating system's real audio device.
    """
    if not chunks:
        return {"underruns": None, "underrun_ms": None, "max_stream_gap_ms": None}
    play_end = chunks[0][0] / 1e9
    count, stalled = 0, 0.0
    for t_ns, frames in chunks:
        arrival = t_ns / 1e9
        gap = arrival - play_end
        if gap > 1e-9:
            count += 1
            stalled += gap
        play_end = max(play_end, arrival) + frames / sample_rate
    gaps = [(b[0] - a[0]) / 1e6 for a, b in zip(chunks, chunks[1:])]
    return {"underruns": count, "underrun_ms": stalled * 1000,
            "max_stream_gap_ms": max(gaps, default=0.0)}


class AudioHTTP:
    endpoint_kind = "audio_http"

    def __init__(self, ep: Endpoint, timeout_s: float = 300.0):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.ep, self.timeout_s = ep, timeout_s

    def generate(self, request: AudioRequest) -> AudioResult:
        res = AudioResult(request.id)
        if (type(request.sample_rate) is not int or not 8000 <= request.sample_rate <= 192000
                or type(request.channels) is not int or not 1 <= request.channels <= 8):
            res.error = "request PCM rate must be 8000..192000 Hz and channels 1..8"
            return res
        cls = http.client.HTTPSConnection if self.ep.scheme == "https" else http.client.HTTPConnection
        conn = cls(self.ep.host, self.ep.port, timeout=self.timeout_s)
        headers = {"Content-Type": "application/json", "Accept": "application/x-ndjson", "Connection": "close"}
        if self.ep.api_key:
            headers["Authorization"] = "Bearer " + self.ep.api_key
        payload = json.dumps(request.body(), allow_nan=False).encode()
        pcm = bytearray()
        metadata, ended = False, False
        playback_end_ns = 0
        res.t_send_ns = time.perf_counter_ns()
        deadline = time.monotonic() + self.timeout_s
        response = None
        expired = threading.Event()
        conn._create_connection = lambda address, timeout, source_address=None: _connect_bounded(
            address, max(0.001, deadline - time.monotonic()), source_address)

        def expire():
            expired.set()
            active = conn.sock or getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            if active is not None:
                try:
                    active.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                active.close()

        # A receive timeout resets on each packet. This watchdog also interrupts
        # a peer dripping header bytes or one never-terminated NDJSON line.
        watchdog = threading.Timer(self.timeout_s, expire)
        watchdog.daemon = True
        watchdog.start()
        try:
            conn.request("POST", self.ep.prefix + AUDIO_PATH, body=payload, headers=headers)
            if conn.sock is not None:
                conn.sock.settimeout(max(0.001, deadline - time.monotonic()))
            response = conn.getresponse()
            res.status = response.status
            res.t_headers_ns = time.perf_counter_ns()
            if response.status != 200:
                # Do not echo bodies: proxies and engines can echo credentials or input.
                raise ValueError(f"HTTP {response.status}")
            ctype = (response.getheader("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in ("application/x-ndjson", "application/ndjson"):
                raise ValueError("response must be application/x-ndjson")
            while True:
                if expired.is_set() or time.monotonic() >= deadline:
                    raise ValueError("response exceeded request deadline")
                active = conn.sock or getattr(getattr(response.fp, "raw", None), "_sock", None)
                if active is not None:
                    active.settimeout(max(0.001, deadline - time.monotonic()))
                line = response.readline(MAX_LINE_BYTES + 1)
                if expired.is_set() or time.monotonic() >= deadline:
                    raise ValueError("response exceeded request deadline")
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("NDJSON event exceeds size limit")
                if not line.endswith(b"\n"):
                    raise ValueError("truncated NDJSON event (missing newline)")
                if not line.strip():
                    continue
                if ended:
                    raise ValueError("event after final")
                doc = json.loads(line)
                if not isinstance(doc, dict):
                    raise ValueError("event must be a JSON object")
                kind = doc.get("type")
                if kind == "metadata":
                    if metadata:
                        raise ValueError("duplicate metadata")
                    for key, expected in (("sample_rate", request.sample_rate), ("channels", request.channels)):
                        if type(doc.get(key)) is not int or doc[key] != expected:
                            raise ValueError(f"metadata {key} mismatch")
                    if doc.get("encoding") != ENCODING:
                        raise ValueError("metadata encoding must be pcm_s16le")
                    res.sample_rate, res.channels, metadata = request.sample_rate, request.channels, True
                elif kind == "audio":
                    if not metadata:
                        raise ValueError("audio before metadata")
                    value = doc.get("pcm_b64")
                    if not isinstance(value, str):
                        raise ValueError("audio pcm_b64 must be a string")
                    data = base64.b64decode(value, validate=True)
                    t_pcm = time.perf_counter_ns()  # PCM is now playable; headers never count
                    if not data or len(data) % (2 * res.channels):
                        raise ValueError("empty or truncated PCM frame")
                    if len(pcm) + len(data) > MAX_PCM_BYTES:
                        raise ValueError("PCM response exceeds size limit")
                    frames = len(data) // (2 * res.channels)
                    res.t_first_ns = res.t_first_ns if res.t_first_ns is not None else t_pcm
                    if res.t_signal_ns is None:
                        first = next((i // res.channels for i, (v,) in enumerate(struct.iter_unpack("<h", data))
                                      if v != 0), None)
                        res.leading_silent_frames += frames if first is None else first
                        if first is not None:
                            res.t_signal_ns = t_pcm
                            res.t_audible_ns = max(t_pcm, playback_end_ns) + round(first / res.sample_rate * 1e9)
                    playback_end_ns = max(t_pcm, playback_end_ns) + round(frames / res.sample_rate * 1e9)
                    res.chunks.append((t_pcm, frames))
                    pcm.extend(data)
                elif kind == "final":
                    if not metadata:
                        raise ValueError("final before metadata")
                    if type(doc.get("sample_count")) is not int or doc["sample_count"] <= 0:
                        raise ValueError("final sample_count must be a positive integer")
                    res.sample_count = doc["sample_count"]
                    if res.sample_count * 2 * res.channels != len(pcm):
                        raise ValueError("final sample_count does not match PCM")
                    expected = {"model": request.model, "seed": request.seed, **request.settings}
                    reported = doc.get("effective")
                    # Retain only requested fields. Unsolicited server metadata may
                    # contain internal endpoints, credentials or launch commands.
                    res.effective = ev.redact({key: reported[key] for key in expected if key in reported}) \
                        if isinstance(reported, dict) else None
                    if not isinstance(res.effective, dict) or canonical_json(res.effective) != canonical_json(expected):
                        raise ValueError("final effective settings missing or mismatched")
                    ended = True
                else:
                    raise ValueError("unknown NDJSON event type")
            if response.length not in (None, 0):
                raise ValueError("truncated HTTP body")
            res.t_done_ns = time.perf_counter_ns()  # complete HTTP body, not merely final announcement
            if not ended:
                raise ValueError("missing final event")
            if res.t_signal_ns is None:
                raise ValueError("silent PCM output (all samples zero)")
            res.pcm, res.ok = bytes(pcm), True
        except (OSError, http.client.HTTPException, ValueError, binascii.Error) as exc:
            # Parser errors are deliberately static; exception text may contain wire data or keys.
            safe = "response exceeded request deadline" if expired.is_set() else \
                str(exc) if type(exc) is ValueError else type(exc).__name__
            if self.ep.api_key:
                safe = safe.replace(self.ep.api_key, "<redacted>")
            res.error = scrub(safe, self.ep)
        finally:
            watchdog.cancel()
            conn.close()
            res.t_end_ns = time.perf_counter_ns()
        return res

    def close(self):
        return None
