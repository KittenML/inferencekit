"""Client-side timing for OpenAI-compatible speech recognition over HTTP."""
from __future__ import annotations

import http.client
import io
import json
import math
import socket
import threading
import time
import wave
from dataclasses import dataclass, field
from uuid import uuid4

from inferencekit import evidence as ev
from inferencekit.serve.audio_client import _connect_bounded
from inferencekit.serve.client import Endpoint

PATH = "/v1/audio/transcriptions"
MAX_AUDIO_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
PIN_FIELDS = ("model", "model_revision", "language", "dtype", "quant", "engine_version")
DECODING_FIELDS = ("temperature", "beam_size", "vad_filter", "chunking")


def wav_info(audio: bytes) -> tuple[int, int, int]:
    """Validate an uncompressed PCM WAV and return sample rate, channels, frames."""
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("WAV exceeds size limit")
    try:
        with wave.open(io.BytesIO(audio), "rb") as stream:
            rate, channels, frames = stream.getframerate(), stream.getnchannels(), stream.getnframes()
            width = stream.getsampwidth()
            if not 8000 <= rate <= 192000 or not 1 <= channels <= 8 or not frames or width not in (1, 2, 3, 4):
                raise ValueError("expected a nonempty PCM WAV at 8–192 kHz with 1–8 channels")
            if len(stream.readframes(frames)) != frames * channels * width:
                raise ValueError("truncated WAV PCM")
    except (EOFError, wave.Error) as exc:
        raise ValueError("invalid PCM WAV") from exc
    return rate, channels, frames


@dataclass
class Transcription:
    text: str = field(default="", repr=False)
    effective: dict = field(default_factory=dict)
    first_partial_ms: float | None = None
    complete_ms: float | None = None
    elapsed_ms: float | None = None
    error: str | None = None
    status: int | None = None


def _multipart(audio: bytes, model: str, language: str, settings: dict, stream: bool) -> tuple[bytes, str]:
    boundary = uuid4().hex
    prefix = ("--" + boundary + "\r\n").encode()
    fields = {"model": model, "language": language, "temperature": settings["temperature"],
              "response_format": "json"}
    if stream:
        fields["stream"] = "true"
    body = [
        prefix + f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    body += [prefix, b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n',
             b"Content-Type: audio/wav\r\n\r\n", audio, ("\r\n--" + boundary + "--\r\n").encode()]
    return b"".join(body), "multipart/form-data; boundary=" + boundary


def _object(raw: bytes) -> dict:
    try:
        doc = json.loads(raw)
    except (ValueError, RecursionError):
        raise ValueError("malformed transcription JSON") from None
    if not isinstance(doc, dict):
        raise ValueError("transcription event must be an object")
    return doc


def _read_stream(response, result: Transcription, start: int) -> tuple[dict | None, str | None]:
    lines, event_name, final, total, partials, error = [], None, None, 0, [], None
    while line := response.readline(MAX_RESPONSE_BYTES + 1):
        arrived = time.perf_counter_ns()
        total += len(line)
        if total > MAX_RESPONSE_BYTES:
            raise ValueError("transcription response exceeds size limit")
        line = line.rstrip(b"\r\n")
        if line.startswith(b"data:"):
            lines.append(line[5:].removeprefix(b" "))
        elif line.startswith(b"event:"):
            event_name = line[6:].strip()
        elif not line:
            payload = b"\n".join(lines)
            name = event_name
            lines, event_name = [], None
            if not payload or payload == b"[DONE]":
                continue
            if final is not None:
                error = error or "transcription event follows terminal event"
                continue
            try:
                doc = _object(payload)
            except ValueError as exc:
                error = error or str(exc)
                continue
            kind = doc.get("type") or (name or b"").decode("ascii", errors="replace")
            if kind == "transcript.text.done":
                final = doc
            elif kind == "transcript.text.delta" and isinstance(doc.get("delta"), str):
                delta = doc["delta"]
                if delta:
                    if result.first_partial_ms is None:
                        result.first_partial_ms = (arrived - start) / 1e6
                    partials.append(delta)
            else:
                error = error or "unexpected transcription event"
    result.text = "".join(partials)
    if lines or final is None:
        error = error or "missing or truncated terminal transcription event"
    return final, error


def transcribe(endpoint: Endpoint, audio: bytes, *, model: str, language: str, settings: dict,
               stream: bool = False, timeout_s: float = 300, request_id: str | None = None) -> Transcription:
    payload, content_type = _multipart(audio, model, language, settings, stream)
    cls = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
    conn = cls(endpoint.host, endpoint.port, timeout=timeout_s)
    headers = {"Content-Type": content_type, "Accept": "text/event-stream" if stream else "application/json",
               "Connection": "close", "X-Request-ID": request_id or uuid4().hex}
    if endpoint.api_key:
        headers["Authorization"] = "Bearer " + endpoint.api_key
    result = Transcription()
    expired = threading.Event()
    response = None
    deadline = time.monotonic() + timeout_s
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

    watchdog = threading.Timer(timeout_s, expire)
    watchdog.daemon = True
    watchdog.start()
    start = time.perf_counter_ns()
    try:
        conn.request("POST", endpoint.prefix + PATH, body=payload, headers=headers)
        response = conn.getresponse()
        result.status = response.status
        media_type = (response.getheader("Content-Type") or "").partition(";")[0].strip().lower()
        expected_type = "text/event-stream" if stream else "application/json"
        if response.status != 200 or media_type != expected_type:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) <= MAX_RESPONSE_BYTES and response.length in (0, None):
                result.complete_ms = (time.perf_counter_ns() - start) / 1e6
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            raise ValueError(f"expected {expected_type} transcription response")
        doc, stream_error = _read_stream(response, result, start) if stream else (None, None)
        raw = b"" if stream else response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("transcription response exceeds size limit")
        if response.length not in (0, None):
            raise ValueError("truncated HTTP body")
        result.complete_ms = (time.perf_counter_ns() - start) / 1e6
        if stream_error:
            raise ValueError(stream_error)
        if stream and result.first_partial_ms is None:
            raise ValueError("stream sent no nonempty partial transcript")
        doc = doc if stream else _object(raw)
        if not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise ValueError("empty or missing transcript")
        result.text = doc["text"]
        reported = doc.get("effective")
        if isinstance(reported, dict):
            keys = {*PIN_FIELDS, *DECODING_FIELDS, "cache", "audio_duration_s", "detected_language", "language_confidence"}
            for key in keys & reported.keys():
                value = reported[key]
                if type(value) in (str, bool, int, float) and (not isinstance(value, float) or math.isfinite(value)):
                    if not endpoint.api_key or value != endpoint.api_key:
                        result.effective[key] = ev.redact(value)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # Values from the wire can contain private input, so retain only parser-defined errors.
        result.error = str(exc) if type(exc) is ValueError else type(exc).__name__
    finally:
        try:
            result.text.encode("utf-8")
        except UnicodeEncodeError:
            result.text = ""
            result.error = "transcript contains invalid Unicode"
        if result.text and (endpoint.api_key and endpoint.api_key in result.text or ev.redact(result.text) != result.text):
            result.text = ""
            result.error = "transcript contains redacted values"
        result.elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        watchdog.cancel()
        if expired.is_set():
            result.error = "transcription request deadline exceeded"
            result.complete_ms = None
        conn.close()
    return result


def effective_errors(reported: dict, expected: dict, duration: float) -> list[str]:
    errors = [f"effective {key} missing or mismatched" for key, value in expected.items()
              if reported.get(key) != value or isinstance(value, bool) != isinstance(reported.get(key), bool)]
    if reported.get("cache") is not False:
        errors.append("effective cache must be false")
    actual = reported.get("audio_duration_s")
    if type(actual) not in (int, float) or not math.isclose(actual, duration, rel_tol=0, abs_tol=0.001):
        errors.append("effective audio_duration_s missing or mismatched")
    return errors
