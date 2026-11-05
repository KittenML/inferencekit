"""Synthetic PCM HTTP only: no models, accelerator or external endpoint."""

import base64
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inferencekit.serve.audio_client import AudioHTTP, AudioRequest, playback_metrics
from inferencekit.serve.client import Endpoint


class FakeAudioServer:
    def __init__(self, mode="honest"):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(request)
                if mode == "drip_headers":
                    try:
                        self.wfile.write(b"HTTP/1.0 200 OK\r\nX-Drip: ")
                        for _ in range(100):
                            self.wfile.write(b"a")
                            self.wfile.flush()
                            time.sleep(0.01)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if mode == "http_error":
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"private-token secret.example:9000")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain" if mode == "wrong_type" else "application/x-ndjson")
                if mode == "truncated_http":
                    self.send_header("Content-Length", "1000000")
                self.end_headers()

                if mode == "drip_body":
                    try:
                        self.wfile.write(b'{"type":"audio","pcm_b64":"')
                        for _ in range(100):
                            self.wfile.write(b"a")
                            self.wfile.flush()
                            time.sleep(0.01)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return

                def send(doc):
                    self.wfile.write(json.dumps(doc).encode() + b"\n")
                    self.wfile.flush()

                meta = {"type": "metadata", "encoding": "pcm_s16le", "sample_rate": request["sample_rate"],
                        "channels": request["channels"]}
                if mode == "wrong_rate":
                    meta["sample_rate"] += 1
                if mode == "wrong_encoding":
                    meta["encoding"] = "float32"
                try:
                    if mode != "no_meta":
                        send(meta)
                    if mode == "duplicate_meta":
                        send(meta)
                    # Headers/metadata arrive first. Delayed PCM must define first_audio.
                    time.sleep(0.02)
                    silent = b"\x00\x00" * request["channels"] * 160
                    if mode == "leading_silence":
                        silent *= 150  # three seconds at the tests' declared 8 kHz rate
                    signal = struct.pack("<h", 2000) * request["channels"] * 8000
                    if mode == "silent":
                        signal = bytes(len(signal))
                    if mode == "odd_pcm":
                        signal = b"\x01"
                    send({"type": "audio", "pcm_b64": base64.b64encode(silent).decode()})
                    if mode == "gap":
                        time.sleep(0.06)
                    if mode != "empty":
                        send({"type": "audio", "pcm_b64": "!" if mode == "bad_base64" else
                              base64.b64encode(signal).decode()})
                    if mode == "missing_final":
                        return
                    effective = {"model": request["model"], "seed": request["seed"], **request["settings"]}
                    if mode.startswith("wrong_pin:"):
                        effective[mode.split(":", 1)[1]] = "wrong-loaded-value"
                    if mode == "extra_metadata":
                        effective["service_url"] = "https://user:private-token@private.example:9000"
                    if mode == "bool_setting":
                        effective["temperature"] = True
                    if mode == "wrong_effective":
                        effective["seed"] += 1
                    count = (len(silent) + (len(signal) if mode != "empty" else 0)) // 2 // request["channels"]
                    final = {"type": "final", "sample_count": count + (1 if mode == "wrong_count" else 0),
                             "effective": effective}
                    if mode == "partial_line":
                        self.wfile.write(json.dumps(final).encode())
                        return
                    send(final)
                    outer.final_at_ns = time.perf_counter_ns()
                    if mode == "after_final":
                        send({"type": "audio", "pcm_b64": base64.b64encode(signal).decode()})
                    if mode == "delayed_eof":
                        time.sleep(0.06)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.requests = []
        self.final_at_ns = None
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()


def test_first_pcm_signal_and_complete_are_client_timed():
    with FakeAudioServer("delayed_eof") as server:
        result = AudioHTTP(Endpoint.parse(server.url)).generate(AudioRequest("a", "Unseen text", "test", sample_rate=8000))
    assert result.ok, result.error
    assert result.sample_count == 8160
    assert len(result.pcm) == 16320
    assert result.t_first_ns > result.t_headers_ns
    assert result.first_audio_ms >= 15  # fake delay after headers
    assert result.first_audible_ms >= result.first_audio_ms + 19
    assert result.leading_silent_frames == 160
    assert result.t_done_ns - server.final_at_ns > 40_000_000
    assert result.complete_audio_ms > result.first_audio_ms


@pytest.mark.parametrize("mode", ["wrong_type", "wrong_rate", "wrong_encoding", "no_meta", "duplicate_meta",
                                  "silent", "empty", "odd_pcm", "missing_final", "partial_line", "wrong_count",
                                  "wrong_effective", "after_final", "bad_base64"])
def test_invalid_streams_never_produce_speed_numbers(mode):
    with FakeAudioServer(mode) as server:
        result = AudioHTTP(Endpoint.parse(server.url)).generate(AudioRequest("a", "test", "test", sample_rate=8000))
    assert not result.ok
    assert result.error
    assert result.first_audio_ms is None
    assert result.audio_duration_s is None
    assert result.pcm == b""


def test_http_failure_does_not_echo_auth_or_response_body():
    with FakeAudioServer("http_error") as server:
        result = AudioHTTP(Endpoint.parse(server.url, "private-token")).generate(AudioRequest("a", "test", "test"))
    assert result.error == "HTTP 500"
    assert "private-token" not in repr(result)
    assert "secret.example" not in repr(result)


def test_stereo_sample_count_is_frames_per_channel():
    with FakeAudioServer() as server:
        result = AudioHTTP(Endpoint.parse(server.url)).generate(AudioRequest("a", "test", "test", sample_rate=8000, channels=2))
    assert result.ok
    assert result.sample_count == 8160
    assert result.audio_duration_s == 1.02
    assert len(result.pcm) == 8160 * 2 * 2


def test_playback_buffer_uses_cumulative_audio_and_detects_stalls():
    result = playback_metrics([(0, 100), (50_000_000, 100), (250_000_000, 100)], 1000)
    assert result["underruns"] == 1
    assert result["underrun_ms"] == pytest.approx(50)
    assert result["max_stream_gap_ms"] == 200
    assert playback_metrics([], 1000)["underruns"] is None


def test_request_declared_pcm_bounds():
    result = AudioHTTP(Endpoint.parse("http://127.0.0.1:1")).generate(AudioRequest("a", "test", "test", channels=0))
    assert not result.ok
    assert result.status is None


def test_truncated_http_body_cannot_pass_even_with_valid_final():
    with FakeAudioServer("truncated_http") as server:
        result = AudioHTTP(Endpoint.parse(server.url)).generate(AudioRequest("a", "test", "test", sample_rate=8000))
    assert not result.ok
    assert result.error == "truncated HTTP body"


def test_timeout_and_size_limits_fail_closed(monkeypatch):
    from inferencekit.serve import audio_client

    with FakeAudioServer() as server:
        result = AudioHTTP(Endpoint.parse(server.url), timeout_s=0.005).generate(AudioRequest("a", "test", "test"))
    assert not result.ok
    assert "Timeout" in result.error or "deadline" in result.error
    monkeypatch.setattr(audio_client, "MAX_PCM_BYTES", 100)
    with FakeAudioServer() as server:
        result = AudioHTTP(Endpoint.parse(server.url)).generate(AudioRequest("a", "test", "test"))
    assert not result.ok
    assert "size limit" in result.error


def test_ndjson_line_limit(monkeypatch):
    from inferencekit.serve import audio_client

    monkeypatch.setattr(audio_client, "MAX_LINE_BYTES", 200)
    with FakeAudioServer() as server:
        result = AudioHTTP(Endpoint.parse(server.url)).generate(AudioRequest("a", "test", "test"))
    assert not result.ok
    assert "size limit" in result.error


def test_unsolicited_effective_metadata_is_not_retained():
    with FakeAudioServer("extra_metadata") as server:
        result = AudioHTTP(Endpoint.parse(server.url)).generate(AudioRequest("a", "test", "test"))
    assert result.ok
    assert result.effective.keys() == {"model", "seed"}
    assert "private-token" not in repr(result)
    assert "private.example" not in repr(result)


def test_boolean_cannot_impersonate_a_numeric_setting():
    with FakeAudioServer("bool_setting") as server:
        result = AudioHTTP(Endpoint.parse(server.url)).generate(AudioRequest("a", "test", "test", settings={"temperature": 1.0}))
    assert not result.ok
    assert result.error == "final effective settings missing or mismatched"


@pytest.mark.parametrize("mode", ["drip_headers", "drip_body"])
def test_whole_response_deadline_interrupts_slow_drips(mode):
    with FakeAudioServer(mode) as server:
        start = time.monotonic()
        result = AudioHTTP(Endpoint.parse(server.url), timeout_s=0.06).generate(AudioRequest("a", "test", "test"))
        elapsed = time.monotonic() - start
    assert not result.ok
    assert elapsed < 0.3
    assert "deadline" in result.error or "Timeout" in result.error
    assert result.first_audio_ms is None


def test_dns_deadline_prevents_a_late_resolver_from_sending(monkeypatch):
    import socket

    original = socket.getaddrinfo
    resolved = threading.Event()

    def slow_resolution(host, port, *args):
        time.sleep(0.3)
        answer = original("127.0.0.1", port, *args)
        resolved.set()
        return answer

    with FakeAudioServer() as server:
        monkeypatch.setattr(socket, "getaddrinfo", slow_resolution)
        endpoint = Endpoint.parse(f"http://speech.invalid:{server.httpd.server_port}")
        start = time.monotonic()
        result = AudioHTTP(endpoint, timeout_s=0.04).generate(AudioRequest("a", "test", "test"))
        elapsed = time.monotonic() - start
        assert resolved.wait(1)
        assert not server.requests
    assert not result.ok
    assert elapsed < 0.25
    assert "deadline" in result.error or "Timeout" in result.error
