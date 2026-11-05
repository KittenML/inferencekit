"""CPU-only image protocol tests, including malformed outputs and bounded transport."""
import base64
import hashlib
import json
import struct
import threading
import time
import zlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inferencekit.serve import image_client as ic
from inferencekit.serve.client import Endpoint


def png(width=4, height=3, seed=0, metadata=None, filt=0):
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
    color = hashlib.sha256(str(seed).encode()).digest()[:3]
    raw, previous = bytearray(), bytes(width * 3)
    for _ in range(height):
        row = color * width
        encoded = row if filt == 0 else bytes((x - previous[i]) & 255 for i, x in enumerate(row))
        raw.extend(bytes([filt]) + encoded)
        previous = row
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + (chunk(b"tEXt", metadata.encode()) if metadata else b"")
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


@contextmanager
def image_server(mode="honest", delay=0.0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.requests.append((self.path, body, self.headers.get("Authorization")))
            effective = {k: v for k, v in body.items() if k not in ("prompt", "size", "steps")}
            effective["steps_run"] = body["steps"]
            image = png(body["width"], body["height"], body["seed"] if mode != "repeat" else 0)
            if mode == "wrong_size":
                image = png(body["width"] + 1, body["height"], body["seed"])
            if mode == "bad_crc":
                image = image[:-1] + bytes([image[-1] ^ 1])
            if mode == "ignore_steps":
                effective["steps_run"] -= 1
            if mode == "ignore_seed":
                effective["seed"] = -1
            if mode == "missing_effective":
                effective = None
            doc = {"image_b64": base64.b64encode(image).decode(), "format": "png",
                   "effective": effective, "cache": ["text_encode"] if mode == "cached" else []}
            if mode == "fixed_negative_cached":
                doc["cache"] = ["fixed_negative_embedding"]
            elif mode == "positive_and_negative_cached":
                doc["cache"] = ["fixed_negative_embedding", "positive_embedding"]
            if mode == "missing_cache":
                doc.pop("cache")
            if mode == "bad_base64":
                doc["image_b64"] = "%%%"
            if mode == "nonfinite":
                doc["effective"]["guidance_scale"] = float("nan")
            raw = json.dumps(doc).encode()
            time.sleep(delay)
            if mode == "slow_headers":
                try:
                    for value in b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n":
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(500 if mode == "http_error" else 200)
            self.send_header("Content-Type", "application/json")
            if mode == "oversize":
                self.send_header("Content-Length", str(ic.MAX_RESPONSE_BYTES + 1))
            elif mode == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                if mode == "chunked":
                    for part in (raw[:15], raw[15:]):
                        self.wfile.write(f"{len(part):X}\r\n".encode() + part + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                elif mode == "truncated":
                    self.wfile.write(raw[:10])
                elif mode == "slow_body":
                    for value in raw:
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(0.01)
                else:
                    self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", httpd.requests
    finally:
        httpd.shutdown()
        httpd.server_close()


def request(**changes):
    return ic.ImageRequest(**{"id": "one", "model": "model@pin", "prompt": "a fox", "width": 4, "height": 3,
                              "steps": 5, "guidance_scale": 3.0, "seed": 4, **changes})


def test_complete_response_and_timer():
    with image_server(delay=0.025) as (url, requests):
        result = ic.ImageHTTP(Endpoint.parse(url, "private-test-key")).generate(request(settings={"scheduler": "euler"}))
    assert result.ok and result.e2e_ms >= 20
    assert result.width == 4 and result.height == 3 and result.pixel_sha256
    assert result.effective_mismatch == [] and result.cache == []
    assert requests[0][0] == "/v1/images/generations"
    assert requests[0][1]["size"] == "4x3" and requests[0][1]["scheduler"] == "euler"
    assert requests[0][2] == "Bearer private-test-key"


def test_timer_includes_image_decoding(monkeypatch):
    validate = ic.validate_png
    def slow_decode(data):
        time.sleep(0.035)
        return validate(data)
    monkeypatch.setattr(ic, "validate_png", slow_decode)
    with image_server() as (url, _):
        result = ic.ImageHTTP(Endpoint.parse(url)).generate(request())
    assert result.ok and result.e2e_ms >= 30
    assert result.t_done_ns - result.t_body_ns >= 30_000_000


@pytest.mark.parametrize("mode", ["wrong_size", "bad_crc", "bad_base64", "missing_cache", "oversize", "truncated", "http_error", "nonfinite"])
def test_malformed_outputs_fail(mode):
    with image_server(mode) as (url, _):
        result = ic.ImageHTTP(Endpoint.parse(url)).generate(request())
    assert not result.ok and result.error and result.image is None and result.e2e_ms is None


@pytest.mark.parametrize("mode,key", [("ignore_steps", "steps_run"), ("ignore_seed", "seed"),
                                      ("missing_effective", "effective settings missing")])
def test_effective_settings_are_checked(mode, key):
    with image_server(mode) as (url, _):
        result = ic.ImageHTTP(Endpoint.parse(url)).generate(request())
    assert result.ok and key in result.effective_mismatch


def test_cache_is_reported_and_chunked_transport_supported():
    with image_server("cached") as (url, _):
        assert ic.ImageHTTP(Endpoint.parse(url)).generate(request()).cache == ["text_encode"]
    with image_server("chunked") as (url, _):
        assert ic.ImageHTTP(Endpoint.parse(url)).generate(request()).ok


def test_pixel_hash_ignores_metadata_and_filter_encoding():
    plain = ic.validate_png(png(seed=1))
    metadata = ic.validate_png(png(seed=1, metadata="hello"))
    up_filtered = ic.validate_png(png(seed=1, filt=2))
    assert plain["pixel_sha256"] == metadata["pixel_sha256"] == up_filtered["pixel_sha256"]
    assert plain["pixel_sha256"] != ic.validate_png(png(seed=2))["pixel_sha256"]


@pytest.mark.parametrize("data", [b"", png()[:-12], png() + b"junk", b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 2**31)])
def test_png_truncation_and_bounds(data):
    with pytest.raises(ValueError):
        ic.validate_png(data)


def test_response_bound_without_large_allocation():
    with image_server() as (url, _):
        result = ic.ImageHTTP(Endpoint.parse(url), max_response_bytes=20).generate(request())
    assert not result.ok and "bound" in result.error


def test_whole_body_deadline_and_redaction():
    with image_server("slow_body") as (url, _):
        started = time.monotonic()
        result = ic.ImageHTTP(Endpoint.parse(url), timeout_s=0.05).generate(request())
        assert time.monotonic() - started < 0.5
    assert not result.ok and result.error
    ep = Endpoint.parse("http://127.0.0.1:1", "secret-access-value")
    result = ic.ImageHTTP(ep, timeout_s=0.1).generate(request())
    assert not result.ok and "127.0.0.1" not in result.error and "secret-access-value" not in result.error


def test_trickled_headers_obey_absolute_deadline():
    with image_server("slow_headers") as (url, _):
        started = time.monotonic()
        result = ic.ImageHTTP(Endpoint.parse(url), timeout_s=0.05).generate(request())
        assert time.monotonic() - started < 0.5
    assert not result.ok


def test_effective_nested_boolean_is_not_integer():
    req = request(settings={"precision": {"flag": True}})
    effective = {k: v for k, v in req.body().items() if k not in ("prompt", "size", "steps")}
    effective["steps_run"] = req.steps
    effective["precision"] = {"flag": 1}
    assert ic.effective_mismatches(req, effective) == ["precision"]


def test_dns_is_bounded_and_does_not_submit_a_late_request(monkeypatch):
    def slow_dns(*_):
        time.sleep(0.4)
        return []
    monkeypatch.setattr(ic.socket, "getaddrinfo", slow_dns)
    start = time.monotonic()
    result = ic.ImageHTTP(Endpoint.parse("http://example.invalid"), timeout_s=0.03).generate(request())
    assert time.monotonic() - start < 0.2
    assert not result.ok and "DNS deadline" in result.error


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 86401])
def test_timeout_must_be_finite(timeout):
    with pytest.raises(ValueError):
        ic.ImageHTTP(Endpoint.parse("http://localhost"), timeout_s=timeout)


def test_request_fields_cannot_be_overridden():
    with pytest.raises(ValueError):
        request(settings={"seed": 1}).body()
