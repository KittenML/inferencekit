"""Bounded client for POST /v1/images/generations, timed through decoded image validation.

Requests carry model, prompt, size (WxH), width, height, steps, guidance_scale, seed,
and any explicitly requested settings. Responses carry image_b64, format="png",
effective (model, width, height, steps_run, guidance_scale, seed and all settings),
and cache=[] (names of any reused stages). Effective values are self-reports;
the PNG's dimensions, CRCs and decompressed scanlines are checked independently.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import ipaddress
import json
import math
import re
import queue
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field

from inferencekit import evidence as ev
from inferencekit.serve.client import Endpoint, scrub

PATH = "/v1/images/generations"
DEFAULT_TIMEOUT_S = 1800.0
MAX_RESPONSE_BYTES = 128 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_PIXELS = 16_777_216
MAX_RAW_BYTES = 256 * 1024 * 1024
_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _connect(address, timeout, source_address=None):
    """socket.create_connection with a bounded DNS lookup as well as connect.

    The daemon resolver can finish after a timeout, but cannot open a socket or
    submit a request. Literal IPs (the normal local benchmark case) need no DNS.
    """
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
            raise TimeoutError("image DNS deadline exceeded") from None
        if error:
            raise error
    error = OSError("image endpoint resolved no addresses")
    for family, kind, protocol, _, destination in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("image connection deadline exceeded")
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
class ImageRequest:
    id: str
    model: str
    prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    settings: dict = field(default_factory=dict)

    def body(self) -> dict:
        body = {"model": self.model, "prompt": self.prompt, "size": f"{self.width}x{self.height}",
                "width": self.width, "height": self.height, "steps": self.steps,
                "guidance_scale": self.guidance_scale, "seed": self.seed}
        if set(self.settings) & set(body):
            raise ValueError("image settings cannot override required request fields")
        return {**body, **self.settings}


@dataclass
class ImageResult:
    id: str
    ok: bool = False
    error: str | None = None
    status: int | None = None
    t_send_ns: int = 0
    t_body_ns: int | None = None
    t_done_ns: int | None = None
    image: bytes | None = field(default=None, repr=False)
    effective: dict | None = None
    effective_mismatch: list[str] = field(default_factory=list)
    cache: list[str] | None = None
    width: int | None = None
    height: int | None = None
    pixel_sha256: str | None = None

    @property
    def e2e_ms(self) -> float | None:
        return (self.t_done_ns - self.t_send_ns) / 1e6 if self.ok and self.t_done_ns is not None else None


def validate_png(data: bytes) -> dict:
    """Validate a non-interlaced PNG without optional imaging packages.

    Reject truncated chunks, bad CRCs/order, unknown critical chunks, malformed
    zlib streams and scanlines. Hash decompressed pixels so changing metadata
    alone cannot evade the repeated-output check. This is a structural gate,
    not an assessment of image quality.
    """
    if not data.startswith(_SIGNATURE) or len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("output is not a bounded PNG")
    pos, ihdr, palette, compressed = 8, None, None, bytearray()
    have_idat, ended_idat, ended = False, False, False
    while pos < len(data):
        if pos + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        n = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4:pos + 8]
        if n > MAX_RESPONSE_BYTES or pos + n + 12 > len(data):
            raise ValueError("truncated or oversized PNG chunk")
        chunk = data[pos + 8:pos + 8 + n]
        crc = struct.unpack_from(">I", data, pos + 8 + n)[0]
        if not all(65 <= c <= 90 or 97 <= c <= 122 for c in kind) or kind[2] & 32:
            raise ValueError("invalid PNG chunk type")
        if zlib.crc32(kind + chunk) & 0xFFFFFFFF != crc:
            raise ValueError("PNG CRC mismatch")
        if ihdr is None and kind != b"IHDR":
            raise ValueError("PNG must start with IHDR")
        if kind == b"IHDR":
            if ihdr is not None or n != 13:
                raise ValueError("invalid PNG IHDR")
            ihdr = struct.unpack(">IIBBBBB", chunk)
            width, height, depth, color, comp, filt, interlace = ihdr
            allowed = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
            if not width or not height or width * height > MAX_PIXELS:
                raise ValueError("PNG dimensions exceed bounds")
            if depth not in allowed.get(color, ()) or comp or filt or interlace:
                raise ValueError("unsupported PNG encoding (non-interlaced PNG required)")
        elif kind == b"PLTE":
            if palette is not None or have_idat or not n or n % 3 or n > 768 or ihdr[3] in (0, 4):
                raise ValueError("invalid PNG palette")
            palette = chunk
            if ihdr[3] == 3 and n // 3 > 2 ** ihdr[2]:
                raise ValueError("PNG palette exceeds indexed bit depth")
        elif kind == b"IDAT":
            if ended_idat or (ihdr[3] == 3 and palette is None):
                raise ValueError("invalid PNG IDAT order")
            have_idat = True
            compressed.extend(chunk)
        elif kind == b"IEND":
            if n or not have_idat or pos + n + 12 != len(data):
                raise ValueError("invalid PNG IEND or trailing data")
            ended = True
            break
        elif not kind[0] & 32:
            raise ValueError("unknown critical PNG chunk")
        if have_idat and kind not in (b"IDAT", b"IEND"):
            ended_idat = True
        pos += n + 12
    if not ended or ihdr is None:
        raise ValueError("PNG has no complete IEND")
    width, height, depth, color, *_ = ihdr
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    stride = (width * channels * depth + 7) // 8
    expected = (stride + 1) * height
    if expected > MAX_RAW_BYTES:
        raise ValueError("PNG decoded size exceeds bounds")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(compressed, expected + 1)
    except zlib.error:
        raise ValueError("invalid PNG compressed data") from None
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("PNG decompressed size or stream mismatch")
    # Undo PNG row filters before hashing: distinct compression/filter choices
    # for identical pixels must not bypass the cache check.
    bpp, previous, pixels = max(1, (channels * depth + 7) // 8), bytearray(stride), hashlib.sha256()
    pixels.update(struct.pack(">IIBB", width, height, depth, color))
    pixels.update(palette or b"")
    for offset in range(0, len(raw), stride + 1):
        filt = raw[offset]
        if filt > 4:
            raise ValueError("invalid PNG scanline filter")
        row = bytearray(raw[offset + 1:offset + stride + 1])
        for i in range(stride) if filt else ():
            left = row[i - bpp] if i >= bpp else 0
            above, upper_left = previous[i], previous[i - bpp] if i >= bpp else 0
            if filt == 1:
                row[i] = (row[i] + left) & 255
            elif filt == 2:
                row[i] = (row[i] + above) & 255
            elif filt == 3:
                row[i] = (row[i] + (left + above) // 2) & 255
            elif filt == 4:
                estimate = left + above - upper_left
                pa, pb, pc = abs(estimate - left), abs(estimate - above), abs(estimate - upper_left)
                predictor = left if pa <= pb and pa <= pc else above if pb <= pc else upper_left
                row[i] = (row[i] + predictor) & 255
        pixels.update(row)
        previous = row
    return {"width": width, "height": height, "pixel_sha256": pixels.hexdigest()}


def effective_mismatches(request: ImageRequest, effective: dict | None) -> list[str]:
    expected = {k: v for k, v in request.body().items() if k not in ("prompt", "size", "steps")}
    expected["steps_run"] = request.steps
    if not isinstance(effective, dict):
        return ["effective settings missing"]
    def same(actual, expected):
        if isinstance(actual, bool) or isinstance(expected, bool):
            return type(actual) is type(expected) and actual == expected
        if isinstance(expected, dict):
            return (isinstance(actual, dict) and actual.keys() == expected.keys()
                    and all(same(actual[k], value) for k, value in expected.items()))
        if isinstance(expected, list):
            return (isinstance(actual, list) and len(actual) == len(expected)
                    and all(same(a, b) for a, b in zip(actual, expected)))
        return actual == expected
    mismatches = []
    for key, value in expected.items():
        actual = effective.get(key)
        # JSON 3 and 3.0 are the same guidance scale, including nested settings.
        if key not in effective or not same(actual, value):
            mismatches.append(key)
    return mismatches


class ImageHTTP:
    endpoint_kind = "image_http"

    def __init__(self, ep: Endpoint, timeout_s: float = DEFAULT_TIMEOUT_S,
                 max_response_bytes: int = MAX_RESPONSE_BYTES):
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 86400:
            raise ValueError("timeout_s must be finite and in (0, 86400]")
        if not isinstance(max_response_bytes, int) or not 0 < max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError("response byte bound is invalid")
        self.ep, self.timeout_s, self.max_response_bytes = ep, timeout_s, max_response_bytes

    def generate(self, request: ImageRequest) -> ImageResult:
        result = ImageResult(request.id)
        connection = response = watchdog = None
        try:
            body = json.dumps(request.body(), allow_nan=False).encode()
            if len(body) > MAX_REQUEST_BYTES:
                raise ValueError("image request exceeds byte bound")
            headers = {"Content-Type": "application/json", "Accept": "application/json", "Connection": "close"}
            if self.ep.api_key:
                headers["Authorization"] = "Bearer " + self.ep.api_key
            result.t_send_ns = time.perf_counter_ns()
            deadline = time.monotonic() + self.timeout_s
            cls = http.client.HTTPSConnection if self.ep.scheme == "https" else http.client.HTTPConnection
            connection = cls(self.ep.host, self.ep.port, timeout=self.timeout_s)
            connection._create_connection = _connect
            # A per-recv socket timeout alone is not a whole-response deadline:
            # a peer could trickle header bytes forever. Interrupt the active
            # socket at the absolute deadline, including while parsing headers.
            def expire():
                sock = connection.sock
                if sock is None and response is not None:
                    sock = getattr(getattr(response.fp, "raw", None), "_sock", None)
                if sock:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()
            watchdog = threading.Timer(self.timeout_s, expire)
            watchdog.daemon = True
            watchdog.start()
            connection.request("POST", self.ep.prefix + PATH, body=body, headers=headers)
            if connection.sock:
                connection.sock.settimeout(max(0.001, deadline - time.monotonic()))
            response = connection.getresponse()
            result.status = response.status
            # Error bodies can contain credentials or arbitrary addresses. They
            # are never included in evidence, nor downloaded for diagnostics.
            if response.status != 200:
                raise ValueError(f"image endpoint returned HTTP {response.status}")
            if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ValueError("image endpoint must return application/json")
            announced = response.getheader("Content-Length")
            if announced is not None and (not announced.isdigit() or int(announced) > self.max_response_bytes):
                raise ValueError("image response exceeds byte bound")
            chunks, count = [], 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("image response deadline exceeded")
                # HTTP/1.1 Connection:close can detach the socket from its
                # connection; response.fp retains it until the final read.
                sock = connection.sock or getattr(getattr(response.fp, "raw", None), "_sock", None)
                if sock:
                    sock.settimeout(remaining)
                chunk = response.read1(min(65536, self.max_response_bytes + 1 - count))
                if not chunk:
                    break
                chunks.append(chunk)
                count += len(chunk)
                if count > self.max_response_bytes:
                    raise ValueError("image response exceeds byte bound")
            result.t_body_ns = time.perf_counter_ns()
            watchdog.cancel()
            if announced is not None and count != int(announced):
                raise ValueError("truncated image HTTP response")
            doc = json.loads(b"".join(chunks), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            if not isinstance(doc, dict) or doc.get("format") != "png" or not isinstance(doc.get("image_b64"), str):
                raise ValueError("image response needs image_b64 and format png")
            try:
                result.image = base64.b64decode(doc["image_b64"], validate=True)
            except (ValueError, binascii.Error):
                raise ValueError("invalid image base64") from None
            shape = validate_png(result.image)
            result.width, result.height, result.pixel_sha256 = shape["width"], shape["height"], shape["pixel_sha256"]
            if (result.width, result.height) != (request.width, request.height):
                raise ValueError("decoded PNG dimensions differ from request")
            # Keep requested settings only. Unsolicited server metadata may
            # include hostnames, paths or credentials and is not evidence of
            # the requested execution contract.
            reported = doc.get("effective")
            keys = set(request.body()) - {"prompt", "size", "steps"} | {"steps_run"}
            result.effective = ev.redact({key: reported[key] for key in keys if key in reported}) if isinstance(reported, dict) else None
            result.effective_mismatch = effective_mismatches(request, result.effective)
            cache = doc.get("cache")
            if not isinstance(cache, list) or any(not isinstance(k, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", k) for k in cache):
                raise ValueError("image response must declare cache as a list")
            result.cache = ev.redact(cache)
            # Completion means the client has decoded and checked the final
            # image. Body EOF is a separate diagnostic timestamp, not the
            # published end-to-end endpoint.
            result.t_done_ns = time.perf_counter_ns()
            result.ok = True
        except (OSError, ValueError, TypeError, RecursionError, http.client.HTTPException, socket.timeout) as exc:
            message = str(exc)
            if self.ep.api_key:
                message = message.replace(self.ep.api_key, "<redacted>")
            result.error = ev.redact(scrub(message, self.ep))[:500]
            result.image = None
        finally:
            if watchdog:
                watchdog.cancel()
            if connection:
                connection.close()
        return result

    def close(self) -> None:
        pass
