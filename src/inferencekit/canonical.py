"""Canonical JSON (RFC 8785 style) and digests.

Why: an evidence id is the sha256 of the record body. For the id to be stable, the
same data must always serialise to the same bytes, on any machine and any Python
version. Plain ``json.dumps`` is close but not exact (it writes ``1e-05`` where
RFC 8785 writes ``0.00001``, and ``1.0`` where RFC 8785 writes ``1``), so the
number formatting below follows RFC 8785 / ECMAScript ``Number.prototype.toString``.

Rules:
  * object keys sorted by their UTF-16 code units, no whitespace anywhere;
  * strings: UTF-8, only the mandatory escapes (``"`` ``\\`` and control chars);
  * numbers: shortest round-trip digits, ECMAScript layout; NaN/Infinity rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _format_number(x: float) -> str:
    """ECMAScript Number-to-string, which RFC 8785 adopts for JSON numbers."""
    if math.isnan(x) or math.isinf(x):
        raise ValueError("NaN and Infinity cannot be represented in canonical JSON")
    if x == 0:
        return "0"  # also maps -0.0 to "0", as ECMAScript does
    sign = "-" if x < 0 else ""
    # repr() gives the shortest digit string that round-trips (same digits ECMAScript picks).
    mantissa, _, exp_text = repr(abs(x)).partition("e")
    int_part, _, frac_part = mantissa.partition(".")
    digits = (int_part + frac_part).lstrip("0")
    exp10 = (int(exp_text) if exp_text else 0) - len(frac_part)  # value = int(digits) * 10**exp10
    stripped = digits.rstrip("0")
    exp10 += len(digits) - len(stripped)
    digits = stripped
    k = len(digits)  # number of significant digits
    n = exp10 + k  # value = 0.d1d2..dk * 10**n
    if k <= n <= 21:  # integer: digits then zeros
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:  # decimal point inside the digits
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:  # small: 0.000ddd
        return sign + "0." + "0" * (-n) + digits
    exponent = n - 1  # scientific: d.ddde+N
    mant = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{sign}{mant}e{'+' if exponent > 0 else '-'}{abs(exponent)}"


def _utf16_key(s: str) -> bytes:
    # RFC 8785 sorts by UTF-16 code units; big-endian UTF-16 bytes sort the same way.
    return s.encode("utf-16-be")


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to canonical JSON text."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, str):
        # ensure_ascii=False keeps non-ASCII as UTF-8; json escapes control chars as \u00xx.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, str):
                raise TypeError(f"canonical JSON object keys must be strings, got {type(k).__name__}")
        items = sorted(value.items(), key=lambda kv: _utf16_key(kv[0]))
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canonical_json(v) for k, v in items) + "}"
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    """``sha256:<hex>`` of the canonical JSON of ``value`` (used for argv digests)."""
    return "sha256:" + sha256_hex(canonical_json(value))


def file_sha256(path, chunk: int = 1 << 20) -> tuple[str, int]:
    """(sha256 hex, size in bytes) of a file, read in 1 MiB chunks so large traces fit in memory."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size
