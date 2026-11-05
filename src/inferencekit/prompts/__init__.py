"""Versioned prompt sets: the inputs every engine sees, identical for all of them.

A prompt set is a JSONL file (one request per line), optionally gzipped (``<set>.jsonl.gz``;
every reader takes either, detected by content), plus ``<set>.manifest.json``:

    {"id": "short-0007", "bucket": "short", "messages": [{"role": "user", "content": "..."}],
     "max_tokens": 256, "source": "databricks-dolly-15k/CC-BY-SA-3.0", "kind": "chat", ...}

    {"name": "dev", "version": "1.0", "n": 64, "buckets": {"short": 16, ...},
     "sha256": "<sha256 of the JSONL bytes>", "sealed": false, ...}

Two sets ship with the package, gzipped: ``dev`` (16 prompts per bucket, single-user runs)
and ``dev-load`` (1,200+ short and medium prompts, so a load test never repeats a prompt
inside a level). A set's sha256 is always that of the uncompressed JSONL bytes, so a
gzipped and a plain copy of one set have the same identity. A ``sealed`` set is built by the same code from *different* source
items and never ships; only its sha256 is published ahead of time.

Lengths are measured with a tokenizer-free proxy, ``proxy_tokens = words * 1.3``
(whitespace-separated words), so bucket membership does not depend on any one model's
tokenizer. For English prose, 1.3 tokens per word is the usual rule of thumb for
BPE tokenizers with 30k-150k vocabularies; the true count for a given model differs by
roughly +/-15%, which the bucket ranges absorb.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from importlib import resources
from pathlib import Path

# Bucket rules. very_short is defined in words (1-5), the others in proxy tokens.
BUCKETS = ("very_short", "short", "medium", "long")
PROXY_FACTOR = 1.3
BUCKET_RULES = {
    "very_short": {"words": [1, 5]},
    "short": {"proxy_tokens": [20, 50]},
    "medium": {"proxy_tokens": [240, 360]},
    "long": {"proxy_tokens": [2000, 4000]},
}
SHIPPED = ("dev", "dev-load")


def words(text: str) -> int:
    return len(text.split())


def proxy_tokens(text: str) -> int:
    """Tokenizer-free length estimate: whitespace words * 1.3, rounded."""
    return round(words(text) * PROXY_FACTOR)


def prompt_text(item: dict) -> str:
    """All message contents of an item joined; what the length rules are applied to."""
    return "\n".join(str(m.get("content", "")) for m in item.get("messages", []))


def in_bucket(text: str, bucket: str) -> bool:
    rule = BUCKET_RULES[bucket]
    if "words" in rule:
        lo, hi = rule["words"]
        return lo <= words(text) <= hi
    lo, hi = rule["proxy_tokens"]
    return lo <= words(text) * PROXY_FACTOR <= hi


def data_dir() -> Path:
    """Directory holding the shipped sets (inside the wheel or the source checkout)."""
    try:
        d = resources.files("inferencekit").joinpath("prompts")
        if d.is_dir():
            return Path(str(d))
    except (ModuleNotFoundError, FileNotFoundError):  # pragma: no cover
        pass
    return Path(__file__).resolve().parent


GZIP_MAGIC = b"\x1f\x8b"


def existing(path: str | Path) -> Path:
    """``path``, or its gzipped / plain twin when only that exists (``x.jsonl`` <-> ``x.jsonl.gz``)."""
    p = Path(path)
    if p.exists():
        return p
    twin = p.with_name(p.name + ".gz") if p.name.endswith(".jsonl") else \
        p.with_name(p.name[:-3]) if p.name.endswith(".jsonl.gz") else None
    return twin if twin is not None and twin.is_file() else p


def read_bytes(path: str | Path) -> bytes:
    """The JSONL bytes of a prompt-set file: gunzipped when the file is gzip (by content, not name).

    A damaged gzip file (bad header, corrupt deflate stream, truncated, wrong CRC or length)
    raises ``ValueError("corrupt gzip (...)")`` (callers add the path); a missing file raises ``OSError``."""
    raw = Path(path).read_bytes()
    if raw[:2] != GZIP_MAGIC:
        return raw
    try:
        return gzip.decompress(raw)
    except (zlib.error, EOFError, gzip.BadGzipFile) as e:
        raise ValueError(f"corrupt gzip ({e})") from None


def read_text(path: str | Path) -> str:
    return read_bytes(path).decode("utf-8")


def gzip_bytes(data: bytes) -> bytes:
    """Deterministic gzip (no file name, mtime 0) of JSONL bytes."""
    return gzip.compress(data, compresslevel=9, mtime=0)


def set_stem(path: str | Path) -> str:
    """File name without ``.jsonl.gz`` / ``.jsonl`` / ``.gz``: ``dev.jsonl.gz`` -> ``dev``."""
    name = Path(path).name
    for suffix in (".jsonl.gz", ".jsonl", ".gz"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def resolve(name_or_path: str) -> Path:
    """'dev' / 'dev-load' -> the shipped set (``.jsonl.gz``, or ``.jsonl`` in a rebuilt checkout);
    anything else is a path to a ``.jsonl`` or ``.jsonl.gz`` file."""
    if name_or_path in SHIPPED:
        d = data_dir()
        gz = d / f"{name_or_path}.jsonl.gz"
        return gz if gz.is_file() else d / f"{name_or_path}.jsonl"
    p = existing(name_or_path)
    if p.is_dir():
        raise ValueError(f"{p} is a directory; pass the .jsonl file")
    return p


def manifest_path(jsonl: Path) -> Path:
    return Path(jsonl).with_name(set_stem(jsonl) + ".manifest.json")


def sha256_file(path: Path) -> str:
    """sha256 of the file's JSONL bytes (after gunzip for a ``.jsonl.gz``): the set's identity."""
    return hashlib.sha256(read_bytes(path)).hexdigest()


def load(name_or_path: str, buckets: list[str] | tuple[str, ...] | None = None) -> list[dict]:
    """Items of a prompt set, optionally only some buckets, in file order."""
    path = resolve(name_or_path)
    items = [json.loads(line) for line in read_text(path).splitlines() if line.strip()]
    if buckets:
        items = [it for it in items if it.get("bucket") in buckets]
    return items


def load_manifest(name_or_path: str) -> dict:
    return json.loads(manifest_path(resolve(name_or_path)).read_text(encoding="utf-8"))


def identity(name_or_path: str) -> dict:
    """``{name, sha256}`` for the evidence workload (sha256 of the bytes actually read)."""
    path = resolve(name_or_path)
    try:
        name = load_manifest(name_or_path).get("name") or path.stem
    except (OSError, ValueError):
        name = set_stem(path)
    return {"name": name, "sha256": sha256_file(path)}
