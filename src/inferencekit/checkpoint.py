"""Read safetensors headers (never the weights) from a local directory or the Hugging Face Hub.

A ``.safetensors`` file starts with an 8-byte little-endian unsigned integer N, then N bytes
of JSON::

    {"<tensor name>": {"dtype": "BF16", "shape": [rows, cols], "data_offsets": [begin, end]},
     ..., "__metadata__": {...}}

and then the tensor data. The first 8 + N bytes give every tensor's dtype and shape, which is
all a bytes-per-token count needs. For a Hub id the local Hugging Face cache is tried first (an
engine box usually has the snapshot it serves on disk, often with no network); otherwise each
shard costs one or two HTTP range requests (a few MB at most, whatever the checkpoint size). The
weights are never downloaded. ``HF_HUB_OFFLINE`` is honoured: offline, only the cache is read.

Every header is recorded as (file name, header length, sha256 of the N header bytes), so a
count can be reproduced against exactly the same tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from inferencekit.registry import OpError

# Bits per element of each safetensors dtype. Packed formats (NVFP4/MXFP4 stored as U8 pairs,
# FP8 scales) are counted as stored: the bytes on disk are the bytes the kernel reads.
DTYPE_BITS = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16,
    "I64": 64, "I32": 32, "I16": 16, "I8": 8,
    "U64": 64, "U32": 32, "U16": 16, "U8": 8, "BOOL": 8,
    "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8,
    "F6_E2M3": 6, "F6_E3M2": 6, "F4": 4, "C64": 64,
}
MAX_HEADER = 100_000_000  # the safetensors format caps the header at 100 MB
FIRST_READ = 1 << 16  # first range request: the length prefix plus the start of the header
INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"


class NotFound(OpError):
    """The Hub has no such file at that revision."""


class HubUnavailable(OpError):
    """The Hub could not be reached (offline mode, network error, server error); not a missing file."""


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    file: str


@dataclass
class Checkpoint:
    """Every tensor of a checkpoint, from its headers, plus how to reproduce the read."""

    tensors: dict[str, Tensor]
    files: list[dict]  # [{file, header_len, header_sha256, tensors, data_bytes}]
    source: str  # "directory" | "hub" | "hub-cache"
    commit: str | None = None  # the Hub commit the headers came from, when the Hub (or the cache) says
    metadata: dict = field(default_factory=dict)
    local_dir: Path | None = None  # the snapshot directory, for headers read from the local HF cache

    @property
    def headers_digest(self) -> str:
        """One sha256 over the sorted (file, header sha256) list."""
        lines = "".join(f"{f['file']}:{f['header_sha256']}\n" for f in sorted(self.files, key=lambda f: f["file"]))
        return hashlib.sha256(lines.encode()).hexdigest()

    @property
    def stored_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())


def tensor_bytes(dtype: str, shape) -> int:
    """Stored bytes of one tensor: elements x bits per element, rounded up to whole bytes."""
    if dtype not in DTYPE_BITS:
        raise OpError(f"unknown safetensors dtype {dtype}; cannot size it")
    return math.ceil(math.prod(int(d) for d in shape) * DTYPE_BITS[dtype] / 8)


def parse_header(raw: bytes, file: str) -> tuple[dict[str, Tensor], dict]:
    """Tensors and the digest entry from the N header bytes of one file."""
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise OpError(f"{file}: safetensors header is not valid JSON ({e})") from None
    if not isinstance(header, dict):
        raise OpError(f"{file}: safetensors header is not a JSON object")
    tensors: dict[str, Tensor] = {}
    data_bytes = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape = tuple(int(d) for d in info["shape"])
        size = tensor_bytes(info["dtype"], shape)
        begin, end = info["data_offsets"]
        if end - begin != size:
            raise OpError(f"{file}: tensor {name} is {info['dtype']} {list(shape)} = {size} bytes, but its "
                          f"data_offsets span {end - begin} bytes")
        tensors[name] = Tensor(name, info["dtype"], shape, size, file)
        data_bytes = max(data_bytes, end)
    entry = {"file": file, "header_len": len(raw), "header_sha256": hashlib.sha256(raw).hexdigest(),
             "tensors": len(tensors), "data_bytes": data_bytes}
    return tensors, entry


def _length(prefix: bytes, file: str) -> int:
    if len(prefix) < 8:
        raise OpError(f"{file}: shorter than the 8-byte safetensors length prefix")
    (n,) = struct.unpack("<Q", prefix[:8])
    if n > MAX_HEADER:
        raise OpError(f"{file}: header length {n} exceeds the safetensors limit; not a safetensors file?")
    return n


def _merge(parts: list[tuple[dict[str, Tensor], dict]], source: str, commit: str | None = None) -> Checkpoint:
    tensors: dict[str, Tensor] = {}
    for t, _ in parts:
        dup = set(t) & set(tensors)
        if dup:
            raise OpError(f"tensor {sorted(dup)[0]} appears in two shards; pass the directory with its index file")
        tensors.update(t)
    return Checkpoint(tensors, [e for _, e in parts], source, commit)


# --------------------------------------------------------------------------- local files

def local_header(path: Path) -> bytes:
    with open(path, "rb") as f:
        n = _length(f.read(8), path.name)
        raw = f.read(n)
    if len(raw) != n:
        raise OpError(f"{path.name}: truncated header ({len(raw)} of {n} bytes)")
    return raw


def local_files(directory: Path) -> list[Path]:
    """Shards named by the index file when there is one (so stray copies are not counted twice)."""
    index = directory / INDEX
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [directory / n for n in names]
    return sorted(directory.glob("*.safetensors"))


def load_local(directory: Path) -> Checkpoint | None:
    files = local_files(directory)
    if not files:
        return None
    try:
        return _merge([parse_header(local_header(p), p.name) for p in files], "directory")
    except OSError as e:
        raise OpError(f"could not read a safetensors header in {directory}: {e}") from None


# --------------------------------------------------------------------------- Hugging Face Hub

_TRUE = {"1", "ON", "YES", "TRUE"}


def _hf_constants():
    """huggingface_hub.constants when that package is installed, else None."""
    try:
        from huggingface_hub import constants
    except Exception:  # noqa: BLE001 - not installed, or broken
        return None
    return constants


def hub_offline() -> bool:
    """HF_HUB_OFFLINE (or TRANSFORMERS_OFFLINE), as huggingface_hub reads it."""
    constants = _hf_constants()
    if constants is not None and getattr(constants, "HF_HUB_OFFLINE", False):
        return True
    return any(os.environ.get(k, "").upper() in _TRUE for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def hub_cache_dir() -> Path:
    """The Hugging Face hub cache: huggingface_hub's own setting when installed, else the same env vars."""
    constants = _hf_constants()
    if constants is not None and getattr(constants, "HF_HUB_CACHE", None):
        return Path(constants.HF_HUB_CACHE)
    env = os.environ
    if env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE"):
        return Path(env.get("HF_HUB_CACHE") or env["HUGGINGFACE_HUB_CACHE"]).expanduser()
    home = env.get("HF_HOME") or os.path.join(env.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                                               "huggingface")
    return Path(home).expanduser() / "hub"


def cached_file(repo: str, filename: str, revision: str | None) -> Path | None:
    """A file of ``repo`` at ``revision`` in the local HF cache, or None. Never touches the network.
    Uses huggingface_hub.try_to_load_from_cache when installed, else the same cache layout
    (models--org--name/refs/<rev> -> snapshots/<commit>/<file>)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - not installed, or too old
        try_to_load_from_cache = None
    if try_to_load_from_cache is not None:
        try:
            p = try_to_load_from_cache(repo, filename, revision=revision)
        except Exception:  # noqa: BLE001 - an unusable cache is the same as no cache
            return None
        return Path(p) if isinstance(p, str) and os.path.isfile(p) else None
    folder = hub_cache_dir() / ("models--" + repo.replace("/", "--"))
    rev = revision or "main"
    ref = folder / "refs" / rev
    try:
        commit = ref.read_text().strip() if ref.is_file() else rev
    except OSError:
        return None
    p = folder / "snapshots" / commit / filename
    return p if p.is_file() else None


def snapshot_commit(path: Path) -> str | None:
    """The commit of a cached file: the directory name after 'snapshots' in its (unresolved) path."""
    parts = Path(path).parts
    if "snapshots" in parts:
        i = len(parts) - 1 - parts[::-1].index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_hub_cache(repo: str, revision: str | None) -> Checkpoint | None:
    """Headers of a Hub checkpoint whose shards are all in the local HF cache, else None."""
    index = cached_file(repo, INDEX, revision)
    if index is not None:
        try:
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        except (OSError, ValueError, KeyError):
            return None
    else:
        names = [SINGLE]
    paths = [cached_file(repo, n, revision) for n in names]
    if not names or any(p is None for p in paths):
        return None
    commits = {snapshot_commit(p) for p in paths} | ({snapshot_commit(index)} if index else set())
    if len(commits) != 1:  # files from different snapshots: not one checkpoint
        return None
    try:
        parts = [parse_header(local_header(p), n) for p, n in zip(paths, names)]
    except OSError:
        return None
    ck = _merge(parts, "hub-cache", commits.pop())
    ck.local_dir = paths[0].parent
    return ck


def hub_endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def hub_token() -> str | None:
    """HF_TOKEN, else the huggingface_hub login when that package is installed. Never recorded."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001 - not installed, or an old version without get_token
        return None


def hub_url(repo: str, filename: str, revision: str | None) -> str:
    rev = urllib.parse.quote(revision or "main", safe="")
    return f"{hub_endpoint()}/{repo}/resolve/{rev}/{urllib.parse.quote(filename)}"


class _CommitRedirects(urllib.request.HTTPRedirectHandler):
    """Follows redirects and keeps the X-Repo-Commit header the Hub puts on the redirect itself."""

    commit: str | None = None

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.commit = self.commit or headers.get("X-Repo-Commit")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get(url: str, byte_range: tuple[int, int] | None, limit: int) -> tuple[bytes, str | None]:
    """GET at most ``limit`` bytes. The token is an unredirected header, so it is not forwarded
    to the storage host a Hub download redirects to (that URL is pre-signed)."""
    req = urllib.request.Request(url)
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    token = hub_token()
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    redirects = _CommitRedirects()
    with urllib.request.build_opener(redirects).open(req, timeout=60) as r:
        commit = redirects.commit or r.headers.get("X-Repo-Commit")
        if byte_range and r.status == 200 and byte_range[0] > 0:  # range ignored: skip to the offset
            r.read(byte_range[0])
        # Read only what is needed even if the server ignored the range: never the weights.
        return r.read(limit), commit


def hub_small_file(repo: str, filename: str, revision: str | None) -> tuple[bytes | None, str | None]:
    """A small file (index, config) from the Hub, or (None, None) when the repo has no such file."""
    if hub_offline():
        raise HubUnavailable(f"could not fetch {filename} for '{repo}': HF_HUB_OFFLINE is set")
    try:
        return _get(hub_url(repo, filename, revision), None, 64 << 20)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, None
        cls = HubUnavailable if e.code >= 500 else OpError
        raise cls(f"could not fetch {filename} for '{repo}': HTTP {e.code}") from None
    except OSError as e:  # connection refused, DNS, timeout: the Hub is unreachable
        raise HubUnavailable(f"could not fetch {filename} for '{repo}': {e}") from None


def hub_header(repo: str, filename: str, revision: str | None) -> tuple[bytes, str | None]:
    if hub_offline():
        raise HubUnavailable(f"could not read the header of {filename} in '{repo}': HF_HUB_OFFLINE is set")
    url = hub_url(repo, filename, revision)
    try:
        first, commit = _get(url, (0, FIRST_READ - 1), FIRST_READ)
        n = _length(first, filename)
        raw = first[8:8 + n]
        if len(raw) < n:
            rest, _ = _get(url, (8 + len(raw), 8 + n - 1), n - len(raw))
            raw += rest
    except urllib.error.HTTPError as e:
        cls = NotFound if e.code == 404 else HubUnavailable if e.code >= 500 else OpError
        raise cls(f"could not read the header of {filename} in '{repo}': HTTP {e.code}") from None
    except OSError as e:  # connection refused, DNS, timeout: the Hub is unreachable
        raise HubUnavailable(f"could not read the header of {filename} in '{repo}': {e}") from None
    if len(raw) != n:
        raise OpError(f"{filename}: got {len(raw)} of {n} header bytes from the Hub")
    return raw, commit


def load_hub(repo: str, revision: str | None) -> Checkpoint | None:
    """Headers of every shard named by the index (or the single model.safetensors): from the local
    HF cache when every shard is there, else by range requests. Offline with no cached copy, or with
    the Hub unreachable, raises HubUnavailable (the caller can fall back to the config method)."""
    cached = load_hub_cache(repo, revision)
    if cached is not None:
        return cached
    if hub_offline():
        raise HubUnavailable(f"HF_HUB_OFFLINE is set and the safetensors shards of '{repo}' at "
                             f"{revision or 'main'} are not all in the local Hugging Face cache ({hub_cache_dir()})")
    index, commit = hub_small_file(repo, INDEX, revision)
    if index is not None:
        try:
            names = sorted(set(json.loads(index)["weight_map"].values()))
        except (ValueError, KeyError) as e:
            raise OpError(f"{repo}: unreadable {INDEX} ({e})") from None
    else:
        names = [SINGLE]
    parts = []
    for name in names:
        try:
            raw, c = hub_header(repo, name, revision)
        except NotFound:
            if index is None:  # no index and no model.safetensors: not a safetensors repo
                return None
            raise
        commit = commit or c
        parts.append(parse_header(raw, name))
    return _merge(parts, "hub", commit)


def is_hub_id(model: str) -> bool:
    return not Path(model).exists() and model.count("/") == 1 and not model.startswith((".", "/"))


def load(model: str, revision: str | None = None) -> Checkpoint | None:
    """Headers for a model directory, a file inside one (e.g. its config.json), or a Hub id.
    None when there are no safetensors files to read."""
    p = Path(model)
    if p.is_dir():
        return load_local(p)
    if p.is_file():
        return load_local(p.parent)
    if is_hub_id(model):
        return load_hub(model, revision)
    return None
