"""Prompt sets ship gzipped; every reader takes .jsonl and .jsonl.gz and hashes the JSONL bytes."""

import gzip
import hashlib
import json
import shutil

import pytest

from inferencekit import prompts as ps
from inferencekit import registry
from inferencekit.ops import gates
from inferencekit.ops.quality import load_prompts as quality_load
from inferencekit.prompts import build
from inferencekit.registry import OpError
from inferencekit.serve import prompts_io
from inferencekit.serve.rounds import read_prompts as rounds_read

# The pre-registered sha256 of the shipped sets (uncompressed JSONL bytes): gzipping must not change them.
PINNED = {"dev": "736717af715d14e64540b64f359372cc0aa0c611584167d61c12d918995592b6",
          "dev-load": "fe42629d968dbf37e75cb0a5bd7c00d0605190845f2c27e1942b7428b6120632"}


def test_shipped_sets_are_gzipped_and_keep_their_hashes():
    for name, sha in PINNED.items():
        path = ps.resolve(name)
        assert path.name == f"{name}.jsonl.gz" and not path.with_name(f"{name}.jsonl").exists()
        assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == sha
        assert ps.load_manifest(name)["sha256"] == sha == ps.sha256_file(path) == ps.identity(name)["sha256"]


@pytest.fixture
def twins(tmp_path):
    """The dev set as plain/dev.jsonl and packed/dev.jsonl.gz, each with its manifest."""
    plain, packed = tmp_path / "plain", tmp_path / "packed"
    plain.mkdir()
    packed.mkdir()
    src = ps.resolve("dev")
    shutil.copy(src, packed / "dev.jsonl.gz")
    (plain / "dev.jsonl").write_bytes(gzip.decompress(src.read_bytes()))
    for d in (plain, packed):
        shutil.copy(ps.manifest_path(src), d / "dev.manifest.json")
    return plain / "dev.jsonl", packed / "dev.jsonl.gz"


def test_every_loader_reads_both_forms_identically(twins):
    plain, packed = twins
    a, b = prompts_io.load(plain), prompts_io.load(packed)
    assert a.sha256 == b.sha256 == PINNED["dev"] and a.prompts == b.prompts and a.name == b.name == "dev"
    assert a.manifest == b.manifest
    ra, rb = rounds_read(plain), rounds_read(packed)
    assert ra["sha256"] == rb["sha256"] == PINNED["dev"] and ra["name"] == rb["name"] == "dev" and ra["n"] == 64
    qa, qb = quality_load(str(plain)), quality_load(str(packed))
    assert qa == qb and qa[1] == PINNED["dev"]
    assert ps.load(str(plain)) == ps.load(str(packed))
    # a .jsonl path whose file was replaced by the .jsonl.gz twin still loads (old docs, old commands)
    missing = packed.with_name("dev.jsonl")
    assert prompts_io.load(missing).sha256 == PINNED["dev"] and rounds_read(missing)["n"] == 64
    assert ps.resolve(str(missing)) == packed


def test_verify_and_show_work_on_both_forms(twins, tmp_path):
    plain, packed = twins
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{PINNED['dev']}  dev.jsonl\n")  # SHA256SUMS name the .jsonl, as prompts seal writes them
    for path in (plain, packed):
        res = registry.call("prompts_verify", {"set": str(path), "expected_sha256": PINNED["dev"],
                                               "sums": str(sums)})
        assert res["verdict"] == "PASS", res["checks"]
        assert res["set"] == "dev" and res["sha256"] == PINNED["dev"]
        shown = registry.call("prompts_show", {"set": str(path)})
        assert shown["set"] == "dev" and shown["n"] == 64 and shown["sha256"] == PINNED["dev"]
    assert registry.call("prompts_verify", {"set": "dev-load"})["verdict"] == "PASS"


def test_tampered_or_truncated_gzip_is_caught(twins):
    _, packed = twins
    items = [json.loads(x) for x in gzip.decompress(packed.read_bytes()).decode().splitlines()]
    items[0]["messages"][0]["content"] += " changed"
    packed.write_bytes(gzip.compress("".join(json.dumps(i) + "\n" for i in items).encode()))
    res = registry.call("prompts_verify", {"set": str(packed)})
    assert res["verdict"] == "FAIL" and not next(c for c in res["checks"] if c["check"] == "manifest_sha256")["ok"]
    with pytest.raises(OpError, match="does not match its manifest"):
        prompts_io.load(packed)
    packed.write_bytes(packed.read_bytes()[:200])
    with pytest.raises(OpError):
        prompts_io.load(packed)
    with pytest.raises(OpError):
        registry.call("prompts_verify", {"set": str(packed)})


@pytest.mark.parametrize("damage", ["mid_stream", "header", "crc", "truncated", "trailing_garbage"])
def test_corrupt_gzip_is_a_clear_error_in_every_reader(twins, damage):
    """A damaged .jsonl.gz is an OpError (or a review warning) everywhere, never a raw zlib traceback."""
    _, packed = twins
    raw = bytearray(packed.read_bytes())
    if damage == "mid_stream":  # flip 40 bytes inside the deflate stream: zlib.error before the CRC check
        mid = len(raw) // 2
        raw[mid:mid + 40] = bytes(b ^ 0xFF for b in raw[mid:mid + 40])
    elif damage == "header":
        raw[2] = 0x07  # compression method other than deflate
    elif damage == "crc":
        raw[-8] ^= 0xFF
    elif damage == "truncated":
        raw = raw[:len(raw) // 3]
    else:
        raw += b"\x00garbage"
    packed.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="corrupt gzip"):
        ps.read_bytes(packed)
    for call in (lambda: registry.call("prompts_verify", {"set": str(packed)}),
                 lambda: registry.call("prompts_show", {"set": str(packed)}),
                 lambda: prompts_io.load(packed),
                 lambda: rounds_read(packed),
                 lambda: quality_load(str(packed)),
                 lambda: gates._load_prompts(str(packed))):
        with pytest.raises(OpError):
            call()


def test_build_into_the_package_directory_stays_gzipped(tmp_path, monkeypatch):
    """scripts/build_prompts.py passes the package directory as --out: the rebuild must not un-gzip the sets."""
    from test_prompts import _fake_sources

    monkeypatch.setenv("INFERENCEKIT_PROMPT_BLOCKLIST", "none")
    monkeypatch.setattr(build, "BOOKS", {1: "Book One", 2: "Book Two", 3: "Book Three", 4: "Book Four",
                                         5: "Book Five", 6: "Book Six"})
    monkeypatch.setattr(build, "PER_BUCKET", 2)
    monkeypatch.setattr(build, "LOAD_TARGET", 40)
    monkeypatch.setattr(build, "LOAD_MEDIUM_MAX", 10)
    src, pkg, other = tmp_path / "src", tmp_path / "pkg" / "prompts", tmp_path / "other"
    src.mkdir()
    pkg.mkdir(parents=True)
    _fake_sources(src)
    monkeypatch.setattr(ps, "data_dir", lambda: pkg)
    (pkg / "dev.jsonl").write_text("stale\n")  # a plain twin left over is replaced by the .gz
    for out in (str(pkg), str(tmp_path / "pkg" / "." / "prompts"), None):  # explicit, unnormalised, default
        registry.call("prompts_build", {"sources": str(src), "out": out})
        assert sorted(x.name for x in pkg.glob("dev*.jsonl*")) == ["dev-load.jsonl.gz", "dev.jsonl.gz"]
    registry.call("prompts_build", {"sources": str(src), "out": str(other)})
    assert sorted(x.name for x in other.glob("dev*.jsonl*")) == ["dev-load.jsonl", "dev.jsonl"]
    for name in ("dev", "dev-load"):  # same set, same identity, whichever form was written
        assert ps.sha256_file(pkg / f"{name}.jsonl.gz") == ps.sha256_file(other / f"{name}.jsonl") == \
            json.loads((pkg / f"{name}.manifest.json").read_text())["sha256"]


def test_gzip_writer_is_deterministic_and_replaces_the_twin(tmp_path):
    items = [{"id": "s-1", "bucket": "short", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
              "source": "x/y", "kind": "chat"}]
    plain = build.write_set(tmp_path, "t", items, False, "single", [])
    assert (tmp_path / "t.jsonl").is_file()
    packed = build.write_set(tmp_path, "t", items, False, "single", [], gz=True)
    assert not (tmp_path / "t.jsonl").exists() and (tmp_path / "t.jsonl.gz").is_file()
    first = (tmp_path / "t.jsonl.gz").read_bytes()
    build.write_set(tmp_path, "t", items, False, "single", [], gz=True)
    assert (tmp_path / "t.jsonl.gz").read_bytes() == first  # no mtime or name in the header
    assert plain["sha256"] == packed["sha256"] == ps.sha256_file(tmp_path / "t.jsonl.gz")
