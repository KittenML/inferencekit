"""GSM8K data: the pure-Python Parquet reader, and the pinned, sha256-checked download (offline: file:// URLs)."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from inferencekit.quality import gsm8k as g8
from inferencekit.quality.parquet import ParquetError, read_table, snappy_decompress
from inferencekit.registry import OpError

FIX = Path(__file__).parent / "fixtures" / "parquet"
TINY = FIX / "gsm8k_tiny.parquet"
QUESTIONS = ["Tom has 3 apples and buys 4 more. How many apples does he have?",
             "A train goes 60 miles per hour for 2 hours. How far does it go?",
             "What is " + "ab" * 40 + " repeated? Café ’quotes’ cost $1,200."]
ANSWERS = ["3 + 4 = <<3+4=7>>7\n#### 7", "60 * 2 = <<60*2=120>>120\n#### 120", "It costs 1,200 dollars.\n#### 1,200"]
TINY_JSONL = "".join(json.dumps({"question": q, "answer": a}) + "\n" for q, a in zip(QUESTIONS, ANSWERS)).encode()


def _sha(path_or_bytes) -> str:
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- parquet reader

def test_reader_matches_pyarrow_written_files():
    # gsm8k_tiny: the Hugging Face file's layout (snappy, dictionary pages, data page v1, two row groups)
    tiny = read_table(TINY)
    assert [b.decode() for b in tiny["question"]] == QUESTIONS
    assert [b.decode() for b in tiny["answer"]] == ANSWERS
    # data page v2, gzip, plain encoding, an optional column with a null
    v2 = read_table(FIX / "nulls_gzip_plain_v2.parquet")
    assert [b.decode() if b is not None else None for b in v2["question"]] == QUESTIONS + [None]
    assert v2["answer"][-1] == b"#### 1"
    # required columns, uncompressed, int64
    assert read_table(FIX / "required_uncompressed.parquet") == {"n": [1, -2, 3000000000], "s": [b"x", b"y", b"z"]}


def test_snappy_overlapping_copy_and_corruption():
    # literal "ab", then copy 6 bytes from offset 2: the copy overlaps its own output
    assert snappy_decompress(bytes([8, 0x04, 0x61, 0x62, 0x09, 0x02])) == b"abababab"
    with pytest.raises(ParquetError):
        snappy_decompress(bytes([8, 0x04, 0x61, 0x62, 0x09, 0x09]))  # offset beyond the output
    with pytest.raises(ParquetError):
        read_table(b"not a parquet file at all")


def test_parquet_to_jsonl_is_byte_exact():
    assert g8.parquet_to_jsonl(TINY) == TINY_JSONL


# --------------------------------------------------------------------------- pinned download

@pytest.fixture
def pinned_tiny(tmp_path, monkeypatch):
    """Pin the loader to the tiny fixture, served from a file:// URL; cache under tmp_path."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("INFERENCEKIT_CACHE_DIR", str(cache))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    served = tmp_path / "hub" / "test-00000-of-00001.parquet"
    served.parent.mkdir()
    shutil.copy(TINY, served)
    monkeypatch.setattr(g8, "source_url", lambda: served.as_uri())
    monkeypatch.setattr(g8, "PARQUET_SHA256", _sha(TINY))
    monkeypatch.setattr(g8, "JSONL_SHA256", _sha(TINY_JSONL))
    return {"cache": cache / "gsm8k", "served": served}


def test_download_checks_both_hashes_and_caches(pinned_tiny):
    items, sha = g8.load()
    assert sha == _sha(TINY_JSONL) and [it["gold"] for it in items] == ["7", "120", "1,200"]
    cached = g8.data_path(fetch=False)
    assert cached.parent == pinned_tiny["cache"] and cached.read_bytes() == TINY_JSONL
    pinned_tiny["served"].unlink()  # cached now: no network needed
    assert g8.load(limit=2)[0][1]["gold"] == "120"


def test_offline_error_gives_the_manual_download_command(pinned_tiny, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(OpError) as e:
        g8.load()
    msg = str(e.value)
    assert "HF_HUB_OFFLINE" in msg and g8.HF_REVISION in msg and g8.PARQUET_SHA256 in msg
    assert f"curl -fL -o '{pinned_tiny['cache'] / 'test-00000-of-00001.parquet'}'" in msg
    # placing the file by hand (as the message says) works offline
    pinned_tiny["cache"].mkdir(parents=True)
    shutil.copy(TINY, pinned_tiny["cache"] / "test-00000-of-00001.parquet")
    assert len(g8.load()[0]) == 3


def test_download_failure_and_wrong_hash_are_errors(pinned_tiny, monkeypatch):
    pinned_tiny["served"].unlink()
    with pytest.raises(OpError, match="download failed"):
        g8.load()
    pinned_tiny["served"].write_bytes(b"PAR1 tampered PAR1")
    with pytest.raises(OpError, match="is not the pinned"):
        g8.load()
    assert not (pinned_tiny["cache"] / "test-00000-of-00001.parquet").exists()  # bad file not kept
    # a cached file with the wrong hash is replaced by a fresh download
    shutil.copy(TINY, pinned_tiny["served"])
    (pinned_tiny["cache"] / "test-00000-of-00001.parquet").write_bytes(b"stale")
    assert len(g8.load()[0]) == 3


def test_data_argument_takes_jsonl_or_parquet(tmp_path):
    jl = tmp_path / "mine.jsonl"
    jl.write_bytes(TINY_JSONL)
    assert g8.load(jl) == g8.load(TINY)
    with pytest.raises(OpError, match="not found"):
        g8.load(tmp_path / "missing.jsonl")


def test_malformed_data_files_are_clear_errors(tmp_path):
    """A damaged --data file is an OpError naming the file, never a struct/zlib/KeyError traceback."""
    import random

    good = TINY.read_bytes()
    cases = {"bad_footer.parquet": good[:-8] + (len(good) * 4).to_bytes(4, "little") + b"PAR1",  # IndexError inside
             "short.parquet": b"PAR1" + bytes(5) + b"PAR1"}
    for name, data in cases.items():
        f = tmp_path / name
        f.write_bytes(data)
        with pytest.raises(OpError, match=name):
            g8.load(f)
    # Any corruption either still parses (Parquet has no checksum) or raises ParquetError: nothing else escapes.
    rng = random.Random(1)
    for _ in range(600):
        data = bytearray(good)
        for _ in range(rng.randint(1, 4)):
            data[rng.randrange(4, len(data) - 4)] = rng.randrange(256)
        try:
            read_table(bytes(data))
        except ParquetError:
            pass
    for cut in range(12, len(good), 11):
        with pytest.raises(ParquetError):
            read_table(good[:cut] + good[-8:])
    bad_rows = {"not_json.jsonl": b'{"question": "q", "answer": "#### 1"}\n{not json\n',
                "no_answer.jsonl": b'{"question": "q"}\n',
                "not_object.jsonl": b'[1, 2]\n',
                "latin1.jsonl": '{"question": "caf\u00e9", "answer": "#### 1"}\n'.encode("latin-1")}
    for name, data in bad_rows.items():
        f = tmp_path / name
        f.write_bytes(data)
        with pytest.raises(OpError, match=name):
            g8.load(f)


@pytest.mark.skipif(not os.environ.get("INFERENCEKIT_NETWORK_TESTS"), reason="set INFERENCEKIT_NETWORK_TESTS=1")
def test_real_pinned_split_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_CACHE_DIR", str(tmp_path))
    items, sha = g8.load()
    assert len(items) == g8.N_TEST and sha == g8.JSONL_SHA256 and items[0]["gold"] == "18"
