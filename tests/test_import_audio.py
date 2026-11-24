import wave

import pytest

from inferencekit import evidence as ev
from inferencekit.quality.audio import load_benchmark
from inferencekit.registry import OpError, call


def test_import_valid_pcm_and_wav_for_quality(tmp_path, store):
    fixture_file = tmp_path / "fixtures.jsonl"
    fixture_file.write_text('{"id":"line","text":"Good morning."}\n')
    params = {"fixtures": str(fixture_file), "dir": str(tmp_path), "label": "native", "model": "model",
              "model_revision": "rev", "engine_version": "engine", "sample_rate": 24000, "rounds": 2}
    pcm = b"\x00\x00\x01\x00\x01\x00"
    for i in range(2):
        (tmp_path / f"r{i}").mkdir()
    (tmp_path / "r0" / "line.pcm").write_bytes(pcm)
    with wave.open(str(tmp_path / "r1" / "line.wav"), "wb") as handle:
        handle.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        handle.writeframes(pcm)

    record = call("import_audio", params)
    assert ev.validate(record) == []
    assert record["data"]["timing_verdict"] == "NOT RUN"
    assert record["metrics"] == []
    loaded, rows = load_benchmark(store, record["evidence_id"])
    assert loaded["data"]["source"] == "import audio"
    assert len(rows) == 2
    assert all(item["pcm"] == pcm for item in rows.values())

    (tmp_path / "r0" / "line.pcm").write_bytes(pcm[:-1])
    with pytest.raises(OpError, match="incomplete"):
        call("import_audio", params)
