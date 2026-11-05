"""Native media run layouts remain reviewable without converting or changing evidence."""

import hashlib
import json
import wave

import pytest

from inferencekit import evidence as ev, registry
from inferencekit.registry import Arg, Call, Operation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value if isinstance(value, str) else json.dumps(value))


def record(store, root, label, kind, rows, verdicts=None):
    op = Operation(f"bench_{kind}", ("bench", kind), "t", (Arg("run_dir", "string", "d"),), lambda **kw: {})
    call = Call(op, {"run_dir": str(root)})
    stage = root.parent / f"stage-{label}"
    write(stage / "requests.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    rec = ev.build(call, kind=f"{kind}_benchmark", claim_class="observed", summary="screen only",
                   identity_=ev.identity(engine={"name": label}),
                   data={"source": f"bench {kind}", "label": label, **(verdicts or {})},
                   artifacts=[ev.artifact(stage / "requests.jsonl", stage, "requests")])
    rec = ev.finalize(rec)
    dest = store.root / rec["evidence_id"]
    write(dest / "evidence.json", rec)
    write(dest / "requests.jsonl", (stage / "requests.jsonl").read_bytes())
    return dest


def build(root, **kw):
    result = registry.call("build_review", {"run_dir": str(root), **kw})
    return result, (root / "review/index.html").read_text(), json.loads((root / "review/data.json").read_text())


def test_native_image_layout_and_failed_request(tmp_path, store):
    root = tmp_path / "images"
    for label in ("reference", "candidate"):
        for iid in ("photo.fox", "failed"):
            write(root / f"inputs/{label}/r0/{iid}.json",
                  {"prompt": "Fox in snow <script>bad</script>", "width": 1024, "height": 1024,
                   "steps": 50, "seed": 42, "guidance_scale": 3})
        write(root / f"outputs/{label}/r0/photo.fox.png", b"PNG display fixture")
        write(root / f"metrics/{label}.image.json", {"source": "bench image", "label": label,
              "rounds": [{"index": 0, "image_ms": None, "images_per_hour": None, "n_failed": 1}]})
        record(store, root, label, "image", [
            {"id": "photo.fox", "round": 0, "ok": True, "e2e_ms": 1234},
            {"id": "failed", "round": 0, "ok": False, "error": "decode failed"}])
    result, page, data = build(root)
    assert result["modality"] == "image" and result["inputs"] == 2
    assert result["engines"] == ["reference", "candidate"]  # no spurious '.image' engine
    assert result["failed"] == 2 and result["not_run"] == 0
    assert '../outputs/candidate/r0/photo.fox.png' in page and 'class="generated"' in page
    assert "e2e 1234 ms" in page and "decode failed" in page
    assert "semantic quality: UNPROVEN" in page and "<script>bad</script>" not in page
    assert "Fox in snow &lt;script&gt;bad&lt;/script&gt;" in page
    assert len(data["cells"]) == 4


def audio_run(root, store):
    label, iid = "speech", "short.01"
    pcm = b"\x01\x00\xff\xff" * 100
    rel = f"outputs/{label}/r0/{iid}.pcm"
    write(root / rel, pcm)
    index = [{"path": rel, "sha256": hashlib.sha256(pcm).hexdigest(), "bytes": len(pcm),
              "sample_rate": 44100, "channels": 1, "encoding": "pcm_s16le"}]
    write(root / f"outputs/{label}/index.json", index)
    write(root / f"metrics/{label}.audio.json", {"source": "bench audio", "label": label,
          "rounds": [{"index": 0, "rtf": .75}], "structural_verdict": "PASS", "timing_verdict": "FAIL",
          "fixture_set": {"fixtures": [{"id": iid}, {"id": "failed"}]}})
    dest = record(store, root, label, "audio", [
        {"id": iid, "round": 0, "ok": True, "first_pcm_ms": 100, "first_audio_ms": 130,
         "rtf": .75, "underruns": 1},
        {"id": "failed", "round": 0, "ok": False, "error": "HTTP 500"}])
    return pcm, index, dest


def test_pcm_playback_preserves_samples_and_labels_simulation(tmp_path, store):
    root = tmp_path / "audio"
    pcm, index, _ = audio_run(root, store)
    fixture = tmp_path / "fixtures.jsonl"
    write(fixture, json.dumps({"id": "short.01", "text": "Hello from the fixture."}) + "\n")
    result, page, data = build(root, prompts=str(fixture))
    assert result["modality"] == "tts" and result["inputs"] == 2
    assert result["engines"] == ["speech"] and result["failed"] == 1
    assert "Hello from the fixture." in page and "first PCM 100 ms" in page
    assert "simulated nonzero onset 130 ms" in page and "audio timing screen: FAIL" in page
    assert "semantic quality: UNPROVEN" in page and "HTTP 500" in page
    assert len(data["playback_copies"]) == 1
    copy = data["playback_copies"][0]
    with wave.open(str(root / copy["path"]), "rb") as wav:
        assert wav.getframerate() == 44100 and wav.getnchannels() == 1
        assert wav.readframes(wav.getnframes()) == pcm
    assert (root / index[0]["path"]).read_bytes() == pcm
    assert {x["path"] for x in data["sources"]} >= {index[0]["path"], "outputs/speech/index.json"}
    assert 'src="../review/media/' in page
    assert data["engines"][0]["speed"]["r0"]["RTF"]["claim_class"] == "derived"


@pytest.mark.parametrize("change", ["hash", "size", "rate", "encoding", "duplicate", "missing", "unaligned"])
def test_pcm_metadata_failure_does_not_play_guessed_audio(tmp_path, store, change):
    root = tmp_path / "audio"
    _, index, _ = audio_run(root, store)
    if change == "hash":
        index[0]["sha256"] = "0" * 64
    elif change == "size":
        index[0]["bytes"] += 2
    elif change == "rate":
        index[0]["sample_rate"] = True
    elif change == "encoding":
        index[0]["encoding"] = "pcm_f32le"
    elif change == "duplicate":
        index.append(dict(index[0]))
    elif change == "missing":
        index = []
    else:
        blob = b"123"
        write(root / index[0]["path"], blob)
        index[0].update(bytes=3, sha256=hashlib.sha256(blob).hexdigest())
    write(root / "outputs/speech/index.json", index)
    result, page, data = build(root)
    assert result["failed"] == 2 and "PCM playback unavailable" in page
    assert "<audio controls" not in page and data["playback_copies"] == []


def test_changed_request_artifact_is_not_shown_as_observed(tmp_path, store):
    root = tmp_path / "audio"
    _, _, dest = audio_run(root, store)
    write(dest / "requests.jsonl", json.dumps({"id": "short.01", "round": 0, "ok": True,
                                              "first_pcm_ms": 1}) + "\n")
    result, page, _ = build(root)
    assert "first PCM 1 ms" not in page
    assert any("does not match" in warning for warning in result["warnings"])


def test_all_failed_image_run_still_has_inputs_and_one_engine(tmp_path, store):
    root = tmp_path / "failed-images"
    write(root / "inputs/broken/r0/photo.fox.json", {"prompt": "Fox", "width": 1024, "height": 1024})
    write(root / "metrics/broken.image.json", {"source": "bench image", "rounds": [{"index": 0}]})
    record(store, root, "broken", "image", [{"id": "photo.fox", "round": 0, "ok": False, "error": "OOM"}])
    result, page, _ = build(root)
    assert result["modality"] == "image" and result["inputs"] == 1 and result["failed"] == 1
    assert result["engines"] == ["broken"] and "OOM" in page and "Fox" in page


def test_existing_temporary_symlink_cannot_redirect_pcm_write(tmp_path, store):
    root = tmp_path / "audio"
    _, index, _ = audio_run(root, store)
    sha = index[0]["sha256"]
    identity = hashlib.sha256(f"{sha}:44100:1".encode()).hexdigest()
    victim = tmp_path / "outside.txt"
    victim.write_text("unchanged")
    temporary = root / f"review/media/{identity}.wav.tmp"
    temporary.parent.mkdir(parents=True)
    temporary.symlink_to(victim)
    result, _, data = build(root)
    assert result["failed"] == 1 and len(data["playback_copies"]) == 1
    assert victim.read_text() == "unchanged" and temporary.is_symlink()


def test_external_playback_directory_is_rejected(tmp_path, store):
    from inferencekit.review.media import prepare_pcm
    from inferencekit.review.rundir import load_run

    root = tmp_path / "audio"
    audio_run(root, store)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "review").mkdir()
    (root / "review/media").symlink_to(outside, target_is_directory=True)
    run = load_run(root)
    assert prepare_pcm(run) == []
    assert run.cells[("speech", "r0/short.01")].status == "failed"
    assert not list(outside.iterdir())


def test_review_page_write_ignores_old_temporary_symlinks(tmp_path, store):
    root = tmp_path / "audio"
    audio_run(root, store)
    victim = tmp_path / "outside.txt"
    victim.write_text("unchanged")
    (root / "review").mkdir()
    (root / "review/index.html.tmp").symlink_to(victim)
    build(root)
    assert victim.read_text() == "unchanged"


def test_external_review_directory_is_rejected(tmp_path, store):
    from inferencekit.registry import OpError

    root = tmp_path / "audio"
    audio_run(root, store)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "review").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OpError, match="escapes"):
        build(root)
    assert not list(outside.iterdir())


def test_audio_evidence_without_summary_keeps_simulation_derived(tmp_path, store):
    root = tmp_path / "audio"
    audio_run(root, store)
    (root / "metrics/speech.audio.json").unlink()
    _, page, _ = build(root)
    chips = [s.split('</div>', 1)[0] for s in page.split('<div class="chips">')[1:]]
    simulated = next(c for c in chips if "simulated nonzero onset" in c)
    assert 'cc-d' in simulated and 'cc-o' not in simulated
    assert "RTF 0.75" in simulated and "underruns 1" in simulated
    assert "semantic quality: UNPROVEN" in page


@pytest.mark.parametrize("edited", [False, True])
def test_imported_audio_verdicts_do_not_require_metrics_summary(tmp_path, store, edited):
    root = tmp_path / "audio"
    write(root / "outputs/speech/r0/example.wav", b"display fixture")
    dest = record(store, root, "speech", "audio", [],
                  {"structural_verdict": "PASS", "timing_verdict": "FAIL", "quality_verdict": "PASS"})
    if edited:
        rec = json.loads((dest / "evidence.json").read_text())
        rec["data"]["timing_verdict"] = "PASS"
        write(dest / "evidence.json", rec)
    _, page, data = build(root)
    quality = {q["name"]: q["verdict"] for q in data["engines"][0]["quality"]}
    assert quality == {"semantic quality": "UNPROVEN", "PCM structure": "UNPROVEN" if edited else "PASS",
                       "audio timing screen": "UNPROVEN" if edited else "FAIL"}
    assert "semantic quality: UNPROVEN" in page
