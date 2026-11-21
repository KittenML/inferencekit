"""Wrap saved PCM in a speech evidence record for registered quality scoring."""

from datetime import datetime
import json
from pathlib import Path
import wave

from inferencekit import evidence as ev
from inferencekit.canonical import canonical_json, sha256_hex
from inferencekit.ops.bench_audio import _ID, load_fixtures, round_seed
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve.audio_client import ENCODING, MAX_PCM_BYTES


def _read_pcm(path, sample_rate, channels):
    frame_bytes = channels * 2
    try:
        if path.suffix == ".wav":
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                if ((handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (sample_rate, channels, 2)
                        or frames * frame_bytes > MAX_PCM_BYTES):
                    raise ValueError
                pcm = handle.readframes(frames)
                if len(pcm) != frames * frame_bytes:
                    raise ValueError
        else:
            with path.open("rb") as handle:
                pcm = handle.read(MAX_PCM_BYTES + 1)
    except (OSError, EOFError, ValueError, wave.Error):
        raise OpError("unreadable or invalid PCM/WAV input") from None
    if not pcm or len(pcm) > MAX_PCM_BYTES or len(pcm) % frame_bytes or not any(pcm):
        raise OpError("saved PCM is empty, silent, oversized or incomplete")
    return pcm


@operation("import_audio", ("import", "audio"), "Import saved PCM/WAV for speech quality without timing claims.", (
    Arg("fixtures", "string", "Original speech fixture JSONL.", required=True, sensitive=True),
    Arg("dir", "string", "Directory of ID.wav/ID.pcm, or r0/, r1/, … for repeated captures.", required=True, sensitive=True),
    Arg("label", "string", "Configuration label.", required=True),
    Arg("model", "string", "Model identifier.", required=True),
    Arg("model_revision", "string", "Checkpoint revision.", required=True),
    Arg("engine_version", "string", "Runtime version or commit.", required=True),
    Arg("model_dtype", "string", "Checkpoint dtype."),
    Arg("model_quant", "string", "Quantization recipe."),
    Arg("settings", "string", "Declared generation settings JSON, including voice.", default="{}"),
    Arg("capture_start", "string", "Declared earliest capture start in UTC, in ISO 8601 with offset."),
    Arg("rounds", "integer", "Number of complete fixture passes.", default=1),
    Arg("seed_base", "integer", "Declared common bench audio round seed base.", default=0),
    Arg("sample_rate", "integer", "Sample rate of the imported PCM/WAV.", required=True),
    Arg("channels", "integer", "Number of audio channels.", default=1),
))
def import_audio(call, fixtures, dir, label, model, model_revision, engine_version, model_dtype, model_quant,
                 settings, capture_start, rounds, seed_base, sample_rate, channels):
    if not _ID.fullmatch(label) or not all(value.strip() for value in (model, model_revision, engine_version)):
        raise OpError("audio import needs a file-safe label and nonempty model and engine pins")
    if not 1 <= rounds <= 1000 or not 8000 <= sample_rate <= 192000 or not 1 <= channels <= 8:
        raise OpError("invalid audio import rounds, sample rate or channel count")
    try:
        settings = json.loads(settings)
        if (not isinstance(settings, dict) or ev.redact(settings) != settings
                or set(settings) & {"model", "seed", "input", "sample_rate", "channels", "encoding", "stream"}):
            raise ValueError
        canonical_json(settings)
        for key, value in (("model_revision", model_revision), ("engine_version", engine_version),
                           ("dtype", model_dtype), ("quant", model_quant)):
            if value is not None and settings.setdefault(key, value) != value:
                raise ValueError
        start = datetime.fromisoformat(capture_start.replace("Z", "+00:00")) if capture_start else None
        if start is not None and start.utcoffset() is None:
            raise ValueError
        if start is not None and start > datetime.fromisoformat(call.started.replace("Z", "+00:00")):
            raise ValueError
        timestamp = start.timestamp() if start else None
    except (ValueError, TypeError, OSError, OverflowError):
        raise OpError("audio import requires finite public settings and a timezone-aware capture start") from None
    items, fixture_sha = load_fixtures(fixtures)
    if ev.redact([label, model, model_revision, engine_version, model_dtype, model_quant]
                 + [item.id for item in items]) != (
                     [label, model, model_revision, engine_version, model_dtype, model_quant]
                     + [item.id for item in items]):
        raise OpError("audio capture identity fields must be safe to retain")
    root = Path(dir).resolve()
    store = ev.Store()
    with store.staging_dir() as staging:
        rows, outputs, artifacts, round_rows = [], [], [], []
        for index in range(rounds):
            source = root / f"r{index}" if rounds > 1 or (root / "r0").is_dir() else root
            seed = round_seed(seed_base, index)
            for fixture in items:
                candidates = [source / (fixture.id + ext) for ext in (".pcm", ".wav")]
                found = [path for path in candidates if path.is_file() and path.resolve().is_relative_to(root)]
                if len(found) != 1:
                    raise OpError(f"expected one PCM/WAV file for fixture {fixture.id} in round {index}")
                pcm = _read_pcm(found[0], sample_rate, channels)
                relative = f"outputs/{label}/r{index}/{fixture.id}.pcm"
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(pcm)
                sha = sha256_hex(pcm)
                rows.append({**fixture.identity(), "round": index, "seed": seed, "ok": True,
                             "sample_rate": sample_rate, "channels": channels, "sample_count": len(pcm) // (2 * channels),
                             "pcm_sha256": sha, "pcm_bytes": len(pcm)})
                outputs.append({"path": relative, "id": fixture.id, "round": index, "encoding": ENCODING,
                                "sample_rate": sample_rate, "channels": channels, "sha256": sha, "bytes": len(pcm)})
                artifacts.append(ev.artifact(target, staging, "audio_pcm"))
            round_rows.append({"index": index, "seed": seed, "t_start": timestamp, "n_failed": 0})
        row_path = staging / "requests.jsonl"
        row_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
        index_path = staging / "outputs" / label / "index.json"
        index_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf8")
        artifacts += [ev.artifact(row_path, staging, "requests"), ev.artifact(index_path, staging, "outputs_index")]
        data = {"source": "import audio", "label": label, "rounds": round_rows,
                "round_plan": {"expected_indices": list(range(rounds))}, "total_requests": len(rows),
                "structural_verdict": "PASS", "timing_verdict": "NOT RUN", "quality_verdict": "UNPROVEN",
                "fixture_set": {"sha256": fixture_sha, "fixtures": [item.identity() for item in items]},
                "audio_config": {"sample_rate": sample_rate, "channels": channels, "encoding": ENCODING,
                                 "settings": settings, "seed_base": seed_base}}
        record = ev.build(call, kind="audio_benchmark", claim_class="observed",
                          summary=f"{label}: imported {len(rows)} PCM outputs for quality",
                          identity_=ev.identity(engine={"name": label, "version": engine_version},
                                                model={"id": model, "revision": model_revision,
                                                       "dtype": model_dtype, "quant": model_quant},
                                                workload={"task": "tts", "endpoint_kind": "imported_pcm"}),
                          artifacts=artifacts, data=data, missing=["speech timing", "capture configuration verification"],
                          limitations=["Model, settings, seeds and capture times are declared by the importer.",
                                       "PCM structure is checked independently of synthesis provenance."])
        return store.save(record, staging)
