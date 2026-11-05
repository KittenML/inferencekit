"""Display saved image and streamed PCM runs without re-measuring or judging quality."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import wave
from pathlib import Path

from inferencekit.canonical import file_sha256

AUDIO_KEYS = ("first_pcm_ms", "first_signal_ms", "first_audio_ms", "complete_audio_ms", "rtf",
              "audio_duration_s", "max_stream_gap_ms", "underruns", "underrun_ms")
DERIVED_AUDIO = {"first_audio_ms", "rtf", "audio_duration_s", "underruns", "underrun_ms"}


def attach_verdicts(info, kind, data, evidence=None, problem=False):
    """Keep imported media screens visible even without a separate metrics summary."""
    fields = [("semantic quality", None)]
    if kind == "audio":
        fields += [("PCM structure", "structural_verdict"), ("audio timing screen", "timing_verdict")]
    for name, key in fields:
        prior = [row for row in info.ev_quality if row.get("name") == name]
        # Older records may omit these fields; they cannot replace an available summary.
        if prior and (not evidence or (key and key not in data and not problem)):
            continue
        verdict = data.get(key, "UNPROVEN") if key and not problem else "UNPROVEN"
        if verdict not in ("PASS", "FAIL", "UNPROVEN", "NOT RUN"):
            verdict = "UNPROVEN"
        row = {"name": name, "verdict": verdict}
        if evidence:
            row["evidence"] = evidence
            info.ev_quality[:] = [old for old in info.ev_quality
                                  if old.get("name") != name or old.get("evidence")]
        info.ev_quality.append(row)


def load_media_metrics(root, label, info, run, items):
    from inferencekit.review.rundir import Item

    for kind in ("image", "audio"):
        path = root / "metrics" / f"{label}.{kind}.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("source") != f"bench {kind}":
                raise ValueError("not a media benchmark summary")
        except (OSError, ValueError) as exc:
            run.warnings.append(f"{path.name}: unreadable ({exc})")
            continue
        info.media_kind = kind
        run.sources.append(path.relative_to(root).as_posix())
        info.metrics_file = info.metrics_file or path.relative_to(root).as_posix()
        attach_verdicts(info, kind, data)
        if kind == "audio":
            # This is the benchmark's structural/timing screen, never a semantic or full gates pass.
            info.metric_classes.update({k: "derived" for k in DERIVED_AUDIO})
        rounds = data.get("rounds")
        for row in rounds if isinstance(rounds, list) else []:
            if not isinstance(row, dict) or type(row.get("index")) is not int:
                continue
            bucket = f"r{row['index']}"
            keys = ("image_ms", "images_per_hour") if kind == "image" else ("rtf",)
            info.per_bucket.setdefault(bucket, {}).update({k: row[k] for k in keys if row.get(k) is not None})
            if kind == "audio":
                fixture_set = data.get("fixture_set")
                fixtures = fixture_set.get("fixtures") if isinstance(fixture_set, dict) else None
                for fixture in fixtures if isinstance(fixtures, list) else []:
                    if isinstance(fixture, dict) and isinstance(fixture.get("id"), str):
                        iid = fixture["id"]
                        items.setdefault(f"{bucket}/{iid}", Item(id=iid, bucket=bucket))
        if kind == "image":
            info.metric_classes["images_per_hour"] = "derived"


def attach_rows(run, info, rec, source):
    from inferencekit.review.rundir import artifact_lines

    data = rec.get("data") or {}
    kind = "image" if rec.get("kind") == "image_benchmark" else "audio"
    if data.get("source") != f"bench {kind}":
        return
    info.media_kind = kind
    if kind == "audio":
        info.metric_classes.update({k: "derived" for k in DERIVED_AUDIO})
    keys = ("e2e_ms",) if kind == "image" else AUDIO_KEYS
    for row in artifact_lines(run, rec, source, "requests", "per-request media metrics") or []:
        if not isinstance(row.get("id"), str) or type(row.get("round")) is not int:
            continue
        key = f"r{row['round']}/{row['id']}"
        filtered = {k: row[k] for k in ("ok", "error", *keys) if k in row}
        old = info.media_rows.get(key)
        if old is not None and old != filtered:
            info.media_rows[key] = {"ok": False, "error": "conflicting benchmark rows for this output"}
        else:
            info.media_rows[key] = filtered


def merge_rows(run):
    for info in run.engines:
        for item in run.items:
            cell = run.cells[(info.label, run.key(item))]
            row = info.media_rows.get(run.key(item), {})
            cell.metrics.update({k: row[k] for k in ("e2e_ms", *AUDIO_KEYS) if k in row})
            if row.get("ok") is False:
                cell.status, cell.reason = "failed", str(row.get("error") or "media request failed")


def prepare_pcm(run):
    """Wrap verified pcm_s16le bytes in WAV under review/media; never edit source outputs.

    A WAV is a playback copy, not a measurement or quality proof. The original index and
    PCM are included in the page's source manifest. Missing or mismatched metadata fails
    the cell rather than guessing a sample rate and playing incorrect audio.
    """
    derived = []
    indexes = {}
    for cell in run.cells.values():
        if not cell.pcm or cell.status == "failed":
            continue
        try:
            index = run.root / "outputs" / cell.engine / "index.json"
            if cell.engine not in indexes:
                obj = json.loads(index.read_text(encoding="utf-8"))
                if not isinstance(obj, list):
                    raise ValueError("PCM index must be a list")
                indexes[cell.engine] = obj
                run.sources.append(index.relative_to(run.root).as_posix())
            matches = [r for r in indexes[cell.engine] if isinstance(r, dict) and r.get("path") == cell.pcm]
            if len(matches) != 1:
                raise ValueError("PCM output needs exactly one index entry")
            meta = matches[0]
            rate, channels = meta.get("sample_rate"), meta.get("channels")
            if (meta.get("encoding") != "pcm_s16le" or type(rate) is not int or not 8000 <= rate <= 192000
                    or type(channels) is not int or not 1 <= channels <= 8):
                raise ValueError("invalid PCM encoding, sample rate or channel count")
            source = run.root / cell.pcm
            if not source.resolve().is_relative_to(run.root.resolve()):
                raise ValueError("PCM output escapes the run directory")
            sha, size = file_sha256(source)
            if (sha != meta.get("sha256") or type(meta.get("bytes")) is not int or size != meta["bytes"]
                    or size == 0 or size % (2 * channels) or size > 0xffffffff - 36):
                raise ValueError("PCM bytes do not match the index or are not frame-aligned")
            identity = hashlib.sha256(f"{sha}:{rate}:{channels}".encode()).hexdigest()
            relative = f"review/media/{identity}.wav"
            dest = run.root / relative
            if not dest.parent.resolve().is_relative_to(run.root.resolve()):
                raise ValueError("playback directory escapes the run directory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = None
            try:
                digest = hashlib.sha256()
                # Exclusive creation prevents an existing .tmp symlink from redirecting writes.
                with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".pcm-", suffix=".wav.tmp",
                                                 delete=False) as temporary:
                    temp = Path(temporary.name)
                    with source.open("rb") as src, wave.open(temporary, "wb") as wav:
                        wav.setnchannels(channels)
                        wav.setsampwidth(2)
                        wav.setframerate(rate)
                        while block := src.read(1024 * 1024):
                            digest.update(block)
                            wav.writeframesraw(block)
                if digest.hexdigest() != sha:
                    raise ValueError("PCM changed while preparing playback")
                os.replace(temp, dest)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
            cell.audio = relative
            out_sha, out_bytes = file_sha256(dest)
            derived.append({"path": relative, "sha256": out_sha, "bytes": out_bytes,
                            "source": cell.pcm, "source_sha256": sha,
                            "transformation": "WAV header around unchanged pcm_s16le bytes"})
        except (OSError, ValueError, wave.Error) as exc:
            cell.status, cell.reason = "failed", f"PCM playback unavailable: {exc}"
    run.sources = sorted(set(run.sources))
    return derived
