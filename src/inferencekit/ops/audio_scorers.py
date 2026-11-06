"""CLI/MCP operations for local speech scorer descriptors and execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from inferencekit import evidence as ev
from inferencekit.ops.bench_audio import load_fixtures
from inferencekit.ops.quality_audio import _source_compatibility
from inferencekit.quality import audio as qa, audio_scorers as scorers
from inferencekit.registry import Arg, OpError, operation


@operation("quality_audio_scorers", ("quality", "audio-scorers"),
           "Describe actual local CPU scorer packages and checkpoint hashes before policy registration.",
           [Arg("config", "string", "Local scorer configuration JSON, including optional Python executable.", required=True, positional=True)])
def quality_audio_scorers(call, config):
    return scorers.describe(scorers.load_config(config))


@operation("quality_audio_score", ("quality", "audio-score"),
           "Run local offline CPU ASR, UTMOSv2 and ECAPA scorers on retained speech PCM; never human listening.",
           [Arg("baseline", "string", "Baseline bench audio evidence ID or path.", required=True),
            Arg("candidate", "string", "Candidate bench audio evidence ID or path.", required=True),
            Arg("fixtures", "string", "Registered speech fixture JSONL.", required=True),
            Arg("protocol", "string", "Registered quality audio-policy evidence ID or path.", required=True),
            Arg("config", "string", "Local scorer configuration JSON.", required=True),
            Arg("out", "string", "New hash-bound audio scores JSON; refuses overwrite.", required=True)])
def quality_audio_score(call, baseline, candidate, fixtures, protocol, config, out):
    output = Path(out)
    if output.exists() or output.is_symlink():
        raise OpError("audio scores output already exists; choose a new file")
    settings = scorers.load_config(config)
    store = ev.Store()
    registered, root = qa.load_record(store, protocol, "quality_audio")
    if registered["data"].get("format") != qa.POLICY_FORMAT:
        raise OpError("protocol must be a quality audio-policy record")
    qa.artifact_bytes(root, qa.one_artifact(registered, "audio_quality_policy"))
    policy = qa.validate_policy(registered["data"]["policy"])
    fixture_rows, fixture_sha = load_fixtures(fixtures)
    if fixture_sha != registered["data"]["fixture_sha256"]:
        raise OpError("audio fixtures differ from the registered policy")
    a, arows = qa.load_benchmark(store, baseline)
    b, brows = qa.load_benchmark(store, candidate)
    _source_compatibility(a, b, policy)
    if a["data"]["fixture_set"]["sha256"] != fixture_sha or set(arows) != set(brows):
        raise OpError("audio benchmark fixtures or paired requests differ")
    roles = set(settings["scorers"])
    expected = set(policy["scorers"]) - {"listening"}
    if roles != expected:
        raise OpError("configured scorer roles must match every registered model scorer")
    if ("asr" in roles and not {"wer", "cer"}.intersection(policy["metrics"])) or any(
            role not in policy["metrics"] for role in roles & {"naturalness", "speaker"}):
        raise OpError("scorers require corresponding registered quality metrics")
    descriptors = scorers.describe(settings)
    if descriptors["scorers"] != {k: v for k, v in policy["scorers"].items() if k != "listening"}:
        raise OpError("actual scorer packages, checkpoints or settings differ from the registered policy")
    references = {f.id: f for f in fixture_rows}
    with tempfile.TemporaryDirectory(prefix="inferencekit-audio-inputs-") as directory:
        temporary = Path(directory)
        jobs = []
        for arm, rows in (("baseline", arows), ("candidate", brows)):
            for (fid, round_index), row in sorted(rows.items()):
                if not row["ok"] or row["pcm"] is None:
                    raise OpError("audio scoring requires complete successful saved PCM")
                pcm = temporary / f"{len(jobs)}.pcm"
                pcm.write_bytes(row["pcm"])
                job = {key: row[key] for key in ("seed", "pcm_sha256", "text_sha256", "sample_rate", "channels", "sample_count", "reference_sha256")}
                job.update(arm=arm, id=fid, round=round_index, pcm_path=str(pcm))
                fixture = references[fid]
                if fixture.reference_audio is not None and "speaker" in roles:
                    reference = temporary / f"reference-{fixture.reference_sha256}"
                    if not reference.exists():
                        reference.write_bytes(fixture.reference_audio)
                    job["reference_path"] = str(reference)
                jobs.append(job)
        result = scorers.run_worker(settings, "score", {"descriptors": descriptors, "jobs": jobs})
        if not isinstance(result, dict) or set(result) != {"descriptors", "rows"} or result["descriptors"] != descriptors:
            raise OpError("audio scoring returned mismatched scorer identities")
        rows = result["rows"]
        if not isinstance(rows, list) or len(rows) != len(jobs):
            raise OpError("audio scorer returned incomplete observations")
        required = ({"asr"} if "asr" in roles else set()) | ({"naturalness"} if "naturalness" in roles else set())
        for row in rows:
            if not isinstance(row, dict) or "error" in row or "listening" in row or not required.issubset(row):
                raise OpError("audio scorer returned failed or incomplete observations")
            if "asr_retry" in roles and (not isinstance(row.get("asr"), dict) or "retry_text" not in row["asr"]):
                raise OpError("audio scorer omitted registered ASR retry")
            if "speaker" in roles and references.get(row.get("id")) and references[row["id"]].reference_audio is not None and "speaker" not in row:
                raise OpError("audio scorer omitted fixture voice reference observation")
        doc = {"format": qa.SCORES_FORMAT, "protocol": registered["evidence_id"], "baseline": a["evidence_id"],
               "candidate": b["evidence_id"], "scorers": policy["scorers"], "rows": rows}
        validate_path = temporary / "scores.json"
        validate_path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        qa.load_scores(validate_path, registered["evidence_id"], a["evidence_id"], b["evidence_id"], policy,
                       {"baseline": arows, "candidate": brows}, {f.id: f.text for f in fixture_rows})
        # Publish a complete validated file using an exclusive hard link; neither
        # an existing target nor a concurrent writer can be overwritten.
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, staging_name = tempfile.mkstemp(prefix=".audio-scores-", dir=output.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                raw = validate_path.read_bytes()
                stream.write(raw)
            os.link(staging_name, output)
        except FileExistsError:
            raise OpError("audio scores output already exists; choose a new file") from None
        except OSError:
            raise OpError("cannot publish audio scores output") from None
        finally:
            Path(staging_name).unlink(missing_ok=True)
    return {"format": qa.SCORES_FORMAT, "sha256": qa.sha(raw), "rows": len(rows),
            "model_scorers": sorted(roles), "listening": "NOT RUN", "quality_verdict": "NOT RUN",
            "next": "Import this file with quality audio --scores; model proxies do not establish listening quality."}
