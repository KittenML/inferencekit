"""Speech artifact validation and CPU-only metrics. No model downloads or execution.

Imported observations are attestations by their producer, bound to the exact PCM,
fixture text, scorer configuration and registered protocol. Hashes do not certify
the producer's methodology. Numerical parity and MOS predictors are not listening.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.registry import OpError

POLICY_FORMAT = "inferencekit.quality.audio.policy.v1"
SCORES_FORMAT = "inferencekit.quality.audio.scores.v1"
RESULT_FORMAT = "inferencekit.quality.audio.v1"
NORMALIZATIONS = ("whitespace_v1", "nfkc_casefold_punctuation_v1")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
        json.dumps(value, allow_nan=False)
        return value, raw
    except (OSError, ValueError, TypeError):
        raise OpError("cannot read finite audio quality JSON") from None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_policy(doc):
    if not isinstance(doc, dict) or set(doc) - {
            "acceptance", "min_fixtures", "min_rounds", "resamples", "seed", "allow_numerics_change",
            "metrics", "scorers", "normalization"}:
        raise OpError("invalid audio policy fields")
    out = {"min_fixtures": 2, "min_rounds": 1, "resamples": 2000, "seed": 0,
           "allow_numerics_change": False, "metrics": {}, "scorers": {},
           "normalization": "nfkc_casefold_punctuation_v1", **doc}
    if out.get("acceptance") not in ("exact", "task_noninferiority", "perceptual_and_task_noninferiority"):
        raise OpError("audio acceptance must be exact, task_noninferiority or perceptual_and_task_noninferiority")
    for key in ("min_fixtures", "min_rounds", "resamples", "seed"):
        if type(out[key]) is not int or out[key] < (0 if key == "seed" else 1):
            raise OpError(f"audio policy {key} must be a positive integer (seed may be zero)")
    if out["min_fixtures"] < 2 or out["resamples"] < 100:
        raise OpError("audio policy needs at least two fixtures and 100 bootstrap resamples")
    if type(out["allow_numerics_change"]) is not bool or out["normalization"] not in NORMALIZATIONS:
        raise OpError("invalid audio numerical-change or normalization policy")
    metrics, scorers = out["metrics"], out["scorers"]
    if not isinstance(metrics, dict) or set(metrics) - {"wer", "naturalness", "speaker", "listening"}:
        raise OpError("unknown audio quality metric")
    if not isinstance(scorers, dict) or set(scorers) - {"asr", "asr_retry", "naturalness", "speaker", "listening"}:
        raise OpError("unknown audio scorer")
    if out["acceptance"] != "exact" and not {"wer", "naturalness"}.issubset(metrics):
        raise OpError("noninferiority requires registered WER and naturalness margins")
    if out["acceptance"] == "perceptual_and_task_noninferiority" and "listening" not in metrics:
        raise OpError("perceptual acceptance requires a registered listening metric")
    for name, spec in metrics.items():
        allowed = {"margin", "outlier_wer"} if name == "wer" else {"margin", "min_raters"} if name == "listening" else {"margin"}
        if not isinstance(spec, dict) or set(spec) - allowed or not finite(spec.get("margin")) or spec["margin"] < 0:
            raise OpError(f"{name} requires a finite nonnegative margin")
        if name == "wer" and (not finite(spec.get("outlier_wer")) or spec["outlier_wer"] <= 0):
            raise OpError("WER requires a positive outlier_wer threshold, registered before decoding")
        if name == "listening" and (type(spec.get("min_raters")) is not int or spec["min_raters"] < 2):
            raise OpError("listening requires min_raters >= 2")
        scorer = "asr" if name == "wer" else name
        if scorer not in scorers:
            raise OpError(f"{name} requires a pinned {scorer} scorer")
    for name, spec in scorers.items():
        required = {"name", "version", "config", "protocol_sha256"} if name == "listening" else {
            "name", "version", "model", "revision", "config"}
        if not isinstance(spec, dict) or not required.issubset(spec) or set(spec) - (required | {"scale"}):
            raise OpError(f"{name} scorer requires name/version, model/revision (or listening protocol hash), and config")
        if any(not isinstance(spec[key], str) or not spec[key].strip() for key in required - {"config"}):
            raise OpError(f"{name} scorer pins must be nonempty strings")
        if not isinstance(spec["config"], dict) or ev.redact(spec) != spec:
            raise OpError("audio scorer config must be an object without secrets")
        if name == "asr_retry" and (spec["config"].get("vad_filter") is not True or
                                     spec["config"].get("condition_on_previous_text") is not False):
            raise OpError("ASR corroboration requires VAD on and condition_on_previous_text false")
        if name in ("naturalness", "speaker", "listening"):
            scale = spec.get("scale")
            if not isinstance(scale, list) or len(scale) != 2 or not all(finite(x) for x in scale) or scale[0] >= scale[1]:
                raise OpError(f"{name} scorer needs a finite increasing scale")
        if name == "listening" and (spec["config"].get("blinded") is not True or
                                     not re.fullmatch(r"[a-f0-9]{64}", spec["protocol_sha256"])):
            raise OpError("listening requires a blinded protocol and its SHA-256")
    return out


def artifact_bytes(root, item):
    """Read only a hash-checked, contained artifact; never trust a path from JSON."""
    try:
        path = (Path(root) / item["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or sha(raw) != item["sha256"]:
            raise ValueError
        return raw
    except (OSError, KeyError, TypeError, ValueError):
        raise OpError("audio evidence artifact missing, changed or outside its record") from None


def one_artifact(record, role):
    items = [a for a in record.get("artifacts", []) if a.get("role") == role]
    if len(items) != 1:
        raise OpError(f"audio evidence requires exactly one {role} artifact")
    return items[0]


def load_record(store, ref, kind):
    record = store.load(ref)
    verification = ev.verify(record)
    if record.get("kind") != kind or not verification["id_ok"] or not verification["schema_ok"]:
        raise OpError(f"audio quality requires intact {kind} evidence")
    p = Path(ref)
    root = p.parent if p.is_file() else p if p.is_dir() else store.run_dir(record["evidence_id"])
    if root.name == "evidence" and p.is_file():
        root = root.parent  # copied run-dir evidence/<id>.json uses run-relative artifacts
    return record, root


def load_benchmark(store, ref):
    record, root = load_record(store, ref, "audio_benchmark")
    data = record["data"]
    if data.get("source") != "bench audio":
        raise OpError("audio quality requires bench audio source evidence")
    try:
        rows = [json.loads(line) for line in artifact_bytes(root, one_artifact(record, "requests")).splitlines()
                if line.strip()]
        index = json.loads(artifact_bytes(root, one_artifact(record, "outputs_index")))
        if not isinstance(index, list):
            raise ValueError
        fixtures = {f["id"]: f for f in data["fixture_set"]["fixtures"]}
        rounds = {r["index"]: r for r in data["rounds"]}
        if len(fixtures) != len(data["fixture_set"]["fixtures"]) or len(rounds) != len(data["rounds"]):
            raise ValueError
        keyed = {}
        for row in rows:
            key = (row["id"], row["round"])
            if key in keyed or row["id"] not in fixtures or row["round"] not in rounds:
                raise ValueError
            if row["seed"] != rounds[row["round"]]["seed"] or any(row[k] != v for k, v in fixtures[row["id"]].items()):
                raise ValueError
            keyed[key] = {**row, "pcm": None}
        if set(keyed) != {(f, r) for f in fixtures for r in rounds} or len(rows) != data["total_requests"]:
            raise ValueError
        artifacts = {a["path"]: a for a in record["artifacts"] if a["role"] == "audio_pcm"}
        if len(artifacts) != sum(a["role"] == "audio_pcm" for a in record["artifacts"]):
            raise ValueError
        used = set()
        label = data["label"]
        for item in index:
            match = re.fullmatch(r"outputs/" + re.escape(label) + r"/r(\d+)/([^/]+)\.pcm", item["path"])
            if not match:
                raise ValueError
            key = (match[2], int(match[1]))
            if key not in keyed or key in used:
                raise ValueError
            row = keyed[key]
            if ("id" in item and item["id"] != key[0]) or ("round" in item and item["round"] != key[1]):
                raise ValueError
            if not row["ok"] or item["encoding"] != "pcm_s16le":
                raise ValueError
            if any(item[k] != row[k] for k in ("sample_rate", "channels")):
                raise ValueError
            if item["sha256"] != row["pcm_sha256"] or item["bytes"] != row["pcm_bytes"]:
                raise ValueError
            raw = artifact_bytes(root, artifacts[item["path"]])
            if sha(raw) != item["sha256"] or len(raw) != item["bytes"] or not raw or len(raw) != 2 * row["sample_count"] * row["channels"]:
                raise ValueError
            row["pcm"] = raw
            used.add(key)
        if set(artifacts) != {item["path"] for item in index}:
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise OpError("audio benchmark request/output identities are inconsistent") from None
    return record, keyed


def parity(baseline: bytes, candidate: bytes, channels: int):
    result = {"bit_exact": baseline == candidate,
              "same_sample_count": len(baseline) == len(candidate),
              "baseline_frames": len(baseline) // (2 * channels),
              "candidate_frames": len(candidate) // (2 * channels),
              "snr_db": None, "maxabs": None, "snr_status": "length_mismatch"}
    if len(baseline) != len(candidate):
        return result  # no alignment, truncation, resampling or padding can hide missing audio
    energy = error = maximum = 0
    for (a,), (b,) in zip(struct.iter_unpack("<h", baseline), struct.iter_unpack("<h", candidate)):
        diff = a - b
        energy += a * a
        error += diff * diff
        maximum = max(maximum, abs(diff))
    result["maxabs"] = maximum / 32768
    result["snr_status"] = "identical" if not error else "zero_reference_energy" if not energy else "finite"
    if energy and error:
        result["snr_db"] = 10 * math.log10(energy / error)
    return result


def words(text, normalization):
    if normalization == "nfkc_casefold_punctuation_v1":
        text = unicodedata.normalize("NFKC", text).casefold()
        text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return text.split()


def word_error(reference, hypothesis, normalization):
    """Word Levenshtein distance; inserted words can make WER exceed one."""
    ref, hyp = words(reference, normalization), words(hypothesis, normalization)
    if not ref:
        raise OpError("fixture has no words after the registered normalization")
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return {"errors": previous[-1], "words": len(ref), "wer": previous[-1] / len(ref),
            "hypothesis_words": len(hyp)}


def scaled(value, scorer):
    if not finite(value) or not scorer["scale"][0] <= value <= scorer["scale"][1]:
        raise OpError("audio imported score is nonfinite or outside its registered scale")
    return value


def window_score(value, frames, scorer):
    """Full-coverage, nonoverlapping windows in original PCM frame coordinates."""
    if not isinstance(value, dict) or set(value) != {"windows"} or not isinstance(value["windows"], list):
        raise OpError("naturalness requires scored windows")
    end, total = 0, 0.0
    for window in value["windows"]:
        if not isinstance(window, dict) or set(window) != {"start_frame", "end_frame", "score"}:
            raise OpError("invalid naturalness window")
        start, stop = window["start_frame"], window["end_frame"]
        if type(start) is not int or type(stop) is not int or start != end or stop <= start or stop > frames:
            raise OpError("naturalness windows must cover complete PCM once, in order")
        total += (stop - start) * scaled(window["score"], scorer)
        end = stop
    if end != frames:
        raise OpError("naturalness windows omit PCM frames")
    return total / frames


def load_scores(path, protocol_id, baseline_id, candidate_id, policy, sources, texts):
    if not path:
        return {}, None
    doc, raw = read_json(path)
    if not isinstance(doc, dict) or set(doc) != {"format", "protocol", "baseline", "candidate", "scorers", "rows"}:
        raise OpError("invalid audio score import envelope")
    if (doc["format"], doc["protocol"], doc["baseline"], doc["candidate"]) != (
            SCORES_FORMAT, protocol_id, baseline_id, candidate_id) or doc["scorers"] != policy["scorers"]:
        raise OpError("audio scores do not match the protocol, source evidence or pinned scorers")
    if not isinstance(doc["rows"], list):
        raise OpError("audio scores rows must be a list")
    result = {}
    for item in doc["rows"]:
        if not isinstance(item, dict) or set(item) - {"arm", "id", "round", "seed", "pcm_sha256", "text_sha256",
                                                    "asr", "naturalness", "speaker", "listening", "error"}:
            raise OpError("invalid imported audio observation")
        try:
            key = (item["arm"], item["id"], item["round"])
            row = sources[item["arm"]][(item["id"], item["round"])]
            if key in result or not row["ok"] or row["pcm"] is None or any(
                    item[k] != row[k] for k in ("seed", "pcm_sha256", "text_sha256")):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise OpError("imported audio observation has a duplicate, unknown or mismatched PCM/text identity") from None
        values = {}
        if "asr" in item:
            asr = item["asr"]
            if "wer" not in policy["metrics"] or not isinstance(asr, dict) or set(asr) - {"text", "retry_text"}:
                raise OpError("invalid ASR observation")
            for source, dest in (("text", "wer"), ("retry_text", "wer_retry")):
                if source == "text" or source in asr:
                    if not isinstance(asr.get(source), str) or (source == "retry_text" and "asr_retry" not in policy["scorers"]):
                        raise OpError("ASR text and a registered retry scorer are required")
                    values[dest] = word_error(texts[item["id"]], asr[source], policy["normalization"])
        if "naturalness" in item:
            if "naturalness" not in policy["metrics"]:
                raise OpError("unregistered naturalness metric")
            values["naturalness"] = window_score(item["naturalness"], row["sample_count"], policy["scorers"]["naturalness"])
        if "speaker" in item:
            speaker = item["speaker"]
            if "speaker" not in policy["metrics"] or not isinstance(speaker, dict) or set(speaker) != {"score", "reference_sha256"}:
                raise OpError("invalid speaker observation")
            if not row["reference_sha256"] or speaker["reference_sha256"] != row["reference_sha256"]:
                raise OpError("speaker score must bind the fixture's voice reference")
            values["speaker"] = scaled(speaker["score"], policy["scorers"]["speaker"])
        if "listening" in item:
            listen = item["listening"]
            if "listening" not in policy["metrics"] or not isinstance(listen, dict) or set(listen) != {"ratings"} or not isinstance(listen["ratings"], list):
                raise OpError("invalid listening observations")
            ratings = {}
            for rating in listen["ratings"]:
                if not isinstance(rating, dict) or set(rating) != {"rater", "score"} or not isinstance(rating["rater"], str) or not rating["rater"] or rating["rater"] in ratings:
                    raise OpError("listening needs distinct anonymous rater IDs")
                ratings[rating["rater"]] = scaled(rating["score"], policy["scorers"]["listening"])
            values["listening"] = ratings
        result[key] = values
    return result, raw
