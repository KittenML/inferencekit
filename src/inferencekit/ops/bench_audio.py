"""Speech screening: client-timed PCM, structural validation, never an audio quality proof.

Fixture JSONL rows: {id, text, reference_audio?: relative file, reference_text?: str,
unseen?: bool}. The unseen flag is an attestation, not independently verified.
One process serves all rounds; final comparative claims need a separate fresh-
process interleaved scheduler. Warmup requests are excluded from all metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from inferencekit import evidence as ev, stats
from inferencekit.canonical import canonical_json
from inferencekit.ops.bench_serve import _record_call
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve.audio_client import AudioHTTP, AudioRequest, ENCODING, playback_metrics
from inferencekit.serve.client import Endpoint, api_key_from_env
from inferencekit.serve.metrics import pct

KIND = "audio_benchmark"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")
ARGS = [
    Arg("base_url", "string", "Speech server root; stored only as a keyed endpoint tag.", required=True, sensitive=True),
    Arg("model", "string", "Pinned model identifier sent to the server.", required=True),
    Arg("model_revision", "string", "Checkpoint revision, sent and required in final effective settings."),
    Arg("model_dtype", "string", "Checkpoint dtype, sent as dtype and checked in effective settings."),
    Arg("model_quant", "string", "Complete quantization recipe, sent as quant and checked in effective settings."),
    Arg("fixtures", "string", "JSONL rows: id, text, reference_audio?, reference_text?, unseen?.", required=True),
    Arg("label", "string", "Configuration label for artifacts.", default="audio"),
    Arg("rounds", "integer", "Timed passes; screening in one server process.", default=3),
    Arg("first_round", "integer", "First round index; default zero or previous maximum plus one."),
    Arg("warmup", "integer", "Untimed distinct-text requests before measurement.", default=1),
    Arg("seed_base", "integer", "Shared deterministic round-seed base.", default=0),
    Arg("sample_rate", "integer", "Required decoded PCM sample rate.", default=44100),
    Arg("channels", "integer", "Required PCM channels.", default=1),
    Arg("settings", "string", "JSON object of sampling/streaming settings; final must report them.", default="{}"),
    Arg("api_key_env", "string", "Environment variable containing the key; never a key value."),
    Arg("engine_version", "string", "Pinned engine version, sent and checked in effective settings."),
    Arg("hardware_name", "string", "Measured hardware label; never inferred from a remote client."),
    Arg("run_dir", "string", "Also copy request/output artifacts and metrics here."),
    Arg("timeout_s", "number", "Whole-response deadline including DNS, connect, headers and streamed body.", default=300.0),
    Arg("resamples", "integer", "Round-cluster bootstrap resamples.", default=stats.DEFAULT_RESAMPLES),
    Arg("save_outputs", "string", "Keep decoded PCM or hashes only.", default="all", choices=("all", "hash_only")),
    Arg("slo_first_audio_p95_ms", "number", "p95 simulated nonzero-signal onset limit.", default=2000.0),
    Arg("slo_rtf", "number", "Maximum complete-generation wall seconds per audio second.", default=1.0),
    Arg("extend", "string", "Append rounds to the same configuration's earlier evidence record."),
]


@dataclass
class AudioFixture:
    id: str
    text: str
    text_sha256: str
    reference_audio: bytes | None
    reference_sha256: str | None
    reference_text: str | None
    reference_text_sha256: str | None
    declared_unseen: bool

    def identity(self):
        return {"id": self.id, "text_sha256": self.text_sha256,
                "reference_sha256": self.reference_sha256,
                "reference_text_sha256": self.reference_text_sha256, "declared_unseen": self.declared_unseen}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_fixtures(path) -> tuple[list[AudioFixture], str]:
    p = Path(path)
    try:
        raw = p.read_bytes()
        docs = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        raise OpError("cannot read speech fixture JSONL") from None
    fixtures, ids = [], set()
    for n, doc in enumerate(docs, 1):
        if not isinstance(doc, dict) or not isinstance(doc.get("id"), str) or not _ID.fullmatch(doc["id"]):
            raise OpError(f"speech fixture row {n}: invalid id")
        if doc["id"] in ids:
            raise OpError(f"speech fixture row {n}: duplicate id")
        ids.add(doc["id"])
        if not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise OpError(f"speech fixture row {n}: nonempty text required")
        if set(doc) - {"id", "text", "reference_audio", "reference_text", "unseen"}:
            raise OpError(f"speech fixture row {n}: unknown field")
        if "unseen" in doc and type(doc["unseen"]) is not bool:
            raise OpError(f"speech fixture row {n}: unseen must be boolean")
        reference, reference_text = None, doc.get("reference_text")
        if "reference_audio" in doc:
            if not isinstance(doc["reference_audio"], str):
                raise OpError(f"speech fixture row {n}: reference_audio must be a file path")
            if not isinstance(reference_text, str) or not reference_text.strip():
                raise OpError(f"speech fixture row {n}: reference audio needs its transcript")
            try:
                reference = (p.parent / doc["reference_audio"]).read_bytes()
            except OSError:
                raise OpError(f"speech fixture row {n}: reference audio unreadable") from None
            if not reference:
                raise OpError(f"speech fixture row {n}: reference audio empty")
        elif reference_text is not None:
            raise OpError(f"speech fixture row {n}: transcript requires reference audio")
        fixtures.append(AudioFixture(doc["id"], doc["text"], _sha(doc["text"].encode()), reference,
                                     _sha(reference) if reference is not None else None, reference_text,
                                     _sha(reference_text.encode()) if reference_text is not None else None,
                                     doc.get("unseen", False)))
    if not fixtures:
        raise OpError("speech fixture set is empty")
    # Content identity includes the reference bytes, independent of local filenames.
    return fixtures, _sha(canonical_json([f.identity() for f in fixtures]).encode())


def round_seed(seed_base: int, round_index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"inferencekit-audio:{seed_base}:{round_index}".encode()).digest()[:4],
                          "big") & 0x7fffffff


def audio_row(result, fixture, round_index, seed):
    duration = result.audio_duration_s
    complete = result.complete_audio_ms
    return {**fixture.identity(), "round": round_index, "seed": seed, "ok": result.ok,
            "error": result.error, "status": result.status, "t_send_ns": result.t_send_ns,
            "t_done_ns": result.t_done_ns, "t_end_ns": result.t_end_ns,
            "first_pcm_ms": result.first_audio_ms, "first_signal_ms": result.first_signal_ms,
            "first_audio_ms": result.first_audible_ms, "complete_audio_ms": complete,
            "audio_duration_s": duration, "rtf": complete / 1000 / duration if duration and complete is not None else None,
            "leading_silence_ms": result.leading_silent_frames / result.sample_rate * 1000
            if result.ok else None, "sample_rate": result.sample_rate, "channels": result.channels,
            "sample_count": result.sample_count if result.ok else None,
            "received_pcm_frames": sum(count for _, count in result.chunks),
            "pcm_bytes": len(result.pcm) if result.ok else None,
            "pcm_sha256": _sha(result.pcm) if result.ok else None, "effective": result.effective,
            "chunks": [{"arrival_ms": (t - result.t_send_ns) / 1e6, "sample_count": count}
                       for t, count in result.chunks],
            **(playback_metrics(result.chunks, result.sample_rate) if result.ok else
               {"underruns": None, "underrun_ms": None, "max_stream_gap_ms": None})}


def audio_metrics(rows: list[dict], resamples: int) -> tuple[list[dict], dict]:
    metrics, table = [], {}
    for key, unit in (("first_audio_ms", "ms"), ("first_pcm_ms", "ms"), ("first_signal_ms", "ms"),
                      ("complete_audio_ms", "ms"), ("rtf", "ratio"), ("max_stream_gap_ms", "ms")):
        valid = [row for row in rows if row["ok"] and row.get(key) is not None]
        values, units = [row[key] for row in valid], [row["round"] for row in valid]
        cis = stats.bootstrap_ci(values, units=units, ps=(50, 95), resamples=resamples)
        table[key] = {"n": len(values), "median": pct(values, 50), "p95": pct(values, 95),
                      "max": max(values, default=None), "ci95_median": list(cis[50]), "ci95_p95": list(cis[95])}
        for p, stat in ((50, "median"), (95, "p95")):
            metrics.append(ev.metric(key, pct(values, p), unit, stat, n=len(values), ci95=cis[p]))
    return metrics, table


def _write(roots, relative: str, data: bytes):
    for root in roots:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _previous(store, reference, config_sha, call):
    if not reference:
        return None, [], []
    record = store.load(reference)
    if record.get("kind") != KIND or record.get("data", {}).get("source") != "bench audio":
        raise OpError("--extend requires a bench audio record")
    if ev.integrity_problem(record):
        raise OpError("--extend evidence integrity check failed")
    if record["data"].get("configuration_sha256") != config_sha:
        raise OpError("--extend requires the same workload, model, version, hardware and seed policy")
    old_args = record["data"].get("request", {}).get("args", {})
    if old_args.get("base_url") != call.request()["args"].get("base_url"):
        raise OpError("--extend requires the same endpoint")
    artifact = next((a for a in record["artifacts"] if a.get("role") == "requests"), None)
    if artifact is None:
        raise OpError("--extend per-request artifact missing")
    previous_root = store.run_dir(record["evidence_id"])
    path = previous_root / artifact["path"]
    if not path.resolve().is_relative_to(previous_root.resolve()):
        raise OpError("--extend per-request artifact escapes its record")
    try:
        raw = path.read_bytes()
        if _sha(raw) != artifact["sha256"]:
            raise ValueError
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    except (OSError, ValueError):
        raise OpError("--extend per-request artifact missing or changed") from None
    return record, rows, record["data"]["rounds"]


@operation("bench_audio", ("bench", "audio"), "Client-timed streamed PCM speech screening; structural gates and "
           "first-audio p95, complete audio, RTF and buffer underruns. Audio quality remains UNPROVEN.", ARGS)
def bench_audio(call, base_url, model, model_revision, model_dtype, model_quant, fixtures, label, rounds, first_round, warmup, seed_base, sample_rate,
                channels, settings, api_key_env, engine_version, hardware_name, run_dir, timeout_s,
                resamples, save_outputs, slo_first_audio_p95_ms, slo_rtf, extend):
    if rounds < 1 or (first_round is not None and first_round < 0) or warmup < 0 or resamples < 1:
        raise OpError("rounds/resamples must be positive; first_round/warmup nonnegative")
    if not _ID.fullmatch(label):
        raise OpError("invalid audio label")
    if not model.strip():
        raise OpError("audio model identifier must be nonempty")
    if not 8000 <= sample_rate <= 192000 or not 1 <= channels <= 8:
        raise OpError("PCM rate must be 8000..192000 Hz and channels 1..8")
    if any(not math.isfinite(v) or v <= 0 for v in (timeout_s, slo_first_audio_p95_ms, slo_rtf)):
        raise OpError("timeout and SLO limits must be finite and positive")
    try:
        setting_doc = json.loads(settings)
        if not isinstance(setting_doc, dict):
            raise ValueError
        json.dumps(setting_doc, allow_nan=False)
        if set(setting_doc) & {"model", "seed", "input", "sample_rate", "channels", "encoding", "stream"}:
            raise ValueError
        if ev.redact(setting_doc) != setting_doc:
            raise ValueError
    except ValueError:
        raise OpError("settings must be a finite JSON object without reserved request fields") from None
    pins = {key: value for key, value in (("model_revision", model_revision), ("dtype", model_dtype),
                                        ("quant", model_quant), ("engine_version", engine_version)) if value is not None}
    for key, value in pins.items():
        if key in setting_doc and setting_doc[key] != value:
            raise OpError(f"settings conflict with declared {key}")
        setting_doc[key] = value
    fixture_rows, fixture_sha = load_fixtures(fixtures)
    try:
        parsed_url = urlsplit(base_url if "://" in base_url else "http://" + base_url)
        if parsed_url.username is not None or parsed_url.password is not None or parsed_url.query or parsed_url.fragment:
            raise ValueError
        endpoint = Endpoint.parse(base_url, api_key_from_env(api_key_env))
    except (ValueError, OpError):
        raise OpError("invalid speech endpoint or API-key environment configuration") from None
    record_call, persistent = _record_call(call, endpoint)
    backend, store = AudioHTTP(endpoint, timeout_s=timeout_s), ev.Store()
    run = Path(run_dir) if run_dir else None
    workload_config = {"fixtures": fixture_sha, "model": model, "revision": model_revision,
                       "dtype": model_dtype, "quant": model_quant, "sample_rate": sample_rate,
                       "channels": channels, "encoding": ENCODING,
                       "settings": {key: value for key, value in setting_doc.items() if key != "engine_version"},
                       "seed_policy": {"name": "per_round", "base": seed_base}}
    request_sha = _sha(canonical_json(workload_config).encode())
    config_sha = _sha(canonical_json({**workload_config, "label": label, "hardware": hardware_name,
                                    "engine_version": engine_version, "effective_settings": setting_doc,
                                    "save_outputs": save_outputs,
                                    "slo_first_audio_p95_ms": slo_first_audio_p95_ms, "slo_rtf": slo_rtf}).encode())
    previous, previous_rows, previous_rounds = _previous(store, extend, config_sha, record_call)
    done = max((row["index"] for row in previous_rounds), default=-1)
    first_round = done + 1 if first_round is None else first_round
    if first_round <= done:
        raise OpError("--first-round conflicts with rounds already present in --extend")
    if previous is not None and first_round != done + 1:
        raise OpError("--extend --first-round must be the previous maximum plus one; skipped rounds are refused")

    def request(fixture, seed, text=None):
        return AudioRequest(fixture.id, fixture.text if text is None else text, model, seed, sample_rate,
                            channels, setting_doc, fixture.reference_audio, fixture.reference_text)

    try:
        with store.staging_dir() as staging:
            roots = [staging] + ([run] if run else [])
            warm_rows, rows, round_rows, outputs = [], list(previous_rows), list(previous_rounds), []
            if previous:
                previous_root = store.run_dir(previous["evidence_id"])
                for artifact in previous["artifacts"]:
                    if artifact.get("role") not in ("audio_pcm", "outputs_index"):
                        continue
                    src = previous_root / artifact["path"]
                    if not src.resolve().is_relative_to(previous_root.resolve()):
                        raise OpError("--extend artifact path escapes its record")
                    try:
                        blob = src.read_bytes()
                    except OSError:
                        raise OpError("--extend output artifact missing") from None
                    if _sha(blob) != artifact["sha256"]:
                        raise OpError("--extend output artifact changed")
                    if artifact["role"] == "outputs_index":
                        outputs = json.loads(blob)
                    else:
                        _write(roots, artifact["path"], blob)
            for i in range(warmup):
                fixture = fixture_rows[i % len(fixture_rows)]
                text = f"Inferencekit untimed speech warmup {i}. " + fixture.text
                while text in {f.text for f in fixture_rows}:
                    text = "Warmup. " + text
                result = backend.generate(request(fixture, round_seed(seed_base, -i - 1), text))
                warm_rows.append({"ok": result.ok, "error": result.error})
            for round_index in range(first_round, first_round + rounds):
                start = time.time()
                current_rows = []
                seed = round_seed(seed_base, round_index)
                for fixture in fixture_rows:
                    result = backend.generate(request(fixture, seed))
                    row = audio_row(result, fixture, round_index, seed)
                    rows.append(row)
                    current_rows.append(row)
                    if result.ok and save_outputs == "all":
                        rel = f"outputs/{label}/r{round_index}/{fixture.id}.pcm"
                        _write(roots, rel, result.pcm)
                        outputs.append({"path": rel, "sha256": row["pcm_sha256"], "bytes": row["pcm_bytes"],
                                        "sample_rate": sample_rate, "channels": channels, "encoding": ENCODING})
                valid = [row for row in current_rows if row["ok"]]
                round_rows.append({"index": round_index, "t_start": start, "t_end": time.time(), "seed": seed,
                                   "n_failed": len(current_rows) - len(valid),
                                   "audio_ms": sum(r["complete_audio_ms"] for r in valid) / len(valid) if valid else None,
                                   "first_audio_p95_ms": pct([r["first_audio_ms"] for r in valid], 95),
                                   "rtf": sum(r["rtf"] for r in valid) / len(valid) if valid else None})
            metrics, table = audio_metrics(rows, resamples)
            failed = sum(not row["ok"] for row in rows)
            structure = "FAIL" if failed else "PASS"
            timing = "FAIL" if failed or table["first_audio_ms"]["p95"] > slo_first_audio_p95_ms or \
                table["rtf"]["max"] > slo_rtf or any(row["underruns"] for row in rows if row["ok"]) else "PASS"
            verdict = stats.combine_claims([structure, timing, "UNPROVEN"])
            data = {"source": "bench audio", "timer": "client perf_counter_ns", "label": label,
                    "endpoint_kind": "audio_http", "base_url_kind": "local" if endpoint.is_local() else "remote",
                    "fixture_set": {"sha256": fixture_sha, "fixtures": [f.identity() for f in fixture_rows]},
                    "configuration_sha256": config_sha, "extends": previous["evidence_id"] if previous else None,
                    "rounds": round_rows, "warmup": {"excluded": True, "requests": warm_rows},
                    "per_request": table, "failed_requests": failed, "total_requests": len(rows),
                    "structural_verdict": structure, "timing_verdict": timing, "quality_verdict": "UNPROVEN",
                    "verdict": verdict, "claim_validity": {"valid": not failed, "checked": [
                        "PCM metadata", "aligned nonempty chunks", "terminal final event", "sample count",
                        "nonzero signal", "effective settings (self-reported)"],
                        "violations": [r["error"] for r in rows if not r["ok"]]},
                    "slo": {"first_audio_p95_ms": slo_first_audio_p95_ms, "rtf_max": slo_rtf, "max_underruns": 0},
                    "memory": {"verdict": "NOT RUN", "source": "external process/Metal instrumentation required"},
                    "metric_claim_classes": {"first_pcm_ms": "observed (client timer)",
                        "first_signal_ms": "observed (client timer, nonzero-sample chunk arrival)",
                        "complete_audio_ms": "observed (client timer)", "max_stream_gap_ms": "observed (client timer)",
                        "first_audio_ms": "derived (PCM playback simulation, not an audio device measurement)",
                        "rtf": "derived (complete time / decoded PCM duration)",
                        "underruns": "derived (PCM playback simulation)", "audio_duration_s": "derived (frames / declared rate)"},
                    "definitions": {"first_pcm_ms": "First decoded nonempty PCM chunk, including silence.",
                        "sample_count": "Validated complete PCM frame count; null for failed responses.",
                        "received_pcm_frames": "Decoded frames received, including partial failed responses; diagnostic only.",
                        "first_signal_ms": "Arrival of the first chunk containing a nonzero int16 sample.",
                        "first_audio_ms": "Simulated nonzero-signal playback onset, including initial zero PCM and stalls.",
                        "complete_audio_ms": "Full HTTP body consumption minus request start, on the client clock.",
                        "rtf": "complete_audio_ms / (1000 * decoded PCM duration seconds); lower is faster.",
                        "underruns": "PCM playback simulation starting at first chunk, zero extra prebuffer."}}
            limitations = ["Structural PCM checks do not establish intelligibility, completion, speaker similarity or perceptual quality; external reference proof required.",
                          "unseen is a fixture attestation; text/reference hashes bind inputs but do not prove the server has never seen them.",
                          "Rounds share one server process: screening only; comparative claims require fresh-process interleaved rounds.",
                          "NDJSON events must be flushed by the server. Arrival includes its buffering, HTTP line framing and network transfer.",
                          "Signal means a nonzero PCM sample, not validated speech; PCM silence is measured at exact zero.",
                          "Effective settings are self-reported, not independently proven.",
                          "Memory, swap and sustained repeated-run memory growth are NOT RUN; collect external telemetry."]
            if not persistent:
                limitations.append("Endpoint tag is process-local because no persistent HMAC key could be written.")
            if warmup == 0 or any(not row["ok"] for row in warm_rows):
                limitations.append("Warmup absent or failed: timed requests may include initialization.")
            if len(round_rows) < 3:
                limitations.append("Fewer than three rounds: insufficient independent-round evidence for a final claim.")
            if save_outputs == "hash_only":
                limitations.append("PCM outputs not retained: hashes alone cannot support later audio quality evaluation.")
            _write(roots, "requests.jsonl", ("".join(json.dumps(ev.redact(row)) + "\n" for row in rows)).encode())
            _write(roots, f"outputs/{label}/index.json", (json.dumps(outputs, indent=2) + "\n").encode())
            artifacts = [ev.artifact(staging / "requests.jsonl", staging, "requests"),
                         ev.artifact(staging / f"outputs/{label}/index.json", staging, "outputs_index")]
            for output in outputs:
                artifacts.append(ev.artifact(staging / output["path"], staging, "audio_pcm"))
            identity = ev.identity(gpu={"name": hardware_name}, engine={"name": label, "version": engine_version},
                                   model={"id": model, "revision": model_revision, "dtype": model_dtype, "quant": model_quant},
                                   workload={"mode": "single", "batch": 1, "endpoint_kind": "audio_http",
                                   "request_set": {"sha256": request_sha, "n": len(fixture_rows)},
                                   "seed_policy": {"name": "per_round", "base": seed_base}})
            data["audio_config"] = {"sample_rate": sample_rate, "channels": channels,
                                    "settings": setting_doc, "seed_base": seed_base, "save_outputs": save_outputs}
            record = ev.build(record_call, kind=KIND, claim_class="observed",
                              summary=f"{label}: {len(rows)} speech requests, {failed} structural failures; quality UNPROVEN; {verdict}",
                              identity_=identity, metrics=metrics, artifacts=artifacts,
                              missing=["audio quality reference proof", "external memory telemetry"],
                              limitations=limitations, data=data)
            if run:
                _write([run], f"metrics/{label}.audio.json", (json.dumps(ev.redact(data), indent=2) + "\n").encode())
            return store.save(record, staging)
    finally:
        backend.close()
