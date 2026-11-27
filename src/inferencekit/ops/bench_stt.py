"""Speech recognition timing and immutable, fixture-bound transcript records."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from inferencekit import evidence as ev
from inferencekit.canonical import canonical_json
from inferencekit.ops.bench_audio import _ID, round_seed
from inferencekit.ops.bench_serve import _record_call
from inferencekit.quality.audio import artifact_bytes, one_artifact, sha
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve.client import Endpoint, api_key_from_env
from inferencekit.serve.stt_client import DECODING_FIELDS, MAX_AUDIO_BYTES, effective_errors, transcribe, wav_info
from inferencekit.stats import nearest_rank

KIND = "stt_benchmark"
LANGUAGE = re.compile(r"[a-z]{2,3}")
METRICS = {"stt_ms": "ms", "first_partial_p95_ms": "ms", "rtf": "x", "audio_s_per_s": "audio s/s"}
MODEL_ARGS = (
    Arg("model", "string", "Served model identifier.", required=True),
    Arg("model_revision", "string", "Pinned checkpoint revision.", required=True),
    Arg("model_dtype", "string", "Effective model dtype.", required=True),
    Arg("model_quant", "string", "Quantization recipe, or none.", required=True),
    Arg("engine_version", "string", "Effective pinned engine version.", required=True),
)
FIXTURE_ARG = Arg("fixtures", "string", "JSONL rows with id, audio, reference_text, language and optional unseen.", required=True)
ENDPOINT_ARGS = (
    Arg("base_url", "string", "Transcription endpoint root; stored as a keyed tag.", required=True, sensitive=True),
    Arg("api_key_env", "string", "Environment variable containing the HTTP API key."),
    Arg("settings", "string", "JSON decoding settings: temperature, beam_size, vad_filter, chunking.", default="{}"),
    Arg("stream", "boolean", "Use transcript.text.delta/done SSE to measure the first partial.", default=False),
    Arg("timeout_s", "number", "Whole-request wall deadline in seconds.", default=300.0),
)


@dataclass(frozen=True)
class Fixture:
    id: str
    path: Path
    reference: str
    language: str
    unseen: bool
    audio_sha: str
    rate: int
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.rate

    def identity(self) -> dict:
        return {"id": self.id, "audio_sha256": self.audio_sha, "reference_sha256": sha(self.reference.encode()),
                "language": self.language, "declared_unseen": self.unseen, "sample_rate": self.rate,
                "channels": self.channels, "frames": self.frames}

    def audio(self) -> bytes:
        raw = read_wav(self.path)
        if sha(raw) != self.audio_sha:
            raise OpError(f"STT fixture {self.id}: WAV changed since validation")
        return raw


def read_wav(path: Path) -> bytes:
    with path.open("rb") as handle:
        audio = handle.read(MAX_AUDIO_BYTES + 1)
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("WAV exceeds size limit")
    return audio


def load_fixtures(path: str) -> tuple[list[Fixture], str]:
    source = Path(path)
    root = source.resolve().parent
    fixtures, seen = [], set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise OpError("cannot read STT fixture JSONL") from None
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected an object")
            fid, reference, language = (row.get(k) for k in ("id", "reference_text", "language"))
            if not isinstance(fid, str) or not _ID.fullmatch(fid) or fid in seen:
                raise ValueError("id must be filename-safe and unique")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("reference_text must be nonempty")
            reference.encode("utf-8")
            if not isinstance(language, str) or not LANGUAGE.fullmatch(language):
                raise ValueError("language must be a lower-case ISO language code")
            unseen = row.get("unseen", False)
            if type(unseen) is not bool:
                raise ValueError("unseen must be boolean")
            relative = row.get("audio")
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise ValueError("audio must be a relative WAV path")
            wav_path = (root / relative).resolve()
            if not wav_path.is_relative_to(root):
                raise ValueError("WAV escapes the fixture directory")
            audio = read_wav(wav_path)
            rate, channels, frames = wav_info(audio)
            if any(key in row and (type(row[key]) is not int or row[key] != actual)
                   for key, actual in (("sample_rate", rate), ("channels", channels))):
                raise ValueError("declared sample rate/channels disagree with WAV")
            fixtures.append(Fixture(fid, wav_path, reference, language, unseen, sha(audio), rate, channels, frames))
            seen.add(fid)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise OpError(f"STT fixture line {i}: cannot read valid JSON and PCM WAV") from None
        except ValueError as exc:
            raise OpError(f"STT fixture line {i}: {exc}") from None
    if not fixtures:
        raise OpError("STT fixture set is empty")
    return fixtures, sha(canonical_json([f.identity() for f in fixtures]).encode())


def client_options(base_url, api_key_env, settings, timeout_s):
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise OpError("STT timeout must be finite and positive")
    try:
        url = urlsplit(base_url if "://" in base_url else "http://" + base_url)
        if url.username is not None or url.password is not None or url.query or url.fragment:
            raise ValueError
        endpoint = Endpoint.parse(base_url, api_key_from_env(api_key_env))
    except (ValueError, OpError):
        raise OpError("invalid transcription endpoint or API key configuration") from None
    return endpoint, decoding_settings(settings)


def decoding_settings(settings):
    try:
        doc = json.loads(settings)
        if not isinstance(doc, dict) or set(doc) - set(DECODING_FIELDS) or doc != ev.redact(doc):
            raise ValueError
        doc.setdefault("temperature", 0.0)
        if type(doc["temperature"]) not in (int, float) or not 0 <= doc["temperature"] <= 1:
            raise ValueError
        if "beam_size" in doc and (type(doc["beam_size"]) is not int or doc["beam_size"] < 1):
            raise ValueError
        if "vad_filter" in doc and type(doc["vad_filter"]) is not bool:
            raise ValueError
        if "chunking" in doc and (not isinstance(doc["chunking"], str) or not doc["chunking"].strip()):
            raise ValueError
        doc["temperature"] = float(doc["temperature"])
    except (ValueError, RecursionError):
        raise OpError("invalid STT settings: expected temperature, beam_size, vad_filter or chunking") from None
    return doc


def model_pins(model, model_revision, model_dtype, model_quant, engine_version):
    pins = {"model": model, "model_revision": model_revision, "dtype": model_dtype,
            "quant": model_quant, "engine_version": engine_version}
    if any(not isinstance(value, str) or not value.strip() or value != ev.redact(value) for value in pins.values()):
        raise OpError("STT requires nonempty model, revision, dtype, quant and engine version pins")
    return pins


def request_row(fixture, result, index, seed, expected, save):
    errors = ([result.error] if result.error else []) + effective_errors(
        result.effective, {**expected, "language": fixture.language}, fixture.duration)
    raw = result.text
    return {**fixture.identity(), "round": index, "seed": seed, "status": result.status,
            "complete_ms": result.complete_ms, "elapsed_ms": result.elapsed_ms,
            "timing_status": "OBSERVED" if result.complete_ms is not None else "NOT RUN",
            "first_partial_ms": result.first_partial_ms, "effective": result.effective,
            "transcript": raw if save else None, "transcript_sha256": sha(raw.encode()) if raw else None,
            "completed": result.error is None, "ok": not errors, "errors": errors}


def summarize_round(rows, index, seed, start, end):
    timed = [r for r in rows if r["complete_ms"] is not None]
    wall_ms = sum(r["complete_ms"] for r in timed)
    audio_s = sum(r["frames"] / r["sample_rate"] for r in timed)
    partials = sorted(r["first_partial_ms"] for r in timed if r["first_partial_ms"] is not None)
    return {"index": index, "seed": seed, "t_start": start, "t_end": end,
            "n_failed": sum(not r["ok"] for r in rows),
            "stt_ms": wall_ms / len(timed) if timed else None,
            "first_partial_p95_ms": nearest_rank(partials, 95),
            "rtf": wall_ms / (audio_s * 1000) if audio_s else None,
            "audio_s_per_s": audio_s * 1000 / wall_ms if wall_ms else None}


def save_capture(call, store, rows, rounds, fixtures, fixture_sha, label, pins, settings, seed_base,
                 run_dir, stream, source, hardware_name, device, gpu_count=1, **extra):
    config = {"settings": settings, "stream": stream}
    identity = ev.identity(
        engine={"name": label, "version": pins["engine_version"]},
        model={"id": pins["model"], "revision": pins["model_revision"], "dtype": pins["dtype"], "quant": pins["quant"]},
        cpu={"model": hardware_name} if device == "cpu" else None,
        gpu={"name": hardware_name, "count": gpu_count} if device == "gpu" else None,
        device={"name": hardware_name, "class": "mobile"} if device == "phone" else None,
        workload={"mode": "single", "batch": 1, "endpoint_kind": "stt_http" if source == "bench stt" else "external",
                  "request_set": {"sha256": sha(canonical_json({"fixtures": fixture_sha, **config}).encode()),
                                  "n": len(fixtures)},
                  "seed_policy": {"name": "per_round", "base": seed_base}},
    )
    failed = sum(not row["ok"] for row in rows)
    violations = sorted({error for row in rows for error in row["errors"]})
    data = {"source": source, "label": label, "stt_config": config,
            "timer": "client perf_counter_ns" if source == "bench stt" else "declared device wall timer",
            "fixture_set": {"sha256": fixture_sha, "fixtures": [f.identity() for f in fixtures]},
            "rounds": rounds, "failed_requests": failed, "total_requests": len(rows),
            "claim_validity": {"valid": not failed, "violations": violations},
            "quality_verdict": "UNPROVEN", "verdict": "FAIL" if failed else "UNPROVEN", **extra}
    metrics = []
    for name, unit in METRICS.items():
        values = sorted(row[name] for row in rounds if row[name] is not None)
        metrics.append(ev.metric(name, nearest_rank(values, 50), unit, "median", n=len(values)))
    with store.staging_dir() as stage:
        # Transcripts are outputs, so keep their exact bytes for WER/CER. The caller
        # must review these local artifacts before sharing any evidence bundle.
        raw = "".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows)
        path = stage / "transcripts.jsonl"
        path.write_text(raw, encoding="utf-8")
        record = ev.build(call, kind=KIND, claim_class="observed" if source == "bench stt" else "derived",
                          summary=f"{label}: {len(rows)} STT requests, {failed} invalid; quality UNPROVEN",
                          identity_=identity, metrics=metrics, artifacts=[ev.artifact(path, stage, "transcripts")],
                          missing=["reference transcript quality", "sustained reliability"],
                          limitations=["Effective settings and unseen-fixture status are attestations.",
                                       "Invalid responses retain timing diagnostics and are ineligible for speed claims.",
                                       "See the source timer; imported timing is an on-device attestation."], data=data)
        if run_dir:
            target = Path(run_dir) / "outputs" / label / "transcripts.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(raw, encoding="utf-8")
        saved = store.save(record, stage)
        if run_dir and not call.args.get("endpoint_id"):
            base = Path(run_dir)
            bundle = base / "artifacts" / saved["evidence_id"]
            bundle.mkdir(parents=True, exist_ok=True)
            (bundle / path.name).write_text(raw, encoding="utf-8")
            evidence_dir = base / "evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            (evidence_dir / f"{saved['evidence_id']}.json").write_text(json.dumps(saved, indent=2) + "\n")
        return saved


def previous_capture(store, extend, configuration, call):
    if not extend:
        return [], []
    old = store.load(extend)
    data = old.get("data", {})
    if (old.get("kind") != KIND or data.get("source") != "bench stt" or data.get("scheduler_final")
            or data.get("configuration_sha256") != configuration or ev.integrity_problem(old)):
        raise OpError("--extend requires intact, unfinalized STT evidence with unchanged configuration")
    request = old["data"]["request"]["args"]
    current = call.request()["args"]
    if request.get("endpoint_id") != current.get("endpoint_id") or (
            not current.get("endpoint_id") and request.get("base_url") != current.get("base_url")):
        raise OpError("--extend STT endpoint changed")
    raw = artifact_bytes(store.run_dir(old["evidence_id"]), one_artifact(old, "transcripts"))
    try:
        return [json.loads(line) for line in raw.splitlines()], list(data["rounds"])
    except (ValueError, TypeError, KeyError):
        raise OpError("invalid retained STT request evidence") from None


@operation("bench_stt", ("bench", "stt"), "Client-timed speech recognition with retained transcripts.", (
    *ENDPOINT_ARGS, *MODEL_ARGS, FIXTURE_ARG,
    Arg("label", "string", "Configuration label.", default="stt"),
    Arg("rounds", "integer", "Timed fixture passes.", default=3),
    Arg("first_round", "integer", "First round index; previous maximum plus one for --extend."),
    Arg("warmup", "integer", "Untimed requests from a disjoint warmup set.", default=0),
    Arg("warmup_fixtures", "string", "Separate JSONL set, required for positive --warmup."),
    Arg("seed_base", "integer", "Round identity seed base; no seed is sent to the endpoint.", default=0),
    Arg("hardware_name", "string", "Server hardware identity.", required=True),
    Arg("device", "string", "Server hardware type.", default="gpu", choices=("gpu", "cpu")),
    Arg("gpu_count", "integer", "Number of measured GPUs.", default=1),
    Arg("run_dir", "string", "Also retain the transcript artifact here."),
    Arg("save_outputs", "string", "Retain text or only hashes.", default="all", choices=("all", "hash_only")),
    Arg("endpoint_id", "string", "Stable server identity for scheduler restarts."),
    Arg("extend", "string", "Append to the previous round evidence."),
))
def bench_stt(call, base_url, model, model_revision, model_dtype, model_quant, engine_version, fixtures,
              label, rounds, first_round, warmup, warmup_fixtures, seed_base, settings, stream, timeout_s,
              hardware_name, device, gpu_count, run_dir, save_outputs, api_key_env, endpoint_id, extend):
    pins = model_pins(model, model_revision, model_dtype, model_quant, engine_version)
    ep, settings = client_options(base_url, api_key_env, settings, timeout_s)
    if (not _ID.fullmatch(label) or endpoint_id is not None and not _ID.fullmatch(endpoint_id)
            or not hardware_name.strip() or not 1 <= rounds <= 10000 or not 0 <= warmup <= 100
            or not 1 <= gpu_count <= 1024 or device == "cpu" and gpu_count != 1):
        raise OpError("invalid STT label, hardware, round count or warmup")
    fixture_rows, fixture_sha = load_fixtures(fixtures)
    warm_rows, warm_sha = load_fixtures(warmup_fixtures) if warmup_fixtures else ([], None)
    if warmup and not warm_rows:
        raise OpError("STT --warmup requires a separate --warmup-fixtures set")
    if {f.audio_sha for f in warm_rows} & {f.audio_sha for f in fixture_rows}:
        raise OpError("STT warmup WAV bytes overlap with measured fixtures")
    call, _ = _record_call(call, ep)
    store = ev.Store()
    config_sha = sha(canonical_json({"fixtures": fixture_sha, "warmup_fixtures": warm_sha, "pins": pins,
                                    "settings": settings, "stream": stream, "hardware": hardware_name,
                                    "device": device, "gpu_count": gpu_count, "label": label, "seed": seed_base,
                                    "save_outputs": save_outputs}).encode())
    rows, round_rows = previous_capture(store, extend, config_sha, call)
    expected_first = max((r["index"] for r in round_rows), default=-1) + 1
    first_round = expected_first if first_round is None else first_round
    if not 0 <= first_round <= 100000 - rounds or extend and first_round != expected_first:
        raise OpError("invalid STT first-round index or gap in --extend")

    def run(fixture, index, seed, save):
        try:
            raw = fixture.audio()
        except (OSError, ValueError):
            raise OpError(f"STT fixture {fixture.id}: cannot read validated WAV") from None
        result = transcribe(ep, raw, model=model, language=fixture.language, settings=settings,
                            stream=stream, timeout_s=timeout_s)
        return request_row(fixture, result, index, seed, {**pins, **settings}, save)

    warms = [run(warm_rows[i % len(warm_rows)], -1, 0, False)["errors"] for i in range(warmup)]
    for index in range(first_round, first_round + rounds):
        start = time.time()
        seed = round_seed(seed_base, index)
        current = [run(fixture, index, seed, save_outputs == "all") for fixture in fixture_rows]
        rows.extend(current)
        round_rows.append(summarize_round(current, index, seed, start, time.time()))
    return save_capture(call, store, rows, round_rows, fixture_rows, fixture_sha, label, pins,
                        settings, seed_base, run_dir, stream, "bench stt", hardware_name, device, gpu_count,
                        configuration_sha256=config_sha, warmup_errors=warms, extends=extend, save_outputs=save_outputs)
