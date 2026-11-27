"""Import on-device speech recognition timings and transcripts."""
from __future__ import annotations

import time

from inferencekit import evidence as ev
from inferencekit.ops.bench_audio import _ID, round_seed
from inferencekit.ops.bench_stt import (FIXTURE_ARG, MODEL_ARGS, decoding_settings, load_fixtures, model_pins,
                                        request_row, save_capture, summarize_round)
from inferencekit.quality.audio import finite, read_json
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve.stt_client import Transcription


@operation("import_stt", ("import", "stt"), "Import attested device timings and fixture-bound transcripts.", (
    Arg("timings", "string", "JSON device capture with declared timer, capture timestamps and request rows.", required=True),
    *MODEL_ARGS, FIXTURE_ARG,
    Arg("label", "string", "Configuration label.", default="stt-phone"),
    Arg("hardware_name", "string", "Device and SoC identity.", required=True),
    Arg("settings", "string", "JSON decoding settings applied during capture.", default="{}"),
    Arg("seed_base", "integer", "The run's shared round identity seed base.", default=0),
    Arg("stream", "boolean", "Device capture had incremental transcription.", default=False),
    Arg("run_dir", "string", "Also retain transcript output here."),
))
def import_stt(call, timings, model, model_revision, model_dtype, model_quant, engine_version, fixtures,
               label, hardware_name, settings, seed_base, stream, run_dir):
    pins = model_pins(model, model_revision, model_dtype, model_quant, engine_version)
    settings = decoding_settings(settings)
    if not _ID.fullmatch(label) or not hardware_name.strip():
        raise OpError("invalid STT device label or hardware identity")
    fixture_rows, fixture_sha = load_fixtures(fixtures)
    by_id = {f.id: f for f in fixture_rows}
    doc, _ = read_json(timings)
    if not isinstance(doc, dict) or doc.get("format") != "inferencekit.import.stt.v1":
        raise OpError("expected inferencekit.import.stt.v1 timing document")
    start, end, timer = (doc.get(k) for k in ("capture_start_unix_s", "capture_end_unix_s", "timer"))
    if not (finite(start) and finite(end) and 0 < start < end <= time.time()
            and isinstance(timer, str) and timer.strip()):
        raise OpError("STT device capture needs ordered Unix timestamps and a nonempty timer description")
    rows = doc.get("rows")
    if not isinstance(rows, list) or not rows:
        raise OpError("STT device capture rows are missing")
    grouped, seen = {}, set()
    for number, row in enumerate(rows, 1):
        try:
            fixture = by_id[row["id"]]
            index = row["round"]
            complete, elapsed, partial = (row.get(k) for k in ("complete_ms", "elapsed_ms", "first_partial_ms"))
            text, error, effective = row.get("text", ""), row.get("error"), row.get("effective", {})
            if (type(index) is not int or not 0 <= index <= 100000 or (fixture.id, index) in seen or
                    row["audio_sha256"] != fixture.audio_sha or not isinstance(text, str) or
                    not isinstance(effective, dict) or error is not None and (not isinstance(error, str) or not error.strip()) or
                    not finite(elapsed) or elapsed <= 0 or
                    complete is not None and (not finite(complete) or not 0 < complete <= elapsed) or
                    partial is not None and (not stream or not finite(partial) or not 0 < partial <= (complete or elapsed)) or
                    error is None and (complete is None or not text.strip())):
                raise ValueError
            text.encode("utf-8")
        except (KeyError, TypeError, ValueError, AttributeError):
            raise OpError(f"STT device row {number} has invalid timing, transcript or fixture identity") from None
        if ev.redact(text) != text:
            text, error = "", "transcript contains redacted values"
        result = Transcription(text=text, effective=ev.redact(effective), status=None,
                               first_partial_ms=partial, complete_ms=complete, elapsed_ms=elapsed,
                               error=ev.redact(error))
        seed = round_seed(seed_base, index)
        grouped.setdefault(index, []).append(request_row(fixture, result, index, seed, {**pins, **settings}, True))
        seen.add((fixture.id, index))
    indices = sorted(grouped)
    if any(len(r) != len(fixture_rows) for r in grouped.values()) or indices != list(range(indices[0], indices[-1] + 1)):
        raise OpError("STT device capture must cover each fixture exactly once in each consecutive round")
    rounds = [summarize_round(grouped[i], i, round_seed(seed_base, i), start, end) for i in indices]
    retained = [row for i in indices for row in grouped[i]]
    return save_capture(call, ev.Store(), retained, rounds, fixture_rows, fixture_sha, label, pins,
                        settings, seed_base, run_dir, stream, "import stt", hardware_name, "phone",
                        declared_timer=ev.redact(timer), save_outputs="all")
