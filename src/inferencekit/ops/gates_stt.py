"""Live STT checks for request caching, perturbation, and fixture validity."""
from __future__ import annotations

import io
import random
import struct
import wave
from uuid import uuid4

from inferencekit import evidence as ev, stats
from inferencekit.ops.bench_serve import _record_call
from inferencekit.ops.bench_stt import ENDPOINT_ARGS, FIXTURE_ARG, LANGUAGE, MODEL_ARGS, client_options, load_fixtures, model_pins
from inferencekit.ops.quality_stt import NORMALIZATIONS, normalize
from inferencekit.quality.audio import finite, word_error
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve.stt_client import effective_errors, transcribe


def _perturb(audio, seed):
    with wave.open(io.BytesIO(audio), "rb") as reader:
        params = reader.getparams()
        if params.sampwidth != 2:
            raise ValueError("noise probe requires 16-bit PCM WAV")
        frames = reader.readframes(params.nframes)
    rng = random.Random(seed)
    shifted = bytearray(len(frames))
    for i, (sample,) in enumerate(struct.iter_unpack("<h", frames)):
        # 32.768 counts RMS is 60 dB below int16 full scale before clipping.
        value = max(-32768, min(32767, sample + round(rng.gauss(0, 32.768))))
        struct.pack_into("<h", shifted, 2 * i, value)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(params)
        writer.writeframes(shifted)
    return output.getvalue()


def _result(verdict, observations, reason=None):
    return {"verdict": verdict, "reason": reason, "observations": observations}


def _wer(reference, hypothesis):
    return word_error(reference, hypothesis, "whitespace_v1")


@operation("gates_stt", ("gates", "stt"), "Live recognition integrity probes with timings for every attempted variant.", (
    *ENDPOINT_ARGS, *MODEL_ARGS, FIXTURE_ARG,
    Arg("label", "string", "Engine label.", default="stt"),
    Arg("normalization", "string", "Text normalization for the probes.", default="nfkc_casefold_punctuation_v1",
        choices=NORMALIZATIONS),
    Arg("repeat_min_ratio", "number", "Minimum repeated / original wall latency ratio.", default=0.5),
    Arg("noise_max_wer", "number", "Maximum perturbation WER against the original transcript.", default=0.1),
    Arg("noise_max_timing_ratio", "number", "Maximum bidirectional perturbation / original latency ratio.", default=2.0),
    Arg("heldout_max_wer_increase", "number", "Allowed held-out WER minus tuning-set WER.", default=0.05),
    Arg("locked_language", "string", "Optional ISO language for the cross-language warning probe."),
    Arg("seed", "integer", "Waveform perturbation random seed.", default=0),
))
def gates_stt(call, base_url, model, model_revision, model_dtype, model_quant, engine_version, fixtures,
              label, settings, stream, timeout_s, api_key_env, normalization, repeat_min_ratio, noise_max_wer,
              noise_max_timing_ratio, heldout_max_wer_increase, locked_language, seed):
    ep, settings = client_options(base_url, api_key_env, settings, timeout_s)
    pins = model_pins(model, model_revision, model_dtype, model_quant, engine_version)
    call, _ = _record_call(call, ep)
    if (not all(finite(x) for x in (repeat_min_ratio, noise_max_wer, noise_max_timing_ratio, heldout_max_wer_increase))
            or not 0 < repeat_min_ratio <= 1 or noise_max_timing_ratio < 1 or noise_max_wer < 0
            or heldout_max_wer_increase < 0 or locked_language and not LANGUAGE.fullmatch(locked_language)):
        raise OpError("invalid STT gate thresholds or language lock")
    rows, fixture_sha = load_fixtures(fixtures)
    references = {f.id: normalize(f.reference, normalization) for f in rows}
    if not all(references.values()):
        raise OpError("STT gate fixture has an empty normalized reference")
    repeat, noise, duration, heldout, lang, settings_checks = [], [], [], [], [], []

    def measure(audio, language, variant):
        response = transcribe(ep, audio, model=model, language=language, settings=settings, stream=stream,
                              timeout_s=timeout_s, request_id=uuid4().hex)
        observed = {"variant": variant, "complete_ms": response.complete_ms, "elapsed_ms": response.elapsed_ms,
                    "first_partial_ms": response.first_partial_ms, "error": response.error,
                    "timing_status": "OBSERVED" if response.complete_ms is not None else "NOT RUN"}
        return response, observed

    for i, fixture in enumerate(rows):
        expected = {**pins, **settings, "language": fixture.language}
        raw = fixture.audio()
        a, first = measure(raw, fixture.language, "original")
        b, second = measure(raw, fixture.language, "repeat")
        probes = [(a, first), (b, second)]
        cache_ok = a.effective.get("cache") is False and b.effective.get("cache") is False
        ratio = b.complete_ms / a.complete_ms if a.complete_ms and b.complete_ms is not None else None
        repeat.append({"id": fixture.id, "verdict": "PASS" if cache_ok and ratio is not None and ratio >= repeat_min_ratio
                       and not a.error and not b.error else "FAIL", "ratio": ratio, "cache_disabled": cache_ok,
                       "timing": [first, second]})
        a_text = normalize(a.text, normalization) if not a.error else ""
        try:
            perturbed = _perturb(raw, seed + i)
        except ValueError as exc:
            noise.append({"id": fixture.id, "verdict": "NOT RUN", "reason": str(exc),
                          "timing": [{"variant": "perturbed", "timing_status": "NOT RUN", "error": str(exc)}]})
        else:
            c, third = measure(perturbed, fixture.language, "perturbed")
            probes.append((c, third))
            c_text = normalize(c.text, normalization) if not c.error else ""
            error = _wer(a_text, c_text)["wer"] if a_text and c_text else None
            ratio = c.complete_ms / a.complete_ms if a.complete_ms and c.complete_ms is not None else None
            good = (error is not None and error <= noise_max_wer and ratio is not None and
                    1 / noise_max_timing_ratio <= ratio <= noise_max_timing_ratio)
            noise.append({"id": fixture.id, "verdict": "PASS" if good else "FAIL", "wer": error,
                          "ratio": ratio, "timing": [first, third]})
        for probe, timing in probes:
            errors = effective_errors(probe.effective, expected, fixture.duration)
            duration.append({"id": fixture.id, **timing, "reported_duration_s": probe.effective.get("audio_duration_s"),
                             "expected_duration_s": fixture.duration,
                             "verdict": "PASS" if not any("audio_duration" in error for error in errors) else "FAIL"})
            settings_checks.append({"id": fixture.id, **timing, "errors": errors,
                                    "verdict": "PASS" if not errors else "FAIL"})
        heldout.append({"id": fixture.id, "unseen": fixture.unseen, "timing": first,
                        **_wer(references[fixture.id], a_text)})
        if locked_language and fixture.language != locked_language:
            cross, timing = measure(raw, locked_language, "language-locked")
            detected, confidence = (cross.effective.get(k) for k in ("detected_language", "language_confidence"))
            lock_ok = cross.effective.get("language") == locked_language
            risk = detected == locked_language and finite(confidence) and confidence >= 0.8
            lang.append({"id": fixture.id, "detected_language": detected, "language_confidence": confidence,
                         "timing": timing, "verdict": "WARN" if risk or not lock_ok else
                         "NOT RUN" if cross.error or not finite(confidence) else "PASS"})

    def aggregate(observations):
        verdicts = [o["verdict"] if o["verdict"] != "NOT RUN" else "UNPROVEN" for o in observations]
        return _result(stats.combine_claims(verdicts), observations)

    tune, hidden = ([r for r in heldout if r["unseen"] is unseen] for unseen in (False, True))
    rates = [sum(r["errors"] for r in group) / sum(r["words"] for r in group) if group else None
             for group in (tune, hidden)]
    delta = rates[1] - rates[0] if all(rate is not None for rate in rates) else None
    hold = _result("UNPROVEN" if delta is None else "PASS" if delta <= heldout_max_wer_increase else "FAIL",
                   heldout, "Both tuning and unseen fixtures are required." if delta is None else None)
    hold.update(tuning_wer=rates[0], heldout_wer=rates[1], difference=delta)
    language = _result("WARN" if any(row["verdict"] == "WARN" for row in lang) else
                       "PASS" if any(row["verdict"] == "PASS" for row in lang) else "NOT RUN", lang,
                       "Language lock is an advisory probe; confidence and language are server-reported.")
    gates = {"no_answer_cache": aggregate(repeat), "perturbation": aggregate(noise),
             "held_out_fixtures": hold, "duration_honesty": aggregate(duration),
             "effective_settings": aggregate(settings_checks), "language_lock": language}
    verdict = stats.combine_claims(g["verdict"] for name, g in gates.items() if name != "language_lock")
    return ev.Store().save(ev.build(call, kind="stt_gates", claim_class="derived",
                                    summary=f"{label}: STT gates {verdict}",
                                    identity_=ev.identity(engine={"name": label, "version": engine_version},
                                                          model={"id": model, "revision": model_revision,
                                                                 "dtype": model_dtype, "quant": model_quant}),
                                    limitations=["Gate probes are diagnostics. They do not independently establish "
                                                 "server cache settings or fixture secrecy."],
                                    data={"label": label, "source": "gates stt", "fixture_sha256": fixture_sha,
                                          "verdict": verdict, "gates": gates}))
