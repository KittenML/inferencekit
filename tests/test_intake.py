"""intake template / validate / budget: the orchestrator's input file and run-directory state."""

import json
from pathlib import Path

import jsonschema
import pytest

from inferencekit import cli, registry
from inferencekit.ops import intake
from inferencekit.registry import OpError

from conftest import needs_repo_file

REPO = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"


def base(**over) -> dict:
    d = {
        "model": {"id": "org/model", "revision": SHA, "precision": "fp8"},
        "modality": "llm",
        "hardware": {"gpu": "RTX 5090", "count": 1, "price_usd_h": 0.5, "access": {"kind": "local"}},
        "budget": {"gpu_hours": 4, "usd": 3, "wall_clock_h": 5, "approved": True},
        "repo": {"path": "/x/repo"},
    }
    for k, v in over.items():
        d[k] = v
    return d


def write_json(tmp_path, data, name="intake.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_schema_is_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(intake.load_schema())


def test_defaults_are_filled_and_precision_rule_is_equal_precision():
    r = intake.check(base())
    i = r["intake"]
    assert i["model"]["precision_allowed"] == ["fp8"]
    assert i["model"]["kv_cache_dtype_allowed"] == ["checkpoint_default"]
    assert i["model"]["tokenizer"] == "org/model"
    assert i["goals"]["lead"] == "single_user" and i["goals"]["targets"] == {}
    assert i["io_modes"] == {"input": "complete", "output": "streaming"}
    assert i["buckets"] == ["very_short", "short", "medium", "long"]
    assert i["workload"]["max_tokens"] == 256 and i["workload"]["rounds"] == 5
    assert i["repo"]["branch"] == "perf/model-rtx-5090" and i["repo"]["push"] == "ask"
    assert i["quality"]["reference"] == {"kind": "none"} and i["quality"]["noise_floor"] == "measure"
    assert r["derived"]["claim_class"] == "derived" and r["derived"]["modes"] == ["single"]


def test_no_target_interactive_means_the_stop_point_asks():
    r = intake.check(base())
    stop = r["derived"]["stop_rule"]
    assert stop["kind"] == "ask" and r["derived"]["targets"] == []
    # a budget cap makes keep_climbing the default: still a question, with that as the offered answer
    assert r["intake"]["goals"]["after_target"] == "keep_climbing"
    assert stop["after_target_source"] == "default: budget.gpu_hours, usd or wall_clock_h is set"
    assert "the default to offer is keep climbing until the budget ends" in stop["text"]
    # without a cap the default is stop: the original question, word for word
    r = intake.check(base(budget={"approved": False}))
    assert r["intake"]["goals"]["after_target"] == "stop"
    assert r["derived"]["stop_rule"]["text"] == ("no target: show the ranked table at stage 6 and ask whether to "
                                                 "stop or continue")
    assert r["derived"]["stop_rule"]["after_target_source"] == "default: interactive, no budget cap"
    assert intake.check(base(goals={"after_target": "stop"}))["derived"]["stop_rule"]["kind"] == "ask"


@pytest.mark.parametrize("over", [{"goals": {"after_target": "keep_climbing"}},
                                  {"constraints": ["after_target: keep climbing"]}])
def test_no_target_interactive_with_keep_climbing_written_keeps_climbing(over):
    stop = intake.check(base(**over))["derived"]["stop_rule"]
    assert stop["kind"] == "budget" and stop["after_target"] == "keep_climbing"
    assert stop["text"].startswith("no target: keep climbing until the budget ends")
    assert "best result over time (screened or verified)" in stop["text"]


def test_no_target_non_interactive_continues_within_budget_and_writes_a_question():
    r = intake.check(base(non_interactive=True))
    assert r["derived"]["stop_rule"]["kind"] == "budget"
    qs = [q["question"] for q in r["derived"]["questions"]]
    assert any("No target" in q for q in qs) and any("Push" in q for q in qs)


def test_non_interactive_without_approved_budget_is_an_error():
    d = base(non_interactive=True, budget={"gpu_hours": 2, "approved": False})
    with pytest.raises(OpError, match="approved"):
        intake.check(d)


def test_non_interactive_without_a_cap_is_an_error():
    d = base(non_interactive=True, budget={"wall_clock_h": 2, "approved": True})
    with pytest.raises(OpError, match="spend cap"):
        intake.check(d)


def test_usd_without_price_is_an_error_when_unattended_and_a_warning_otherwise():
    hw = {"gpu": "RTX 5090"}
    with pytest.raises(OpError, match="price_usd_h"):
        intake.check(base(hardware=hw, non_interactive=True, budget={"usd": 5, "gpu_hours": 2, "approved": True}))
    r = intake.check(base(hardware=hw, budget={"usd": 5, "approved": False}))
    assert any("price_usd_h" in w for w in r["warnings"])


def test_capacity_lead_needs_an_slo_and_adds_load_mode():
    with pytest.raises(OpError, match="ttft_p95_ms"):
        intake.check(base(goals={"lead": "capacity"}))
    r = intake.check(base(goals={"lead": "capacity", "slo": {"ttft_p95_ms": 300, "tpot_p95_ms": 40},
                                 "targets": {"n_star": 32}}))
    assert r["derived"]["modes"] == ["single", "load"]
    assert r["intake"]["workload"]["concurrency"] == intake.DEFAULT_CONCURRENCY
    assert r["derived"]["targets"][0] == {"name": "n_star", "value": 32, "better": "higher",
                                          "unit": "streams/GPU", "buckets": list(intake.BUCKETS), "applies": True}


def test_ceiling_fraction_target_is_supported_and_warns_for_nvfp4():
    d = base(goals={"targets": {"decode_ceiling_fraction": 0.7, "buckets": ["short"]}})
    d["model"]["precision"] = "nvfp4"
    r = intake.check(d)
    row = r["derived"]["targets"][0]
    assert row["name"] == "decode_ceiling_fraction" and row["buckets"] == ["short"] and row["better"] == "higher"
    assert r["derived"]["stop_rule"]["kind"] == "target"
    assert any("scripts/bytes_per_token.py" in w for w in r["warnings"])
    with pytest.raises(OpError, match="decode_ceiling_fraction"):
        intake.check(base(goals={"targets": {"decode_ceiling_fraction": 1.5}}))


def test_target_buckets_must_be_intake_buckets():
    with pytest.raises(OpError, match="not in buckets"):
        intake.check(base(buckets=["short"], goals={"targets": {"ttft_ms": 100, "buckets": ["long"]}}))


def test_unknown_keys_are_warnings_and_dropped():
    d = base(targetz={"decode_tok_s": 1})
    d["model"]["revison"] = "x"
    r = intake.check(d)
    assert "targetz" not in r["intake"] and "revison" not in r["intake"]["model"]
    assert any("targetz" in w for w in r["warnings"]) and any("revison" in w for w in r["warnings"])


def test_schema_errors_are_listed_together():
    d = base(modality="unsupported")
    d["hardware"]["count"] = 0
    with pytest.raises(OpError) as e:
        intake.check(d)
    assert "modality" in str(e.value) and "hardware.count" in str(e.value)


def test_ssh_access_needs_host_and_user():
    d = base()
    d["hardware"]["access"] = {"kind": "ssh", "host": "gpu-host"}
    with pytest.raises(OpError, match="user"):
        intake.check(d)
    d["hardware"]["access"]["user"] = "me"
    r = intake.check(d)
    assert r["intake"]["hardware"]["access"]["port"] == 22


def test_placeholders_are_refused_unless_allowed():
    d = base()
    d["model"]["revision"] = "<commit sha>"
    with pytest.raises(OpError, match="placeholder"):
        intake.check(d)
    r = intake.check(d, allow_placeholders=True)
    assert r["placeholders"] == ["model.revision"]


def test_branch_revision_is_a_warning():
    d = base()
    d["model"]["revision"] = "main"
    assert any("not a 40-hex" in w for w in intake.check(d)["warnings"])


def test_hour_cap_takes_the_binding_limit():
    i = intake.check(base(hardware={"gpu": "RTX 5090", "count": 2, "price_usd_h": 1.0},
                          budget={"gpu_hours": 8, "usd": 3, "wall_clock_h": 10, "approved": True}))["intake"]
    cap = intake.hour_cap(i)
    assert cap["caps_h"] == {"gpu_hours": 4.0, "usd": 3.0, "wall_clock_h": 10}
    assert cap["binding"] == "usd" and cap["host_hours_cap"] == 3.0


def test_time_boxes_scale_with_the_cap_and_sum_to_it():
    r = intake.check(base(budget={"gpu_hours": 10, "approved": True}))
    boxes = r["derived"]["time_boxes_h"]
    assert list(boxes) == [k for k, _ in intake.STAGES]
    assert abs(sum(boxes.values()) - 10.0) < 1e-6
    assert r["derived"]["report_reserve_h"] == boxes["8_report"]


def test_tts_targets_and_warnings():
    r = intake.check(base(modality="tts", goals={"targets": {"ttfa_ms": 200, "decode_tok_s": 10}}))
    rows = {t["name"]: t for t in r["derived"]["targets"]}
    assert rows["ttfa_ms"]["applies"] and not rows["decode_tok_s"]["applies"]
    assert any("do not apply to modality tts" in w for w in r["warnings"])
    assert any("UNPROVEN" in w for w in r["warnings"])


# ---------------------------------------------------------------- fidelity, quality budget, after_target

QB = {"gsm8k_drop_pts_max": 1.0, "top1_agreement_min": 0.95, "kl_max": 0.01, "nll_increase_max": 0.02,
      "tv_max": 0.05}


def test_new_rules_have_defaults_and_old_intakes_still_validate():
    r = intake.check(base())  # an older intake: none of the new fields
    i, d = r["intake"], r["derived"]
    assert i["fidelity"] == "lossless" and "budget" not in i["quality"]
    assert d["fidelity_rule"]["value"] == d["fidelity_rule"]["fidelity"] == "lossless"
    assert d["fidelity_rule"]["source"] == "default"
    assert "UNPROVEN" in d["fidelity_rule"]["text"] and d["quality_budget"] is None
    assert d["speculation"]["value"] == "auto" and d["speculation"]["source"] == "default"
    assert "only exact speculation counts" in d["speculation"]["text"]
    assert d["route"]["value"] == "build" and d["route"]["source"] == "default: goals.lead single_user"
    assert not any("fidelity" in w or "after_target" in w for w in r["warnings"])
    # every rule has the same shape: value, source, text
    assert all({"value", "source", "text"} <= set(d[k]) for k in ("fidelity_rule", "speculation", "route"))


@pytest.mark.parametrize("over, expected, source", [
    ({"budget": {"approved": False}}, "stop", "default: interactive, no budget cap"),
    ({"budget": {"wall_clock_h": 3}}, "keep_climbing", "default: budget.gpu_hours, usd or wall_clock_h is set"),
    ({"budget": {"usd": 3}}, "keep_climbing", "default: budget.gpu_hours, usd or wall_clock_h is set"),
    ({"non_interactive": True, "budget": {"gpu_hours": 2, "approved": True}}, "keep_climbing",
     "default: non_interactive"),
    ({"goals": {"after_target": "stop"}}, "stop", "goals.after_target"),
    ({"budget": {"approved": False}, "goals": {"after_target": "keep_climbing"}}, "keep_climbing",
     "goals.after_target"),
    ({"budget": {"approved": False}, "constraints": ["after_target: keep_climbing"]}, "keep_climbing",
     "constraints"),
])
def test_after_target_default_and_source(over, expected, source):
    r = intake.check(base(**over))
    assert r["intake"]["goals"]["after_target"] == expected
    assert r["derived"]["stop_rule"]["after_target"] == expected
    assert r["derived"]["stop_rule"]["after_target_source"] == source


def test_stop_rule_text_reads_after_target():
    goals = {"targets": {"decode_tok_s": 150}}
    climb = intake.check(base(goals=goals))["derived"]["stop_rule"]
    assert climb["kind"] == "target"  # the targets stay: they become milestones
    assert climb["text"].startswith("keep climbing until the budget ends; record the best result over time (screened or verified)")
    stop = intake.check(base(goals={**goals, "after_target": "stop"}))["derived"]["stop_rule"]
    assert stop["kind"] == "target" and stop["text"].startswith("stop when every target is met")
    # keep climbing without any cap (interactive, explicit) climbs until the person stops it, with a warning
    r = intake.check(base(goals={**goals, "after_target": "keep_climbing"}, budget={"approved": False}))
    assert "until you stop it" in r["derived"]["stop_rule"]["text"]
    assert any("no budget cap is set" in w for w in r["warnings"])


def test_non_interactive_no_target_with_after_target_stop_stops_at_the_stop_point():
    r = intake.check(base(non_interactive=True, goals={"after_target": "stop"}))
    assert r["derived"]["stop_rule"]["kind"] == "stop_point"
    assert "stage 7 NOT RUN" in r["derived"]["stop_rule"]["text"]
    q = [x for x in r["derived"]["questions"] if "No target" in x["question"]]
    assert q and "stop after stage 6" in q[0]["choice"]


def test_non_interactive_records_the_defaulted_rules_as_questions():
    r = intake.check(base(non_interactive=True, goals={"targets": {"decode_tok_s": 150}}))
    qs = {q["question"]: q["choice"] for q in r["derived"]["questions"]}
    assert qs["Once every target is met: stop, or keep climbing?"] == "keep_climbing (default: non_interactive)"
    assert any("lossy" in q for q in qs) and any(q.startswith("Build a new engine") for q in qs)
    # rules given explicitly are not open questions
    r = intake.check(base(non_interactive=True, fidelity="lossless", constraints=["route: build"],
                          goals={"targets": {"decode_tok_s": 150}, "after_target": "keep_climbing"}))
    qs = [q["question"] for q in r["derived"]["questions"]]
    assert not any("keep climbing" in q or "lossy" in q or "Build a new engine" in q for q in qs)


def test_constraint_strings_set_the_rules():
    r = intake.check(base(constraints=["Fidelity: lossy, FP8 KV cache is fine", "speculation: off (the task says so)",
                                       "route: patch", "after_target: keep-climbing", "engines must be open source"],
                          quality={"budget": {"kl_max": 0.02}}))
    i, d = r["intake"], r["derived"]
    assert i["fidelity"] == "lossy" and d["fidelity_rule"]["source"] == "constraints"
    assert d["quality_budget"] == {"kl_max": 0.02}
    assert d["speculation"]["value"] == "off" and d["speculation"]["source"] == "constraints"
    assert d["route"]["value"] == "patch" and d["route"]["source"] == "constraints"
    assert i["goals"]["after_target"] == "keep_climbing"
    assert any("does not bound ['gsm8k_drop_pts_max', 'top1_agreement_min', 'nll_increase_max', 'tv_max']" in w
               for w in r["warnings"])
    assert "Not bounded: ['gsm8k_drop_pts_max'" in d["fidelity_rule"]["text"]
    assert "inexact acceptance counts only within quality_budget" in intake.check(base(
        fidelity="lossy", quality={"budget": QB}))["derived"]["speculation"]["text"]


@pytest.mark.parametrize("constraints, match", [
    (["after_target: forever"], "after_target must be one of"),
    (["after_target: keep"], "after_target must be one of"),
    (["speculation: on"], "speculation must be one of"),
    (["speculation: offline"], "speculation must be one of"),  # a value is a whole word, not a prefix
    (["route: build-ish"], "route must be one of"),
    (["route: rewrite"], "route must be one of"),
    (["fidelity:"], "fidelity must be one of"),
    (["route: build", "route: patch"], "route is given twice"),
    (["after target: forever"], "after_target must be one of"),
    (["after-target: stop", "after_target: keep_climbing"], "after_target is given twice"),
    (["speculation: off; route: rewrite"], "route must be one of"),  # every rule in an entry is read
    (["speculation: off, route: patch"], "holds more than one rule"),  # a second rule inside a reason
    (["speculation: off (and fidelity: lossy)"], "holds more than one rule"),
])
def test_constraint_strings_with_unknown_or_conflicting_values_are_errors(constraints, match):
    with pytest.raises(OpError, match=match):
        intake.check(base(constraints=constraints))


@pytest.mark.parametrize("text, value", [
    ("after_target: keep_climbing", "keep_climbing"), ("after_target: keep-climbing", "keep_climbing"),
    ("after_target: Keep climbing until the budget ends", "keep_climbing"), ("AFTER_TARGET: stop.", "stop"),
    ("speculation: OFF", "off"), ("route: build, a single-purpose engine", "build"),
    ("after target: stop", "stop"), ("after-target: keep climbing", "keep_climbing"), ("AfterTarget: stop", "stop"),
])
def test_constraint_values_read_the_words_after_the_colon(text, value):
    found, errors = intake.constraint_rules([text])
    assert not errors and list(found.values()) == [value]


def test_one_constraint_entry_may_hold_several_rules_separated_by_semicolons():
    found, errors = intake.constraint_rules(["speculation: off; route: patch",
                                             "engines must be open source; after target: stop"])
    assert not errors and found == {"speculation": "off", "route": "patch", "after_target": "stop"}
    d = intake.check(base(constraints="speculation: off; route: patch"))["derived"]
    assert d["speculation"]["value"] == "off" and d["route"]["value"] == "patch"


def test_a_field_wins_over_a_constraint_and_they_must_agree():
    r = intake.check(base(fidelity="lossless", constraints=["fidelity: lossless"]))
    assert r["derived"]["fidelity_rule"]["source"] == "fidelity"
    with pytest.raises(OpError, match="fidelity is 'lossless' but constraints say 'fidelity: lossy'"):
        intake.check(base(fidelity="lossless", constraints=["fidelity: lossy"]))
    with pytest.raises(OpError, match="goals.after_target is 'stop' but constraints say"):
        intake.check(base(goals={"after_target": "stop"}, constraints=["after_target: keep_climbing"]))


def test_unquoted_constraint_mapping_is_refused_with_a_hint():
    with pytest.raises(OpError, match="quoted string"):
        intake.check(base(constraints=[{"speculation": "off"}]))


def test_lossless_warns_about_other_precisions_lossy_kv_and_an_unused_budget():
    d = base(quality={"budget": QB})
    d["model"].update(precision_allowed=["fp8", "nvfp4"], kv_cache_dtype_allowed=["checkpoint_default", "fp8_e4m3"])
    w = intake.check(d)["warnings"]
    assert any("fidelity is lossless but model.precision_allowed" in x for x in w)
    assert any("kv_cache_dtype_allowed has ['fp8_e4m3']" in x for x in w)
    assert any("quality.budget is set but fidelity is lossless" in x for x in w)
    d = base()
    d["model"].update(precision_allowed=["fp8"], kv_cache_dtype_allowed=["checkpoint_default", "bf16"])
    assert not any("fidelity" in x for x in intake.check(d)["warnings"])


@pytest.mark.parametrize("dtype, lossy", [("fp8", True), ("FP8_E5M2", True), ("int8", True), ("int4", True),
                                          ("nvfp4", True), ("mxfp4", True), ("q8_0", True), ("float8_e4m3fn", True),
                                          ("fp4_e2m1", True), ("iq4_nl", True), ("bf16", False), ("fp16", False),
                                          ("fp32", False), ("float16", False), ("auto", False),
                                          ("checkpoint_default", False)])
def test_low_precision_kv_dtypes(dtype, lossy):
    assert bool(intake._LOSSY_KV.match(dtype)) is lossy


def test_lossy_needs_a_quality_budget():
    with pytest.raises(OpError, match="fidelity is lossy but quality.budget is missing"):
        intake.check(base(non_interactive=True, fidelity="lossy"))
    r = intake.check(base(fidelity="lossy"))
    assert any("quality.budget is missing" in w and "agrees them with you" in w for w in r["warnings"])
    assert r["derived"]["quality_budget"] is None and "not set yet" in r["derived"]["fidelity_rule"]["text"]
    r = intake.check(base(non_interactive=True, fidelity="lossy", quality={"budget": QB}))
    assert r["derived"]["quality_budget"] == QB and "separate track" in r["derived"]["fidelity_rule"]["text"]
    assert not any("quality.budget" in w for w in r["warnings"])


RELATIVE_IMAGE = {"modality": "image", "lpips_excess_max": 0.02, "ssim_loss_max": 0.01,
                  "ocr_cer_increase_max": 0.02, "semantic_score_drop_max": 0.05,
                  "new_severe_regressions_max": 0}
RELATIVE_TTS = {"modality": "tts", "wer_increase_max": 0.02, "cer_increase_max": 0.03,
                "completion_failure_increase_max": 0.02, "speaker_similarity_drop_max": 0.03,
                "perceptual_score_drop_max": 0.25, "new_severe_regressions_max": 0}


def relative_intake(modality="image", relative=None):
    return base(modality=modality, fidelity="lossy", non_interactive=True,
                quality={"protocol": "external-quality.json", "protocol_sha256": "a" * 64,
                         "budget": {"relative": dict(relative or RELATIVE_IMAGE)}})


@pytest.mark.parametrize("modality,relative", [("image", RELATIVE_IMAGE), ("tts", RELATIVE_TTS),
                                              ("stt", {"modality": "stt", "wer_increase_max": 0.02})])
def test_relative_modality_budget_preserved_and_requires_external_proof(modality, relative):
    r = intake.check(relative_intake(modality, relative))
    assert r["intake"]["quality"]["budget"] == r["derived"]["quality_budget"] == {"relative": relative}
    proof = r["derived"]["external_quality_proof"]
    assert proof["required"] is True and proof["quality_verdict"] == "UNPROVEN"
    assert proof["protocol_sha256"] == "a" * 64 and "not a quality PASS" in proof["text"]
    assert not any("lpips_max" in x or "wer_max" in x for x in r["warnings"])


@pytest.mark.parametrize("field,value", [("protocol", None), ("protocol", "   "), ("protocol_sha256", None),
                                       ("protocol_sha256", "a" * 40), ("protocol_sha256", "G" * 64)])
def test_relative_budget_needs_protocol_and_sha256(field, value):
    d = relative_intake()
    d["quality"][field] = value
    with pytest.raises(OpError, match="protocol"):
        intake.check(d)


@pytest.mark.parametrize("modality,relative", [("llm", RELATIVE_IMAGE), ("image", RELATIVE_TTS),
                                              ("video", {"modality": "image", "lpips_excess_max": 0.02}),
                                              ("stt", {"modality": "stt", "speaker_similarity_drop_max": 0.03})])
def test_relative_wrong_modality_is_an_error(modality, relative):
    with pytest.raises(OpError):
        intake.check(relative_intake(modality, relative))


@pytest.mark.parametrize("value", [0.1, None])
def test_unknown_relative_keys_are_errors_never_dropped(value):
    d = relative_intake()
    d["quality"]["budget"]["relative"]["lpips_exces_max"] = value
    with pytest.raises(OpError, match="lpips_exces_max"):
        intake.check(d)


@pytest.mark.parametrize("key,value", [("lpips_excess_max", -0.1), ("lpips_excess_max", float("nan")),
                                     ("lpips_excess_max", float("inf")), ("lpips_excess_max", None),
                                     ("ssim_loss_max", 2.01), ("semantic_score_drop_max", 1.01),
                                     ("new_severe_regressions_max", 0.5),
                                     ("new_severe_regressions_max", True)])
def test_relative_numbers_are_finite_oriented_bounded_and_typed(key, value):
    d = relative_intake()
    d["quality"]["budget"]["relative"][key] = value
    with pytest.raises(OpError):
        intake.check(d)


def test_relative_budget_not_silently_null_or_ignored_under_lossless():
    d = relative_intake()
    d["quality"]["budget"]["relative"] = None
    with pytest.raises(OpError):
        intake.check(d)
    d = relative_intake()
    d["fidelity"] = "lossless"
    with pytest.raises(OpError, match="requires fidelity: lossy"):
        intake.check(d)


def test_absolute_and_relative_bounds_remain_distinct_and_absolute_compatibility():
    d = relative_intake()
    d["quality"]["budget"]["lpips_max"] = 0.5
    r = intake.check(d)
    assert r["derived"]["quality_budget"]["lpips_max"] == 0.5
    assert r["derived"]["quality_budget"]["relative"]["lpips_excess_max"] == 0.02
    legacy = base(modality="tts", fidelity="lossy", quality={"budget": {"wer_max": 0.1}})
    r = intake.check(legacy)
    assert r["derived"]["quality_budget"] == {"wer_max": 0.1}
    assert "external_quality_proof" not in r["derived"]


@pytest.mark.parametrize("key,value", [("speaker_similarity_drop_max", 2.01),
                                     ("completion_failure_increase_max", 1.01),
                                     ("perceptual_score_drop_max", -0.1)])
def test_speech_relative_metric_bounds(key, value):
    d = relative_intake("tts", RELATIVE_TTS)
    d["quality"]["budget"]["relative"][key] = value
    with pytest.raises(OpError):
        intake.check(d)


def test_json_schema_itself_refuses_another_modality_metric():
    d = relative_intake()
    d["quality"]["budget"]["relative"]["wer_increase_max"] = 0.02
    errors, _ = intake.schema_problems(d)
    assert errors and any("wer_increase_max" in x for x in errors)


def perceptual_intake(modality="image"):
    relative = ({k: v for k, v in RELATIVE_IMAGE.items() if k not in ("lpips_excess_max", "ssim_loss_max")}
                if modality == "image" else RELATIVE_TTS)
    d = relative_intake(modality, relative)
    d["quality"]["acceptance"] = "perceptual_and_task_noninferiority"
    d["quality"]["diagnostic_only"] = ["lpips", "ssim", "psnr"] if modality == "image" else ["cross_engine_pcm_equality"]
    return d


@pytest.mark.parametrize("modality", ["image", "tts"])
def test_perceptual_acceptance_needs_no_dummy_numeric_similarity_limits(modality):
    r = intake.check(perceptual_intake(modality))
    assert not any("does not bound" in w for w in r["warnings"])
    proof = r["derived"]["external_quality_proof"]
    assert proof["blinded_perceptual_review_required"] is True and proof["perceptual_verdict"] == "UNPROVEN"
    assert proof["quality_verdict"] == "UNPROVEN" and "not an independent rejection gate" in proof["text"]
    assert proof["diagnostic_only"] == r["intake"]["quality"]["diagnostic_only"]
    if modality == "image":
        assert "lpips_excess_max" not in r["derived"]["quality_budget"]["relative"]


@pytest.mark.parametrize("relative_key,absolute_key", [("lpips_excess_max", "lpips_max"),
                                                     ("ssim_loss_max", "ssim_min")])
def test_perceptual_acceptance_rejects_conflicting_similarity_gates(relative_key, absolute_key):
    for location, key in [("relative", relative_key), ("absolute", absolute_key)]:
        d = perceptual_intake()
        target = d["quality"]["budget"]["relative"] if location == "relative" else d["quality"]["budget"]
        target[key] = 0.02
        with pytest.raises(OpError, match="diagnostic only"):
            intake.check(d)


@pytest.mark.parametrize("modality,key", [("image", "semantic_score_drop_max"),
                                        ("image", "new_severe_regressions_max"),
                                        ("image", "ocr_cer_increase_max"),
                                        ("tts", "perceptual_score_drop_max")])
def test_perceptual_acceptance_requires_explicit_task_blind_and_severe_limits(modality, key):
    d = perceptual_intake(modality)
    d["quality"]["budget"]["relative"].pop(key)
    with pytest.raises(OpError, match="requires relative bounds"):
        intake.check(d)


def test_diagnostic_only_cannot_quietly_disable_legacy_gates():
    d = relative_intake()
    d["quality"]["diagnostic_only"] = ["lpips"]
    with pytest.raises(OpError, match="requires explicit quality.acceptance"):
        intake.check(d)


@pytest.mark.parametrize("value", [["cross_engine_pcm_equality"], ["made_up_metric"], None])
def test_perceptual_diagnostics_are_typed_and_modality_specific(value):
    d = perceptual_intake()
    d["quality"]["diagnostic_only"] = value
    with pytest.raises(OpError):
        intake.check(d)


def test_perceptual_mode_does_not_invent_other_modality_or_protocol_evidence():
    d = perceptual_intake()
    d["quality"].pop("protocol_sha256")
    with pytest.raises(OpError, match="protocol"):
        intake.check(d)
    d = relative_intake("stt", {"modality": "stt", "wer_increase_max": 0.02})
    d["quality"]["acceptance"] = "perceptual_and_task_noninferiority"
    with pytest.raises(OpError, match="image or tts"):
        intake.check(d)


def test_video_targets_budget_and_warnings():
    r = intake.check(base(modality="video", goals={"targets": {"clip_ms": 60000, "decode_tok_s": 10}}))
    rows = {t["name"]: t for t in r["derived"]["targets"]}
    assert rows["clip_ms"]["applies"] and rows["clip_ms"]["better"] == "lower" and not rows["decode_tok_s"]["applies"]
    assert any("do not apply to modality video" in w for w in r["warnings"])
    assert any("modality video: time clips with bench clip" in w for w in r["warnings"])
    assert r["derived"]["modes"] == ["clip"] and "quality clip" in r["derived"]["fidelity_rule"]["text"]
    # a clip target holds over the whole clip set (bench rounds --mode clip ranks bucket all), not per text bucket
    assert rows["clip_ms"]["buckets"] == ["all"] and "over the whole clip set" in r["derived"]["stop_rule"]["text"]
    assert "bucket" not in r["derived"]["stop_rule"]["text"] and "buckets" not in r["intake"]["goals"]["targets"]
    assert "KV cache" not in r["derived"]["precision_rule"]
    assert r["derived"]["speculation"]["text"].startswith("does not apply to modality video")
    r = intake.check(base(modality="video", goals={"targets": {"clip_ms": 60000, "buckets": ["short"]}}))
    assert any("goals.targets.buckets do not apply to modality video" in w for w in r["warnings"]), r["warnings"]
    assert r["derived"]["targets"][0]["buckets"] == ["all"]  # not refused as a bucket outside `buckets`
    r = intake.check(base(modality="video", goals={"targets": {"clips_per_hour": 40}}, fidelity="lossy",
                          quality={"budget": {"psnr_db_min": 30.0, "kl_max": 0.1}}))
    w = " ".join(r["warnings"])
    assert "does not bound ['ssim_min', 'lpips_max']" in w and "['kl_max'] do not apply to modality video" in w
    full = {"psnr_db_min": 30.0, "ssim_min": 0.95, "lpips_max": 0.05}
    r = intake.check(base(modality="video", non_interactive=True, fidelity="lossy", quality={"budget": full}))
    assert r["derived"]["quality_budget"] == full and not any("quality.budget" in x for x in r["warnings"])


def test_lossy_with_some_bounds_names_the_unbounded_ones():
    r = intake.check(base(fidelity="lossy", quality={"budget": {"kl_max": 0.02, "tv_max": 0.05}}))
    w = [x for x in r["warnings"] if "does not bound" in x]
    assert len(w) == 1 and "['gsm8k_drop_pts_max', 'top1_agreement_min', 'nll_increase_max']" in w[0]
    assert "agree them" in w[0] and "Not bounded" in r["derived"]["fidelity_rule"]["text"]
    # a full budget: no such warning, and the rule names no gap
    r = intake.check(base(non_interactive=True, fidelity="lossy", quality={"budget": QB}))
    assert not any("does not bound" in x for x in r["warnings"])
    assert "Not bounded" not in r["derived"]["fidelity_rule"]["text"]


def test_lossless_precision_rule_ranks_only_the_as_shipped_numerics():
    d = base()
    d["model"].update(precision_allowed=["fp8", "nvfp4"], kv_cache_dtype_allowed=["checkpoint_default", "bf16", "fp8"])
    der = intake.check(d)["derived"]
    rule = der["precision_rule"]
    # agrees with fidelity_rule: no lower precision, no sub-16-bit KV cache among ranked engines
    assert rule.startswith("every ranked engine runs weights/activations in one of ['fp8'] and KV cache in one of "
                           "['checkpoint_default', 'bf16'];")
    assert "precision ['nvfp4'], KV cache ['fp8']" in rule and "not run or ranked" in rule
    assert "no KV cache below 16 bits" in der["fidelity_rule"]["text"]
    # only a low-precision KV dtype listed: checkpoint_default is still the ranked one
    d["model"].update(precision_allowed=["fp8"], kv_cache_dtype_allowed=["fp8"])
    assert "KV cache in one of ['checkpoint_default']" in intake.check(d)["derived"]["precision_rule"]
    # lossy with a budget: every listed entry may be ranked, each precision on its own track
    d = base(fidelity="lossy", quality={"budget": QB})
    d["model"].update(precision_allowed=["fp8", "nvfp4"], kv_cache_dtype_allowed=["checkpoint_default", "fp8"])
    rule = intake.check(d)["derived"]["precision_rule"]
    assert "['fp8', 'nvfp4']" in rule and "['checkpoint_default', 'fp8']" in rule and "not run" not in rule
    # nothing extra listed: the original text, unchanged
    assert intake.check(base())["derived"]["precision_rule"] == (
        "every ranked engine runs weights/activations in one of ['fp8'] and KV cache in one of "
        "['checkpoint_default']; different precisions are separate tracks")


def test_usd_without_a_price_is_no_cap_so_climbing_runs_until_stopped():
    hw = {"gpu": "RTX 5090"}
    r = intake.check(base(hardware=hw, budget={"usd": 5}, goals={"targets": {"decode_tok_s": 100}}))
    stop = r["derived"]["stop_rule"]
    # usd still makes keep_climbing the default, but without a price it cannot end the run
    assert stop["after_target"] == "keep_climbing" and r["derived"]["budget"]["host_hours_cap"] is None
    assert stop["text"].startswith("keep climbing until you stop it (no budget cap is set)")
    assert any("no budget cap is set (budget.usd needs hardware.price_usd_h" in w for w in r["warnings"])
    ask = intake.check(base(hardware=hw, budget={"usd": 5}))["derived"]["stop_rule"]
    assert ask["kind"] == "ask" and "keep climbing until you stop it" in ask["text"]
    assert "because the intake sets budget.usd)" in ask["text"]
    # with a price, the same budget ends the run
    r = intake.check(base(budget={"usd": 5}, goals={"targets": {"decode_tok_s": 100}}))
    assert r["derived"]["stop_rule"]["text"].startswith("keep climbing until the budget ends")
    assert not any("no budget cap is set" in w for w in r["warnings"])


@pytest.mark.parametrize("over", [
    {}, {"budget": {"approved": False}}, {"non_interactive": True},
    {"non_interactive": True, "goals": {"after_target": "stop"}},
    {"goals": {"after_target": "keep_climbing"}}, {"constraints": ["after_target: keep climbing", "route: patch"]},
    {"goals": {"targets": {"decode_tok_s": 150}}, "non_interactive": True},
    {"fidelity": "lossy", "quality": {"budget": QB}}, {"constraints": "fidelity: lossy"},
])
def test_checking_a_normalized_intake_again_gives_the_same_plan(over):
    first = intake.check(base(**over))
    again = intake.check(first["intake"])
    assert again["derived"] == first["derived"] and again["warnings"] == first["warnings"]
    assert again["intake"]["rule_sources"] == first["intake"]["rule_sources"]
    assert first["intake"]["rule_sources"] == {k: v["source"] for k, v in intake.rule_values(first["intake"]).items()}
    assert intake.derive(first["intake"]) == first["derived"]


def test_a_rule_changed_by_hand_in_a_normalized_intake_counts_as_written():
    norm = intake.check(base())["intake"]  # interactive, no target, capped: the stop point asks
    assert norm["rule_sources"]["after_target"].startswith("default")
    norm["goals"]["after_target"] = "stop"
    stop = intake.check(norm)["derived"]["stop_rule"]
    assert stop["after_target"] == "stop" and stop["after_target_source"] == "goals.after_target"


@pytest.mark.parametrize("over, where", [
    ({"fidelity": "approximate"}, "fidelity"),
    ({"goals": {"after_target": "forever"}}, "goals.after_target"),
    ({"quality": {"budget": {}}}, "quality.budget"),
    ({"quality": {"budget": {"top1_agreement_min": 1.5}}}, "quality.budget.top1_agreement_min"),
    ({"quality": {"budget": {"tv_max": -0.1}}}, "quality.budget.tv_max"),
    ({"quality": {"budget": {"kl_max": "small"}}}, "quality.budget.kl_max"),
])
def test_schema_checks_the_new_fields(over, where):
    with pytest.raises(OpError, match=where.replace(".", r"\.")):
        intake.check(base(**over))


def test_unknown_quality_budget_key_is_a_warning():
    r = intake.check(base(fidelity="lossy", quality={"budget": {"kl_max": 0.01, "kl_maxx": 0.5}}))
    assert r["derived"]["quality_budget"] == {"kl_max": 0.01}
    assert any("kl_maxx" in w for w in r["warnings"])


def test_validate_op_starts_and_resumes_a_run_dir(tmp_path):
    src = write_json(tmp_path, base(non_interactive=True, run_id="r1"))
    run = tmp_path / "runs" / "r1"
    r = registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})
    assert r["run"]["resumed"] is False
    assert set(r["run"]["written"]) >= {"intake.json", "intake.normalized.json", "state.json", "experiments.jsonl",
                                        "QUESTIONS.md"}
    state = json.loads((run / "state.json").read_text())
    assert state["run_id"] == "r1" and state["current_stage"] == "1_intake"
    assert state["stages"]["1_intake"]["status"] == "in_progress"
    assert state["intake_sha256"] == r["intake_sha256"]
    questions = (run / "QUESTIONS.md").read_text()
    assert "No target" in questions and "fidelity lossless (the default)" in questions
    norm = json.loads((run / "intake.normalized.json").read_text())
    assert norm["intake"]["fidelity"] == "lossless" and norm["intake"]["goals"]["after_target"] == "keep_climbing"
    assert norm["derived"]["stop_rule"]["kind"] == "budget" and norm["derived"]["route"]["value"] == "build"
    assert norm["intake"]["rule_sources"] == {"after_target": "default: non_interactive", "fidelity": "default",
                                              "speculation": "default", "route": "default: goals.lead single_user"}
    # the agent edits state; a second validate resumes without clobbering it
    state["current_stage"] = "4_tuning"
    (run / "state.json").write_text(json.dumps(state))
    (run / "experiments.jsonl").write_text('{"x": 1}\n')
    r2 = registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})
    assert r2["run"]["resumed"] is True
    assert json.loads((run / "state.json").read_text())["current_stage"] == "4_tuning"
    assert (run / "experiments.jsonl").read_text() == '{"x": 1}\n'


def test_a_changed_intake_is_refused_and_nothing_in_the_run_dir_changes(tmp_path):
    goals = {"targets": {"decode_ceiling_fraction": 0.70}}
    src = write_json(tmp_path, base(non_interactive=True, run_id="r1", goals=goals))
    run = tmp_path / "runs" / "r1"
    registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    # lowering a pre-registered target mid-run is a new run, never an overwrite
    src.write_text(json.dumps(base(non_interactive=True, run_id="r1",
                                   goals={"targets": {"decode_ceiling_fraction": 0.30}})))
    with pytest.raises(OpError, match="intake changed.*new run id"):
        registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    norm = json.loads((run / "intake.normalized.json").read_text())
    assert norm["intake"]["goals"]["targets"]["decode_ceiling_fraction"] == 0.70
    # a leftover normalized file from another intake (no state.json yet) is refused too
    (run / "state.json").unlink()
    with pytest.raises(OpError, match="intake changed"):
        registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})
    assert not (run / "state.json").exists()


def _ssh_intake(tmp_path, home):
    d = base(non_interactive=True, run_id="r1",
             quality={"reference": {"kind": "endpoint", "base_url": "http://gpu.example:8001/v1"}},
             repo={"path": str(home / "work" / "ledger")},
             harness={"wheel": str(home / "dl" / "inferencekit_cli-0.1.0-py3-none-any.whl")})
    d["hardware"]["access"] = {"kind": "ssh", "host": "gpu.example", "user": "root", "port": 40022,
                               "key_path": str(home / ".ssh" / "id_gpu")}
    return d


def _assert_no_access_details(text, home):
    for secret in ("gpu.example", '"root"', "root@", "40022", "id_gpu", str(home)):
        assert secret not in text, secret


def test_run_dir_copies_are_redacted_but_the_sha_is_of_the_original(tmp_path, monkeypatch):
    home = tmp_path / "home" / "someuser"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    src = write_json(tmp_path, _ssh_intake(tmp_path, home))
    run = tmp_path / "runs" / "r1"
    r = registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})
    sha = intake.sha256_hex(src.read_bytes())
    assert r["intake"]["hardware"]["access"]["host"] == "gpu.example"  # the caller still gets the details
    for name in ("intake.json", "intake.normalized.json", "state.json"):
        _assert_no_access_details((run / name).read_text(), home)
    copy = json.loads((run / "intake.json").read_text())
    assert copy["hardware"]["access"] == {"kind": "ssh", "host": "<redacted>", "user": "<redacted>",
                                          "port": "<redacted>", "key_path": "<redacted>"}
    assert copy["quality"]["reference"]["base_url"] == "http://<redacted>:8001/v1"
    assert copy["repo"]["path"] == "~/work/ledger"
    norm = json.loads((run / "intake.normalized.json").read_text())
    state = json.loads((run / "state.json").read_text())
    assert norm["intake_sha256"] == state["intake_sha256"] == sha
    assert norm["intake"]["hardware"]["access"]["host"] == "<redacted>"
    assert "gpu.example" not in state.get("intake_source", "")
    # resume from the original works; validating the redacted copy is refused with a pointer to the original
    assert registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})["run"]["resumed"] is True
    with pytest.raises(OpError, match="redacted run-directory copy"):
        registry.call("intake_validate", {"file": str(run / "intake.json"), "run_dir": str(run)})


def test_redacted_yaml_copy(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    home = tmp_path / "home" / "someuser"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    src = tmp_path / "intake.yaml"
    src.write_text("# my comment\n" + yaml.safe_dump(_ssh_intake(tmp_path, home)))
    run = tmp_path / "runs" / "r1"
    registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})
    text = (run / "intake.yaml").read_text()
    _assert_no_access_details(text, home)
    assert intake.sha256_hex(src.read_bytes()) in text
    assert yaml.safe_load(text)["hardware"]["access"]["host"] == "<redacted>"


def test_double_underscore_fill_in_markers_are_placeholders():
    d = base(quality={"reference": {"kind": "endpoint", "base_url": "http://__HOST__:__PORT__/v1"}})
    with pytest.raises(OpError, match="placeholder.*quality.reference.base_url"):
        intake.check(d)
    d = base(quality={"reference": {"kind": "endpoint", "base_url": "__REFERENCE_BASE_URL__"}})
    assert intake.check(d, allow_placeholders=True)["placeholders"] == ["quality.reference.base_url"]
    d = base(repo={"path": "/x/pkg/__init__.py"})  # dunder names are not fill-in markers
    assert intake.check(d)["placeholders"] == []


def test_run_dir_refuses_placeholders(tmp_path):
    d = base()
    d["model"]["id"] = "<org/name>"
    src = write_json(tmp_path, d)
    with pytest.raises(OpError, match="placeholder"):
        registry.call("intake_validate", {"file": str(src), "run_dir": str(tmp_path / "run"),
                                          "allow_placeholders": True})


def _state(cap=10.0, started="2026-10-13T10:00:00Z", stage="3_candidates", stage_started="2026-10-13T10:30:00Z",
           box=1.8, meter=None):
    return {"current_stage": stage,
            "stages": {stage: {"started": stage_started, "time_box_h": box}},
            "budget": {"host_hours_cap": cap, "usd_cap": 5.0, "price_usd_h": 0.5, "run_started": started,
                       "meter": meter if meter is not None else [{"start": "2026-10-13T10:00:00Z", "end": None}]}}


def _now(s):
    return intake._parse_time(s)


def test_budget_ok_then_stage_over_then_reserve_then_over():
    s = _state()
    r = intake.budget_status(s, _now("2026-10-13T11:00:00Z"))
    assert r["verdict"] == "ok" and r["spent"]["host_hours"] == 1.0 and r["spent"]["usd"] == 0.5
    assert r["left"]["host_hours"] == 9.0 and r["spent"]["open_meter_segments"] == 1
    assert intake.budget_status(s, _now("2026-10-13T12:30:00Z"))["verdict"] == "stage_over"
    assert intake.budget_status(s, _now("2026-10-13T19:30:00Z"))["verdict"] == "reserve_only"
    assert intake.budget_status(s, _now("2026-10-13T20:01:00Z"))["verdict"] == "over_budget"


def test_budget_usd_cap_and_closed_segments():
    meter = [{"start": "2026-10-13T08:00:00Z", "end": "2026-10-13T09:00:00Z", "price_usd_h": 3.0},
             {"start": "2026-10-13T10:00:00Z", "end": None}]
    r = intake.budget_status(_state(meter=meter), _now("2026-10-13T11:00:00Z"))
    assert r["spent"]["host_hours"] == 2.0 and r["spent"]["usd"] == 3.5
    meter[0]["price_usd_h"] = 5.0
    assert intake.budget_status(_state(meter=meter), _now("2026-10-13T11:00:00Z"))["verdict"] == "over_budget"


def test_budget_op_reads_the_run_dir(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps(_state()))
    r = registry.call("intake_budget", {"run_dir": str(tmp_path), "now": "2026-10-13T11:00:00Z"})
    assert r["verdict"] == "ok" and r["claim_class"] == "derived"
    with pytest.raises(OpError, match="not found"):
        registry.call("intake_budget", {"run_dir": str(tmp_path / "nope")})


def test_cli_template_prints_yaml(capsys):
    assert cli.main(["intake", "template"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# inferencekit intake v1") and "non_interactive:" in out


def test_template_out_refuses_to_overwrite(tmp_path):
    out = tmp_path / "intake.yaml"
    registry.call("intake_template", {"out": str(out)})
    assert out.read_text() == intake.TEMPLATE
    with pytest.raises(OpError, match="exists"):
        registry.call("intake_template", {"out": str(out)})


def test_json_needs_no_yaml_and_bad_text_is_an_error():
    assert intake.parse_text('{"a": 1}') == {"a": 1}
    with pytest.raises(OpError, match="invalid JSON"):
        intake.parse_text("{nope")


# ---------------------------------------------------------------- YAML (PyYAML is a declared dependency)

def test_template_is_a_valid_intake_with_placeholders():
    pytest.importorskip("yaml")
    r = intake.check(intake.parse_text(intake.TEMPLATE), allow_placeholders=True)
    assert r["ok"] and "model.revision" in r["placeholders"] and "hardware.access.host" in r["placeholders"]
    assert r["derived"]["stop_rule"]["kind"] == "target"


# Example intakes outside the plugin repository: checked when present.
EXAMPLES = REPO / "examples" / "intake"
needs_examples = needs_repo_file(EXAMPLES, "examples/intake/")


@needs_examples
@pytest.mark.parametrize("name", ["llm-qwen-5090.yaml", "llm-lossy-capacity.yaml", "tts-streaming.yaml",
                                  "stt-streaming.yaml"])
def test_examples_validate(name):
    pytest.importorskip("yaml")
    data, _ = intake.load_file(EXAMPLES / name)
    r = intake.check(data, allow_placeholders=True)
    assert r["ok"]
    assert r["intake"]["modality"] in name.split("-")[0]


# an older llm-qwen-5090 example intake (before after_target and fidelity existed; comments trimmed)
QWEN_5090_V020 = """\
intake_version: 1
model:
  id: <hf-org>/<qwen-checkpoint-nvfp4>
  revision: "<40-hex commit sha>"
  precision: nvfp4
  kv_cache_dtype_allowed: [checkpoint_default]
modality: llm
hardware:
  gpu: RTX 5090
  count: 1
  price_usd_h: 0.60
  access: {kind: ssh, host: <gpu-host>, port: 22, user: <user>, key_path: ~/.ssh/<key>}
goals:
  lead: single_user
  targets:
    decode_ceiling_fraction: 0.70
    buckets: [short, medium]
  slo: {ttft_p95_ms: 500, tpot_p95_ms: 50}
io_modes: {input: complete, output: streaming}
buckets: [very_short, short, medium, long]
workload: {max_tokens: 256, rounds: 5}
budget: {gpu_hours: 8, usd: 6, wall_clock_h: 8, approved: true}
quality: {reference: none, noise_floor: measure}
repo: {path: <path to the ledger repo on the agent's machine>, push: never}
constraints:
  - no changes to precision
  - engines must be open source
non_interactive: true
"""


def test_the_0_2_0_qwen_example_still_validates_with_the_new_defaults():
    pytest.importorskip("yaml")
    r = intake.check(intake.parse_text(QWEN_5090_V020), allow_placeholders=True)
    stop, d = r["derived"]["stop_rule"], r["derived"]
    assert stop["kind"] == "target" and stop["after_target"] == "keep_climbing"
    assert stop["after_target_source"] == "default: non_interactive"
    assert stop["text"].startswith("keep climbing until the budget ends")
    assert d["fidelity_rule"]["value"] == "lossless" and d["fidelity_rule"]["source"] == "default"
    assert d["precision_rule"].startswith("every ranked engine runs weights/activations in one of ['nvfp4'] and KV "
                                          "cache in one of ['checkpoint_default'];")
    assert not any("fidelity" in w or "after_target" in w for w in r["warnings"])
    # the older way to opt in, a constraint string, still works with a target
    text = QWEN_5090_V020.replace("non_interactive: true", "non_interactive: false").replace(
        "  - engines must be open source", '  - engines must be open source\n  - "after_target: keep_climbing"')
    stop = intake.check(intake.parse_text(text), allow_placeholders=True)["derived"]["stop_rule"]
    assert stop["kind"] == "target" and stop["after_target_source"] == "constraints"
    assert stop["text"].startswith("keep climbing until the budget ends")
    # and without it an attended run with this budget still keeps climbing by default (the budget is fixed)
    stop = intake.check(intake.parse_text(text.replace('  - "after_target: keep_climbing"\n', "")),
                        allow_placeholders=True)["derived"]["stop_rule"]
    assert stop["after_target_source"] == "default: budget.gpu_hours, usd or wall_clock_h is set"


@needs_examples
def test_lossy_example_carries_its_budget_and_rules():
    pytest.importorskip("yaml")
    data, _ = intake.load_file(EXAMPLES / "llm-lossy-capacity.yaml")
    d = intake.check(data, allow_placeholders=True)["derived"]
    assert d["fidelity_rule"]["value"] == "lossy" and set(d["quality_budget"]) == set(intake.QUALITY_BUDGET_KEYS)
    assert d["speculation"]["value"] == "off" and d["speculation"]["source"] == "constraints"
    assert d["route"] == {**d["route"], "value": "patch", "source": "default: goals.lead capacity"}
    assert d["stop_rule"]["after_target"] == "stop" and d["stop_rule"]["text"].startswith("stop when every target")


def test_cli_validate_yaml_file(tmp_path, capsys):
    pytest.importorskip("yaml")
    p = tmp_path / "intake.yaml"
    p.write_text(f"""
model: {{id: org/model, revision: "{SHA}", precision: bf16}}
modality: llm
hardware: {{gpu: RTX 5090}}
goals:
  targets: {{decode_tok_s: 100}}
""")
    assert cli.main(["intake", "validate", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["derived"]["targets"][0]["name"] == "decode_tok_s"
    p.write_text("model: [unclosed\n")
    assert cli.main(["intake", "validate", str(p)]) == 1
    assert "invalid YAML" in capsys.readouterr().err


def test_template_shows_fidelity_quality_budget_and_after_target():
    pytest.importorskip("yaml")
    assert "fidelity: lossless" in intake.TEMPLATE and "# after_target: stop" in intake.TEMPLATE
    assert all(f"#   {k}:" in intake.TEMPLATE for k in intake.QUALITY_BUDGET_KEYS)
    r = intake.check(intake.parse_text(intake.TEMPLATE), allow_placeholders=True)
    assert r["intake"]["fidelity"] == "lossless" and r["intake"]["goals"]["after_target"] == "keep_climbing"
    # the commented-out budget block, uncommented, is a valid lossy intake
    text = intake.TEMPLATE.replace("fidelity: lossless", "fidelity: lossy", 1).replace("  # budget:", "  budget:", 1)
    for k in intake.QUALITY_BUDGET_KEYS:
        text = text.replace(f"  #   {k}:", f"    {k}:", 1)
    r = intake.check(intake.parse_text(text), allow_placeholders=True)
    assert set(r["derived"]["quality_budget"]) == set(intake.QUALITY_BUDGET_KEYS)


def test_empty_yaml_blocks_mean_not_given():
    d = base(goals={"lead": "single_user", "targets": None}, repo=None)
    r = intake.check(d)
    assert r["intake"]["goals"]["targets"] == {} and r["intake"]["repo"] is None


def test_run_dir_names_the_run_when_the_intake_does_not(tmp_path):
    src = write_json(tmp_path, base())
    run = tmp_path / "runs" / "my-run"
    r = registry.call("intake_validate", {"file": str(src), "run_dir": str(run)})
    assert r["intake"]["run_id"] == "my-run"
    assert json.loads((run / "state.json").read_text())["run_id"] == "my-run"
    src2 = write_json(tmp_path, base(run_id="other"), name="i2.json")
    r2 = registry.call("intake_validate", {"file": str(src2), "run_dir": str(tmp_path / "runs" / "x")})
    assert any("differs from the run directory" in w for w in r2["warnings"])


# ---------------------------------------------------------------- the skill's bytes/token helper (ceilings.md)

BPT = REPO / "plugin" / "skills" / "optimizing-inference" / "scripts" / "bytes_per_token.py"


def _checkpoint(d: Path, cfg: dict, tensors: dict) -> Path:
    """A fake checkpoint: config.json + one safetensors file holding only a header (name -> byte size)."""
    import struct

    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    off, hdr = 0, {"__metadata__": {"format": "pt"}}
    for name, size in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": [size], "data_offsets": [off, off + size]}
        off += size
    blob = json.dumps(hdr).encode()
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return d


def _bpt(*args):
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "-I", str(BPT), *map(str, args)], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else None), p.stderr


CFG = {"num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
       "hidden_size": 32, "vocab_size": 10, "tie_word_embeddings": False}
DENSE = {"model.embed_tokens.weight": 640, "lm_head.weight": 640,
         "model.layers.0.mlp.w": 1000, "model.layers.1.mlp.w": 1000}
KV_PER_LAYER_TOKEN = 2 * 2 * 8 * 2  # 2 (K and V) x kv_heads x head_dim x 2 bytes


def test_bytes_per_token_dense_full_attention(tmp_path):
    rc, out, err = _bpt(_checkpoint(tmp_path / "m", CFG, DENSE), 100)
    assert rc == 0, err
    assert out["weights_per_token"] == 2000 + 640 + 64  # every-token weights + lm_head + one embedding row
    assert out["kv_bytes"] == KV_PER_LAYER_TOKEN * 100 * 2
    assert out["bytes_per_token"] == out["weights_per_token"] + out["kv_bytes"]
    assert out["checkpoint_bytes"] == 3280 and out["claim_class"] == "derived"


def test_bytes_per_token_refuses_moe_config_without_matching_expert_tensors(tmp_path):
    cfg = {**CFG, "num_experts": 8, "num_experts_per_tok": 2}
    tensors = {**DENSE, **{f"model.layers.0.moe.e{i}.w": 100 for i in range(8)}}
    m = _checkpoint(tmp_path / "m", cfg, tensors)
    rc, _, err = _bpt(m, 100)
    assert rc == 2 and "--expert-pattern" in err and "model.layers.0.moe.e0.w" in err
    rc, out, err = _bpt(m, 100, "--expert-pattern", r"\.moe\.e\d+\.")
    assert rc == 0, err
    assert out["moe"]["expert_fraction"] == 0.25 and out["groups_bytes"]["routed_experts"] == 800
    assert out["weights_per_token"] == 2000 + 640 + 64 + 800 * 0.25


def test_bytes_per_token_uses_layer_types_for_sliding_and_linear_layers(tmp_path):
    cfg = {**CFG, "sliding_window": 16, "layer_types": ["sliding_attention", "full_attention"]}
    rc, out, err = _bpt(_checkpoint(tmp_path / "a", cfg, DENSE), 100)
    assert rc == 0, err
    assert out["kv_bytes"] == KV_PER_LAYER_TOKEN * (16 + 100)
    cfg = {**CFG, "layer_types": ["linear_attention", "full_attention"], "linear_num_key_heads": 4}
    rc, out, err = _bpt(_checkpoint(tmp_path / "b", cfg, DENSE), 100)
    assert rc == 0, err
    assert out["kv_bytes"] == KV_PER_LAYER_TOKEN * 100 and out["attention"]["layers"]["state"] == 1
    assert any("NOT counted" in n for n in out["notes"])


def test_bytes_per_token_never_assumes_full_attention_silently(tmp_path):
    m = _checkpoint(tmp_path / "m", {**CFG, "sliding_window": 16, "sliding_window_pattern": 2}, DENSE)
    rc, _, err = _bpt(m, 100)
    assert rc == 2 and "--layer-types" in err
    rc, out, err = _bpt(m, 100, "--layer-types", "sliding,full")
    assert rc == 0 and out["kv_bytes"] == KV_PER_LAYER_TOKEN * (16 + 100)
    rc, out, err = _bpt(m, 100, "--layer-types", "full")  # an explicit, recorded statement
    assert rc == 0 and out["attention"]["source"] == "--layer-types"
    # a dense config whose sliding window is switched off is plain full attention
    off = {**CFG, "sliding_window": None, "use_sliding_window": False, "max_window_layers": 2}
    rc, out, err = _bpt(_checkpoint(tmp_path / "q", off, DENSE), 100)
    assert rc == 0, err
    # per-layer-type KV shapes or MLA need a hand-derived KV term
    rc, _, err = _bpt(_checkpoint(tmp_path / "g", {**CFG, "num_global_key_value_heads": 1}, DENSE), 100)
    assert rc == 2 and "--kv-bytes-total" in err


def test_explicit_uncapped_deadline_and_no_stage_or_reserve_cutoff():
    result = intake.check(base(non_interactive=True, budget={
        "approved": True, "spend_cap": "none", "deadline": "2026-10-26T12:00:00+04:00",
        "stage_timeboxes": False, "report_reserve_h": 0}))
    derived = result["derived"]
    assert derived["budget"]["usd_cap"] is None and derived["budget"]["host_hours_cap"] is None
    assert derived["budget"]["spend_cap"] == "none"
    assert set(derived["time_boxes_h"].values()) == {None} and derived["report_reserve_h"] == 0
    state = intake.initial_state(result, "fixture")
    state["budget"]["run_started"] = "2026-10-24T08:00:00Z"
    state["stages"]["1_intake"].update(started="2026-10-24T08:00:00Z", time_box_h=0.01)
    before = intake.budget_status(state, _now("2026-10-26T07:59:59Z"))
    assert before["verdict"] == "ok" and before["cap"]["report_reserve_h"] == 0
    assert before["stage"]["time_box_h"] is None
    assert intake.budget_status(state, _now("2026-10-26T08:00:00Z"))["verdict"] == "over_budget"


@pytest.mark.parametrize("budget, message", [
    ({"spend_cap": "none", "approved": True}, "finite"),
    ({"spend_cap": "none", "approved": False, "wall_clock_h": 2}, "approved"),
    ({"spend_cap": "none", "approved": True, "wall_clock_h": 2, "usd": 10}, "conflicts"),
    ({"spend_cap": "none", "approved": True, "deadline": "2026-10-26T12:00:00"}, "timezone"),
    ({"spend_cap": "none", "approved": True, "deadline": "not a date"}, "ISO8601"),
])
def test_uncapped_spend_requires_explicit_approval_finite_deadline_and_no_conflicting_caps(budget, message):
    with pytest.raises(OpError, match=message):
        intake.check(base(non_interactive=True, budget=budget))


def test_deadline_default_reserve_applies_without_disabling_existing_stage_policies():
    result = intake.check(base(budget={"approved": True, "spend_cap": "none", "wall_clock_h": 10,
                                      "deadline": "2026-10-26T08:00:00Z"}))
    assert result["derived"]["budget"]["usd_cap"] is None
    assert result["derived"]["stage_timeboxes"] is True
    assert result["derived"]["time_boxes_h"]["7_custom"] == 2.3
    state = intake.initial_state(result, "fixture")
    state["budget"]["run_started"] = "2026-10-26T00:00:00Z"
    assert intake.budget_status(state, _now("2026-10-26T07:30:00Z"))["verdict"] == "reserve_only"


@pytest.mark.parametrize("modality, target, budget", [
    ("image", {"image_ms": 20000}, {"ssim_min": 0.9, "psnr_db_min": 30, "lpips_max": 0.1}),
    ("tts", {"ttfa_p95_ms": 2000, "rtf": 1}, {"wer_max": 0.05, "cer_max": 0.03, "speaker_similarity_min": 0.8}),
])
def test_affordability_preregisters_memory_latency_and_modality_quality(modality, target, budget):
    r = intake.check(base(modality=modality, non_interactive=True, fidelity="lossy", goals={
        "lead": "affordability", "affordability": {"memory_gb_max": 24, "stretch_memory_gb_max": 16,
            "memory_kind": "unified", "offload_allowed": True}, "targets": target},
        quality={"protocol": "Pinned prompts/seeds; reference A/A; paired CI; human review for artifacts.", "budget": budget}))
    assert all(t["applies"] for t in r["derived"]["targets"])
    assert r["derived"]["affordability"]["stretch_memory_gb_max"] == 16
    assert r["derived"]["quality_budget"] == budget
    assert "no LLM token/GSM8K proxy" in r["derived"]["fidelity_rule"]["text"]
    assert not any("do not apply to modality" in w for w in r["warnings"])
    if modality == "image":
        assert r["derived"]["modes"] == ["image"]
        assert "KV cache" not in r["derived"]["precision_rule"]


def test_affordability_cannot_silently_drop_latency_quality_or_stretch_constraints():
    with pytest.raises(OpError, match="quality.protocol"):
        intake.check(base(goals={"lead": "affordability", "affordability": {
            "memory_gb_max": 16, "stretch_memory_gb_max": 24, "memory_kind": "vram"}}))


def test_affordability_latency_target_must_apply_to_its_modality():
    with pytest.raises(OpError, match="explicit latency target"):
        intake.check(base(modality="image", goals={"lead": "affordability", "affordability": {
            "memory_gb_max": 16, "memory_kind": "vram"}, "targets": {"ttft_ms": 2000}},
            quality={"protocol": "Pinned image adherence/perceptual review with A/A floor."}))


def test_deadline_only_interactive_policy_keeps_climbing_by_default():
    r = intake.check(base(goals={"targets": {"decode_tok_s": 100}}, budget={
        "approved": True, "spend_cap": "none", "deadline": "2026-10-26T12:00:00+04:00", "report_reserve_h": 0}))
    stop = r["derived"]["stop_rule"]
    assert stop["after_target"] == "keep_climbing" and "budget.deadline" in stop["after_target_source"]
    assert "no early reserve" in stop["text"] and "report reserve is kept" not in stop["text"]


@pytest.mark.parametrize("preference,fidelity", [
    ("exact", "lossless"), ("balanced", "lossy"), ("speed", "lossy"),
])
def test_simple_quality_preference_roundtrips_without_granting_permissions(preference, fidelity):
    data = base(quality={"preference": preference},
                interaction={"checkpoints": "required", "show_plan": False})
    result = intake.check(data)
    assert result["intake"]["fidelity"] == fidelity
    assert result["derived"]["fidelity_rule"]["source"] == "quality.preference"
    assert result["derived"]["quality_preference"] == preference
    assert result["derived"]["architecture_changes"] == "forbid"
    assert result["derived"]["quality_budget"] is None
    assert result["intake"]["model"]["precision_allowed"] == ["fp8"]
    again = intake.check(result["intake"])
    assert again["intake"] == result["intake"]
    assert again["derived"] == result["derived"]


@pytest.mark.parametrize("preference,conflict", [("exact", "lossy"), ("balanced", "lossless"), ("speed", "lossless")])
@pytest.mark.parametrize("via_constraint", [False, True])
def test_quality_preference_conflicts_are_errors(preference, conflict, via_constraint):
    data = base(quality={"preference": preference})
    data.update({"constraints": [f"fidelity: {conflict}"]} if via_constraint else {"fidelity": conflict})
    with pytest.raises(OpError, match="quality.preference"):
        intake.check(data)


def test_speed_preference_needs_real_quality_budget_and_spend_authorization():
    with pytest.raises(OpError, match="quality.budget"):
        intake.check(base(non_interactive=True, quality={"preference": "speed"}))
    with pytest.raises(OpError, match="approved"):
        intake.check(base(non_interactive=True, budget={"gpu_hours": 1},
                          quality={"preference": "speed", "budget": {"gsm8k_drop_pts_max": 1}}))
    data = base(non_interactive=True, quality={"preference": "speed", "budget": {"gsm8k_drop_pts_max": 1}})
    data["model"]["architecture_changes"] = "ask"
    result = intake.check(data)
    assert result["derived"]["architecture_changes"] == "ask"
    assert result["derived"]["quality_budget"]["gsm8k_drop_pts_max"] == 1
    assert not result["derived"]["interaction"]["show_plan"]


@pytest.mark.parametrize("checkpoint", ["quality", "milestones"])
def test_unattended_runs_cannot_promise_human_checkpoints(checkpoint):
    with pytest.raises(OpError, match="human"):
        intake.check(base(non_interactive=True, interaction={"checkpoints": checkpoint}))


def test_required_checkpoints_do_not_reask_to_continue_authorized_budget():
    result = intake.check(base(interaction={"checkpoints": "required"}))
    assert result["derived"]["stop_rule"]["kind"] == "budget"
    assert intake.check(base())["derived"]["stop_rule"]["kind"] == "ask"  # legacy intake
    result = intake.check(base(interaction={"checkpoints": "required"}, goals={"after_target": "stop"}))
    assert result["derived"]["stop_rule"]["kind"] == "stop_point"


def test_exact_preference_is_stronger_than_legacy_lossless_description():
    exact = intake.check(base(quality={"preference": "exact"}))
    legacy = intake.check(base())
    assert exact["derived"]["fidelity_rule"]["text"].startswith("exact: equality")
    assert exact["derived"]["fidelity_rule"]["text"] != legacy["derived"]["fidelity_rule"]["text"]
    assert any("identical outputs" in warning for warning in exact["warnings"])
