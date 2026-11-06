"""Resume checks must stay read-only and must never turn missing evidence into proof."""

import json
from pathlib import Path

import pytest

from inferencekit import cli, evidence as ev, registry
from inferencekit.mcp_server import fit
from inferencekit.registry import OpError

from conftest import fake_call


NOW = "2026-01-01T01:00:00Z"


def write(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = tmp_path / "original.json"
    write(original, {
        "model": {"id": "org/model", "revision": "a" * 40, "precision": "fp8"},
        "modality": "llm", "hardware": {"gpu": "Test GPU", "count": 1, "access": {"kind": "local"}},
        "budget": {"wall_clock_h": 4, "approved": True, "stage_timeboxes": False, "report_reserve_h": 0},
    })
    root = tmp_path / "runs" / "resume-test"
    registry.call("intake_validate", {"file": str(original), "run_dir": str(root)})
    state = json.loads((root / "state.json").read_text())
    state["budget"]["run_started"] = "2026-01-01T00:00:00Z"
    state["budget"]["stage_timeboxes"] = False
    state["budget"]["report_reserve_h"] = 0
    state["current_stage"] = "4_tuning"
    state["stages"]["4_tuning"]["status"] = "in_progress"
    state["stages"]["4_tuning"]["started"] = "2026-01-01T00:00:00Z"
    state["stages"]["3_candidates"]["status"] = "done"
    state["next_action"] = "screen the pending candidate; reuse completed baseline"
    store = ev.Store(root / "evidence")
    rec = store.save(ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="baseline screen",
                             data={"verdict": "UNPROVEN"}, missing=["full quality proof"],
                             limitations=["screening only"]))
    state["stages"]["3_candidates"]["evidence"] = [rec["evidence_id"]]
    state["candidates"]["baseline"] = {"evidence": [rec["evidence_id"]]}
    state["best_history"] = [{"checks": "screened", "evidence": [rec["evidence_id"]]}]
    write(root / "state.json", state)
    return root, original, state, rec


def resume(root, **args):
    return registry.call("intake_resume", {"run_dir": str(root), "now": NOW, **args})


def snapshot(root):
    return {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_resume_checks_deduplicated_evidence_and_preserves_state_and_incomplete_proof(run):
    root, _, state, rec = run
    before = snapshot(root.parent.parent)
    result = resume(root)
    assert snapshot(root.parent.parent) == before
    assert result["status"] == "context_ready" and result["evidence_summary"]["invalid"] == 0
    assert result["current_stage"] == state["current_stage"] and result["next_action"] == state["next_action"]
    assert result["stages"]["3_candidates"]["status"] == "done"
    assert result["evidence_root"] == str(root / "evidence")
    assert len(result["items"]) == 1
    row = result["items"][0]
    assert row["evidence_id"] == rec["evidence_id"] and row["status"] == "available"
    assert row["integrity"] == {"id_ok": True, "schema_ok": True, "schema_errors": [], "cited_id_ok": True}
    assert row["coverage"] == {"complete": False, "missing": ["full quality proof"]}
    assert row["limitations"] == ["screening only"]
    assert row["cited_by"] == ["stages.3_candidates.evidence", "candidates.baseline.evidence",
                                "best_history.0.evidence"]
    assert "not a benchmark or quality proof" in result["scope"]


def test_resume_matches_existing_budget_and_keeps_explicit_overrides(run):
    root, _, _, _ = run
    result = resume(root)
    expected = registry.call("intake_budget", {"run_dir": str(root), "now": NOW})
    assert {k: v for k, v in result["budget"].items() if k != "basis"} == {
        k: v for k, v in expected.items() if k != "run_dir"}
    assert "mutable checkpoint" in result["budget"]["basis"]
    assert "not verified authorization" in result["budget"]["basis"]
    assert result["budget"]["stage"]["time_box_h"] is None
    assert result["budget"]["cap"]["report_reserve_h"] == 0
    assert result["budget"]["verdict"] == "ok"


def test_resume_policy_brief_uses_validated_intake_without_inferring_session_authorization(run):
    root, original, _, _ = run
    validated = registry.call("intake_validate", {"file": str(original)})
    result = resume(root)
    assert result["policy"] == {
        "source": "validated intake only; session authorization is not evaluated",
        "budget_approved": validated["intake"]["budget"]["approved"],
        "fidelity": validated["intake"]["fidelity"],
        "quality": validated["intake"]["quality"],
        "architecture_changes": validated["derived"]["architecture_changes"],
    }


def test_resume_keeps_over_budget_action(run):
    root, _, _, _ = run
    result = resume(root, now="2026-01-02T00:00:00Z")
    assert result["status"] == "needs_attention"
    assert result["budget"]["verdict"] == "over_budget"
    assert result["budget"]["action"].startswith("stop GPU work now")


def test_missing_evidence_is_reported_even_when_global_store_has_it(run, monkeypatch, tmp_path):
    root, _, _, rec = run
    folder = root / "evidence" / rec["evidence_id"]
    other = tmp_path / "other-store"
    other.mkdir()
    folder.rename(other / folder.name)
    monkeypatch.setenv("INFERENCEKIT_RUNS_DIR", str(other))
    result = resume(root)
    assert result["status"] == "needs_attention"
    assert result["evidence_summary"]["missing"] == 1
    assert result["items"][0]["evidence_id"] == rec["evidence_id"]
    assert result["items"][0]["status"] == "missing"
    assert result["stages"]["3_candidates"]["status"] == "done"
    assert result["budget"]["verdict"] == "ok"


@pytest.mark.parametrize("damage", ["hash", "schema", "id", "json", "array", "nonfinite"])
def test_corrupt_evidence_never_becomes_available(run, damage):
    root, _, _, rec = run
    path = root / "evidence" / rec["evidence_id"] / "evidence.json"
    if damage == "hash":
        rec["summary"] = "tampered"
    elif damage == "schema":
        rec.pop("summary")
        rec["evidence_id"] = ev.compute_id(rec)
    elif damage == "id":
        rec = ev.finalize(ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="another"))
    elif damage == "array":
        rec = []
    elif damage == "nonfinite":
        rec["data"]["x"] = float("nan")
    write(path, rec)
    if damage == "json":
        path.write_text("{invalid")
    result = resume(root)
    assert result["status"] == "needs_attention"
    assert result["items"][0]["status"] == "invalid"
    assert result["items"][0]["reason"]


def test_flat_run_directory_evidence_copy_is_supported(run):
    root, _, _, rec = run
    path = root / "evidence" / rec["evidence_id"] / "evidence.json"
    path.rename(root / "evidence" / f"{rec['evidence_id']}.json")
    result = resume(root)
    assert result["status"] == "context_ready" and result["items"][0]["status"] == "available"


@pytest.mark.parametrize("file", ["original", "state.json", "intake.normalized.json"])
def test_changed_intake_pin_is_refused_without_mutation(run, file):
    root, original, _, _ = run
    path = original if file == "original" else root / file
    data = json.loads(path.read_text())
    if file == "original":
        data["budget"]["wall_clock_h"] = 100
    else:
        data["intake_sha256"] = "changed"
    write(path, data)
    before = snapshot(root.parent.parent)
    with pytest.raises(OpError, match="intake changed"):
        resume(root)
    assert snapshot(root.parent.parent) == before


def test_redacted_source_is_refused(run):
    root, original, _, _ = run
    data = json.loads(original.read_text())
    data["hardware"]["access"] = {"kind": "ssh", "host": "<redacted>"}
    write(original, data)
    with pytest.raises(OpError, match="redacted"):
        resume(root)


def test_source_override_reuses_pinned_bytes_from_another_working_directory(run, monkeypatch, tmp_path):
    root, original, _, _ = run
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(OpError, match="pass --file"):
        resume(root)
    assert resume(root, file=str(original))["status"] == "context_ready"


@pytest.mark.parametrize("model_id", ["/home/first/models/test", "/home/second/models/test", "~/models/test"])
def test_resume_preserves_pinned_intake_across_home_directories(run, monkeypatch, model_id):
    from inferencekit.ops import intake

    root, original, _, _ = run
    data = json.loads(original.read_text())
    data["model"]["id"] = model_id
    source = original.with_name("moved-intake.json")
    fresh = root.with_name("moved-run")
    write(source, data)
    monkeypatch.setattr(intake, "_home", lambda: "/home/first")
    registry.call("intake_validate", {"file": str(source), "run_dir": str(fresh)})
    state = json.loads((fresh / "state.json").read_text())
    state["budget"]["run_started"] = "2026-01-01T00:00:00Z"
    write(fresh / "state.json", state)
    before = snapshot(fresh)
    monkeypatch.setattr(intake, "_home", lambda: "/home/second")
    assert resume(fresh, file=str(source))["status"] == "context_ready"
    assert snapshot(fresh) == before


@pytest.mark.parametrize("damage", ["suffix", "conflicting_home"])
def test_resume_home_redaction_cannot_hide_changed_policy(run, monkeypatch, damage):
    from inferencekit.ops import intake

    root, original, _, _ = run
    data = json.loads(original.read_text())
    data["model"].update(id="/home/first/models/test", tokenizer="/home/second/models/test")
    source = original.with_name("path-intake.json")
    fresh = root.with_name("path-run")
    write(source, data)
    monkeypatch.setattr(intake, "_home", lambda: "/home/first")
    registry.call("intake_validate", {"file": str(source), "run_dir": str(fresh)})
    path = fresh / "intake.normalized.json"
    normalized = json.loads(path.read_text())
    if damage == "suffix":
        normalized["intake"]["model"]["id"] = "~/models/other"
    else:
        normalized["intake"]["model"]["tokenizer"] = "~/models/test"
    write(path, normalized)
    monkeypatch.setattr(intake, "_home", lambda: "/home/third")
    before = snapshot(fresh)
    with pytest.raises(OpError, match="pinned run policy changed"):
        resume(fresh, file=str(source))
    assert snapshot(fresh) == before


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -float("inf")])
def test_resume_refuses_nonfinite_checkpoint_spending(run, price):
    root, _, state, _ = run
    state["budget"]["meter"] = [{"start": "2026-01-01T00:00:00Z", "end": NOW, "price_usd_h": price}]
    write(root / "state.json", state)
    before = snapshot(root)
    with pytest.raises(OpError, match="malformed budget/stage fields"):
        resume(root)
    assert snapshot(root) == before


@pytest.mark.parametrize("mutation", ["stage", "stage_type", "version", "citation", "budget", "time"])
def test_malformed_state_fields_are_refused(run, mutation):
    root, _, state, _ = run
    if mutation == "stage":
        state["current_stage"] = "missing"
    elif mutation == "stage_type":
        state["current_stage"] = []
    elif mutation == "version":
        state["state_version"] = 2
    elif mutation == "citation":
        state["stages"]["3_candidates"]["evidence"] = ["not-an-id"]
    elif mutation == "budget":
        state["budget"]["meter"] = [{"start": "bad-time"}]
    write(root / "state.json", state)
    with pytest.raises(OpError):
        resume(root, **({"now": "bad-time"} if mutation == "time" else {}))


def test_cli_and_registry_share_resume_arguments(run, capsys):
    root, original, _, _ = run
    assert cli.main(["intake", "resume", str(root), "--file", str(original), "--now", NOW]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == resume(root, file=str(original))


def test_resume_redacts_secrets_in_state_notes_and_next_action(run, monkeypatch):
    root, _, state, _ = run
    monkeypatch.setenv("HF_TOKEN", "hf_not_real_resume_secret")
    state["next_action"] = "test hf_not_real_resume_secret"
    state["stages"]["3_candidates"]["notes"] = ['api_key="not_real_inline_secret"']
    write(root / "state.json", state)
    text = json.dumps(resume(root))
    assert "hf_not_real_resume_secret" not in text and "not_real_inline_secret" not in text


@pytest.mark.parametrize("key,value", [
    ("host_hours_cap", 100), ("usd_cap", 100), ("spend_cap", "none"),
    ("deadline", "2099-01-01T00:00:00Z"), ("stage_timeboxes", True),
    ("report_reserve_h", 1), ("price_usd_h", 0.01),
])
def test_state_budget_policy_cannot_override_the_pinned_intake(run, key, value):
    root, _, state, _ = run
    state["budget"][key] = value
    write(root / "state.json", state)
    before = snapshot(root)
    with pytest.raises(OpError, match=f"state.json.budget.{key}"):
        resume(root, now="2026-01-01T05:00:00Z")
    assert snapshot(root) == before


@pytest.mark.parametrize("section,mutate", [
    ("intake", lambda d: d["quality"].update(fidelity="lossy")),
    ("intake", lambda d: d["workload"].update(chat_template_kwargs={"enable_thinking": True})),
    ("intake", lambda d: d["budget"].update(approved=False)),
    ("derived", lambda d: d["budget"].update(host_hours_cap=100)),
    ("derived", lambda d: d["fidelity_rule"].update(value="lossy")),
])
def test_normalized_policy_edits_are_detected_even_when_hash_strings_are_unchanged(run, section, mutate):
    root, _, _, _ = run
    path = root / "intake.normalized.json"
    data = json.loads(path.read_text())
    mutate(data[section])
    write(path, data)
    with pytest.raises(OpError, match=f"intake.normalized.json.{section}"):
        resume(root)


@pytest.fixture
def legacy_media_run(tmp_path):
    def create(modality, workload=None):
        original = tmp_path / "original.json"
        root = tmp_path / "run"
        data = {"run_id": "legacy-media", "model": {"id": "org/model", "revision": "a" * 40, "precision": "fp32"},
                "modality": modality, "hardware": {"gpu": "Test GPU", "count": 1, "access": {"kind": "local"}},
                "budget": {"wall_clock_h": 4, "approved": True, "stage_timeboxes": False, "report_reserve_h": 0}}
        if workload is not None:
            data["workload"] = workload
        write(original, data)
        registry.call("intake_validate", {"file": str(original), "run_dir": str(root)})
        state = json.loads((root / "state.json").read_text())
        state["budget"]["run_started"] = "2026-01-01T00:00:00Z"
        write(root / "state.json", state)
        normalized = json.loads((root / "intake.normalized.json").read_text())
        # a612865 normalized every modality with these defaults, retaining explicit values.
        normalized["intake"]["workload"].setdefault("max_tokens", 256)
        normalized["intake"]["workload"].setdefault("prompt_set", "dev")
        write(root / "intake.normalized.json", normalized)
        return root, original, normalized
    return create


@pytest.mark.parametrize("modality", ["tts", "image", "stt", "video"])
def test_legacy_implied_media_defaults_resume_without_changing_saved_policy(legacy_media_run, modality):
    root, original, normalized = legacy_media_run(modality)
    before = snapshot(root.parent)
    result = resume(root)
    assert result["status"] == "context_ready"
    assert result["intake"]["sha256"] == normalized["intake_sha256"]
    assert "workload" not in json.loads(original.read_text())
    assert snapshot(root.parent) == before


@pytest.mark.parametrize("modality", ["tts", "image"])
@pytest.mark.parametrize("key,value", [("max_tokens", 512), ("prompt_set", "changed")])
def test_legacy_media_defaults_do_not_allow_other_saved_values(legacy_media_run, modality, key, value):
    root, _, normalized = legacy_media_run(modality)
    normalized["intake"]["workload"][key] = value
    write(root / "intake.normalized.json", normalized)
    before = snapshot(root)
    with pytest.raises(OpError, match=f"intake.normalized.json.intake.workload.{key}"):
        resume(root)
    assert snapshot(root) == before


@pytest.mark.parametrize("modality", ["tts", "image"])
@pytest.mark.parametrize("key,explicit,saved", [("max_tokens", 512, 256), ("prompt_set", "custom", "dev"),
                                                ("max_tokens", 256, 512), ("prompt_set", "dev", "changed")])
def test_legacy_media_compatibility_preserves_explicit_source_fields(legacy_media_run, modality, key, explicit, saved):
    root, _, normalized = legacy_media_run(modality, {key: explicit})
    assert resume(root)["status"] == "context_ready"
    normalized["intake"]["workload"][key] = saved
    write(root / "intake.normalized.json", normalized)
    with pytest.raises(OpError, match=f"intake.normalized.json.intake.workload.{key}"):
        resume(root)


def test_legacy_media_default_compatibility_does_not_change_original_hash_check(legacy_media_run):
    root, original, _ = legacy_media_run("tts")
    data = json.loads(original.read_text())
    data["workload"] = {"max_tokens": 256, "prompt_set": "dev"}
    write(original, data)
    with pytest.raises(OpError, match="intake changed since this run started"):
        resume(root)


@pytest.mark.parametrize("field,value", [
    ("model", {"id": "different-model", "revision": "a" * 40}),
    ("workload_pin", {"chat_template_kwargs": {"enable_thinking": True}}),
])
def test_state_model_or_workload_pin_cannot_drift(run, field, value):
    root, _, state, _ = run
    state[field] = value
    write(root / "state.json", state)
    with pytest.raises(OpError, match=f"state.json.{field}"):
        resume(root)


def test_documented_progress_meter_and_stage_box_updates_are_allowed(run):
    root, _, state, _ = run
    state["budget"]["meter"] = [{"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:30:00Z",
                                  "price_usd_h": 0.6}]
    state["stages"]["4_tuning"].update(time_box_h=3, started="2026-01-01T00:10:00Z", notes=["time reassigned"])
    state["model"]["revision_resolved"] = "a" * 40
    state["decisions"].append({"choice": "continue", "asked_user": False})
    write(root / "state.json", state)
    result = resume(root)
    assert result["status"] == "context_ready"
    assert result["budget"]["spent"]["host_hours"] == 0.5
    assert result["budget"]["spent"]["usd"] == 0.3


@pytest.mark.parametrize("resolved", ["b" * 40, "main", "", 123])
def test_immutable_revision_pin_rejects_a_conflicting_resolution(run, resolved):
    root, _, state, _ = run
    state["model"]["revision_resolved"] = resolved
    write(root / "state.json", state)
    before = snapshot(root)
    with pytest.raises(OpError) as exc:
        resume(root)
    rejection = json.loads(str(exc.value))
    assert rejection["status"] == "blocked"
    assert "state.json.model.revision_resolved" in rejection["reason"]
    assert "Do not resume" in rejection["action"]
    assert snapshot(root) == before


@pytest.mark.parametrize("revision,resolved", [("a" * 40, "A" * 40), ("main", "b" * 40),
                                               ("release-v1", "c" * 40), (None, "d" * 40)])
def test_matching_sha_and_legitimate_branch_resolutions_are_allowed(run, revision, resolved):
    root, original, _, _ = run
    data = json.loads(original.read_text())
    data["model"]["revision"] = revision
    fresh_original = original.with_name("resolved-intake.json")
    write(fresh_original, data)
    fresh_root = root.with_name("resolved-run")
    registry.call("intake_validate", {"file": str(fresh_original), "run_dir": str(fresh_root)})
    state = json.loads((fresh_root / "state.json").read_text())
    state["model"]["revision_resolved"] = resolved
    state["budget"]["run_started"] = "2026-01-01T00:00:00Z"
    write(fresh_root / "state.json", state)
    assert resume(fresh_root)["status"] == "context_ready"


def test_mutable_clock_and_meter_never_establish_remaining_authorization(run):
    root, _, state, _ = run
    state["budget"]["run_started"] = "2025-12-31T20:00:00Z"
    state["budget"]["meter"] = [{"start": "2025-12-31T20:00:00Z", "end": NOW, "price_usd_h": 0.6}]
    write(root / "state.json", state)
    exhausted = resume(root)
    assert exhausted["status"] == "needs_attention"
    assert exhausted["budget"]["verdict"] == "over_budget"
    assert exhausted["budget"]["left"]["host_hours"] == -1
    state["budget"]["run_started"] = NOW
    state["budget"]["meter"] = []
    write(root / "state.json", state)
    before = snapshot(root)
    reset = resume(root)
    assert reset["status"] == "context_ready"
    assert reset["budget"]["left"]["host_hours"] == 4
    assert reset["budget"]["spent"]["usd"] == 0
    for result in (exhausted, reset):
        assert result["trust"]["checkpoint_history_authenticated"] is False
        assert result["trust"]["remaining_authorization_verified"] is False
        assert "do not authorize work" in result["trust"]["authorization"]
        assert "not verified authorization" in result["budget"]["basis"]
    assert snapshot(root) == before


def test_large_resume_result_keeps_late_failure_and_pages_without_losing_evidence(run):
    root, _, state, first = run
    store = ev.Store(root / "evidence")
    ids = [first["evidence_id"]]
    for i in range(199):
        rec = store.save(ev.build(fake_call(), kind="benchmark", claim_class="observed",
                                 summary=f"record {i}: " + "detail " * 80))
        ids.append(rec["evidence_id"])
    path = store.run_dir(ids[-1]) / "evidence.json"
    rec = json.loads(path.read_text())
    rec["summary"] = "edited after capture"
    write(path, rec)
    state["stages"]["3_candidates"]["evidence"] = ids
    write(root / "state.json", state)
    result = fit(resume(root, limit=200), max_chars=12000)
    assert result["status"] == "needs_attention"
    assert result["budget"]["verdict"] == "ok"
    assert result["evidence_summary"] == {"checked": 200, "available": 199, "missing": 0, "invalid": 1,
                                           "order": "missing/invalid first"}
    assert result["items"][0]["evidence_id"] == ids[-1]
    assert result["items"][0]["status"] == "invalid"
    assert result["next_offset"] == len(result["items"])
    seen = [item["evidence_id"] for item in result["items"]]
    while result["next_offset"] is not None:
        result = fit(resume(root, offset=result["next_offset"], limit=200), max_chars=12000)
        seen.extend(item["evidence_id"] for item in result["items"])
    assert len(seen) == len(set(seen)) == 200 and set(seen) == set(ids)


def test_resume_page_continuation_preserves_source_override_and_fixed_time(run):
    root, original, state, rec = run
    state["stages"]["3_candidates"]["evidence"].append("ev_" + "b" * 64)
    write(root / "state.json", state)
    first = resume(root, file=str(original), limit=1)
    assert first["next_call"] == {"tool": "intake_resume", "arguments": {
        "run_dir": str(root), "file": str(original), "now": NOW, "offset": 1, "limit": 1}}
    second = registry.call(first["next_call"]["tool"], first["next_call"]["arguments"])
    assert second["items"][0]["evidence_id"] == rec["evidence_id"] and second["next_offset"] is None
    assert second["budget"] == first["budget"]


@pytest.mark.parametrize("field", ["notes", "next_action", "best"])
def test_oversized_checkpoint_details_keep_resume_failures_and_usable_continuations(run, field):
    from inferencekit.ops.store import get_path

    root, original, state, rec = run
    huge = ["first log " + "a" * 20000, "second log " + "b" * 20000]
    if field == "notes":
        state["stages"]["4_tuning"]["notes"] = huge
    elif field == "best":
        state["best"] = {"single_user": {"notes": huge}}
    else:
        state["next_action"] = "next action " + " ".join(huge)
    state["stages"]["3_candidates"]["evidence"].append("ev_" + "b" * 64)
    write(root / "state.json", state)
    before = snapshot(root)
    raw = resume(root, file=str(original))
    raw_bytes = json.dumps(raw)
    shown = fit(raw)
    assert json.dumps(raw) == raw_bytes and snapshot(root) == before
    assert len(json.dumps(shown)) <= 32000
    assert shown["status"] == "needs_attention"
    assert shown["trust"] == raw["trust"] and shown["budget"] == raw["budget"]
    assert shown["policy"] == raw["policy"] and shown["intake"] == raw["intake"]
    assert shown["evidence_summary"] == raw["evidence_summary"]
    assert shown["items"][0]["status"] == "missing"
    assert shown["state_path"] == str(root / "state.json")
    if shown.get("truncated"):
        assert "neither proof nor authorization" in shown["note"]
        for pointer in shown["detail_sources"]:
            assert get_path(json.loads(Path(pointer["file"]).read_text()), pointer["path"])
    else:  # A single long string can already fit after selecting one evidence row.
        assert len(json.dumps(raw)) <= 32000
    seen = [row["evidence_id"] for row in shown["items"]]
    for _ in range(3):
        if shown["next_offset"] is None:
            break
        previous_offset = shown["offset"]
        call = shown["next_call"]
        assert call["arguments"]["file"] == str(original) and call["arguments"]["now"] == NOW
        shown = fit(registry.call(call["tool"], call["arguments"]))
        assert shown["offset"] > previous_offset
        seen.extend(row["evidence_id"] for row in shown["items"])
    assert shown["next_offset"] is None
    assert len(seen) == len(set(seen)) == 2 and rec["evidence_id"] in seen


def test_small_resume_result_keeps_exact_payload_and_oversized_rejection_stays_blocked(run):
    root, original, state, _ = run
    raw = resume(root)
    before = json.dumps(raw)
    assert fit(raw) is raw and json.dumps(fit(raw)) == before
    state["next_action"] = "saved action " * 5000
    state["stages"]["4_tuning"]["notes"] = ["a" * 50000, "b" * 50000]
    state["best"] = {"single_user": "c" * 50000}
    write(root / "state.json", state)
    data = json.loads(original.read_text())
    data["budget"]["wall_clock_h"] = 40
    write(original, data)
    with pytest.raises(OpError) as exc:
        resume(root)
    rejection = json.loads(str(exc.value))
    assert len(str(exc.value)) <= 24000
    assert rejection["status"] == "blocked" and rejection["action"].startswith("Do not resume.")
    assert rejection["saved_state"]["next_action"]["truncated"] is True
    assert "diagnostic only" in rejection["trust"]["authorization"]


def test_oversized_evidence_qualifiers_keep_coverage_and_direct_source_paths(run):
    from inferencekit.ops.store import get_path

    root, _, state, _ = run
    rec = ev.Store(root / "evidence").save(ev.build(
        fake_call(), kind="benchmark", claim_class="observed", summary="screen " + "a" * 20000,
        missing=["missing proof " + "b" * 20000], limitations=["limitation " + "c" * 20000]))
    state["stages"]["3_candidates"]["evidence"] = [rec["evidence_id"]]
    state["candidates"] = {}
    state["best_history"] = []
    write(root / "state.json", state)
    shown = fit(resume(root))
    assert len(json.dumps(shown)) <= 32000
    assert shown["trust"]["remaining_authorization_verified"] is False
    row = shown["items"][0]
    assert row["status"] == "available" and row["coverage"]["complete"] is False
    assert row["summary"] == rec["summary"]
    for field, key in ((row, "limitations"), (row["coverage"], "missing")):
        cut = field[key]
        assert cut["truncated"] is True
        source = cut["source"]
        assert get_path(json.loads(Path(source["file"]).expanduser().read_text()), source["path"])
    assert shown["next_offset"] is None


def test_unshrinkable_resume_metadata_stops_with_bounded_cli_recovery(run):
    root, original, state, _ = run
    # Stage names have no length constraint. This valid checkpoint also repeats the name in budget.stage.
    name = "custom_stage_" + "x" * 40000
    state["stages"][name] = dict(state["stages"]["4_tuning"])
    state["current_stage"] = name
    state["stages"][name]["evidence"] = ["ev_" + "b" * 64]
    write(root / "state.json", state)
    raw = resume(root, file=str(original))
    shown = fit(raw)
    assert len(json.dumps(shown)) <= 32000
    assert shown["status"] == "needs_attention" and shown["resume_context_status"] == raw["status"]
    assert shown["trust"] == raw["trust"] and shown["evidence_summary"] == raw["evidence_summary"]
    assert shown["evidence_summary"]["missing"] == 1
    assert "budget" in shown["omitted_fields"] and "CLI" in shown["recovery"]
    assert "next_call" not in shown and "next_offset" not in shown
    assert shown["state_path"] == str(root / "state.json")
    assert shown["cli_arguments"] == {"run_dir": str(root), "file": str(original), "now": NOW}


@pytest.mark.parametrize("args", [{"offset": -1}, {"limit": 0}])
def test_resume_rejects_invalid_page_arguments(run, args):
    with pytest.raises(OpError, match="offset.*limit"):
        resume(run[0], **args)


def blocked(root, **args):
    with pytest.raises(OpError) as exc:
        resume(root, **args)
    result = json.loads(str(exc.value))
    assert result["status"] == "blocked" and result["action"].startswith("Do not resume.")
    assert "diagnostic only" in result["trust"]["authorization"]
    assert "budget" not in result and "policy" not in result  # no validated success fields
    return result


def drift(original):
    data = json.loads(original.read_text())
    data["budget"]["wall_clock_h"] = 80
    write(original, data)


def test_drift_rejection_retains_read_only_independent_evidence_and_unvalidated_state_diagnostics(run):
    root, original, state, rec = run
    drift(original)
    before = snapshot(root.parent.parent)
    result = blocked(root)
    assert snapshot(root.parent.parent) == before
    assert result["saved_state"]["current_stage"] == "4_tuning"
    assert result["saved_state"]["next_action"] == state["next_action"]
    assert result["saved_policy_snapshot_unvalidated"]["budget"]["wall_clock_h"] == 4
    assert result["supplied_intake_snapshot_not_pinned"]["budget"]["wall_clock_h"] == 80
    assert result["saved_budget_calculation_unvalidated"]["left"]["host_hours"] == 3
    assert result["evidence_summary"]["checked"] == result["evidence_summary"]["available"] == 1
    assert result["evidence_summary"]["incomplete_records"] == 1
    assert result["evidence_summary"]["coverage_missing"]["items"] == ["full quality proof"]
    assert result["evidence_summary"]["limitations"]["items"] == ["screening only"]
    assert result["evidence_issue_page"]["total"] == 0


@pytest.mark.parametrize("mutation", ["state_cap", "normalized_quality", "state_model"])
def test_policy_rejections_remain_blocked_even_when_diagnostic_budget_says_continue(run, mutation):
    root, _, state, _ = run
    if mutation == "state_cap":
        state["budget"]["host_hours_cap"] = 100
        write(root / "state.json", state)
    elif mutation == "state_model":
        state["model"]["id"] = "another-model"
        write(root / "state.json", state)
    else:
        path = root / "intake.normalized.json"
        data = json.loads(path.read_text())
        data["intake"]["fidelity"] = "lossy"
        write(path, data)
    result = blocked(root)
    assert "pinned run policy changed" in result["reason"]
    assert result["saved_budget_calculation_unvalidated"]["action"] == "continue"
    assert result["trust"]["saved_policy_and_budget"].startswith("UNPROVEN")


def test_drift_diagnostics_check_corrupt_evidence_independently(run):
    root, original, _, rec = run
    drift(original)
    rec["summary"] = "edited after capture"
    write(root / "evidence" / rec["evidence_id"] / "evidence.json", rec)
    result = blocked(root)
    assert result["evidence_summary"]["invalid"] == 1
    assert result["evidence_issue_page"]["items"][0]["evidence_id"] == rec["evidence_id"]


def test_blocked_issue_pages_are_bounded_and_preserve_all_failure_ids(run):
    root, original, state, _ = run
    drift(original)
    ids = [f"ev_{i:064x}" for i in range(37)]
    state["stages"]["3_candidates"]["evidence"] += ids
    state["next_action"] = "long diagnostic action " * 1000
    write(root / "state.json", state)
    result = blocked(root, file=str(original), limit=1000)
    assert len(json.dumps(result)) < 24000
    assert result["saved_state"]["next_action"]["truncated"]
    seen = []
    while True:
        assert result["evidence_summary"]["missing"] == 37
        seen.extend(item["evidence_id"] for item in result["evidence_issue_page"]["items"])
        if "next_call" not in result:
            break
        args = result["next_call"]["arguments"]
        assert args["now"] == NOW and args["limit"] == 10
        with pytest.raises(OpError) as exc:
            registry.call(result["next_call"]["tool"], args)
        result = json.loads(str(exc.value))
        assert result["status"] == "blocked"
    assert len(seen) == 37 and set(seen) == set(ids)


def test_drift_with_malformed_citations_does_not_report_an_empty_successful_audit(run):
    root, original, state, _ = run
    drift(original)
    state["stages"]["3_candidates"]["evidence"] = "invalid"
    write(root / "state.json", state)
    result = blocked(root)
    assert result["evidence_summary"]["complete_check"] is False
    assert "error" in result["evidence_summary"]
    assert "available" not in result["evidence_summary"]


def test_cli_drift_diagnostics_still_exit_one_with_no_success_output(run, capsys):
    root, original, _, _ = run
    drift(original)
    assert cli.main(["intake", "resume", str(root), "--file", str(original), "--now", NOW]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    result = json.loads(out.err.removeprefix("inferencekit: error: "))
    assert result["status"] == "blocked" and result["evidence_summary"]["checked"] == 1


def test_nonfinite_saved_policy_still_produces_a_structured_rejection(run):
    root, _, _, _ = run
    path = root / "intake.normalized.json"
    data = json.loads(path.read_text())
    data["intake"]["budget"]["wall_clock_h"] = float("nan")
    write(path, data)
    result = blocked(root)
    assert result["saved_policy_snapshot_unvalidated"]["budget"]["wall_clock_h"] == "see source (non-finite value)"
