"""Speech quality uses synthetic PCM and imported observations only, never models."""

import copy
import json
import math
import shutil
import struct
import time

import pytest

from inferencekit import evidence as ev
from inferencekit.ops import quality_audio as op
from inferencekit.ops.bench_audio import load_fixtures, round_seed
from inferencekit.ops.report import build_report
from inferencekit.quality import audio as qa
from inferencekit.registry import Call, OpError, REGISTRY
from conftest import fake_call


def invoke(name, **kwargs):
    operation = REGISTRY[name]
    args = operation.validate(kwargs)
    return operation.handler(Call(operation, args), **args)


def fixture_file(tmp_path, n=3, reference=False):
    path = tmp_path / "fixtures.jsonl"
    rows = [{"id": f"f{i}", "text": "The meeting starts tomorrow morning.", "unseen": True} for i in range(n)]
    if reference:
        (tmp_path / "voice.wav").write_bytes(b"synthetic voice reference")
        for row in rows:
            row.update(reference_audio="voice.wav", reference_text="Reference voice.")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def scorer(name):
    value = {"name": name, "version": "test-1", "model": "synthetic-scorer", "revision": "test-pinned",
             "config": {"device": "cpu", "vad_filter": True, "condition_on_previous_text": False}}
    if name in ("naturalness", "speaker"):
        value["scale"] = [1, 5] if name == "naturalness" else [-1, 1]
    return value


def policy_file(tmp_path, acceptance="exact", retry=False, listening=False, speaker=False):
    doc = {"acceptance": acceptance, "resamples": 100}
    if acceptance != "exact":
        doc.update(metrics={"wer": {"margin": 0.02, "outlier_wer": 1}, "naturalness": {"margin": 0.1}},
                   scorers={"asr": scorer("asr"), "naturalness": scorer("naturalness")})
        if retry:
            doc["scorers"]["asr_retry"] = scorer("asr_retry")
        if speaker:
            doc["metrics"]["speaker"] = {"margin": 0.02}
            doc["scorers"]["speaker"] = scorer("speaker")
        if listening:
            doc["metrics"]["listening"] = {"margin": 0.1, "min_raters": 2}
            doc["scorers"]["listening"] = {"name": "blinded_panel", "version": "1",
                                                   "protocol_sha256": "a" * 64, "scale": [1, 5],
                                                   "config": {"blinded": True}}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(doc))
    return path


def bench(store, fixtures, label, *, different=False, frames=16, rounds=2, keep=True, failed=False, start=None):
    fixture_rows, fixture_sha = load_fixtures(fixtures)
    config = {"sample_rate": 8000, "channels": 1, "seed_base": 0, "settings": {"temperature": 0.7}}
    rows, outputs = [], []
    now = time.time() + 1 if start is None else start
    round_rows = [{"index": r, "seed": round_seed(0, r), "t_start": now + r * 5, "t_end": now + r * 5 + 1}
                  for r in range(rounds)]
    with store.staging_dir() as staging:
        for r in range(rounds):
            for fixture in fixture_rows:
                pcm = struct.pack("<" + "h" * frames, *([101 if different else 100] * frames))
                row = {**fixture.identity(), "round": r, "seed": round_seed(0, r), "ok": not failed,
                       "sample_count": frames, "sample_rate": 8000, "channels": 1,
                       "pcm_sha256": qa.sha(pcm) if not failed else None, "pcm_bytes": len(pcm) if not failed else None}
                rows.append(row)
                if keep and not failed:
                    rel = f"outputs/{label}/r{r}/{fixture.id}.pcm"
                    target = staging / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(pcm)
                    outputs.append({"path": rel, "sha256": qa.sha(pcm), "bytes": len(pcm),
                                    "sample_rate": 8000, "channels": 1, "encoding": "pcm_s16le"})
        requests = staging / "requests.jsonl"
        requests.write_text("".join(json.dumps(row) + "\n" for row in rows))
        index = staging / f"outputs/{label}/index.json"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(outputs))
        artifacts = [ev.artifact(requests, staging, "requests"), ev.artifact(index, staging, "outputs_index")]
        artifacts.extend(ev.artifact(staging / x["path"], staging, "audio_pcm") for x in outputs)
        rec = ev.build(fake_call(), kind="audio_benchmark", claim_class="observed", summary="synthetic speech",
                       identity_=ev.identity(engine={"name": label}, model={"id": "test", "revision": "abc", "dtype": "fp32", "quant": "none"}),
                       artifacts=artifacts, data={"source": "bench audio", "label": label,
                       "fixture_set": {"sha256": fixture_sha, "fixtures": [f.identity() for f in fixture_rows]},
                       "rounds": round_rows, "audio_config": config, "total_requests": len(rows),
                       "structural_verdict": "FAIL" if failed else "PASS", "quality_verdict": "UNPROVEN"})
        return store.save(rec, staging)


def setup_run(tmp_path, store, acceptance="exact", *, retry=False, listening=False, reference=False, **candidate_args):
    fixtures = fixture_file(tmp_path, reference=reference)
    protocol = invoke("quality_audio_policy", policy=str(policy_file(tmp_path, acceptance, retry, listening, reference)),
                      fixtures=str(fixtures))
    baseline = bench(store, fixtures, "baseline")
    candidate = bench(store, fixtures, "candidate", **candidate_args)
    return fixtures, protocol, baseline, candidate


def score_file(tmp_path, store, run, *, candidate_text=None, retry_text=None, mos=4.0, listening=False, speaker=False):
    fixtures, protocol, baseline, candidate = run
    texts = {f.id: f.text for f in load_fixtures(fixtures)[0]}
    rows = []
    for arm, source in (("baseline", baseline), ("candidate", candidate)):
        _, requests = qa.load_benchmark(store, source["evidence_id"])
        for (fid, round_index), row in requests.items():
            asr = {"text": candidate_text if arm == "candidate" and candidate_text is not None else texts[fid]}
            if retry_text is not None:
                asr["retry_text"] = retry_text if arm == "candidate" else texts[fid]
            item = {"arm": arm, "id": fid, "round": round_index, "seed": row["seed"],
                    "pcm_sha256": row["pcm_sha256"], "text_sha256": row["text_sha256"], "asr": asr,
                    "naturalness": {"windows": [{"start_frame": 0, "end_frame": row["sample_count"],
                                                  "score": mos if arm == "candidate" else 4.0}]}}
            if listening:
                item["listening"] = {"ratings": [{"rater": "anon-1", "score": 4.0}, {"rater": "anon-2", "score": 4.0}]}
            if speaker:
                item["speaker"] = {"score": 0.8, "reference_sha256": row["reference_sha256"]}
            rows.append(item)
    doc = {"format": qa.SCORES_FORMAT, "protocol": protocol["evidence_id"], "baseline": baseline["evidence_id"],
           "candidate": candidate["evidence_id"], "scorers": protocol["data"]["policy"]["scorers"], "rows": rows}
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(doc))
    return path


def quality(run, scores=None):
    fixtures, protocol, baseline, candidate = run
    return invoke("quality_audio", baseline=baseline["evidence_id"], candidate=candidate["evidence_id"],
                  fixtures=str(fixtures), protocol=protocol["evidence_id"], scores=str(scores) if scores else None)


def test_parity_signed_stereo_and_no_truncation():
    a = struct.pack("<4h", -1000, 1000, -2000, 2000)
    b = struct.pack("<4h", -1001, 1001, -2002, 2002)
    result = qa.parity(a, b, 2)
    assert result["baseline_frames"] == 2
    assert result["maxabs"] == 2 / 32768
    assert result["snr_db"] == pytest.approx(60)
    assert qa.parity(a, a, 2)["snr_status"] == "identical"
    assert qa.parity(a, b[:-4], 2)["snr_db"] is None
    assert qa.parity(a, b[:-4], 2)["maxabs"] is None


def test_wer_is_recomputed_not_capped_or_silently_normalized():
    assert qa.word_error("Hello, WORLD!", "hello world", "nfkc_casefold_punctuation_v1")["errors"] == 0
    assert qa.word_error("hello", "hello extra extra extra", "whitespace_v1")["wer"] == 3
    assert qa.word_error("a b c", "a x", "whitespace_v1")["errors"] == 2
    with pytest.raises(OpError, match="no words"):
        qa.word_error("!!!", "speech", "nfkc_casefold_punctuation_v1")


def test_exact_preservation_and_changed_pcm_fail(tmp_path, store):
    run = setup_run(tmp_path, store)
    good = quality(run)
    assert good["data"]["verdict"] == "PASS"
    assert "baseline outputs" in good["data"]["scope"]
    assert ev.validate(good) == []
    assert build_report([good], "improved", "quality")["verdict"] == "PASS"
    changed = bench(store, run[0], "candidate", different=True)
    bad = quality((*run[:3], changed))
    assert bad["data"]["verdict"] == "FAIL"
    assert bad["data"]["pairs"][0]["snr_db"] == pytest.approx(40)


def test_missing_pcm_is_unproven_and_never_zero(tmp_path, store):
    run = setup_run(tmp_path, store, keep=False)
    result = quality(run)
    assert result["data"]["verdict"] == "UNPROVEN"
    assert next(m for m in result["metrics"] if m["name"] == "bit_exact_pct")["value"] is None


def test_failed_source_cannot_become_quality_pass(tmp_path, store):
    run = setup_run(tmp_path, store, failed=True)
    assert quality(run)["data"]["verdict"] == "FAIL"


def test_posthoc_policy_cannot_pass(tmp_path, store):
    run = setup_run(tmp_path, store, start=0)
    result = quality(run)
    assert result["data"]["verdict"] == "UNPROVEN"
    assert result["data"]["checks"]["parity"]["verdict"] == "PASS"
    assert not result["data"]["preregistered"]


def test_proxy_policy_allows_nonidentical_pcm_without_claiming_listening(tmp_path, store):
    run = setup_run(tmp_path, store, "task_noninferiority", different=True)
    result = quality(run, score_file(tmp_path, store, run))
    assert result["data"]["verdict"] == "PASS"
    assert result["data"]["checks"]["parity"]["verdict"] == "FAIL"
    assert result["data"]["checks"]["parity"]["required"] is False
    assert "not human listening" in result["data"]["scope"]
    assert result["data"]["checks"]["wer"]["n_prompts"] == 3  # not six fixture-rounds
    assert result["data"]["checks"]["wer"]["ci95"] == [0.0, 0.0]


def test_missing_and_failed_metrics_preserve_verdicts(tmp_path, store):
    run = setup_run(tmp_path, store, "task_noninferiority")
    assert quality(run)["data"]["verdict"] == "UNPROVEN"
    path = score_file(tmp_path, store, run, mos=3.0)
    assert quality(run, path)["data"]["verdict"] == "FAIL"
    path = score_file(tmp_path, store, run)
    doc = json.loads(path.read_text())
    doc["rows"].pop()
    path.write_text(json.dumps(doc))
    assert quality(run, path)["data"]["verdict"] == "UNPROVEN"


@pytest.mark.parametrize("mutation", ["duplicate", "pcm", "text", "scorer", "window_gap", "window_overlap", "nan"])
def test_invalid_import_rejected(tmp_path, store, mutation):
    run = setup_run(tmp_path, store, "task_noninferiority")
    path = score_file(tmp_path, store, run)
    doc = json.loads(path.read_text())
    if mutation == "duplicate":
        doc["rows"].append(doc["rows"][0])
    elif mutation in ("pcm", "text"):
        doc["rows"][0][f"{mutation}_sha256"] = "0" * 64
    elif mutation == "scorer":
        doc["scorers"]["asr"]["revision"] = "other"
    elif mutation.startswith("window"):
        window = doc["rows"][0]["naturalness"]["windows"][0]
        if mutation == "window_gap":
            window["end_frame"] -= 1
        else:
            doc["rows"][0]["naturalness"]["windows"].append(window)
    else:
        doc["rows"][0]["naturalness"]["windows"][0]["score"] = math.nan
    path.write_text(json.dumps(doc))
    with pytest.raises(OpError):
        quality(run, path)


def test_pcm_tampering_is_refused(tmp_path, store):
    run = setup_run(tmp_path, store)
    candidate = run[-1]
    item = next(a for a in candidate["artifacts"] if a["role"] == "audio_pcm")
    (store.run_dir(candidate["evidence_id"]) / item["path"]).write_bytes(b"changed")
    with pytest.raises(OpError, match="missing, changed"):
        quality(run)


def test_asr_outliers_never_choose_best_decode_or_use_snr_to_waive_wer(tmp_path, store):
    run = setup_run(tmp_path, store, "task_noninferiority", retry=True)
    hallucination = "invented " * 20
    path = score_file(tmp_path, store, run, candidate_text=hallucination)
    result = quality(run, path)
    assert result["data"]["checks"]["wer"]["primary_verdict"] == "FAIL"
    assert result["data"]["verdict"] == "UNPROVEN"
    path = score_file(tmp_path, store, run, candidate_text=hallucination, retry_text="The meeting starts tomorrow morning.")
    assert quality(run, path)["data"]["verdict"] == "UNPROVEN"
    path = score_file(tmp_path, store, run, candidate_text=hallucination, retry_text=hallucination)
    assert quality(run, path)["data"]["verdict"] == "FAIL"


def test_length_weighting_and_complete_tail():
    value = {"windows": [{"start_frame": 0, "end_frame": 75, "score": 4},
                         {"start_frame": 75, "end_frame": 100, "score": 2}]}
    assert qa.window_score(value, 100, scorer("naturalness")) == 3.5


def test_perceptual_policy_requires_blinded_paired_listeners(tmp_path, store):
    run = setup_run(tmp_path, store, "perceptual_and_task_noninferiority", listening=True)
    assert quality(run, score_file(tmp_path, store, run))["data"]["verdict"] == "UNPROVEN"
    path = score_file(tmp_path, store, run, listening=True)
    assert quality(run, path)["data"]["verdict"] == "PASS"
    doc = json.loads(path.read_text())
    doc["rows"][-1]["listening"]["ratings"][0]["rater"] = "unpaired"
    path.write_text(json.dumps(doc))
    assert quality(run, path)["data"]["verdict"] == "UNPROVEN"


def test_speaker_score_binds_reference_audio(tmp_path, store):
    run = setup_run(tmp_path, store, "task_noninferiority", reference=True)
    path = score_file(tmp_path, store, run, speaker=True)
    assert quality(run, path)["data"]["verdict"] == "PASS"
    doc = json.loads(path.read_text())
    doc["rows"][0]["speaker"]["reference_sha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(OpError, match="voice reference"):
        quality(run, path)


def test_report_retains_missing_quality_observations(tmp_path, store):
    run = setup_run(tmp_path, store, "task_noninferiority")
    result = quality(run, score_file(tmp_path, store, run))
    root = store.run_dir(result["evidence_id"])
    artifact = next(a for a in result["artifacts"] if a["role"] == "audio_quality_observations")
    (root / artifact["path"]).unlink()
    assert build_report([result], "improved", "quality")["verdict"] == "UNPROVEN"


def test_run_directory_retains_policy_and_observations_without_store(tmp_path, store):
    run = setup_run(tmp_path, store, "task_noninferiority")
    path = score_file(tmp_path, store, run)
    out = tmp_path / "portable"
    result = invoke("quality_audio", baseline=run[2]["evidence_id"], candidate=run[3]["evidence_id"],
                    fixtures=str(run[0]), protocol=run[1]["evidence_id"], scores=str(path), run_dir=str(out))
    assert all((out / a["path"]).is_file() for a in result["artifacts"])
    shutil.rmtree(store.run_dir(result["evidence_id"]))
    assert op.report_verdict(result)[0] == "PASS"
    policy = invoke("quality_audio_policy", policy=str(policy_file(tmp_path)), fixtures=str(run[0]), run_dir=str(out))
    loaded, root = qa.load_record(store, str(out / "evidence" / f"{policy['evidence_id']}.json"), "quality_audio")
    assert qa.artifact_bytes(root, qa.one_artifact(loaded, "audio_quality_policy"))


def test_longer_output_parity_has_no_truncated_score(tmp_path, store):
    run = setup_run(tmp_path, store, "task_noninferiority", frames=20)
    result = quality(run, score_file(tmp_path, store, run))
    assert result["data"]["verdict"] == "PASS"
    assert all(pair["snr_status"] == "length_mismatch" for pair in result["data"]["pairs"])
    assert all(pair["maxabs"] is None for pair in result["data"]["pairs"])


def comparison(store, baseline, candidate):
    return store.save(ev.build(fake_call(), kind="comparison", claim_class="derived", summary="synthetic timing",
                               data={"source_kind": "audio_benchmark", "baseline": baseline["evidence_id"],
                                     "candidate": candidate["evidence_id"], "verdict": "improved", "interleaved": True,
                                     "required_proofs": {"media quality": "UNPROVEN", "fresh-process lifecycle": "UNPROVEN"}}))


def finalize_sources(store, run, planned_rounds):
    from inferencekit.serve.round_evidence import finalize_arm_records

    cells = [{"label": arm, "round": r, "status": "ok" if r <= 2 else "skipped_after_cleanup_failure",
              "ready": {"status": "ready"}, "stop": {"group_gone": True}, "server_env": {}}
             for arm in ("baseline", "candidate") for r in range(1, planned_rounds + 1)]
    final = finalize_arm_records({"baseline": run[2]["evidence_id"], "candidate": run[3]["evidence_id"]},
                                 cells, {"rounds": planned_rounds, "design": {"design": "Williams", "balanced": True}})
    return (*run[:2], store.load(final["baseline"]), store.load(final["candidate"]))


def test_matching_scheduler_prefix_cannot_pass_quality(tmp_path, store):
    run = finalize_sources(store, setup_run(tmp_path, store), planned_rounds=4)
    result = quality(run)
    assert result["data"]["verdict"] == "UNPROVEN"
    assert result["coverage"]["complete"] is False
    assert result["data"]["checks"]["round_coverage"]["verdict"] == "UNPROVEN"
    assert result["data"]["checks"]["round_coverage"]["arms"]["baseline"]["missing_indices"] == [2, 3]
    assert all(pair["bit_exact"] for pair in result["data"]["pairs"])
    compared = comparison(store, run[2], run[3])
    report = build_report([compared, result], "improved", "quality")
    assert all(c["verdict"] == "UNPROVEN" for c in report["claims"]
               if c["claim"] == "speech quality preservation against baseline")


def test_final_source_ids_and_original_round_times_bind_quality(tmp_path, store):
    raw = setup_run(tmp_path, store)
    raw_quality = quality(raw)
    final = finalize_sources(store, raw, planned_rounds=2)
    compared = comparison(store, final[2], final[3])
    unbound = build_report([compared, raw_quality], "improved", "quality")
    assert next(c for c in unbound["claims"] if c["claim"] == "media quality")["verdict"] == "UNPROVEN"
    final_quality = quality(final)
    assert final_quality["data"]["preregistered"] is True
    assert final_quality["data"]["verdict"] == "PASS"
    bound = build_report([compared, final_quality], "improved", "quality")
    assert next(c for c in bound["claims"] if c["source"] == compared["evidence_id"] and
                c["claim"] == "speech quality preservation against baseline")["verdict"] == "PASS"
    assert final[3]["data"]["quality_verdict"] == "UNPROVEN"  # source timing record is immutable
    late = finalize_sources(store, (*raw[:3], bench(store, raw[0], "candidate", start=0)), planned_rounds=2)
    assert quality(late)["data"]["preregistered"] is False
    assert quality(late)["data"]["verdict"] == "UNPROVEN"


def test_report_attaches_exact_pair_and_retains_lifecycle_and_failure(tmp_path, store):
    run = setup_run(tmp_path, store)
    good = quality(run)
    compare = comparison(store, run[2], run[3])
    report = build_report([compare, good], "improved", "quality")
    assert report["verdict"] == "UNPROVEN"
    linked = [c for c in report["claims"] if c["source"] == compare["evidence_id"]]
    assert next(c for c in linked if c["claim"] == "speech quality preservation against baseline")["verdict"] == "PASS"
    assert next(c for c in linked if c["claim"] == "fresh-process lifecycle")["verdict"] == "UNPROVEN"
    unrelated = copy.deepcopy(compare)
    unrelated["data"]["candidate"] = "ev_" + "0" * 64
    unrelated = store.save(unrelated)
    report = build_report([unrelated, good], "improved", "quality")
    assert next(c for c in report["claims"] if c["claim"] == "media quality")["verdict"] == "UNPROVEN"
    changed = bench(store, run[0], "candidate", different=True)
    failed = quality((*run[:3], changed))
    assert build_report([compare, good, failed], "improved", "quality")["verdict"] == "FAIL"


@pytest.mark.parametrize("field,value", [("resamples", 0), ("min_fixtures", 1), ("seed", -1), ("min_rounds", True)])
def test_invalid_registered_plan_is_rejected(field, value):
    with pytest.raises(OpError):
        qa.validate_policy({"acceptance": "exact", field: value})
