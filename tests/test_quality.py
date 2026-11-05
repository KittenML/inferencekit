"""quality trajectory / score / gsm8k end to end against deterministic fake OpenAI servers."""

import json
import math

import pytest

from inferencekit import evidence as ev
from inferencekit import registry
from inferencekit.quality import metrics as qm
from inferencekit.quality.fake import FakeLM, serve


def write_prompts(path, n=6, max_tokens=24):
    rows = [{"id": f"p{i:02d}", "bucket": "short", "kind": "chat", "source": "test",
             "messages": [{"role": "user", "content": f"question {i}: " + "abc"[: i % 3 + 1] * (i + 1)}],
             "max_tokens": max_tokens} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def trajectory(url, prompts, run, label, **kw):
    args = {"base_url": url, "model": "fake-model", "prompts": str(prompts), "label": label, "run_dir": str(run),
            "engine_kind": "vllm", **kw}
    return registry.call("quality_trajectory", args)


def score(url, run, engines, baseline=None, **kw):
    args = {"reference_url": url, "reference_model": "fake-model",
            "trajectories": [str(run / "quality" / e) for e in engines], "resamples": 300, **kw}
    if baseline:
        args["baseline"] = str(run / "quality" / baseline)
    rec = registry.call("quality_score", args)
    report = json.loads((run / "quality" / "score.json").read_text())
    return rec, report


def expected(ref: FakeLM, eng: FakeLM, run, label):
    """Exact pooled agreement / KL / NLL recomputed from the fakes' full distributions."""
    agree = kl = nll = 0.0
    n = 0
    for f in sorted((run / "quality" / label).glob("p*.json")):
        doc = json.loads(f.read_text())
        ctx = ref.template(doc["messages"])
        for tok in doc["token_ids"]:
            p, q = ref.logprobs(ctx), eng.logprobs(ctx)
            agree += tok == max(range(ref.vocab), key=lambda j: p[j])
            kl += qm.kl_full(p, q)
            nll += -p[tok]
            n += 1
            ctx = ctx + [tok]
    return agree / n, kl / n, nll / n


def test_engine_identical_to_reference_scores_kl_zero_and_full_agreement(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl")
    run = tmp_path / "runs" / "r1"
    lm = FakeLM()
    with serve(lm) as url:
        rec = trajectory(url, prompts, run, "engine")
        trajectory(url, prompts, run, "reference")
        srec, report = score(url, run, ["engine"], baseline="reference")
    assert ev.verify(rec)["id_ok"] and ev.verify(srec)["id_ok"]
    assert rec["data"]["token_id_methods"] == ["token_id_strings"]
    assert rec["data"]["top_logprobs_returned_min"] == lm.vocab  # 20 requested, the toy vocab has 16
    doc = json.loads((run / "quality" / "engine" / "p00.json").read_text())
    assert len(doc["token_ids"]) == 24 and doc["prompt_token_ids_source"] == "response.prompt_token_ids"
    eng = report["engines"]["engine"]
    assert eng["all"]["top1_agreement"]["value"] == 1.0
    assert eng["all"]["kl_mean"]["value"] == 0.0
    assert eng["first_divergence"]["identical"] == 6
    assert eng["all"]["ref_nll_mean"]["value"] == pytest.approx(report["floor"]["all"]["ref_nll_mean"]["value"])
    assert eng["verdict"]["verdict"] == "PASS"
    assert srec["data"]["verdict"] == "PASS"
    assert srec["data"]["prompt_ids_sources"] == ["reference_tokenize"]
    # the record is copied into the run dir, and the teacher-forcing request was the vLLM extension
    assert (run / "evidence" / f"{srec['evidence_id']}.json").is_file()
    tf = [r["body"] for r in lm.requests if r["path"] == "/v1/completions"]
    assert tf and tf[0]["max_tokens"] == 1 and tf[0]["prompt_logprobs"] == 20 and isinstance(tf[0]["prompt"], list)


def test_known_kl_under_a_constructed_distribution_shift(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=5, max_tokens=70)  # reaches bucket 64-255
    run = tmp_path / "runs" / "r2"
    ref_lm, eng_lm = FakeLM(), FakeLM(perturb=1.5)
    with serve(ref_lm) as ref_url, serve(eng_lm) as eng_url:
        trajectory(eng_url, prompts, run, "shifted")
        trajectory(ref_url, prompts, run, "reference")
        srec, report = score(ref_url, run, ["shifted"], baseline="reference",
                             margin_agreement=0.001, margin_kl=1e-4, margin_nll=1e-4)
    agree, kl, nll = expected(ref_lm, eng_lm, run, "shifted")
    eng = report["engines"]["shifted"]
    assert kl > 0.01  # the shift is real
    assert eng["all"]["kl_mean"]["value"] == pytest.approx(kl, rel=1e-6)  # full support -> exact KL
    assert eng["all"]["top1_agreement"]["value"] == pytest.approx(agree)
    assert eng["all"]["ref_nll_mean"]["value"] == pytest.approx(nll, rel=1e-9)
    assert eng["64-255"]["top1_agreement"]["n_tokens"] == 5 * 6
    assert eng["verdict"]["verdict"] == "FAIL"
    assert eng["verdict"]["checks"]["kl_mean"]["verdict"] == "FAIL"
    assert any(m["name"] == "kl_ref_engine" and m["subject"] == "shifted @pos 16-63" for m in srec["metrics"])
    # floor: the reference against itself
    assert report["floor"]["all"]["kl_mean"]["value"] == 0.0


def test_engine_logprob_cap_is_detected_and_recorded(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=3)
    run = tmp_path / "runs" / "r3"
    with serve(FakeLM(max_logprobs=5)) as eng_url, serve(FakeLM()) as ref_url:
        rec = trajectory(eng_url, prompts, run, "capped")
        trajectory(ref_url, prompts, run, "reference")
        _, report = score(ref_url, run, ["capped"], baseline="reference")
    assert rec["data"]["top_logprobs_returned_min"] == 5
    assert rec["data"]["cap_errors"]
    assert any("requested 20" in lim for lim in rec["limitations"])
    kl = report["engines"]["capped"]["all"]["kl_mean"]["value"]
    assert math.isfinite(kl) and kl == pytest.approx(0.0, abs=1e-12)  # same model, compared at top-5 on both sides
    assert report["engines"]["capped"]["kl_top_k"] == [5]
    assert report["engines"]["capped"]["all"]["top1_agreement"]["value"] == 1.0  # same model, truncated lists


def test_llama_server_style_id_fields(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=2)
    run = tmp_path / "runs" / "r4"
    with serve(FakeLM(ids="llamacpp")) as url:
        rec = trajectory(url, prompts, run, "llama", engine_kind="llamacpp")
    assert rec["data"]["token_id_methods"] == ["id_field"]
    doc = json.loads((run / "quality" / "llama" / "p00.json").read_text())
    assert doc["prompt_token_ids_source"] == "engine_tokenize"


def test_no_baseline_means_unproven(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=2)
    run = tmp_path / "runs" / "r5"
    with serve(FakeLM()) as url:
        trajectory(url, prompts, run, "engine")
        srec, report = score(url, run, ["engine"])
    assert report["engines"]["engine"]["verdict"]["verdict"] == "UNPROVEN"
    assert "noise floor" in srec["coverage"]["missing"]


def test_edited_trajectory_is_caught_and_never_passes(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=4)
    run = tmp_path / "runs" / "r6"
    with serve(FakeLM()) as url:
        trajectory(url, prompts, run, "engine")
        trajectory(url, prompts, run, "reference")
        f = run / "quality" / "engine" / "p01.json"
        doc = json.loads(f.read_text())
        doc["text"] = "tampered"
        f.write_text(json.dumps(doc))
        srec, report = score(url, run, ["engine"], baseline="reference")
    assert "input integrity" in srec["coverage"]["missing"]
    assert report["engines"]["engine"]["verdict"]["verdict"] == "UNPROVEN"


class OtherTemplate(FakeLM):
    def template(self, messages):  # an engine that silently uses a different chat template
        return super().template(messages)[:-1] + [0, self.ASSISTANT]


def test_chat_template_mismatch_is_flagged(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=3)
    run = tmp_path / "runs" / "r7"
    with serve(OtherTemplate()) as eng_url, serve(FakeLM()) as ref_url:
        trajectory(eng_url, prompts, run, "engine")
        trajectory(ref_url, prompts, run, "reference")
        srec, report = score(ref_url, run, ["engine"], baseline="reference", margin_agreement=1.0, margin_kl=100.0,
                             margin_nll=100.0)
    assert len(report["engines"]["engine"]["template_mismatch"]) == 3
    assert report["engines"]["engine"]["verdict"]["verdict"] == "UNPROVEN"  # downgraded from PASS
    assert any("chat template" in lim for lim in srec["limitations"])


# --------------------------------------------------------------------------- gsm8k

def write_gsm8k(path):
    rows = [{"question": f"Q{i}: how many apples?", "answer": f"work <<{i}*1={i}>>\n#### {i * 10}"}
            for i in range(1, 6)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def gsm8k(url, run, data, label, **kw):
    return registry.call("quality_gsm8k", {"base_url": url, "model": "fake-model", "run_dir": str(run),
                                           "data": str(data), "label": label, "concurrency": 4, **kw})


def test_gsm8k_accuracy_wilson_and_mcnemar(tmp_path, store):
    data = write_gsm8k(tmp_path / "gsm8k.jsonl")
    run = tmp_path / "runs" / "g1"
    good = {"Q1:": "1 times 10 is 10.\n#### 10", "Q2:": "So \\boxed{20}", "Q3:": "It is 30 apples.",
            "Q4:": "#### 41", "Q5:": "#### 50"}
    bad = {"Q1:": "#### 11", "Q2:": "#### 21", "Q3:": "#### 31", "Q4:": "#### 41", "Q5:": "#### 50"}
    with serve(FakeLM(answers=good)) as a_url, serve(FakeLM(answers=bad)) as b_url:
        a = gsm8k(a_url, run, data, "a")
        b = gsm8k(b_url, run, data, "b", paired_with=str(run / "quality" / "gsm8k" / "a.jsonl"))
        c = gsm8k(b_url, run, data, "c", paired_with=a["evidence_id"])  # pairing by evidence id
    assert a["data"]["correct"] == 4 and a["data"]["n"] == 5
    assert a["data"]["extraction_rules_used"] == {"hash": 3, "boxed": 1, "last": 1, "none": 0}
    acc = next(m for m in a["metrics"] if m["name"] == "gsm8k_accuracy_pct")
    lo, hi = qm.wilson_ci(4, 5)
    assert acc["value"] == 80.0 and acc["ci95"] == [pytest.approx(100 * lo), pytest.approx(100 * hi)]
    paired = b["data"]["paired"]
    assert (paired["only_this_correct"], paired["only_other_correct"], paired["both_correct"]) == (0, 3, 1)
    assert paired["mcnemar_exact_p"] == pytest.approx(qm.mcnemar_exact(0, 3))
    rows = [json.loads(x) for x in (run / "quality" / "gsm8k" / "a.jsonl").read_text().splitlines()]
    assert [r["idx"] for r in rows] == [0, 1, 2, 3, 4] and rows[0]["extracted"] == "10"
    assert a["data"]["backend"] == "builtin"
    assert c["data"]["paired"]["n_paired"] == 5 and c["data"]["errors"] == 0
    assert c["data"]["paired"]["only_other_correct"] == 3


def test_lm_eval_backend_reports_missing_binary(tmp_path, store, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(registry.OpError, match="lm_eval"):
        registry.call("quality_gsm8k", {"base_url": "http://127.0.0.1:9", "model": "m", "run_dir": str(tmp_path),
                                        "backend": "lm-eval", "limit": 3})


def test_lm_eval_results_parsing():
    from inferencekit.ops.quality import parse_lm_eval_results

    obj = {"results": {"gsm8k": {"exact_match,strict-match": 0.75, "exact_match,flexible-extract": 0.8}},
           "n-samples": {"gsm8k": {"original": 1319, "effective": 100}}, "versions": {"gsm8k": 3}}
    assert parse_lm_eval_results(obj) == {"strict": 0.75, "flexible": 0.8, "n": 100, "lm_eval_version": 3}


def test_prompt_ids_fall_back_to_hf_tokenizer_then_engine_ids(tmp_path, store, monkeypatch):
    from inferencekit.ops import quality as q

    prompts = write_prompts(tmp_path / "prompts.jsonl", n=3)
    run = tmp_path / "runs" / "r8"
    ref_lm = FakeLM(has_tokenize=False)
    with serve(FakeLM()) as eng_url, serve(ref_lm) as ref_url:
        trajectory(eng_url, prompts, run, "engine")
        trajectory(eng_url, prompts, run, "reference")
        # an HF tokenizer with the same chat template
        monkeypatch.setattr(q._PromptIds, "_hf_ids", lambda self, messages, kwargs: ref_lm.template(messages))
        srec, report = score(ref_url, run, ["engine"], baseline="reference")
        assert srec["data"]["prompt_ids_sources"] == ["hf_tokenizer"]
        assert report["engines"]["engine"]["verdict"]["verdict"] == "PASS"
        # no HF tokenizer either: the engine's own prompt ids, flagged as unchecked
        monkeypatch.setattr(q._PromptIds, "_hf_ids", lambda self, messages, kwargs: None)
        srec, _ = score(ref_url, run, ["engine"], baseline="reference")
    assert srec["data"]["prompt_ids_sources"] == ["engine_reported"]
    assert any("not independently checked" in lim for lim in srec["limitations"])


def test_input_bucket_summaries_and_cli(tmp_path, store, capsys):
    from inferencekit import cli

    prompts = tmp_path / "prompts.jsonl"
    rows = [{"id": f"{b}-{i}", "bucket": b, "messages": [{"role": "user", "content": f"{b} {i} " * (i + 1)}],
             "max_tokens": 20} for b in ("short", "medium") for i in range(3)]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = tmp_path / "runs" / "r9"
    with serve(FakeLM(perturb=0.4)) as eng_url, serve(FakeLM()) as ref_url:
        trajectory(eng_url, prompts, run, "engine")
        trajectory(ref_url, prompts, run, "reference")
        code = cli.main(["--runs-dir", str(store.root), "quality", "score", "--reference-url", ref_url,
                         "--reference-model", "fake-model", "--trajectories", str(run / "quality"),
                         "--baseline", str(run / "quality" / "reference"), "--resamples", "200"])
    assert code == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["data"]["verdicts"].keys() == {"engine"}
    report = json.loads((run / "quality" / "score.json").read_text())
    by_input = report["engines"]["engine"]["by_input_bucket"]
    assert set(by_input) == {"short", "medium"}
    assert by_input["short"]["top1_agreement"]["n_tokens"] == 60


class Reader(FakeLM):
    """Answers the needle question by reading the document, but only up to 2,000 characters of it."""

    def chat(self, body):
        import re as _re

        content = body["messages"][-1]["content"]
        q = _re.search(r"for the (\w+) (\w+) vault\?", content)
        m = _re.search(rf"code for the {q.group(1)} {q.group(2)} vault is (\d+)", content[:2000])
        reply = f"The code is {int(m.group(1)):,}." if m else "I could not find it."
        return 200, {"choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": reply}}],
                     "usage": {"prompt_tokens": len(content), "completion_tokens": 5}}


def test_niah_cells_and_haystack_properties(tmp_path, store):
    from inferencekit.quality import niah

    item = niah.build(1000, 0.5, seed=3, chars_per_token=4.0)
    text = item["messages"][0]["content"]
    assert item["code"] in text and len(re_digits(text)) == 1  # the only digit string is the code
    assert 3000 < len(text) < 4500
    assert niah.build(1000, 0.5, seed=3) == item  # deterministic
    run = tmp_path / "runs" / "n1"
    with serve(Reader()) as url:
        rec = registry.call("quality_niah", {"base_url": url, "model": "fake-model", "run_dir": str(run),
                                             "lengths": [256, 4096], "depths": [0.1, 0.9], "trials": 2})
    cells = {(c["target_tokens"], c["depth"]): c["correct"] for c in rec["data"]["cells"]}
    assert cells == {(256, 0.1): 2, (256, 0.9): 2, (4096, 0.1): 2, (4096, 0.9): 0}
    assert rec["data"]["chars_per_token_method"] == "measured with /tokenize"
    assert rec["metrics"][0]["value"] == 75.0


def re_digits(text):
    import re as _re

    return _re.findall(r"\d+", text)


def test_second_reference_run_is_the_a_a_floor(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=4)
    run = tmp_path / "runs" / "r10"
    with serve(FakeLM()) as ref_url, serve(FakeLM(perturb=0.6)) as eng_url:
        trajectory(eng_url, prompts, run, "engine")
        trajectory(ref_url, prompts, run, "ref-a")
        trajectory(ref_url, prompts, run, "ref-b")
        srec, report = score(ref_url, run, ["engine"], baseline="ref-a", baseline_repeat=str(run / "quality" / "ref-b"))
    assert report["noise_floor"] == "ref-b" and report["floor"]["source"] == "baseline_repeat"
    assert report["floor"]["first_divergence"]["identical"] == 4  # a deterministic reference
    assert report["engines"]["engine"]["first_divergence"]["n_prompts"] == 4
    assert set(srec["data"]["verdicts"]) == {"engine"}
    with pytest.raises(registry.OpError, match="needs --baseline"):
        score(ref_url, run, ["engine"], baseline_repeat=str(run / "quality" / "ref-b"))


# --------------------------------------------------------------------------- review fixes: coverage, KL at the engine's k,
# floor labelling, index provenance, gsm8k pairing

class FailsOn(FakeLM):
    """An engine that answers HTTP 500 on prompts containing ``marker`` (a crash on some inputs)."""

    def __init__(self, marker, **kw):
        super().__init__(**kw)
        self.marker = marker

    def chat(self, body):
        if any(self.marker in str(m.get("content")) for m in body.get("messages") or []):
            return 500, {"error": {"message": "engine crashed"}}
        return super().chat(body)


def score_aa(url, run, engines, **kw):
    """Score with ref-a as the greedy path and ref-b as the run-to-run A/A floor."""
    return score(url, run, engines, baseline="ref-a", baseline_repeat=str(run / "quality" / "ref-b"), **kw)


def test_engine_scored_on_fewer_prompts_than_the_baseline_is_never_a_pass(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=4)
    run = tmp_path / "runs" / "r11"
    with serve(FakeLM()) as ref_url, serve(FailsOn("question 2:")) as bad_url:
        trajectory(ref_url, prompts, run, "ref-a")
        trajectory(ref_url, prompts, run, "ref-b")
        trajectory(ref_url, prompts, run, "short", limit=2)  # the reference model itself, on half the prompts
        rec = trajectory(bad_url, prompts, run, "crashy")  # the same model, crashing on one prompt
        srec, report = score_aa(ref_url, run, ["short", "crashy"])
    assert set(rec["data"]["failed"]) == {"p02"}
    for label, missing_ids in (("short", ["p02", "p03"]), ("crashy", ["p02"])):
        eng = report["engines"][label]
        assert eng["all"]["top1_agreement"]["value"] == 1.0  # perfect on what it did score...
        assert eng["coverage"]["required"] == 4
        assert eng["coverage"]["missing"] == len(missing_ids)
        assert eng["coverage"]["missing_ids"] == missing_ids
        assert eng["verdict"]["verdict"] == "UNPROVEN"  # ... but never a PASS
        assert "not scored" in eng["verdict"]["reason"]
    assert report["engines"]["crashy"]["coverage"]["failed_in_trajectory"] == ["p02"]
    assert srec["data"]["verdicts"] == {"short": "UNPROVEN", "crashy": "UNPROVEN"}
    assert "scored prompts" in srec["coverage"]["missing"]
    assert any(lim.startswith("short: 2 of 4 prompts not scored") for lim in srec["limitations"])
    scored = {m["subject"]: m for m in srec["metrics"] if m["name"] == "prompts_scored"}
    assert scored["short"]["value"] == 2 and scored["short"]["n"] == 4


def test_trajectories_from_a_different_prompt_file_never_pass(tmp_path, store):
    run = tmp_path / "runs" / "r12"
    base_prompts = write_prompts(tmp_path / "a.jsonl", n=3)
    other = write_prompts(tmp_path / "b.jsonl", n=4)  # the same three prompts plus one
    with serve(FakeLM()) as url:
        trajectory(url, base_prompts, run, "ref-a")
        trajectory(url, base_prompts, run, "ref-b")
        trajectory(url, other, run, "engine")
        _, report = score_aa(url, run, ["engine"])
    eng = report["engines"]["engine"]
    assert eng["coverage"]["missing"] == 0
    assert eng["verdict"]["verdict"] == "UNPROVEN" and "prompt file" in eng["verdict"]["reason"]


def test_capped_engine_identical_to_reference_is_compared_at_its_own_k(tmp_path, store):
    """top-5 from the engine against top-20 from the reference used to read as KL > 0 (a spurious FAIL)."""
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=4, max_tokens=16)
    run = tmp_path / "runs" / "r13"
    peaked = {"vocab": 200, "sharpness": 3.0}
    with serve(FakeLM(max_logprobs=5, **peaked)) as eng_url, serve(FakeLM(**peaked)) as ref_url:
        rec = trajectory(eng_url, prompts, run, "capped")
        trajectory(ref_url, prompts, run, "ref-a")
        trajectory(ref_url, prompts, run, "ref-b")
        srec, report = score_aa(ref_url, run, ["capped"])
    assert rec["data"]["top_logprobs_returned_min"] == 5
    eng = report["engines"]["capped"]
    assert eng["kl_top_k"] == [5] and report["floor"]["kl_top_k"] == [20]
    assert eng["all"]["kl_mean"]["value"] == pytest.approx(0.0, abs=1e-12)
    kl = eng["verdict"]["checks"]["kl_mean"]
    assert kl["top_k"] == 5 and kl["verdict"] == "PASS"
    assert eng["verdict"]["verdict"] == "PASS"
    assert any("top-5" in lim for lim in srec["limitations"])


def test_capped_engine_with_a_real_shift_still_fails(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=4, max_tokens=16)
    run = tmp_path / "runs" / "r13b"
    with serve(FakeLM(vocab=200, sharpness=3.0, perturb=3.0, max_logprobs=5)) as eng_url, \
            serve(FakeLM(vocab=200, sharpness=3.0)) as ref_url:
        trajectory(eng_url, prompts, run, "capped")
        trajectory(ref_url, prompts, run, "ref-a")
        trajectory(ref_url, prompts, run, "ref-b")
        _, report = score_aa(ref_url, run, ["capped"], margin_kl=0.01)
    kl = report["engines"]["capped"]["verdict"]["checks"]["kl_mean"]
    assert kl["top_k"] == 5 and kl["verdict"] == "FAIL"


def test_floor_without_a_repeat_run_is_labelled_and_has_no_divergence_floor(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=3)
    run = tmp_path / "runs" / "r15"
    with serve(FakeLM()) as url:
        trajectory(url, prompts, run, "engine")
        trajectory(url, prompts, run, "reference")
        srec, report = score(url, run, ["engine"], baseline="reference")
    assert report["floor"]["source"] == "baseline"
    assert "first_divergence" not in report["floor"]
    assert "first_divergence" in report["engines"]["engine"]
    assert any("--baseline-repeat" in lim for lim in srec["limitations"])
    assert not any(m["name"] == "identical_to_reference_greedy_pct" and "noise floor" in m["subject"]
                   for m in srec["metrics"])


def test_rewritten_index_is_caught_against_the_trajectory_evidence_record(tmp_path, store):
    from inferencekit.canonical import file_sha256

    prompts = write_prompts(tmp_path / "prompts.jsonl", n=4)
    run = tmp_path / "runs" / "r14"
    with serve(FakeLM()) as ref_url, serve(FakeLM(perturb=2.0)) as eng_url:
        trajectory(eng_url, prompts, run, "engine")
        trajectory(ref_url, prompts, run, "ref-a")
        trajectory(ref_url, prompts, run, "ref-b")
        # make the bad engine look like the reference: copy the reference's files in and rewrite index.json
        eng_dir, ref_dir = run / "quality" / "engine", run / "quality" / "ref-a"
        index = json.loads((eng_dir / "index.json").read_text())
        for pid, meta in index["files"].items():
            doc = json.loads((ref_dir / meta["file"]).read_text())
            doc["label"] = "engine"
            (eng_dir / meta["file"]).write_text(json.dumps(doc) + "\n")
            meta["sha256"] = file_sha256(eng_dir / meta["file"])[0]
        (eng_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
        srec, report = score_aa(ref_url, run, ["engine"])
    eng = report["engines"]["engine"]
    assert eng["all"]["top1_agreement"]["value"] == 1.0  # the forgery is good...
    assert eng["verdict"]["verdict"] == "UNPROVEN"  # ... but it cannot pass
    assert "input integrity" in srec["coverage"]["missing"]
    assert any("no quality_trajectory evidence record" in lim for lim in srec["limitations"])


def test_gsm8k_pairing_checks_that_both_runs_used_the_same_settings(tmp_path, store):
    data = write_gsm8k(tmp_path / "gsm8k.jsonl")
    other_data = tmp_path / "other.jsonl"
    other_data.write_text(data.read_text().replace("apples", "pears"))
    run = tmp_path / "runs" / "g2"
    lm = FakeLM(answers={f"Q{i}:": f"#### {i * 10}" for i in range(1, 6)})
    with serve(lm) as url:
        a = gsm8k(url, run, data, "a")
        sent = len(lm.requests)
        with pytest.raises(registry.OpError, match="max_tokens"):
            gsm8k(url, run, data, "b", max_tokens=256, paired_with=a["evidence_id"])
        with pytest.raises(registry.OpError, match="dataset_sha256"):
            gsm8k(url, run, other_data, "c", paired_with=str(run / "quality" / "gsm8k" / "a.jsonl"))
        assert len(lm.requests) == sent  # refused before any request was sent
        ok = gsm8k(url, run, data, "d", paired_with=str(run / "quality" / "gsm8k" / "a.jsonl"))
        # results with no evidence record behind them (here: one row dropped) pair, but are flagged
        loose = tmp_path / "elsewhere" / "a.jsonl"
        loose.parent.mkdir()
        loose.write_text("".join((run / "quality" / "gsm8k" / "a.jsonl").read_text().splitlines(True)[:4]))
        unchecked = gsm8k(url, run, data, "e", paired_with=str(loose))
    assert ok["data"]["paired"]["settings_checked"] is True and ok["data"]["paired"]["n_paired"] == 5
    assert unchecked["data"]["paired"]["settings_checked"] is False
    assert unchecked["data"]["paired"]["n_paired"] == 4
    assert any("could not be checked" in lim for lim in unchecked["limitations"])
    # an edited results file behind an evidence id is refused
    (run / "quality" / "gsm8k" / "a.jsonl").write_text(loose.read_text())
    with serve(lm) as url, pytest.raises(registry.OpError, match="changed since"):
        gsm8k(url, run, data, "f", paired_with=a["evidence_id"])


# --------------------------------------------------------------------------- TV, text-only trajectories

def expected_tv(ref: FakeLM, eng: FakeLM, run, label):
    """Exact mean TV per token over the engine's own trajectories, from the fakes' full distributions."""
    tv, n = 0.0, 0
    for f in sorted((run / "quality" / label).glob("p*.json")):
        doc = json.loads(f.read_text())
        ctx = ref.template(doc["messages"])
        for tok in doc["token_ids"]:
            tv += qm.tv_full(ref.logprobs(ctx), eng.logprobs(ctx))
            n += 1
            ctx = ctx + [tok]
    return tv / n


def test_tv_matches_the_exact_value_and_has_its_own_margin(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=5, max_tokens=30)
    run = tmp_path / "runs" / "t1"
    ref_lm, eng_lm = FakeLM(), FakeLM(perturb=0.6)
    with serve(ref_lm) as ref_url, serve(eng_lm) as eng_url:
        trajectory(eng_url, prompts, run, "shifted")
        trajectory(ref_url, prompts, run, "reference")
        srec, report = score(ref_url, run, ["shifted"], baseline="reference", margin_agreement=1.0, margin_kl=100.0,
                             margin_nll=100.0, margin_tv=1e-4)
        _, loose = score(ref_url, run, ["shifted"], baseline="reference", margin_agreement=1.0, margin_kl=100.0,
                         margin_nll=100.0, margin_tv=1.0)
        _, unjudged = score(ref_url, run, ["shifted"], baseline="reference", margin_agreement=1.0, margin_kl=100.0,
                            margin_nll=100.0)
    eng = report["engines"]["shifted"]
    tv = expected_tv(ref_lm, eng_lm, run, "shifted")
    assert 0.01 < tv < 1.0
    assert eng["all"]["tv_mean"]["value"] == pytest.approx(tv, rel=1e-6)  # full support -> exact TV
    assert eng["0-15"]["tv_mean"]["n_tokens"] == 5 * 16 and report["floor"]["all"]["tv_mean"]["value"] == 0.0
    checks = eng["verdict"]["checks"]
    assert checks["tv_mean"]["verdict"] == "FAIL" and checks["tv_mean"]["margin"] == 1e-4
    assert checks["tv_mean"]["top_k"] == 16 and eng["verdict"]["verdict"] == "FAIL"
    assert report["margins"]["tv"] == 1e-4
    assert any(m["name"] == "tv_ref_engine" and m["subject"] == "shifted" for m in srec["metrics"])
    assert loose["engines"]["shifted"]["verdict"]["checks"]["tv_mean"]["verdict"] == "PASS"
    # no --margin-tv: TV and its CI against the floor are reported, the verdict comes from the other checks
    plain = unjudged["engines"]["shifted"]["verdict"]
    assert plain["checks"]["tv_mean"]["verdict"] == "NOT RUN" and plain["checks"]["tv_mean"]["ci95"][0] > 0.01
    assert plain["verdict"] == "PASS" and unjudged["margins"]["tv"] is None


def test_capped_engine_tv_is_compared_at_its_own_k(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=3)
    run = tmp_path / "runs" / "t2"
    with serve(FakeLM(max_logprobs=5)) as eng_url, serve(FakeLM()) as ref_url:
        trajectory(eng_url, prompts, run, "capped")
        trajectory(ref_url, prompts, run, "reference")
        _, report = score(ref_url, run, ["capped"], baseline="reference", margin_tv=0.02)
    eng = report["engines"]["capped"]
    assert eng["all"]["tv_mean"]["value"] == pytest.approx(0.0, abs=1e-12)  # same model at top-5 on both sides
    assert eng["verdict"]["checks"]["tv_mean"]["top_k"] == 5 and eng["verdict"]["verdict"] == "PASS"


@pytest.mark.parametrize("mode", ["ignore", "refuse"])
def test_engine_without_logprobs_gets_a_text_only_trajectory(tmp_path, store, mode):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=4)
    run = tmp_path / "runs" / f"t3-{mode}"
    with serve(FakeLM(logprob_mode=mode)) as eng_url, serve(FakeLM()) as ref_url:
        rec = trajectory(eng_url, prompts, run, "plain")
        trajectory(ref_url, prompts, run, "reference")
        srec, report = score(ref_url, run, ["plain"], baseline="reference")
    assert rec["data"]["token_id_methods"] == ["text"] and rec["data"]["text_only_prompts"] == 4
    # no prompt failed, but a required feature is missing: the record says so in coverage, not only in text
    assert not rec["data"]["failed"] and rec["coverage"]["missing"] == ["logprobs"]
    assert any(lim.startswith("text only for 4/4 prompts") for lim in rec["limitations"])
    assert ("refused" in rec["limitations"][0]) == (mode == "refuse")
    assert bool(rec["data"]["logprobs_refused"]) == (mode == "refuse") and not rec["data"]["cap_errors"]
    doc = json.loads((run / "quality" / "plain" / "p00.json").read_text())
    assert doc["token_id_method"] == "text" and doc["token_ids"] is None and doc["text"]
    eng = report["engines"]["plain"]
    assert eng["retokenized_prompts"] == 4 and eng["n_scored"] == 4
    assert eng["all"]["top1_agreement"]["value"] == 1.0 and eng["all"]["kl_mean"]["value"] is None
    checks = eng["verdict"]["checks"]
    assert checks["kl_mean"]["verdict"] == checks["tv_mean"]["verdict"] == "NOT RUN"
    assert checks["top1_agreement"]["verdict"] == "PASS" and eng["verdict"]["verdict"] == "PASS"
    assert any("re-tokenized text" in lim for lim in srec["limitations"])
    # the checks that could not run are listed per engine, so "required feature missing" can be applied
    assert srec["data"]["not_run"] == report["not_run"] == {"plain": ["kl_mean", "tv_mean"]}
    assert srec["data"]["text_only"] == {"plain": 4} and "logprobs" in srec["coverage"]["missing"]
    assert report["floor_retokenized"]["n_scored"] == 4


class MergingTokenizer(FakeLM):
    """Reads the id pair ``pair`` in a text as the single id ``into``: re-tokenized text does not give back the
    generated ids (as when a model emits a token sequence its tokenizer would never produce)."""

    pair, into = None, 0

    def tokenize(self, body):
        status, out = super().tokenize(body)
        if "messages" in body or self.pair is None:  # the chat template (prompt ids) is unchanged
            return status, out
        merged = []
        for t in out["tokens"]:
            if merged and (merged[-1], t) == self.pair:
                merged[-1] = self.into
            else:
                merged.append(t)
        return status, {**out, "tokens": merged, "count": len(merged)}


@pytest.mark.parametrize("repeat", [False, True])
def test_text_only_engine_is_compared_with_a_floor_re_tokenized_the_same_way(tmp_path, store, repeat):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=4, max_tokens=32)
    run = tmp_path / "runs" / f"t3b-{repeat}"
    ref_lm = MergingTokenizer()
    with serve(ref_lm) as ref_url, serve(FakeLM(logprob_mode="ignore")) as eng_url:
        trajectory(eng_url, prompts, run, "plain")  # the reference model itself, without logprobs
        trajectory(ref_url, prompts, run, "ref-a")
        trajectory(ref_url, prompts, run, "ref-b")
        # the most common pair of generated ids is read back as one other id
        pairs = {}
        for f in (run / "quality" / "ref-a").glob("p*.json"):
            ids = json.loads(f.read_text())["token_ids"]
            for p in zip(ids, ids[1:]):
                pairs[p] = pairs.get(p, 0) + 1
        ref_lm.pair = max(pairs, key=lambda p: (pairs[p], p))
        ref_lm.into = (ref_lm.pair[0] + 5) % ref_lm.vocab
        _, report = score(ref_url, run, ["plain"], baseline="ref-a", margin_agreement=0.001, margin_nll=1e-4,
                          **({"baseline_repeat": str(run / "quality" / "ref-b")} if repeat else {}))
    eng, text_floor = report["engines"]["plain"], report["floor_retokenized"]
    # the re-tokenized reading costs agreement on the engine's side: compared with the floor's own ids (agreement
    # 1.0) that alone would be a FAIL; the floor's text read the same way costs exactly the same
    assert report["floor"]["all"]["top1_agreement"]["value"] == 1.0
    assert eng["all"]["top1_agreement"]["value"] == text_floor["all"]["top1_agreement"]["value"] < 1.0
    assert eng["all"]["ref_nll_mean"]["value"] == text_floor["all"]["ref_nll_mean"]["value"]
    checks = eng["verdict"]["checks"]
    assert checks["top1_agreement"]["worse_than_floor"] == checks["ref_nll_mean"]["worse_than_floor"] == 0.0
    assert eng["verdict"]["verdict"] == "PASS", eng["verdict"]["reason"]
    # first divergence is measured from the baseline's text read the same way
    assert eng["first_divergence"]["identical"] == eng["first_divergence"]["n_prompts"] == 4


def test_text_only_trajectory_still_catches_a_shift(tmp_path, store):
    prompts = write_prompts(tmp_path / "prompts.jsonl", n=5, max_tokens=40)
    run = tmp_path / "runs" / "t4"
    ref_lm, eng_lm = FakeLM(), FakeLM(perturb=1.5, logprob_mode="refuse")
    with serve(ref_lm) as ref_url, serve(eng_lm) as eng_url:
        trajectory(eng_url, prompts, run, "shifted")
        trajectory(ref_url, prompts, run, "reference")
        _, report = score(ref_url, run, ["shifted"], baseline="reference", margin_agreement=0.001, margin_nll=1e-4)
    agree = n = 0
    for f in sorted((run / "quality" / "shifted").glob("p*.json")):
        doc = json.loads(f.read_text())
        ctx = ref_lm.template(doc["messages"])
        for tok in ref_lm.tokenize_text(doc["text"]):  # the toy tokenizer reads <tN> back as token N
            p = ref_lm.logprobs(ctx)
            agree += tok == max(range(ref_lm.vocab), key=lambda j: p[j])
            n += 1
            ctx = ctx + [tok]
    eng = report["engines"]["shifted"]
    assert n == 5 * 40 and eng["all"]["top1_agreement"]["value"] == pytest.approx(agree / n) and agree < n
    assert eng["verdict"]["verdict"] == "FAIL"


# --------------------------------------------------------------------------- gsm8k concurrency and budget

GOOD = {f"Q{i}:": f"#### {i * 10}" for i in range(1, 30)}


def write_gsm8k_n(path, n):
    rows = [{"question": f"Q{i}: how many apples?", "answer": f"#### {i * 10}"} for i in range(1, n + 1)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.mark.parametrize("info,props,used,source", [
    ({"max_running_requests": 6}, None, 6, "get_server_info"),
    ({"max_running_requests": None, "internal_states": [{"max_running_requests": 256}]}, None, 32,
     "get_server_info"),
    (None, {"total_slots": 3}, 3, "llamacpp_props"),
])
def test_gsm8k_auto_concurrency_reads_the_server_capacity(tmp_path, store, info, props, used, source):
    data = write_gsm8k(tmp_path / "gsm8k.jsonl")
    lm = FakeLM(answers=GOOD, server_info=info, props=props)
    with serve(lm) as url:
        rec = gsm8k(url, tmp_path / "runs" / "c1", data, "auto", concurrency=0)
    conc = rec["data"]["concurrency"]
    assert (conc["requested"], conc["used"], conc["source"]) == (0, used, source)
    assert rec["identity"]["workload"]["concurrency"] == rec["identity"]["workload"]["batch"] == used
    assert f"concurrency {used} ({source})" in rec["summary"]
    assert len([r for r in lm.requests if r["path"] == "/v1/chat/completions"]) == 5  # no timing probe
    assert rec["data"]["correct"] == 5


def test_gsm8k_concurrency_source_is_in_identity_workload(tmp_path, store):
    data = write_gsm8k(tmp_path / "gsm8k.jsonl")
    with serve(FakeLM(answers=GOOD, props={"total_slots": 3})) as url:
        rec = gsm8k(url, tmp_path / "runs" / "c1w", data, "auto", concurrency=0)
    assert rec["identity"]["workload"]["concurrency_source"] == "llamacpp_props"
    check = ev.verify(rec)  # the schema must list the field too (workload has additionalProperties false)
    assert check["id_ok"] and check["schema_ok"], check["schema_errors"]


def test_gsm8k_auto_concurrency_timing_probe(tmp_path, store):
    data = write_gsm8k(tmp_path / "gsm8k.jsonl")
    with serve(FakeLM(answers=GOOD, slots=2, delay_s=0.2)) as two, serve(FakeLM(answers=GOOD, delay_s=0.2)) as wide:
        a = gsm8k(two, tmp_path / "runs" / "c2", data, "two-slots", concurrency=0)
        b = gsm8k(wide, tmp_path / "runs" / "c2", data, "batching", concurrency=0)
    ca, cb = a["data"]["concurrency"], b["data"]["concurrency"]
    assert ca["source"] == cb["source"] == "timing_probe"
    # one request alone, then a burst of 8: a server with 2 slots runs it in 4 waves, so 8 * t1 / t8 = 2
    assert ca["used"] == 2 and [x["size"] for x in ca["detail"]["bursts"]] == [8]
    assert "get_server_info" in ca["detail"]["server_info"] and "props" in ca["detail"]["server_info"]
    # a server that runs every request at once passes the burst of 8, then of 32 (the cap)
    assert [x["size"] for x in cb["detail"]["bursts"]] == [8, 32] and 16 <= cb["used"] <= 32
    assert a["data"]["correct"] == b["data"]["correct"] == 5


def test_gsm8k_flag_concurrency_is_recorded_as_given(tmp_path, store):
    data = write_gsm8k(tmp_path / "gsm8k.jsonl")
    with serve(FakeLM(answers=GOOD, server_info={"max_running_requests": 6})) as url:
        rec = gsm8k(url, tmp_path / "runs" / "c3", data, "flag")
    assert rec["data"]["concurrency"] == {"requested": 4, "used": 4, "source": "flag", "detail": {}}
    assert rec["data"]["time_budget"] is None and "concurrency" not in rec["summary"]


def test_gsm8k_time_budget_stops_with_the_first_n(tmp_path, store):
    data = write_gsm8k_n(tmp_path / "gsm8k.jsonl", 20)
    run = tmp_path / "runs" / "c4"
    with serve(FakeLM(answers=GOOD, delay_s=0.1)) as url:
        rec = gsm8k(url, run, data, "budget", concurrency=1, time_budget_s=1)
    n = rec["data"]["n"]
    assert 1 <= n < 20
    rows = [json.loads(x) for x in (run / "quality" / "gsm8k" / "budget.jsonl").read_text().splitlines()]
    assert [r["idx"] for r in rows] == list(range(n)) and rec["data"]["correct"] == n
    budget = rec["data"]["time_budget"]
    assert budget["stopped_early"] and budget["issued"] == n and budget["budget_s"] == 1
    # the record rounds the projection to 0.1 s and the mean latency to 1 ms: up to 0.05 + 20 * 0.0005 apart
    assert budget["projected_full_s"] == pytest.approx(20 * budget["mean_latency_s"], abs=0.061)
    assert any(lim.startswith(f"subset: first {n} of 20 problems") for lim in rec["limitations"])
    assert f"time budget: first {n} of 20" in rec["summary"]


def test_gsm8k_concurrency_and_budget_arguments_are_checked(tmp_path, store):
    data = write_gsm8k(tmp_path / "gsm8k.jsonl")
    for kw, msg in (({"concurrency": -1}, "concurrency"), ({"time_budget_s": 0}, "time_budget_s"),
                    ({"time_budget_s": 60, "backend": "lm-eval"}, "builtin backend")):
        with pytest.raises(registry.OpError, match=msg):
            gsm8k("http://127.0.0.1:9", tmp_path / "runs" / "c5", data, "x", **kw)
    with serve(FakeLM(answers=GOOD)) as url:  # a generous budget changes nothing
        rec = gsm8k(url, tmp_path / "runs" / "c5", data, "roomy", time_budget_s=600)
    assert rec["data"]["n"] == 5 and not rec["data"]["time_budget"]["stopped_early"]
    assert not any(lim.startswith("subset") for lim in rec["limitations"])
