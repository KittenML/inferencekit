"""review build: escaping, first-divergence marking, bucket grouping, NOT RUN cells, offline page."""

import json
import re
import time

import pytest

from inferencekit import cli, registry
from inferencekit.registry import OpError
from inferencekit.review.diff import align_words, first_token_divergence, word_diff
from inferencekit.review.page import MARK, render_diff
from inferencekit.review.rundir import load_run

XSS = "<script>alert(1)</script>"
IMG = '<img src=x onerror="alert(2)">'


def _w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def make_llm_run(root):
    """A small LLM run: reference + two engines, four buckets, one missing and one failed output."""
    prompts = {("very_short", "vs-0001"): "Hi there",
               ("short", "short-0002"): f"Repeat this exactly: {XSS}",
               ("short", "short-0010"): "Name three primes.",
               ("medium", "med-0001"): "Summarise the paragraph. " * 5,
               ("long", "long-0001"): "Context sentence number one. " * 40}
    for (bucket, iid), text in prompts.items():
        _w(root / "inputs" / bucket / f"{iid}.json",
           {"id": iid, "bucket": bucket, "messages": [{"role": "user", "content": text}], "max_tokens": 64})
    ref = {("very_short", "vs-0001"): ("Hello! How can I help?", [1, 2, 3, 4, 5]),
           ("short", "short-0002"): (f"Sure: {XSS}", [10, 11, 12]),
           ("short", "short-0010"): ("Two, three and five.", [20, 21, 22, 23]),
           ("medium", "med-0001"): ("The paragraph says the cat sat on the mat.", [30, 31, 32, 33, 34, 35]),
           ("long", "long-0001"): ("It repeats one sentence.", [40, 41, 42])}
    for (bucket, iid), (text, ids) in ref.items():
        _w(root / "outputs" / "reference" / bucket / f"{iid}.json",
           {"text": text, "token_ids": ids, "finish_reason": "stop", "usage": {"completion_tokens": len(ids)}})
    vllm = dict(ref)
    vllm[("medium", "med-0001")] = ("The paragraph says the dog sat on the mat.", [30, 31, 32, 99, 34, 35])
    vllm[("short", "short-0002")] = (f"Sure: {IMG}", [10, 11, 77])
    for (bucket, iid), (text, ids) in vllm.items():
        _w(root / "outputs" / "vllm-tuned" / bucket / f"{iid}.json", {"text": text, "token_ids": ids})
    for (bucket, iid), (text, ids) in ref.items():
        if iid == "short-0010":
            continue  # never produced: must show NOT RUN
        body = {"error": "HTTP 500 from server"} if iid == "long-0001" else {"text": text, "token_ids": ids}
        _w(root / "outputs" / "sglang" / bucket / f"{iid}.json", body)
    _w(root / "metrics" / "vllm-tuned.json", {
        "engine": "vllm-tuned", "role": "tuned",
        "per_input": {"med-0001": {"ttft_ms": 41.5, "decode_tok_s": 181.2, "tpot_ms": 5.5},
                      "vs-0001": {"ttft_ms": 30.1, "decode_tok_s": {"value": 190.0, "ci95": [189, 191]}}},
        "per_bucket": {"medium": {"decode_tok_s": {"value": 181.2, "ci95": [180.0, 182.5]}, "ttft_ms": 41.5}},
        "quality": {"tf_agreement": {"verdict": "PASS", "value": 98.9, "unit": "%"}},
        "gates": {"verdict": "PASS", "violations": []},
        "cost": {"usd_per_1m_output_tokens": 0.91, "gpu_price_usd_h": 0.62},
        "ceiling_tok_s": 250.0,
        "evidence": ["ev_" + "a" * 64]})
    return root


@pytest.fixture
def llm_run(tmp_path):
    return make_llm_run(tmp_path / "runs" / "run-1")


def build(run_dir, **kw):
    return registry.call("build_review", {"run_dir": str(run_dir), **kw})


def page_of(run_dir) -> str:
    return (run_dir / "review" / "index.html").read_text()


def cell_of(page: str, iid: str, engine_idx: int) -> str:
    row = page[page.index(f'<div class="iid">{iid}</div>'):]
    row = row[:row.index("</tr>")]
    start = row.rindex("<td ", 0, row.index(f'data-e="{engine_idx}"'))
    end = row.find("<td ", start + 1)
    return row[start:end if end != -1 else None]


# --------------------------------------------------------------------------- the page

def test_build_writes_one_page_and_data(llm_run):
    res = build(llm_run)
    assert res["path"].endswith("review/index.html") and res["inputs"] == 5
    assert res["engines"] == ["reference", "sglang", "vllm-tuned"] and res["reference"] == "reference"
    assert res["not_run"] == 1 and res["failed"] == 1 and res["modality"] == "llm"
    data = json.loads((llm_run / "review" / "data.json").read_text())
    assert any(s["path"] == "metrics/vllm-tuned.json" and len(s["sha256"]) == 64 for s in data["sources"])


def test_model_text_is_escaped(llm_run):
    build(llm_run)
    page = page_of(llm_run)
    assert XSS not in page and "<img" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=&quot;alert(2)&quot;&gt;" in page
    assert page.count("<script>") == 1  # only the page's own filter script


def test_engine_labels_and_paths_are_escaped(tmp_path):
    root = tmp_path / "r"
    _w(root / "inputs" / "short" / "a.txt", "hello")
    _w(root / "outputs" / 'x"><b>bold' / "short" / "a.txt", "out")
    build(root)
    page = page_of(root)
    assert '"><b>bold' not in page and "x&quot;&gt;&lt;b&gt;bold" in page


def test_page_is_offline_and_themed(llm_run):
    build(llm_run)
    page = page_of(llm_run)
    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:)?//""", page)
    assert "<link" not in page and "@import" not in page
    assert "prefers-color-scheme:dark" in page and 'name="viewport"' in page


def test_buckets_are_grouped_in_order(llm_run):
    build(llm_run)
    page = page_of(llm_run)
    heads = re.findall(r'<section class="bucket" data-b="\d+"><h2>([a-z ]+) ', page)
    assert heads == ["very short", "short", "medium", "long"]
    sections = page.split('<section class="bucket"')[1:]
    assert "short-0002" in sections[1] and "short-0010" in sections[1] and "med-0001" in sections[2]
    assert "long-0001" in sections[3] and "short-0002" not in sections[2]
    # rows inside a bucket in natural id order
    assert sections[1].index("short-0002") < sections[1].index("short-0010")


def test_missing_output_is_not_run_not_blank(llm_run):
    build(llm_run)
    page = page_of(llm_run)
    cell = cell_of(page, "short-0010", 1)  # sglang never wrote it
    assert "NOT RUN" in cell and "st-notrun" in cell and "no output file" in cell
    failed = cell_of(page, "long-0001", 1)
    assert "FAILED" in failed and "HTTP 500" in failed
    assert "1 NOT RUN" in page and "1 failed" in page
    # every engine cell has visible content
    for td in re.findall(r'<td class="cell[^"]*" data-e="\d+">(.*?)</td>', page, flags=re.S):
        assert re.sub(r"<[^>]+>", "", td).strip()


def test_first_divergence_is_marked_at_the_first_differing_word(llm_run):
    build(llm_run)
    page = page_of(llm_run)
    cell = cell_of(page, "med-0001", 2)
    assert "diverges @ token 3/6" in cell and "st-diff" in cell
    text = cell[cell.index('<div class="text">'):]
    assert text.index("The paragraph says the ") < text.index(MARK) < text.index("<del")
    assert '<del title="only in the reference">cat </del><ins>dog </ins>' in text
    same = cell_of(page, "vs-0001", 2)
    assert "identical" in same and MARK not in same
    assert "(reference)" in page and "3/5 identical" in page


def test_summary_shows_speed_ceiling_quality_gates_cost(llm_run):
    build(llm_run)
    page = page_of(llm_run)
    summary = page[page.index('<table class="summary">'):page.index("</table>")]
    vllm_row = summary[summary.index("vllm-tuned"):].split("</tr>")[0]
    assert "181.2" in vllm_row and "180" in vllm_row and "182.5" in vllm_row  # bucket median with CI
    assert "tf_agreement: PASS" in vllm_row and "b-pass" in vllm_row
    assert "$0.91" in vllm_row
    assert "74.24%" in vllm_row  # median(190, 181.2) / 250 tok/s ceiling, derived
    sg_row = summary[summary.index(">sglang"):].split("</tr>")[0]
    assert "NOT RUN" in sg_row and "unknown" in sg_row and "not measured" in sg_row


def test_filters_present_for_buckets_and_engines(llm_run):
    build(llm_run)
    page = page_of(llm_run)
    assert len(re.findall(r'data-fb="\d+"', page)) == 4
    assert len(re.findall(r'data-fe="\d+"', page)) == 3
    assert 'id="only-diff"' in page and 'data-flag="1"' in page and 'id="worst-first"' in page
    by_id = {iid: float(sev) for sev, iid in re.findall(r'data-sev="([\d.]+)">.*?<div class="iid">([^<]+)</div>',
                                                         page)}
    assert by_id["short-0010"] == 2.0 and by_id["vs-0001"] == 0.0  # NOT RUN worst; all identical best
    assert 1.0 < by_id["med-0001"] < 2.0


def test_reference_flag_and_errors(llm_run):
    res = build(llm_run, reference="vllm-tuned")
    assert res["engines"][0] == "vllm-tuned"
    with pytest.raises(OpError, match="no-such-engine"):
        build(llm_run, reference="no-such-engine")
    with pytest.raises(OpError):
        build(llm_run / "nope")


def test_no_reference_still_renders(tmp_path):
    root = tmp_path / "r"
    _w(root / "outputs" / "a" / "short" / "x.json", {"text": "hello"})
    res = build(root)
    assert res["reference"] is None and any("no reference" in w for w in res["warnings"])
    assert "hello" in page_of(root)


def test_cli_review_build(llm_run, capsys):
    assert cli.main(["review", "build", str(llm_run), "--reference", "reference"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["path"].endswith("index.html") and (llm_run / "review" / "index.html").is_file()


def test_scales_to_64_inputs_by_6_engines(tmp_path):
    root = tmp_path / "big"
    buckets = ["very_short", "short", "medium", "long"]
    words = [f"w{i}" for i in range(220)]
    for n in range(64):
        b = buckets[n % 4]
        _w(root / "inputs" / b / f"{b}-{n:04d}.json", {"messages": [{"role": "user", "content": "x " * 3000}]})
        _w(root / "outputs" / "reference" / b / f"{b}-{n:04d}.json",
           {"text": " ".join(words), "token_ids": list(range(220))})
        for e in range(5):
            text = list(words)
            text[(n * 7 + e * 13) % 220] = "CHANGED"
            ids = list(range(220))
            ids[(n * 7 + e * 13) % 220] = -1
            _w(root / "outputs" / f"engine-{e}" / b / f"{b}-{n:04d}.json", {"text": " ".join(text), "token_ids": ids})
    t0 = time.perf_counter()
    res = build(root)
    assert time.perf_counter() - t0 < 20
    assert res["cells"] == 64 * 6 and res["not_run"] == 0
    page = page_of(root)
    assert page.count('class="item"') == 64 and "@media (max-width:760px)" in page
    assert res["bytes"] < 4_000_000


# --------------------------------------------------------------------------- speech

def test_tts_audio_players_and_asr_word_errors(tmp_path):
    root = tmp_path / "tts"
    _w(root / "intake.yaml", "modality: tts\n")
    _w(root / "inputs" / "short" / "s1.txt", "The quick brown fox jumps.")
    (root / "outputs" / "kokoro" / "short").mkdir(parents=True)
    (root / "outputs" / "kokoro" / "short" / "s1.s0.wav").write_bytes(b"RIFF0000WAVE")
    _w(root / "metrics" / "kokoro.json", {"per_input": {"s1": {"transcript": "the quick brown box jumps",
                                                              "ttfa_ms": 120.0, "rtf": 0.05, "utmos": 4.1}}})
    res = build(root)
    assert res["modality"] == "tts"
    page = page_of(root)
    assert '<audio controls preload="none" src="../outputs/kokoro/short/s1.s0.wav">' in page
    assert '<span class="w-sub" title="substitution; ground truth: fox">box</span>' in page
    assert "WER 20.0%" in page and "TTFA 120 ms" in page


def test_stt_transcript_errors_against_ground_truth(tmp_path):
    root = tmp_path / "stt"
    (root / "inputs" / "medium").mkdir(parents=True)
    (root / "inputs" / "medium" / "m1.wav").write_bytes(b"RIFF0000WAVE")
    _w(root / "inputs" / "medium" / "m1.gt.txt", "please call stella ask her to bring these things from the store")
    _w(root / "outputs" / "whisper" / "medium" / "m1.json", {"text": "Please call Stella, ask her to bring things from the store today"})
    res = build(root)
    assert res["modality"] == "stt"
    page = page_of(root)
    assert '<span class="w-del" title="deletion: missing from the output">these</span>' in page
    assert '<span class="w-ins" title="insertion">today</span>' in page
    assert "ground truth:" in page and 'src="../inputs/medium/m1.wav"' in page


# --------------------------------------------------------------------------- comparison helpers

def test_first_token_divergence():
    assert first_token_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_token_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_token_divergence([1, 2], [1, 2, 3]) == 2
    assert first_token_divergence(None, []) is None


def test_word_diff_round_trips_and_finds_first_word():
    td = word_diff("a b c d", "a b x d e")
    assert "".join(c for _, _, c in td.segments) == "a b x d e"
    assert td.first_word == 2 and not td.identical
    assert word_diff("same  text\n", "same  text\n").identical


def test_token_level_marker_inside_a_word():
    # Both outputs share the text up to "unbelie", then token 2 differs: the marker sits mid-word.
    td = word_diff("unbelievable story", "unbelieving story")
    info = {"diff": td, "marker_char": len("unbelie")}
    html = render_diff(info, "unbelieving story")
    assert html.index(MARK) > html.index("<ins>unbelie") and "<ins>ving </ins>" in html.replace(MARK, "")


def test_token_marker_from_logprob_pieces(tmp_path):
    root = tmp_path / "r"
    _w(root / "inputs" / "short" / "a.json", {"prompt": "go"})
    _w(root / "outputs" / "reference" / "short" / "a.json", {"text": "unbelievable story", "token_ids": [1, 2, 3, 4]})
    _w(root / "outputs" / "e" / "short" / "a.json",
       {"text": "unbelieving story", "token_ids": [1, 2, 9, 4],
        "logprobs": {"content": [{"token": "un", "logprob": 0}, {"token": "belie", "logprob": 0},
                                 {"token": "ving", "logprob": 0}, {"token": " story", "logprob": 0}]}})
    run = load_run(root)
    assert run.cells[("e", "short/a")].pieces == ["un", "belie", "ving", " story"]
    build(root)
    cell = cell_of(page_of(root), "a", 1)
    assert "diverges @ token 2/4" in cell
    assert cell.index("unbelie") < cell.index(MARK) < cell.index("ving")


def test_word_alignment_counts():
    al = align_words("the cat sat on the mat", "the cat sat on a mat today")
    assert (al.count("sub"), al.count("del"), al.count("ins")) == (1, 0, 1)
    assert al.wer == pytest.approx(2 / 6) and al.exact
    assert align_words("", "anything").wer is None
    long_ref = " ".join(f"w{i}" for i in range(800))
    long_hyp = long_ref.replace("w400 ", "")
    big = align_words(long_ref, long_hyp)
    assert not big.exact and big.count("del") == 1 and big.count("sub") == 0


# --------------------------------------------------------------------------- bench serve / quality / gates files

def _ev_file(root, kind, data, metrics=(), engine=None):
    from conftest import fake_call

    from inferencekit import evidence as ev

    rec = ev.build(fake_call(), kind="correctness", claim_class="derived", summary=kind,
                   identity_=ev.identity(engine={"name": engine}), metrics=list(metrics), data=data)
    rec["kind"] = kind  # v2 kinds (serve / quality / gates), written as those commands do
    rec = ev.finalize(rec)
    _w(root / "evidence" / f"{rec['evidence_id']}.json", rec)
    return rec


def test_sampling_record_shows_its_excess_tv_not_a_percentage(tmp_path):
    from inferencekit import evidence as ev
    from inferencekit.review.rundir import EngineInfo, Run, attach_evidence

    rec = _ev_file(tmp_path, "quality_task", {"label": "myengine", "task": "sampling", "verdict": "PASS"},
                   [ev.metric("first_token_tv", 0.21, "probability", "mean", n=32, subject="myengine"),
                    ev.metric("first_token_tv_excess", 0.0011, "probability", "paired_difference", n=32,
                              ci95=[-0.0085, 0.0116], subject="myengine")])
    run = Run(root=tmp_path, modality="llm", engines=[EngineInfo("myengine")], reference=None, items=[], cells={})
    attach_evidence(run, [("evidence", rec)])
    assert run.engines[0].ev_quality == [{"name": "sampling", "unit": "probability", "evidence": rec["evidence_id"],
                                          "value": 0.0011, "verdict": "PASS"}]


def test_reads_bench_serve_layout_and_lane_evidence(tmp_path):
    from inferencekit import evidence as ev

    root = tmp_path / "serve-run"
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text("\n".join(json.dumps({"id": f"short-000{i}", "bucket": "short", "max_tokens": 32,
                                             "messages": [{"role": "user", "content": f"question {i}"}]})
                                 for i in range(2)) + "\n")
    for label, texts in (("reference", ["a b c", "d e f"]), ("vllm-tuned", ["a b c", "d X f"])):
        _w(root / "outputs" / label / "index.json", {"label": label, "files": {}})
        for i, text in enumerate(texts):
            _w(root / "outputs" / label / "short" / f"short-000{i}.json",
               {"id": f"short-000{i}", "bucket": "short", "text": text, "token_ids": [1, 2, 3 + i * (label != "reference")],
                "finish_reason": "length", "usage": {"completion_tokens": 3}, "error": None, "timings": {}})
    _w(root / "outputs" / "vllm-tuned" / "c8" / "short" / "short-0000.json", {"text": "load"})
    _w(root / "metrics" / "vllm-tuned.json", {"label": "vllm-tuned", "speed": {
        "per_bucket": {"short": {"n_requests": 6, "ttft_ms": {"median": 21.5, "p95": 30.0, "ci95_median": [20.0, 23.0]},
                                 "decode_tok_s": {"median": 150.0, "p95": 160.0, "ci95_median": [149.0, 151.0]}},
                       "all": {"n_requests": 6, "decode_tok_s": {"median": 150.0}}},
        "per_input_median": {"short-0000": {"bucket": "short", "ttft_ms": 21.0, "decode_tok_s": 151.0, "e2e_ms": 40.0}}}})
    _w(root / "metrics" / "vllm-tuned.load.json", {"label": "vllm-tuned", "levels": []})
    _ev_file(root, "quality_trajectory", {"v2_kind": "quality_trajectory", "verdict": "PASS",
                                          "verdicts": {"vllm-tuned": "PASS"}, "noise_floor": "reference"},
             [ev.metric("top1_agreement_pct", 99.1, "%", "pooled_mean", subject="vllm-tuned")])
    _ev_file(root, "quality_task", {"label": "vllm-tuned", "task": "gsm8k"},
             [ev.metric("gsm8k_accuracy_pct", 85.5, "%", "rate", n=100, subject="vllm-tuned")])
    _ev_file(root, "gates", {"verdict": "FAIL", "gates_format": "inferencekit.gates.v1",
                             "gates": {"output_length": {"verdict": "FAIL"}, "unique_prompts": {"verdict": "PASS"}}},
             engine="vllm-tuned")
    _ev_file(root, "capacity", {"label": "vllm-tuned", "n_star": 24, "slo": {"ttft_p95_ms": 300},
                                "gpu_price_usd_h": 0.62},
             [ev.metric("usd_per_1m_output_tokens", 0.31, "USD", "at_n_star")])
    res = build(root, prompts=str(prompts))
    assert res["engines"] == ["reference", "vllm-tuned"] and res["inputs"] == 2  # no 'index' or 'c8' rows
    assert any("load-level" in w for w in res["warnings"])
    page = page_of(root)
    assert "question 0" in page  # the input column filled from the prompt set
    summary = page[page.index('<table class="summary">'):page.index("</table>")]
    row = summary[summary.index(">vllm-tuned"):].split("</tr>")[0]
    assert "decode <b>150</b> tok/s" in row and "(149&ndash;151)" in row and "TTFT <b>21.5</b>" in row
    assert "top-1 agreement: PASS" in row and "gsm8k 85.5%" in row
    assert 'b-fail' in row and "output_length: FAIL" in row  # gates verdict + failing gate in the tooltip
    assert "<b>24</b> streams" in row and "ttft_p95_ms &le; 300" in row and "$0.31" in row
    cell = cell_of(page, "short-0001", 1)
    assert "diverges @ token 2/3" in cell and "TTFT 21" not in cell
    assert "TTFT 21 ms" in cell_of(page, "short-0000", 1)


def test_reasoning_is_shown_collapsed_and_escaped(tmp_path):
    root = tmp_path / "r"
    _w(root / "outputs" / "reference" / "short" / "a.json", {"text": "42"})
    _w(root / "outputs" / "e" / "short" / "a.json", {"text": "42", "reasoning": f"think {XSS}"})
    build(root)
    cell = cell_of(page_of(root), "a", 1)
    assert '<details class="reasoning">' in cell and "think &lt;script&gt;" in cell and "identical" in cell


def test_edited_lane_evidence_is_unproven(tmp_path):
    root = tmp_path / "r"
    _w(root / "outputs" / "e" / "short" / "a.json", {"text": "x"})
    rec = _ev_file(root, "gates", {"verdict": "PASS", "gates_format": "inferencekit.gates.v1", "gates": {}},
                   engine="e")
    path = root / "evidence" / f"{rec['evidence_id']}.json"
    body = json.loads(path.read_text())
    body["summary"] = "tampered"
    path.write_text(json.dumps(body))
    res = build(root)
    assert any("UNPROVEN" in w for w in res["warnings"])
    page = page_of(root)
    row = page[page.index('<table class="summary">'):page.index("</table>")]
    assert "b-warn" in row and ">UNPROVEN<" in row


# --------------------------------------------------------------------------- review fixes

def test_whitespace_only_difference_is_not_identical(tmp_path):
    root = tmp_path / "ws"
    ref = "Line one.\n\n- item a\n- item b"
    _w(root / "outputs" / "reference" / "short" / "a.json", {"text": ref})
    _w(root / "outputs" / "reference" / "short" / "b.json", {"text": "same"})
    _w(root / "outputs" / "e" / "short" / "a.json", {"text": "Line one. - item a - item b "})
    _w(root / "outputs" / "e" / "short" / "b.json", {"text": "same"})
    build(root)
    page = page_of(root)
    cell = cell_of(page, "a", 1)
    assert "whitespace differs @ char 9" in cell and ">identical<" not in cell and "st-diff" in cell
    text = cell[cell.index('<div class="text">'):]
    assert text.index("Line one.") < text.index(MARK) < text.index("- item a")
    assert "1/2 identical" in page and ">identical<" in cell_of(page, "b", 1)
    assert re.search(r'data-flag="1"[^>]*>\s*<th scope="row" class="input"><div class="iid">a<', page)
    data = json.loads((root / "review" / "data.json").read_text())
    a = next(c for c in data["cells"] if c["input"] == "a" and c["engine"] == "e")
    assert a["identical"] is False and a["whitespace_only"] and a["first_divergence_char"] == 9


def test_leading_whitespace_marker_sits_at_the_start():
    td = word_diff("a b", "  a b")
    out = render_diff({"diff": td, "marker_char": 0}, "  a b")
    assert out.startswith(MARK)


def test_capacity_n_star_none_is_fail_only_when_the_slo_failed(tmp_path):
    from inferencekit import evidence as ev

    root = tmp_path / "cap"
    cases = {"e1": {"n_star": None, "reason": "SLO first fails at level 1 (ttft_p95_ms)"},
             "e2": {"n_star": None, "reason": "no SLO given"},
             "e3": {"n_star": None, "reason": "no load levels ran"},
             "e4": {"n_star": 4.0, "level_kind": "rate", "slo": {"ttft_p95_ms": 300},
                    "reason": "SLO holds at every level tried: the true N* may be above the grid"},
             "e5": {"n_star": 8, "level_kind": "concurrency", "slo": {XSS: 300, "tpot_p95_ms": 50},
                    "reason": "SLO first fails at level 16 (tpot_p95_ms)"}}
    for label, data in cases.items():
        _w(root / "outputs" / label / "short" / "a.json", {"text": "x"})
        _ev_file(root, "capacity", {"label": label, **data},
                 [ev.metric("n_star", data["n_star"], "streams", "threshold")])
    build(root)
    page = page_of(root)
    summary = page[page.index('<table class="summary">'):page.index("</table>")]
    row = {lb: summary[summary.index(f">{lb}<"):].split("</tr>")[0] for lb in cases}
    assert ">FAIL<" in row["e1"] and "SLO missed at every level" in row["e1"]
    assert ">UNPROVEN<" in row["e2"] and "no SLO given" in row["e2"] and "FAIL" not in row["e2"]
    assert ">NOT RUN<" in row["e3"] and "FAIL" not in row["e3"]
    assert "at least <b>4</b> req/s" in row["e4"] and "streams" not in row["e4"]
    assert "<b>8</b> streams" in row["e5"] and "tpot_p95_ms &le; 50" in row["e5"]
    assert XSS not in page and "&lt;script&gt;alert(1)&lt;/script&gt; &le; 300" in row["e5"]


def test_no_evidence_attached_is_said_and_store_records_for_the_run_are_found(tmp_path, store):
    from conftest import fake_call  # noqa: F401  (store fixture sets the evidence root)

    from inferencekit import evidence as ev
    from inferencekit.registry import Arg, Call, Operation

    root = tmp_path / "runs" / "r1"
    _w(root / "outputs" / "e" / "short" / "a.json", {"text": "x"})
    res = build(root)
    assert any(w.startswith("no evidence attached") for w in res["warnings"])
    page = page_of(root)
    assert "no evidence attached" in page[page.index('<table class="summary">'):page.index("</table>")]
    # A gates record saved in the store (not copied into evidence/) by a call made with --run-dir <this run>.
    op = Operation("run_gates", ("gates", "run"), "t", (Arg("run_dir", "string", "d"),), lambda call, **kw: {})
    rec = ev.build(Call(op, {"run_dir": str(root)}), kind="correctness", claim_class="derived", summary="gates",
                   identity_=ev.identity(engine={"name": "e"}),
                   data={"gates_format": "inferencekit.gates.v1", "verdict": "PASS", "gates": {}})
    store.save(rec)
    other = ev.build(Call(op, {"run_dir": str(tmp_path / "elsewhere")}), kind="correctness", claim_class="derived",
                     summary="g2", identity_=ev.identity(engine={"name": "e"}),
                     data={"gates_format": "inferencekit.gates.v1", "verdict": "FAIL", "gates": {}})
    store.save(other)
    res = build(root)
    assert not any(w.startswith("no evidence attached") for w in res["warnings"])
    summary = page_of(root)[page_of(root).index('<table class="summary">'):]
    row = summary[summary.index(">e<"):].split("</tr>")[0]
    assert ">PASS<" in row and ">FAIL<" not in row
