"""Video: the review page for clip runs.

The run directories here are written in bench clip's layout (outputs/<label>/r<round>/<id>.mp4, the exact request
bodies at inputs/<label>/r<round>/<id>.json, metrics/<label>.clip.json, and the record's requests.jsonl in the
evidence store), so the page is tested against what bench clip writes, without running it.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from inferencekit import evidence as ev
from inferencekit import registry
from inferencekit.review.rundir import load_run

XSS = "<script>alert(1)</script>"


def _w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, bytes):
        path.write_bytes(obj)
    else:
        path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def _record(store, kind, data, metrics=(), engine=None, artifacts=None, run_dir=None):
    """A record of a video kind, built as the video ops build theirs (the kind set after building, as the other
    review tests do for kinds this schema may not list yet), saved under the store root. With ``run_dir`` its
    request names that run directory, as a record made with ``--run-dir`` does."""
    from inferencekit.registry import Arg, Call, Operation

    op = Operation("bench_clip", ("bench", "clip"), "t", (Arg("run_dir", "string", "d"),), lambda call, **kw: {})
    call = Call(op, {"run_dir": str(run_dir)} if run_dir else {}, started="2026-01-01T00:00:00.000Z")
    rec = ev.build(call, kind="correctness", claim_class="observed", summary=kind,
                   identity_=ev.identity(engine={"name": engine}), metrics=list(metrics), data=data)
    rec["kind"] = kind
    if artifacts is not None:
        rec["artifacts"] = artifacts
    rec = ev.finalize(rec)
    _w(store.root / rec["evidence_id"] / "evidence.json", rec)  # where Store.save puts a record
    return rec


CLIPS = [{"id": "dog-run", "task": "t2v", "prompt": f"A dog runs on a beach {XSS}"},
         {"id": "cat-jump", "task": "i2v", "prompt": "The cat jumps onto the sofa"}]
SHAPE = {"width": 768, "height": 512, "num_frames": 121, "fps": 24, "steps": 40, "guidance_scale": 3.0}


def make_clip_run(root, store, rounds=2, fast_steps=None):
    """reference + 'tuned', two rounds of two clips, in bench clip's layout; 'tuned' fails one clip in round 1."""
    for label in ("reference", "tuned"):
        rows = []
        for r in range(rounds):
            seed = 1000 + r
            for c in CLIPS:
                body = {"id": c["id"], "task": c["task"], "prompt": c["prompt"], **SHAPE, "seed": seed}
                if c["task"] == "i2v":
                    body["image"] = {"sha256": "ab" * 32, "bytes": 2048}
                _w(root / "inputs" / label / f"r{r}" / f"{c['id']}.json", body)
                failed = label == "tuned" and r == 1 and c["id"] == "cat-jump"
                data = f"{label}:{c['id']}:{seed}".encode()
                if not failed:
                    _w(root / "outputs" / label / f"r{r}" / f"{c['id']}.mp4", data)
                steps = fast_steps if (fast_steps and label == "tuned") else SHAPE["steps"]
                rows.append({"id": c["id"], "task": c["task"], "round": r, "seed": seed, "ok": not failed,
                             "error": "HTTP 500" if failed else None, "e2e_ms": None if failed else 40000.0 - 5000 * (
                                 label == "tuned") + 100 * r,
                             "stage_ms": None if failed else {"text_encode": 900.0, "denoise": 30000.0,
                                                               "vae_decode": 2500.0},
                             "stage_sum_ok": True, "steps_run": steps, "w": 768, "h": 512, "frames": 121, "fps": 24,
                             "dtype": "bf16", "attention": "flash", "effective_checked": 5,
                             "effective_mismatch": [f"steps_run {steps} (asked 40)"] if steps != 40 else [],
                             "cache": [], "out_sha256": None if failed else hashlib.sha256(data).hexdigest()})
        staging = root.parent / f"staging-{label}"
        _w(staging / "requests.jsonl", "".join(json.dumps(x) + "\n" for x in rows))
        art = ev.artifact(staging / "requests.jsonl", staging, "requests")
        rec = _record(store, "clip_benchmark", {"source": "bench clip", "label": label}, engine=label,
                      artifacts=[art], run_dir=root)
        _w(store.root / rec["evidence_id"] / "requests.jsonl", (staging / "requests.jsonl").read_text())
        speed = {"all": {"clip_ms": {"median": 40000.0 - 5000 * (label == "tuned"),
                                     "ci95_median": [39000.0, 41000.0]},
                         "clips_per_hour": {"median": 90.0, "ci95_median": [88.0, 92.0]}},
                 "stages_self_reported": {"denoise": {"median": 30000.0, "n": 4}, "vae_decode": {"median": 2500.0,
                                                                                                 "n": 4}}}
        _w(root / "metrics" / f"{label}.clip.json",
           {"label": label, "speed": speed,
            "rounds": [{"index": r, "clip_ms": 40000.0 + 100 * r, "clips_per_hour": 90.0} for r in range(rounds)]})
    return root


@pytest.fixture
def clip_run(tmp_path, store):
    return make_clip_run(store.root / "run-v1", store)


def build(run_dir, **kw):
    return registry.call("build_review", {"run_dir": str(run_dir), **kw})


def page_of(run_dir) -> str:
    return (run_dir / "review" / "index.html").read_text()


def cell_of(page: str, section: int, iid: str, engine_idx: int) -> str:
    sec = page.split('<section class="bucket"')[1 + section]
    row = sec[sec.index(f'<div class="iid">{iid}</div>'):]
    row = row[:row.index("</tr>")]
    start = row.rindex("<td ", 0, row.index(f'data-e="{engine_idx}"'))
    end = row.find("<td ", start + 1)
    return row[start:end if end != -1 else None]


def test_clip_run_is_video_with_rounds_as_tables(clip_run):
    res = build(clip_run)
    assert res["modality"] == "video" and res["reference"] == "reference"
    assert res["engines"] == ["reference", "tuned"] and res["buckets"] == ["r0", "r1"] and res["inputs"] == 4
    assert res["failed"] == 1 and res["not_run"] == 0  # the failed clip is FAILED with bench clip's error
    page = page_of(clip_run)
    heads = re.findall(r'<section class="bucket" data-b="\d+"><h2>([a-z0-9 ]+) <span class="sub">\(([^)]*)\)', page)
    assert heads == [("round 0", "2 inputs, seed 1000"), ("round 1", "2 inputs, seed 1001")]
    assert not any("no matching input" in w for w in res["warnings"])


def test_video_players_side_by_side_with_relative_links(clip_run):
    build(clip_run)
    page = page_of(clip_run)
    ref, tuned = cell_of(page, 0, "dog-run", 0), cell_of(page, 0, "dog-run", 1)
    assert '<video controls muted playsinline preload="metadata" src="../outputs/reference/r0/dog-run.mp4">' in ref
    assert 'src="../outputs/tuned/r0/dog-run.mp4"' in tuned and "reference</span>" in ref
    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:)?//""", page)  # offline: no external URL
    assert page.count("<script>") == 1 and 'data-play="1"' in page


def test_clip_chips_stages_and_failed_clip(clip_run):
    build(clip_run)
    page = page_of(clip_run)
    tuned = cell_of(page, 0, "dog-run", 1)
    assert "clip 35 s" in tuned  # client-timed e2e, observed
    assert "denoise 30 s" in tuned and "768x512, 121 frames, 24 fps, 40 steps" in tuned and "cc-s" in tuned
    failed = cell_of(page, 1, "cat-jump", 1)
    assert "FAILED" in failed and "HTTP 500" in failed and "<video" not in failed


def test_input_column_shows_the_clip_request_escaped(clip_run):
    build(clip_run)
    page = page_of(clip_run)
    assert XSS not in page and "A dog runs on a beach &lt;script&gt;" in page
    assert "t2v · 768x512 · 121 frames @ 24 fps · 40 steps · CFG 3.0 · seed 1000" in page
    assert "image sha256 abababababab" in page


def test_summary_shows_clip_time_clips_per_hour_and_stages(clip_run):
    build(clip_run)
    page = page_of(clip_run)
    summary = page[page.index('<table class="summary">'):page.index("</table>")]
    assert "all rounds" in summary and "round 0" not in summary
    row = summary[summary.index(">tuned<"):].split("</tr>")[0]
    assert "clip <b>35</b> s" in row and "(39&ndash;41)" in row and "clips <b>90</b> /h" in row
    assert "stages: denoise 30 s, vae_decode 2.5 s" in row


def test_engine_reporting_fewer_steps_is_flagged(tmp_path, store):
    root = make_clip_run(store.root / "run-fast", store, fast_steps=20)
    build(root)
    page = page_of(root)
    tuned = cell_of(page, 0, "dog-run", 1)
    assert "ran something else" in tuned and "steps_run 20 (asked 40)" in tuned and "st-diff" in tuned
    data = json.loads((root / "review" / "data.json").read_text())
    assert any(c.get("effective_mismatch") for c in data["cells"] if c["engine"] == "tuned")


def quality_clip_record(store, verdicts, rows, metric_values):
    """A clip_quality record in the shape quality clip's _save writes: one record for every engine given with
    --outputs, data.engines[label] = its verdict, checks and metrics; metrics psnr / ssim / lpips with subject =
    the label (and floor:<name> for the floors); per-clip scores in the clip_metrics artifact, one row per clip
    pair keyed '<side>|<clip key>'."""
    staging = store.root.parent / "staging-quality"
    _w(staging / "clip_metrics.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    art = ev.artifact(staging / "clip_metrics.jsonl", staging, "clip_metrics")
    metrics = []
    for label, vals in metric_values.items():
        for name, value in vals.items():
            metrics.append(ev.metric(name, value, {"psnr": "dB", "ssim": "index", "lpips": "distance"}[name],
                                     "mean_over_frames", n=2, ci95=[value * 0.99, value * 1.01], subject=label))
    metrics.append(ev.metric("psnr", 52.0, "dB", "mean_over_frames", n=2, subject="floor:alt"))
    engines = {label: {"verdict": v, "n_clips_scored": 2, "metrics": {}, "caps": [], "reason": "",
                       "checks": {m: {"verdict": v} for m in ("psnr", "ssim", "lpips")}}
               for label, v in verdicts.items()}
    overall = "FAIL" if "FAIL" in verdicts.values() else "PASS" if set(verdicts.values()) == {"PASS"} else "UNPROVEN"
    data = {"verdict": overall, "fidelity": "lossless", "engines": engines, "floor_used": "alt",
            "metrics": ["psnr", "ssim", "lpips"], "format": "inferencekit.quality.clip.v1"}
    rec = _record(store, "clip_quality", data, metrics, engine=None if len(verdicts) > 1 else next(iter(verdicts)),
                  artifacts=[art])
    _w(store.root / rec["evidence_id"] / "clip_metrics.jsonl", (staging / "clip_metrics.jsonl").read_text())
    return rec


def test_quality_and_gates_records_show_per_engine_and_per_clip(clip_run, store):
    rows = [{"key": "engine:tuned|r0/dog-run", "ok": True, "mean": {"psnr": 39.5, "ssim": 0.98, "lpips": 0.015}},
            {"key": "engine:tuned|r0/cat-jump", "ok": True, "mean": {"psnr": 41.0, "ssim": 0.99, "lpips": 0.01}},
            {"key": "engine:other|r0/dog-run", "ok": True, "mean": {"psnr": 20.0}},  # no such engine here
            {"key": "floor:alt|r0/dog-run", "ok": True, "mean": {"psnr": 52.0}}]
    quality_clip_record(store, {"tuned": "PASS", "other": "FAIL"}, rows,
                        {"tuned": {"psnr": 38.2, "ssim": 0.97, "lpips": 0.021}, "other": {"psnr": 20.0}})
    _record(store, "gates", {"label": "tuned", "verdict": "FAIL", "gates_format": "inferencekit.gates.clip.v1",
                             "gates": {"steps_honoured": {"gate": "steps_honoured", "verdict": "FAIL"},
                                       "no_cache": {"gate": "no_cache", "verdict": "PASS"}}}, engine="tuned")
    rec_ids = [p.name for p in store.root.iterdir() if p.name.startswith("ev_")]
    build(clip_run, evidence=rec_ids)
    page = page_of(clip_run)
    summary = page[page.index('<table class="summary">'):page.index("</table>")]
    row = summary[summary.index(">tuned<"):].split("</tr>")[0]
    assert "psnr: PASS" in row and "lpips: PASS" in row and "steps_honoured: FAIL" in row and "b-fail" in row
    info = load_run(clip_run, extra_evidence=[json.loads(p.read_text()) for p in store.root.glob("ev_*/evidence.json")
                                              ]).engine("tuned")
    assert {q["name"]: q["value"] for q in info.ev_quality} == {"psnr": 38.2, "ssim": 0.97, "lpips": 0.021}
    ref_row = summary[summary.index(">reference<"):].split("</tr>")[0]
    assert "psnr" not in ref_row  # the overall verdict does not land on an engine the record did not score
    cell = cell_of(page, 0, "dog-run", 1)
    assert "PSNR 39.5 dB" in cell and "SSIM 0.98" in cell and "LPIPS 0.015" in cell
    assert "PSNR" not in cell_of(page, 0, "dog-run", 0) and "PSNR 41 dB" in cell_of(page, 0, "cat-jump", 1)


def test_quality_scores_file_changed_after_capture_is_not_shown(clip_run, store):
    rows = [{"key": "engine:tuned|r0/dog-run", "ok": True, "mean": {"psnr": 39.5}}]
    rec = quality_clip_record(store, {"tuned": "PASS"}, rows, {"tuned": {"psnr": 38.2}})
    (store.root / rec["evidence_id"] / "clip_metrics.jsonl").write_text(json.dumps(
        {"key": "engine:tuned|r0/dog-run", "ok": True, "mean": {"psnr": 99.0}}) + "\n")
    res = build(clip_run, evidence=[rec["evidence_id"]])
    assert any("clip_metrics file" in w and "does not match" in w for w in res["warnings"])
    page = page_of(clip_run)
    assert "PSNR 99" not in page and "psnr: PASS" in page


def test_clip_ids_with_dots_stay_separate(tmp_path, store):
    root = store.root / "run-dots"
    rows = []
    for label in ("reference", "tuned"):
        for cid in ("scene.01", "scene.02", "scene"):
            _w(root / "inputs" / label / "r0" / f"{cid}.json", {"id": cid, "task": "t2v", "prompt": f"p {cid}",
                                                                **SHAPE, "seed": 5})
            if not (label == "tuned" and cid == "scene.02"):
                _w(root / "outputs" / label / "r0" / f"{cid}.mp4", f"{label}:{cid}".encode())
        if label == "tuned":
            rows = [{"id": "scene.01", "task": "t2v", "round": 0, "seed": 5, "ok": True, "e2e_ms": 12000.0},
                    {"id": "scene.02", "task": "t2v", "round": 0, "seed": 5, "ok": False, "error": "HTTP 503"}]
    staging = root.parent / "staging-dots"
    _w(staging / "requests.jsonl", "".join(json.dumps(x) + "\n" for x in rows))
    rec = _record(store, "clip_benchmark", {"source": "bench clip", "label": "tuned"}, engine="tuned",
                  artifacts=[ev.artifact(staging / "requests.jsonl", staging, "requests")], run_dir=root)
    _w(store.root / rec["evidence_id"] / "requests.jsonl", (staging / "requests.jsonl").read_text())
    run = load_run(root, extra_evidence=[rec])
    assert sorted(i.id for i in run.items) == ["scene", "scene.01", "scene.02"]
    assert not any("not sent the same clip request" in w for w in run.warnings), run.warnings
    one = run.cells[("tuned", "r0/scene.01")]
    assert one.video == "outputs/tuned/r0/scene.01.mp4" and one.metrics["clip_s"] == 12.0
    assert run.cells[("tuned", "r0/scene")].video == "outputs/tuned/r0/scene.mp4"  # not scene.01.mp4
    two = run.cells[("tuned", "r0/scene.02")]
    assert two.status == "failed" and "HTTP 503" in two.reason


def test_requests_file_changed_after_capture_is_not_shown(clip_run, store):
    for p in store.root.glob("ev_*/requests.jsonl"):
        p.write_text(p.read_text().replace("40000.0", "1.0"))
    res = build(clip_run)
    assert any("does not match the record's sha256" in w for w in res["warnings"])
    assert "clip 1e-03 s" not in page_of(clip_run)


def test_engines_sent_different_requests_are_warned(clip_run):
    body = json.loads((clip_run / "inputs" / "tuned" / "r0" / "dog-run.json").read_text())
    body["seed"] = 7
    _w(clip_run / "inputs" / "tuned" / "r0" / "dog-run.json", body)
    run = load_run(clip_run)
    assert any("not sent the same clip request" in w and "r0/dog-run" in w for w in run.warnings)


def test_rounds_sort_naturally_and_text_runs_are_unchanged(tmp_path, store):
    root = make_clip_run(store.root / "run-many", store, rounds=11)
    run = load_run(root)
    assert run.buckets()[:3] == ["r0", "r1", "r2"] and run.buckets()[-1] == "r10"
    text = tmp_path / "text"
    _w(text / "inputs" / "short" / "a.json", {"messages": [{"role": "user", "content": "hi"}]})
    _w(text / "outputs" / "e" / "short" / "a.json", {"text": "hello"})
    assert load_run(text).modality == "llm"


def test_intake_or_manifest_can_declare_video(tmp_path):
    root = tmp_path / "v"
    _w(root / "intake.yaml", "modality: video\n")
    _w(root / "outputs" / "e" / "r0" / "a.json", {"text": "no clip came back"})
    assert load_run(root).modality == "video"
