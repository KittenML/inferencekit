"""The whole harness end to end, as an agent would drive it, with no GPU.

Every step runs the real CLI as a subprocess against fake engine processes (``tests/fakes/e2e_engine.py``):
intake -> run manifest -> bench rounds (launches the fakes from the manifest, calls bench serve) -> bench
serve load -> gates per engine -> quality trajectory per engine + quality score against a fake reference ->
review page -> ledger trailers -> compare -> report. Then the artifacts must agree with each other: one prompt-set
sha everywhere, one outputs layout that gates, review and quality all read, evidence ids that ledger,
compare and report accept, and the same chat_template_kwargs on every request every engine received.

Finally the run directory is copied elsewhere (what one rsync of runs/<run-id>/ from a GPU host does) and
review and ledger must work from the copy alone, with an empty evidence store.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "tests" / "fakes" / "e2e_engine.py"
DEV = REPO / "src" / "inferencekit" / "prompts" / "dev.jsonl.gz"  # shipped gzipped; manifests hash the plain bytes
DEV_MANIFEST = REPO / "src" / "inferencekit" / "prompts" / "dev.manifest.json"
# The intake the run starts from.
INTAKE = REPO / "tests" / "fixtures" / "intake" / "llm-qwen-5090.yaml"
REV = "220b46e3b2180893580a4454f21f22d3ebb187d3"
CTK = {"enable_thinking": False}
CTK_JSON = json.dumps(CTK)

pytestmark = pytest.mark.skipif(not DEV.is_file(), reason="needs the shipped dev prompt set")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Env:
    def __init__(self, tmp: Path, store: Path):
        self.tmp, self.store = tmp, store
        self.log: list[str] = []

    def env(self, store: Path | None = None) -> dict:
        e = {k: v for k, v in os.environ.items() if k not in ("INFERKIT_RUNS_DIR", "INFERENCEKIT_RUNS_DIR")}
        e["INFERENCEKIT_RUNS_DIR"] = str(store or self.store)
        e["INFERENCEKIT_KEY_FILE"] = str(self.tmp / "endpoint.key")
        e["INFERENCEKIT_GPU_LOCK"] = str(self.tmp / "gpu.lock")
        e["INFERENCEKIT_OUTPUT"] = "json"  # ik() parses stdout as JSON; the default output is the terse view
        e["PYTHONPATH"] = str(REPO / "src") + os.pathsep + e.get("PYTHONPATH", "")
        return e

    def ik(self, *args: str, store: Path | None = None, timeout: float = 300, expect_json: bool = True):
        argv = [sys.executable, "-m", "inferencekit", *[str(a) for a in args]]
        p = subprocess.run(argv, capture_output=True, text=True, env=self.env(store), timeout=timeout,
                           cwd=str(self.tmp))
        self.log.append(" ".join(argv[2:]))
        assert p.returncode == 0, f"inferencekit {' '.join(map(str, args))[:300]}\n--- stderr\n{p.stderr[-4000:]}"
        return json.loads(p.stdout) if expect_json else p


def _wait_ready(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"fake engine on port {port} never became ready")


def _engine_argv(port: int, log: Path, *extra: str) -> list[str]:
    return [sys.executable, str(ENGINE), "--port", str(port), "--request-log", str(log), *extra]


def run_end_to_end(tmp_path) -> dict:
    """Steps 1-11 and the checks on their artifacts; returns what a later step needs."""
    from inferencekit import __version__

    store = tmp_path / "store"
    run = tmp_path / "runs" / "e2e-001"
    run.parent.mkdir(parents=True)
    x = Env(tmp_path, store)
    reqlog = tmp_path / "requests.jsonl"
    dev_sha = json.loads(DEV_MANIFEST.read_text())["sha256"]
    pa, pb, pref = _free_port(), _free_port(), _free_port()

    # A wheel stand-in (the manifest records its hash) and hashed requirements.
    wdir = tmp_path / "wheel"
    wdir.mkdir()
    wheel = wdir / f"inferencekit_cli-{__version__}-py3-none-any.whl"
    wheel.write_bytes(b"stand-in wheel for the end-to-end test\n")
    (wdir / "requirements-hashed.txt").write_text(
        f"inferencekit-cli=={__version__} \\\n    --hash=sha256:" + "1" * 64 + "\n")

    # 1. Intake: the example (its copy in tests/), filled in, with the thinking-off pin.
    text = INTAKE.read_text()
    for old, new in {"<hf-org>/<qwen-checkpoint-nvfp4>": "Qwen/Qwen3-8B", "<40-hex commit sha>": REV,
                     "<gpu-host>": "203.0.113.10", "<user>": "ubuntu", "~/.ssh/<key>": "~/.ssh/id_test",
                     "<path to the ledger repo on the agent's machine>": str(tmp_path / "ledger"),
                     "<path to inferencekit_cli-0.1.0-py3-none-any.whl>": str(wheel)}.items():
        assert old in text, old
        text = text.replace(old, new)
    text = text.replace("workload:\n", "workload:\n  chat_template_kwargs: {enable_thinking: false}\n", 1)
    intake = tmp_path / "intake.yaml"
    intake.write_text(text)
    res = x.ik("intake", "validate", intake, "--run-dir", run)
    assert (run / "state.json").is_file(), res
    # The pin is binding: recorded for the run, and every later command with --run-dir applies it.
    assert json.loads((run / "state.json").read_text())["workload_pin"] == {"chat_template_kwargs": CTK}

    # 2. Run manifest: two fake engines, an honest one and a slower, slightly perturbed one.
    manifest = {
        "manifest_version": 1,
        "model": {"id": "Qwen/Qwen3-8B", "revision": REV, "precision": "bf16", "tokenizer": "Qwen/Qwen3-8B"},
        "hardware": {"gpu": "RTX 5090", "count": 1},
        "engines": [
            {"name": "vllm", "label": "vllm-a", "version": "0.0.0+fake", "install": ["true"],
             "launch": _engine_argv(pa, reqlog), "env": {}, "port": pa, "ready_check": "/v1/models",
             "resolved_config": {"max_num_seqs": 1}, "role": "tuned"},
            {"name": "sglang", "label": "sglang-b", "version": "0.0.0+fake", "install": ["true"],
             "launch": _engine_argv(pb, reqlog, "--per-token-ms", "4", "--perturb", "0.3"), "env": {},
             "port": pb, "ready_check": "/v1/models", "resolved_config": {"max_running_requests": 1},
             "role": "custom"},
        ],
        "workload": {"modes": ["single"], "buckets": ["very_short", "short"], "max_tokens": 8,
                     "slo": {"ttft_p95_ms": 2000, "tpot_p95_ms": 200, "max_fail": 0},
                     "chat_template_kwargs": CTK},
        "harness": {"inferencekit_version": __version__, "wheel_sha256": _sha(wheel),
                    "dev_prompt_set_sha256": dev_sha},
        "agent_run": {"run_id": "e2e-001", "started": "2026-10-13T19:00:00Z", "ended": "2026-10-13T20:00:00Z"},
    }
    mpath = run / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    res = x.ik("manifest", "validate", mpath)
    assert res["valid"], res
    # A manifest that drops the intake's pin is refused (intake.normalized.json sits next to it).
    nopin = run / "manifest-nopin.json"
    nopin.write_text(json.dumps({**manifest, "workload": {k: v for k, v in manifest["workload"].items()
                                                          if k != "chat_template_kwargs"}}))
    res = x.ik("manifest", "validate", nopin, "--no-strict")
    assert not res["valid"] and any("intake's pin" in e for e in res["errors"]), res
    nopin.unlink()

    # 3. bench rounds: launches each engine from the manifest, 2 interleaved rounds, bench serve per cell.
    # chat_template_kwargs comes from the manifest's workload (no flag needed).
    agg = x.ik("bench", "rounds", "--manifest", mpath, "--prompts", DEV, "--rounds", "2", "--mode", "single",
               "--run-dir", run, "--warmup", "1", "--serve-args", "--limit-per-bucket 2 --buckets very_short "
               "--buckets short --max-tokens 8", timeout=600)
    assert agg["kind"] == "serve_benchmark", agg["kind"]
    last = {}
    for label in ("vllm-a", "sglang-b"):
        cells = sorted((run / "rounds").glob(f"{label}.r*.json"))
        assert len(cells) == 2, cells
        last[label] = json.loads(cells[-1].read_text())["serve"]["evidence_id"]
        assert last[label], cells[-1].read_text()

    # 4. Long-lived engines for load, gates and quality, plus a fake reference endpoint.
    procs = []
    try:
        for argv in (_engine_argv(pa, reqlog), _engine_argv(pb, reqlog, "--per-token-ms", "4", "--perturb", "0.3"),
                     _engine_argv(pref, reqlog)):
            procs.append(subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for port in (pa, pb, pref):
            _wait_ready(port)
        url = {"vllm-a": f"http://127.0.0.1:{pa}", "sglang-b": f"http://127.0.0.1:{pb}",
               "reference": f"http://127.0.0.1:{pref}"}

        # 5. One load-mode bench serve on a small rate grid.
        cap = x.ik("bench", "serve", "--base-url", url["vllm-a"], "--model", "fake-model", "--prompts", DEV,
                   "--mode", "load", "--rates", "2", "--rates", "4", "--duration-s", "1.5", "--buckets", "very_short",
                   "--buckets", "short", "--max-tokens", "8", "--engine-kind", "vllm", "--label", "vllm-a",
                   "--run-dir", run, "--ref-e2e-ms", "100", "--slo-ttft-p95-ms", "2000", "--slo-tpot-p95-ms", "200",
                   "--gpu-price-usd-h", "0.6", "--gpu-name", "RTX 5090", "--gpu-count", "1")
        # (no --chat-template-kwargs: the intake pin in the run dir is applied; the request log checks it)
        assert cap["kind"] == "capacity", cap["kind"]

        # 6. gates per engine (live probes + static checks on the run dir).
        gates = {}
        for label in ("vllm-a", "sglang-b"):
            extra = ["--reference-engine", "vllm-a"] if label != "vllm-a" else []
            gates[label] = x.ik("gates", "run", "--base-url", url[label], "--model", "fake-model", "--engine", label,
                                "--run-dir", run, "--prompts", DEV, "--chat-template-kwargs", CTK_JSON, *extra)
            assert gates[label]["kind"] in ("gates", "correctness"), gates[label]["kind"]

        # 7. quality: the reference's own greedy run (the baseline), each engine, then the score.
        traj = {}
        for label in ("reference", "vllm-a", "sglang-b"):
            flag = ["--chat-template-kwargs", CTK_JSON] if label != "reference" else []  # reference: the pin
            traj[label] = x.ik("quality", "trajectory", "--base-url", url[label], "--model", "fake-model",
                               "--prompts", DEV, "--label", label, "--run-dir", run, "--max-tokens", "8",
                               "--limit", "6", "--engine-kind", "vllm", *flag)
        score = x.ik("quality", "score", "--reference-url", url["reference"], "--reference-model", "fake-model",
                     "--trajectories", run / "quality" / "vllm-a", "--trajectories", run / "quality" / "sglang-b",
                     "--baseline", run / "quality" / "reference", "--out", run / "quality" / "score.json")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()

    # 8. Review page.
    x.ik("review", "build", run, "--reference", "vllm-a", "--prompts", DEV)
    assert (run / "review" / "index.html").is_file()
    review = json.loads((run / "review" / "data.json").read_text())
    # Rows come from inputs/<bucket>/<id> only: bench serve's per-engine copies (inputs/<label>/...) are not buckets.
    assert set(review["buckets"]) <= {"very_short", "short", "medium", "long"}, review["buckets"]

    # 9. Ledger trailers for the tuned engine.
    trailers = x.ik("ledger", "trailers", last["vllm-a"], cap["evidence_id"], gates["vllm-a"]["evidence_id"],
                    score["evidence_id"], "--label", "vllm-a", "--format", "json")

    # 10. compare: the two engines' serve_benchmark records (baseline = the slower one).
    cmp_ = x.ik("compare", last["sglang-b"], last["vllm-a"], "--metric", "decode_tok_s")
    reason = cmp_["data"]["decision"].get("reason") or ""
    # Accepted as comparable serve records: any 'inconclusive' may only come from the CI width (2 rounds).
    for bad in ("not a benchmark", "workloads differ", "different prompt sets", "failed request", "different timers"):
        assert bad not in reason, reason
    assert cmp_["data"]["verdict"] != "inconclusive" or cmp_["data"]["decision"].get("powered") is False, cmp_["data"]
    ratio = next(m for m in cmp_["metrics"] if m["name"] == "decode_tok_s_ratio")["value"]
    assert ratio > 1.5, ratio  # vllm-a (1 ms/token) is much faster than sglang-b (4 ms/token)

    # 11. report over the serve records.
    rep = x.ik("report", last["sglang-b"], last["vllm-a"], "--format", "json")

    # ------------------------------------------------------------------ consistency of the artifacts
    store_recs = {}
    for f in store.glob("ev_*/evidence.json"):
        rec = json.loads(f.read_text())
        store_recs[rec["evidence_id"]] = rec
    serve_recs = [r for r in store_recs.values() if r["kind"] in ("serve_benchmark", "capacity")]
    assert serve_recs
    for r in serve_recs:
        ps = (r["identity"].get("workload") or {}).get("prompt_set") or {}
        assert ps.get("sha256") == dev_sha, (r["evidence_id"], ps)
    for label in ("reference", "vllm-a", "sglang-b"):
        idx = json.loads((run / "quality" / label / "index.json").read_text())
        assert idx.get("prompts_sha256") == dev_sha, (label, {k: idx[k] for k in idx if k != "files"})
    assert manifest["harness"]["dev_prompt_set_sha256"] == dev_sha

    # One outputs layout: bench serve wrote it, gates and review read it.
    for label in ("vllm-a", "sglang-b"):
        outs = sorted((run / "outputs" / label).glob("*/*.json"))
        assert {p.parent.name for p in outs} >= {"very_short", "short"}, outs
    assert sorted((run / "outputs" / "vllm-a").glob("load-*/*/*.json")), "load outputs missing"
    for label in ("vllm-a", "sglang-b"):
        gate_rows = {k: g["verdict"] for k, g in gates[label]["data"]["gates"].items()}
        assert (gates[label]["data"].get("prompt_set") or {}).get("sha256") == dev_sha, gates[label]["data"]["prompt_set"]
        assert gate_rows.get("output_length") not in (None, "NOT RUN"), gate_rows
        assert gate_rows.get("unique_prompts") not in (None, "NOT RUN") or label != "vllm-a", gate_rows

    # Every record made with --run-dir also landed inside the run directory.
    in_run = {p.stem for p in (run / "evidence").glob("ev_*.json")}
    for rid in (last["vllm-a"], last["sglang-b"], cap["evidence_id"], gates["vllm-a"]["evidence_id"],
                gates["sglang-b"]["evidence_id"], score["evidence_id"], agg["evidence_id"]):
        assert rid in in_run, rid

    # The review page saw the gates and quality verdicts (not NOT RUN) for both engines.
    by_engine = {e["engine"]: e for e in review["engines"]}
    assert set(by_engine) >= {"vllm-a", "sglang-b"}, list(by_engine)
    for label, q in (("vllm-a", "PASS"), ("sglang-b", "FAIL")):
        assert by_engine[label]["gates"]["verdict"] in ("PASS", "FAIL"), by_engine[label]["gates"]
        assert [x["verdict"] for x in by_engine[label]["quality"]] == [q], by_engine[label]["quality"]
    assert by_engine["vllm-a"]["capacity"]["n_star"] is not None, by_engine["vllm-a"].get("capacity")

    # Quality: the engine identical to the reference agrees on every scored token; the perturbed one does not.
    sc = json.loads((run / "quality" / "score.json").read_text())
    assert sc["verdicts"]["vllm-a"] == "PASS" and sc["verdicts"]["sglang-b"] == "FAIL", sc["verdicts"]
    assert sc["engines"]["vllm-a"]["all"]["top1_agreement"]["value"] == 1.0, sc["engines"]["vllm-a"]["all"]
    assert sc["engines"]["vllm-a"]["all"]["top1_agreement"]["n_tokens"] == 6 * 8

    # Ledger: speed, capacity and evidence trailers came out; compare and report accepted serve records.
    keys = {t["key"] if isinstance(t, dict) else t for t in trailers.get("trailers", [])}
    assert {"Perf-Workload", "Perf-Decode-Tok-S", "Perf-Evidence"} <= keys, trailers
    assert "Perf-Capacity-N" in keys, trailers
    rep_rows = rep["report"]["results"]
    assert rep_rows and all(str(r["subject"]).startswith("single") for r in rep_rows), rep_rows
    env_rows = rep["report"]["environment"]
    assert env_rows and all("prompt None" not in r["workload"] and r["workload"].startswith("single")
                            for r in env_rows), env_rows

    # Chat-template settings reached every engine identically, from every command that generates.
    rows = [json.loads(line) for line in reqlog.read_text().splitlines()]
    chat = [r for r in rows if r["path"].endswith("/chat/completions")]
    assert {r["port"] for r in chat} == {pa, pb, pref}
    bad = [r for r in chat if r["chat_template_kwargs"] != CTK]
    assert not bad, f"{len(bad)} of {len(chat)} chat requests without the pinned chat_template_kwargs: {bad[:3]}"

    # ------------------------------------------------------------------ the rsync check
    # Copy only runs/<run-id>/ elsewhere, with an EMPTY evidence store: review and ledger must still work.
    local = tmp_path / "laptop" / "runs" / "e2e-001"
    shutil.copytree(run, local)
    empty = tmp_path / "laptop" / "store"
    shutil.rmtree(local / "review")
    x.ik("review", "build", local, "--reference", "vllm-a", store=empty)
    review2 = json.loads((local / "review" / "data.json").read_text())
    assert "no evidence attached" not in json.dumps(review2.get("warnings", []))
    synced = {e["engine"]: e for e in review2["engines"]}
    for label in ("vllm-a", "sglang-b"):
        assert synced[label]["gates"]["verdict"] in ("PASS", "FAIL"), synced[label].get("gates")
        assert synced[label]["quality"] and synced[label]["quality"][0]["verdict"] in ("PASS", "FAIL")
        assert synced[label]["speed"], label
    assert synced["vllm-a"]["capacity"]["n_star"] == by_engine["vllm-a"]["capacity"]["n_star"]
    ev_paths = [local / "evidence" / f"{i}.json" for i in (last["vllm-a"], cap["evidence_id"])]
    x.ik("ledger", "trailers", *ev_paths, "--label", "vllm-a", "--format", "json", store=empty)

    return {"x": x, "manifest": mpath, "wheel": wheel}


def test_end_to_end_with_fake_engines(tmp_path):
    run_end_to_end(tmp_path)
