"""CLI output: the terse view of evidence records by default, full JSON with --json or INFERENCEKIT_OUTPUT=json.

tests/conftest.py pins INFERENCEKIT_OUTPUT=json for every other test; these tests set the mode themselves.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


from inferencekit import cli, registry
from inferencekit import evidence as ev
from inferencekit.evidence import set_default_root
from conftest import fake_call

REPO = Path(__file__).resolve().parents[1]
LONG_REASON = "samples equal greedy on 8/8 prompts\n(seed 1 vs seed 2) " + "x" * 300


def gates_record(store, run_dir: Path | None = None) -> dict:
    gates = {
        "token_count": {"gate": "token_count", "verdict": "PASS", "reason": "all counts agree"},
        "hidden_probes": {"gate": "hidden_probes", "verdict": "FAIL", "reason": LONG_REASON},
        "output_length": {"gate": "output_length", "verdict": "UNPROVEN", "reason": "3 tokens not streamed"},
        "spec_exact": {"gate": "spec_exact", "verdict": "NOT RUN", "reason": "no draft model", "applicable": False},
        "rare_tokens": {"gate": "rare_tokens", "verdict": "NOT RUN", "reason": "no reference", "applicable": True},
    }
    # fake_call's operation declares no arguments, so the request (where run_dir is read from) is given here
    request = {"op": "run_gates", "cli": "inferencekit gates run",
               "args": {"run_dir": str(run_dir) if run_dir else None}}
    rec = ev.build(fake_call("run_gates", ("gates", "run")), kind="gates", claim_class="observed",
                   summary="gates FAIL: token_count PASS, hidden_probes FAIL, output_length UNPROVEN, ...",
                   missing=["spec_exact"], limitations=["hidden_probes: 8 prompts", "token_count: re-tokenized"],
                   data={"verdict": "FAIL", "gates": gates, "request": request})
    return store.save(rec)


def run_cli(monkeypatch, capsys, argv, result, mode: str | None = "terse"):
    if mode is None:
        monkeypatch.delenv("INFERENCEKIT_OUTPUT", raising=False)
    else:
        monkeypatch.setenv("INFERENCEKIT_OUTPUT", mode)
    monkeypatch.setattr(registry, "call", lambda name, args: result)
    try:
        code = cli.main(argv)
    finally:
        set_default_root(None)
    assert code == 0
    return capsys.readouterr()


def test_terse_view_of_a_gates_record(monkeypatch, capsys, store, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rec = gates_record(store, run)
    out = run_cli(monkeypatch, capsys, ["--runs-dir", str(store.root), "evidence", "list"], rec).out
    lines = out.rstrip("\n").split("\n")
    assert lines[0] == rec["summary"]
    assert lines[1].startswith("FAIL hidden_probes: samples equal greedy on 8/8 prompts (seed 1 vs seed 2) xxx")
    assert len(lines[1]) == len("FAIL hidden_probes: ") + cli.REASON_CHARS and lines[1].endswith("...")
    assert lines[2] == "UNPROVEN output_length: 3 tokens not streamed"
    assert lines[3] == "NOT RUN rare_tokens: no reference"  # expected but not run; spec_exact was not expected
    assert lines[4] == "2 limitation(s), 1 missing item(s): listed in the evidence file"
    stored = (store.root / rec["evidence_id"] / "evidence.json").resolve()
    assert lines[5] == f"evidence: {stored}" and stored.is_file()
    assert lines[6] == f"run-dir copy: {(run / 'evidence' / (rec['evidence_id'] + '.json')).resolve()}"
    assert len(lines) == 7 and "PASS token_count" not in out and "{" not in out


def test_terse_is_the_default_and_json_is_one_flag_or_variable_away(monkeypatch, capsys, store):
    rec = gates_record(store)
    base = ["--runs-dir", str(store.root)]
    assert run_cli(monkeypatch, capsys, base + ["evidence", "list"], rec, mode=None).out.startswith("gates FAIL")
    for argv, mode in ((base + ["--json", "evidence", "list"], None), (base + ["evidence", "list", "--json"], None),
                       (base + ["evidence", "list"], "json"), (base + ["evidence", "list"], " JSON "),
                       (base + ["evidence", "list", "--json"], "terse")):
        assert json.loads(run_cli(monkeypatch, capsys, argv, rec, mode=mode).out) == rec, (argv, mode)


def test_unknown_mode_warns_and_stays_terse(monkeypatch, capsys, store):
    io = run_cli(monkeypatch, capsys, ["--runs-dir", str(store.root), "evidence", "list"], gates_record(store),
                 mode="yaml")
    assert "INFERENCEKIT_OUTPUT='yaml'" in io.err and io.out.startswith("gates FAIL")


def test_results_that_are_not_records_print_as_before(monkeypatch, capsys, store, tmp_path):
    listing = {"total": 1, "items": [{"evidence_id": "ev_1", "kind": "gates"}]}
    assert json.loads(run_cli(monkeypatch, capsys, ["evidence", "list"], listing).out) == listing
    shown = {"evidence_id": "ev_1", "path": "data.verdict", "value": "FAIL"}  # evidence show --path: no summary
    assert json.loads(run_cli(monkeypatch, capsys, ["evidence", "list"], shown).out) == shown
    md = {"format": "md", "markdown": "# report\nline", "path": str(tmp_path / "r.md")}
    io = run_cli(monkeypatch, capsys, ["evidence", "list"], md)
    assert io.out == "# report\nline\n" and "(written to" in io.err


def test_hard_gates_and_record_verdicts_get_a_line(monkeypatch, capsys, store):
    bench = store.save(ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="decode 99 tok/s",
                                limitations=["HARD GATE: CUDA graphs off (enforce_eager=True)"], data={}))
    out = run_cli(monkeypatch, capsys, ["evidence", "list"], bench).out
    assert "FAIL hard gate: CUDA graphs off (enforce_eager=True)" in out.splitlines()
    corr = store.save(ev.build(fake_call("check_correctness", ("correctness",)), kind="correctness",
                               claim_class="derived", summary="FAIL: 3/8 sequences identical",
                               data={"verdict": "FAIL", "reason": "5 sequences diverge before token 32"}))
    out = run_cli(monkeypatch, capsys, ["evidence", "list"], corr).out
    assert out.splitlines()[1] == "FAIL check_correctness: 5 sequences diverge before token 32"
    passed = store.save(ev.build(fake_call("check_correctness", ("correctness",)), kind="correctness",
                                 claim_class="derived", summary="PASS: 8/8", data={"verdict": "PASS"}))
    out = run_cli(monkeypatch, capsys, ["evidence", "list"], passed).out
    assert out.splitlines()[0] == "PASS: 8/8" and out.splitlines()[1].startswith("evidence: ")


def test_one_line_per_engine_verdict(monkeypatch, capsys, store):
    rec = store.save(ev.build(fake_call("quality_score", ("quality", "score")), kind="quality_trajectory",
                              claim_class="derived", summary="FAIL: cand FAIL (agreement 91%), other PASS",
                              data={"verdict": "FAIL", "verdicts": {"cand": "FAIL", "other": "PASS",
                                                                    "third": "UNPROVEN"}}))
    lines = run_cli(monkeypatch, capsys, ["evidence", "list"], rec).out.splitlines()
    assert lines[1:3] == ["FAIL cand", "UNPROVEN third"] and lines[3].startswith("evidence: ")


def _row(engine, best, ratio, ci, bucket="all", metric="decode_tok_s"):
    return {"engine": engine, "best": best, "bucket": bucket, "metric": metric, "median": 100.0,
            "ratio_to_best": ratio, "ratio_ci95": ci}


def test_bench_rounds_ratios_get_one_line(monkeypatch, capsys, store):
    rows = [_row("a", "a", 1.0, [1.0, 1.0]), _row("b", "a", 0.52, [0.5, 0.54]), _row("c", "a", 0.9, [None, None]),
            _row("d", "a", None, [None, None]), _row("b", "b", 1.0, [1.0, 1.0], bucket="short")]
    rec = store.save(ev.build(fake_call("bench_rounds", ("bench", "rounds")), kind="benchmark",
                              claim_class="observed", summary="bench rounds (single): 4 engines x 8 rounds; best "
                              "decode_tok_s [all]: a 100 [observed]", data={"plan": {"rounds": 8}, "rows": rows}))
    lines = run_cli(monkeypatch, capsys, ["evidence", "list"], rec).out.splitlines()
    assert lines[0] == rec["summary"]
    assert lines[1] == "ratio to best decode_tok_s [all] (a): b 0.52 (95% CI 0.5-0.54), c 0.9"
    assert lines[2].startswith("evidence: ")
    assert cli.ratio_line({"data": {"plan": {}, "rows": [_row("a", "a", 1.0, [1, 1])]}}) is None  # one engine
    assert cli.ratio_line({"data": {"rows": rows}}) is None  # not a bench rounds record


def test_a_record_with_no_file_on_disk_says_so(monkeypatch, capsys, tmp_path):
    rec = ev.finalize(ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="s"))
    out = run_cli(monkeypatch, capsys, ["--runs-dir", str(tmp_path / "empty"), "evidence", "list"], rec).out
    assert out.splitlines()[-1].startswith(f"evidence: {rec['evidence_id']} (no evidence file found")


def test_every_command_accepts_json_after_it():
    parser = cli.build_parser()
    for op in registry.load_all().values():
        required = [a for a in op.args if a.required and not (a.name == "argv" and op.passthrough)]
        argv = list(op.cli)
        for a in required:
            scalar_type = a.items if a.type == "array" else a.type
            value = a.choices[0] if a.choices else "1" if scalar_type in ("integer", "number") else "x"
            argv += [value] if a.positional else [cli._flag(a.name), value]
        ns = parser.parse_args(argv + ["--json"])
        assert ns.json_output is True, op.cli
        assert parser.parse_args(argv).json_output is False, op.cli


def test_real_command_terse_then_json(tmp_path):
    """The installed entry point, no output variable set: terse; the evidence path printed is the record."""
    env = {k: v for k, v in os.environ.items() if k != "INFERENCEKIT_OUTPUT"}
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    runs = tmp_path / "runs"
    p = subprocess.run([sys.executable, "-m", "inferencekit", "--runs-dir", str(runs), "doctor"], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    last = p.stdout.strip().splitlines()[-1]
    assert last.startswith("evidence: ") and "{" not in p.stdout
    rec = json.loads(Path(last.split(" ", 1)[1]).read_text())
    assert rec["kind"] == "environment" and p.stdout.startswith(rec["summary"])
    p = subprocess.run([sys.executable, "-m", "inferencekit", "--runs-dir", str(runs), "doctor", "--json"], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and json.loads(p.stdout)["kind"] == "environment"


def test_conftest_pins_json_for_other_tests():
    assert os.environ["INFERENCEKIT_OUTPUT"] == "json"  # the autouse fixture; this module overrides it per test
