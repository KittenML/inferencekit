"""ikstep's proof jobs (job baseline | screen | light | full, wait, digest) against a fake inferencekit CLI that
writes records built from real records (tests/fakes_ikstep), fake engine servers (tests/fakes_rounds), and
the real records and manifest of an earlier run (tests/fixtures/ikstep). No GPU, no ssh, no network."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "plugin" / "skills" / "optimizing-inference" / "scripts" / "ikstep.py"
FIX = ROOT / "tests" / "fixtures" / "ikstep"
FAKE_CLI = ROOT / "tests" / "fakes_ikstep" / "fake_inferencekit.py"
FAKE_SERVER = ROOT / "tests" / "fakes_rounds" / "fake_server.py"
pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists() or not shutil.which("setsid"),
                                reason="ikstep reads process groups from /proc; job scripts use setsid")
FAST = ["--poll", "0.1"]
BASE_ENV = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "CUDA_VISIBLE_DEVICES", "CUDA_HOME",
            "CUDA_PATH", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ik(tmp_path, monkeypatch):
    """The script as a module, run in a fresh run directory with a fake CLI on PATH; no GPU is ever queried."""
    spec = importlib.util.spec_from_file_location("ikstep_jobs_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(run)
    monkeypatch.setattr(mod, "gpu_mem", lambda: None)
    monkeypatch.setattr(mod, "UVM", str(tmp_path / "no-such-device"))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "inferencekit").write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I {FAKE_CLI} "$@"\n')
    (bindir / "inferencekit").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    for k, v in {"FAKE_IK_PROMPTS": "/sets/dev.jsonl.gz", "FAKE_IK_LOG": str(tmp_path / "calls.jsonl"), "FAKE_IK": "{}",
                 "INFERENCEKIT_OUTPUT": "terse"}.items():
        monkeypatch.setenv(k, v)
    for k in ("HOST", "PORT", "USER", "KEY", "PREFIX"):
        monkeypatch.delenv(f"IK_{k}", raising=False)
    yield mod
    for f in [*run.glob("logs/*.job.json"), *run.glob("logs/*.server")]:  # never leave a job or a server running
        try:
            pgid = json.loads(f.read_text())["pgid"] if f.suffix == ".json" else int(f.read_text().split()[0])
            os.killpg(pgid, signal.SIGKILL)
        except (OSError, ValueError):
            pass


def step(ik, capsys, *argv) -> tuple[int, list[str]]:
    rc = ik.main(list(map(str, argv)))
    return rc, capsys.readouterr().out.splitlines()


def fake_manifest(tmp_path: Path, cand_flags=(), env_dump=True) -> dict:
    """Two fake engines (baseline 'base', candidate 'fd'), each with its own port, as a run manifest."""
    def engine(label, role, flags, env, ready):
        port = free_port()
        launch = [sys.executable, "-I", str(FAKE_SERVER), "--label", label, "--port", str(port), *flags]
        launch += ["--env-dump", str(tmp_path / f"env-{label}.json")] * env_dump
        return {"name": "vllm" if role != "custom" else "vllm-fastdec (custom)", "label": label, "launch": launch,
                "env": env, "port": port, "ready_check": ready, "role": role}
    m = {"manifest_version": 1, "model": {"id": "Qwen/Qwen3-8B-FP8"},
         "engines": [engine("base", "baseline-recipe", (), {"VLLM_NO_USAGE_STATS": "1"}, "/v1/models"),
                     engine("fd", "custom", cand_flags, {"DO_NOT_TRACK": "1", "IK_TEST_TOKEN": None}, "/health")],
         "workload": {"modes": ["single"], "buckets": ["very_short", "short", "medium", "long"], "max_tokens": 256,
                      "chat_template_kwargs": {"enable_thinking": False},
                      "extra_body": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 0}}}
    Path("manifest.json").write_text(json.dumps(m))
    return m


def calls(tmp_path: Path) -> list[list[str]]:
    return [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]


def run_job(ik, capsys, *argv) -> tuple[str, list[str]]:
    rc, out = step(ik, capsys, "--local", "job", *argv)
    assert rc == 0, out
    assert out[0].startswith("STOPPED ") and out[1].startswith("started ")
    name = out[-1].split()[1].rstrip(":")
    assert out[-1].endswith(f"Next: ikstep wait {name} (Bash timeout 600000)")
    return name, out


def wait_job(ik, capsys, name, *more) -> tuple[int, list[str]]:
    rc, out = step(ik, capsys, "--local", "wait", name, *FAST, *more)
    assert len(out) <= 25, "\n".join(out)
    return rc, out


# --- the job scripts: exactly the planned checks, with flags the installed CLI has ---------------------------------

def scripts(ik, capsys, *more) -> dict[str, str]:
    shutil.copy(FIX / "manifest.json", "manifest.json")  # the fixture run's manifest: baseline-recipe and fastdec
    out = {}
    for kind in ("baseline", "screen", "light", "full"):
        rc, lines = step(ik, capsys, "--local", "job", kind, "--dry-run", *more)
        assert rc == 0 and lines[0] == "#!/bin/bash", lines
        out[kind] = "\n".join(lines)
    assert not Path("jobs").exists() and not Path("logs").exists()  # a dry run writes and starts nothing
    return out


def steps(text: str) -> list[tuple[str, list[str]]]:
    """(step, argv) per `run` line, and ('up', argv) per server start, in order."""
    out = []
    for line in text.splitlines():
        if line.startswith(("run ", "up ", 'echo "== MARK')):
            w = shlex.split(line)
            out.append((w[1], w[2:]) if w[0] == "run" else (w[0] if w[0] == "up" else "MARK", w[1:]))
    return out


def flag(argv: list[str], name: str) -> list[str]:
    return [argv[i + 1] for i, x in enumerate(argv[:-1]) if x == name]


def test_light_and_full_scripts_hold_exactly_the_owner_checks_in_order(ik, capsys):
    sh = scripts(ik, capsys)
    light, full = steps(sh["light"]), steps(sh["full"])
    order = [s if s != "up" else f"up {a[0]}" for s, a in light]
    assert order == ["rounds", "up fastdec", "gates", "MARK", "trajectory", "up baseline-recipe", "score", "up fastdec",
                     "gsm8k"]  # no sampler check unless the sampling code changed
    assert [s if s != "up" else f"up {a[0]}" for s, a in full] == order + ["sampler"]
    L, F = dict(light), dict(full)
    assert L["rounds"][:2] == F["rounds"][:2] == ["inferencekit", "bench"] and L["rounds"][2] == "rounds"
    assert flag(L["rounds"], "--rounds") == ["2"] and flag(F["rounds"], "--rounds") == ["3"]
    assert flag(L["rounds"], "--engines") == flag(F["rounds"], "--engines") == ["baseline-recipe,fastdec"]
    assert shlex.split(flag(L["rounds"], "--serve-args")[0])[-2:] == ["--limit-per-bucket", "8"]  # 4 x 8 = 32
    assert "--limit-per-bucket" not in flag(F["rounds"], "--serve-args")[0]  # all 64 prompts
    assert flag(L["gates"], "--gates") == ["caching", "token_count", "rare_tokens"] and not flag(F["gates"], "--gates")
    assert flag(L["gates"], "--engine") == ["fastdec"] and flag(L["gates"], "--reference-engine") == ["baseline-recipe"]
    assert flag(L["trajectory"], "--limit") == ["8"] and not flag(F["trajectory"], "--limit")
    assert flag(L["score"], "--baseline") == ["quality/reference-8"]  # 8-prompt references: see the baseline job
    assert flag(L["score"], "--baseline-repeat") == ["quality/reference-r2-8"]
    assert flag(F["score"], "--baseline") == ["quality/reference"] and flag(F["score"], "--baseline-repeat") == [
        "quality/reference-r2"]
    for s in (L["score"], F["score"]):
        assert [flag(s, f"--margin-{k}") for k in ("agreement", "kl", "nll")] == [["0.01"]] * 3
        assert flag(s, "--trajectories") == [f"quality/{flag(s, '--out')[0][len('quality/score-'):-5]}"]
    assert flag(L["gsm8k"], "--limit") == ["100"] and flag(L["gsm8k"], "--concurrency") == ["0"]
    assert not flag(L["gsm8k"], "--paired-with")
    assert not flag(F["gsm8k"], "--limit") and flag(F["gsm8k"], "--paired-with") == [
        "quality/gsm8k/baseline-recipe.jsonl"]  # all 1,319, paired with the baseline job's run
    assert json.loads(flag(F["sampler"], "--extra-body")[0]) == {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    assert sh["light"].count("inferencekit ") == sh["full"].count("inferencekit ") - 1 == 5


def test_light_takes_the_sampler_check_only_when_the_sampling_code_changed(ik, capsys):
    assert [s for s, _ in steps(scripts(ik, capsys, "--sampler-changed")["light"])][-1] == "sampler"


def test_baseline_and_screen_scripts(ik, capsys):
    sh = scripts(ik, capsys)
    base = steps(sh["baseline"])
    assert [s if s != "up" else f"up {a[0]}" for s, a in base] == [
        "up baseline-recipe", "aa1", "gates", "reference", "reference-8", "gsm8k", "sampler",
        "up baseline-recipe", "aa2", "reference-r2", "reference-r2-8"]  # the second launch: the A/A run
    B = dict(base)
    for s in ("aa1", "aa2"):  # the same screen as `job screen`, on the baseline
        assert flag(B[s], "--rounds") == ["1"] and flag(B[s], "--limit-per-bucket") == ["4"]
    assert not flag(B["gates"], "--gates") and not flag(B["gsm8k"], "--limit")
    assert [flag(B[s], "--limit") for s in ("reference", "reference-8", "reference-r2", "reference-r2-8")] == [
        [], ["8"], [], ["8"]]
    assert [flag(B[s], "--label") for s in ("aa1", "aa2", "gsm8k", "sampler")] == [
        ["baseline-recipe-aa1"], ["baseline-recipe-aa2"], ["baseline-recipe"], ["baseline-recipe"]]
    screen = steps(sh["screen"])
    assert [s for s, _ in screen] == ["up", "screen"] and screen[0][1][0] == "fastdec"
    S = dict(screen)["screen"]
    assert flag(S, "--limit-per-bucket") == ["4"] and flag(S, "--rounds") == ["1"] and flag(S, "--label") == [
        "screen-fastdec"]
    assert json.loads(flag(S, "--extra-body")[0]) == {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 0}


def test_servers_start_from_the_manifest_launch_line_as_bench_rounds_starts_them(ik, capsys):
    """One proof lost three bench rounds attempts to a launch line nobody had run. A screen starts the
    candidate from the manifest's own argv and env, so the same mistake fails the first screen instead."""
    sh = scripts(ik, capsys)
    m = json.loads((FIX / "manifest.json").read_text())
    for e in m["engines"]:
        ups = [shlex.split(x) for x in sh["full"].splitlines() if x.startswith(f"up {e['label']} ")]
        assert ups and ups[0][2:4] == [f"http://127.0.0.1:{e['port']}", "/v1/models"]
        own = ["FLASHINFER_WORKSPACE_BASE=cache/vllm_fastdec"] * (e["role"] != "baseline-recipe")  # its own venv
        assert ups[0][4:] == own + [f"{k}={v}" for k, v in e["env"].items()] + e["launch"]


def test_a_candidate_in_its_own_venv_gets_its_own_flashinfer_jit_dir(ik, capsys, tmp_path):
    """Both venvs once shared one FlashInfer JIT workspace, whose build files hold each venv's absolute source
    and include paths, so every switch between baseline and candidate rebuilt it: server ready 129-137 s after a
    switch vs 49.6-50.1 s on the same venv, 487 s of a 1,969-s proof. Each launch program (one venv per wrapper)
    other than the baseline's gets cache/<program>; bench rounds reads it from a per-job manifest copy, and the
    run's manifest.json is never changed. Checked on an RTX 5090 (two vLLM 0.31.0 venvs): ready after a
    switch 158.7-177.8 s on one shared dir (build.ninja rewritten with the other venv's paths, then rebuilt),
    55.1-57.1 s with a dir per venv; the light job's rounds step (B C C B) went from 56/171/56/178 s to 56/56/55 s
    after the one cold build."""
    sh = scripts(ik, capsys)
    for kind in ("light", "full"):
        [r] = [argv for s, argv in steps(sh[kind]) if s == "rounds"]
        J = f"{kind}-fastdec"
        assert flag(r, "--manifest") == [f"jobs/{J}.manifest.json"] and flag(r, "--run-dir") == ["."]
    # Keep each server up for all its consecutive checks: every launch after bench rounds is one the check order
    # forces (rounds stops its own servers; the quality score needs the baseline between trajectory and GSM8K).
    for kind, n in (("screen", 1), ("light", 3), ("full", 3)):
        ups = [a[0] for s, a in steps(sh[kind]) if s == "up"]
        assert len(ups) == n and all(x != y for x, y in zip(ups, ups[1:])), (kind, ups)
    before = Path("manifest.json").read_text()
    m = json.loads(before)
    for e in m["engines"]:  # fake servers instead of vLLM: the same labels and launch programs
        e["port"] = free_port()
        e["launch"] = [e["launch"][0], "--port", str(e["port"])]
    Path("manifest.json").write_text(before := json.dumps(m))
    bindir = tmp_path / "bin"
    for prog in ("vllm", "vllm-fastdec"):
        (bindir / prog).write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I {FAKE_SERVER} --label {prog} '
                                   f'"$@" --env-dump {tmp_path}/env-{prog}.json\n')
        (bindir / prog).chmod(0o755)
    name, _ = run_job(ik, capsys, "screen")
    assert wait_job(ik, capsys, name)[0] in (0, 1)
    assert json.loads((tmp_path / "env-vllm-fastdec.json").read_text()).count("FLASHINFER_WORKSPACE_BASE") == 1
    copy = json.loads(Path(f"jobs/{name}.manifest.json").read_text())
    assert [e.get("env", {}).get("FLASHINFER_WORKSPACE_BASE") for e in copy["engines"]] == [None, "cache/vllm_fastdec"]
    assert Path("manifest.json").read_text() == before  # what a re-run replays


@pytest.mark.parametrize("change", ["same program", "own dir set"])
def test_the_baseline_venv_or_the_manifest_own_jit_dir_wins(ik, capsys, change):
    shutil.copy(FIX / "manifest.json", "manifest.json")
    m = json.loads(Path("manifest.json").read_text())
    cand = m["engines"][1]
    if change == "same program":
        cand["launch"][0] = m["engines"][0]["launch"][0]
    else:
        cand["env"]["FLASHINFER_WORKSPACE_BASE"] = "cache/mine"
    Path("manifest.json").write_text(json.dumps(m))
    rc, lines = step(ik, capsys, "--local", "job", "full", "--dry-run")
    assert rc == 0
    [r] = [argv for s, argv in steps("\n".join(lines)) if s == "rounds"]
    assert flag(r, "--manifest") == ["manifest.json" if change == "same program" else "jobs/full-fastdec.manifest.json"]
    ups = [a for s, a in steps("\n".join(lines)) if s == "up" and a[0] == "fastdec"]
    assert [x for x in ups[0] if x.startswith("FLASHINFER_WORKSPACE_BASE=")] == (
        [] if change == "same program" else ["FLASHINFER_WORKSPACE_BASE=cache/mine"])


def test_every_flag_is_in_the_frozen_cli(ik, capsys):
    """Each command line uses only flags the installed inferencekit prints in --help."""
    sh = scripts(ik, capsys, "--sampler-changed")
    seen, helps = set(), {}
    for text in sh.values():
        for _, argv in steps(text):
            if argv[:1] != ["inferencekit"]:
                continue
            cmd = tuple(argv[1:3])
            if cmd not in helps:
                helps[cmd] = subprocess.run([sys.executable, "-m", "inferencekit", *cmd, "--help"], capture_output=True,
                                            text=True, env={**os.environ, "COLUMNS": "200"}).stdout
            words = [w for w in argv[3:] if w.startswith("--") and " " not in w]
            words += [w for w in shlex.split(flag(argv, "--serve-args")[0]) if w.startswith("--")] if \
                "--serve-args" in argv else []
            for w in words:
                serve = "--serve-args" in argv and w not in helps[cmd]
                text_ = helps.setdefault(("bench", "serve"), subprocess.run(
                    [sys.executable, "-m", "inferencekit", "bench", "serve", "--help"], capture_output=True,
                    text=True).stdout) if serve else helps[cmd]
                assert f"{w} " in text_ or f"{w}\n" in text_ or f"[{w}" in text_, (cmd, w)
                seen.add((cmd, w))
    assert {c for c, _ in seen} == {("bench", "rounds"), ("bench", "serve"), ("gates", "run"), ("quality", "trajectory"),
                                    ("quality", "score"), ("quality", "gsm8k"), ("quality", "sampling")}


# --- the jobs end to end, on fake servers --------------------------------------------------------------------------

def test_baseline_then_screen_then_light_then_full(ik, capsys, tmp_path, monkeypatch):
    fake_manifest(tmp_path, cand_flags=("--startup-delay", "0.5"))
    monkeypatch.setenv("IK_TEST_TOKEN", "inherited-not-written")
    monkeypatch.setenv("FAKE_IK", json.dumps({"tok": {"base-aa1": 154.6, "base-aa2": 155.4, "screen-fd": 187.0,
                                                      "screen-fd-2": 190.0, "screen-fd-3": 240.0},
                                              "rounds": {"base": 155.2, "cand": 186.3}, "engine": "fastdec-flashattn"}))
    name, _ = run_job(ik, capsys, "baseline")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 0 and out[0].startswith("BASELINE baseline-base: finished")
    assert "screen A/A: 154.6 then 155.4 tok/s after two launches: screen noise 0.5%" in out
    assert "references: reference 64/64 prompts, reference-8 8/8 prompts, reference-r2 64/64 prompts, " \
           "reference-r2-8 8/8 prompts" in out
    assert json.loads((tmp_path / "env-base.json").read_text()) == sorted(  # a clean environment, as bench rounds
        {k for k in BASE_ENV if k in os.environ} | {"VLLM_NO_USAGE_STATS"})
    name, _ = run_job(ik, capsys, "screen")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 0 and out[0].startswith("SCREEN screen-fd: finished")
    assert ("screen fd: 187.0 tok/s = 1.206x the baseline screen 155.0; gain above the noise: yes; first >= 1.2x: "
            "light check next") in out
    assert json.loads(out[-1])["ratio"] == 1.206 and json.loads(out[-1])["checks"] == "screen job screen-fd: screen only"
    env = json.loads((tmp_path / "env-fd.json").read_text())
    assert {"DO_NOT_TRACK", "IK_TEST_TOKEN"} <= set(env) and "FAKE_IK" not in env  # null in the manifest: inherited
    assert "inherited-not-written" not in Path("jobs/screen-fd.sh").read_text()
    calls_before = len(calls(tmp_path))
    name, _ = run_job(ik, capsys, "light", "--candidate", "fd")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 0 and out[0].startswith("LIGHT light-fd: finished")
    assert out[1] == "speed PASS: fd 186.3 vs base 155.2 tok/s = 1.200x [1.198-1.203], 2 rounds, order balanced True"
    assert out[3] == "gates PASS: caching, token_count, rare_tokens"
    assert out[4].startswith("quality PASS: agreement 97.57% (floor 97.48%), KL 0.0065 (floor 0.0064) nats")
    assert out[6].startswith("gsm8k PASS: 93/100 correct, baseline 93/100 on the same problems, one-sided McNemar p 1")
    assert out[7] == "sampler NOT RUN: sampling code unchanged (--sampler-changed adds it)"
    assert [c[:2] for c in calls(tmp_path)[calls_before:] if c[:2] != ["prompts", "verify"]] == [
        ["bench", "rounds"], ["gates", "run"], ["quality", "trajectory"], ["quality", "score"], ["quality", "gsm8k"]]
    progress = json.loads(out[-1])
    assert list(progress) == ["time", "commit", "engine", "tok_s", "baseline_tok_s", "ratio", "ratio_ci95", "sampling",
                              "checks", "evidence"]
    assert progress["ratio"] == 1.2 and progress["ratio_ci95"] == [1.198, 1.203] and progress["sampling"] == "default"
    assert progress["checks"] == "light job light-fd: speed PASS, gates PASS, quality PASS, gsm8k PASS"
    assert out[-2].startswith("evidence: rounds ev_") and len(progress["evidence"]) == 5
    assert json.loads(Path("logs/light-fd.job.json").read_text())["screen_ratio"] == 1.206  # the screen it checks
    # light checks at the first 1.2x and the first 1.5x, no others
    for want in ("1.226x the baseline screen 155.0; gain above the noise: yes; no light check: milestone checked",
                 "1.548x the baseline screen 155.0; gain above the noise: yes; first >= 1.5x: light check next"):
        name, _ = run_job(ik, capsys, "screen")
        assert any(x.endswith(want) for x in wait_job(ik, capsys, name)[1]), want
    name, _ = run_job(ik, capsys, "full")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 0 and out[0].startswith("FULL full-fd: finished")
    assert ", 3 rounds, order balanced False" in out[1]  # the hard cap of 3 rounds is not a whole Williams period
    assert out[3] == "gates PASS: caching, token_count, output_length, hidden_probes, rare_tokens"
    assert any(x.startswith("gsm8k PASS: 1231/1319 correct, baseline 1231/1319") for x in out)
    assert any(x.startswith("sampler PASS: sampled first tokens") for x in out)
    assert not list(Path("logs").glob("*.server")) or all(  # every server the jobs started is gone
        not Path(f"/proc/{p.read_text().split()[0]}").exists() for p in Path("logs").glob("*.server"))


def test_light_and_full_need_the_baseline_job_and_one_job_runs_at_a_time(ik, capsys, tmp_path, monkeypatch):
    fake_manifest(tmp_path)
    for kind, miss in (("light", "quality/reference-8, quality/reference-r2-8"),
                       ("full", "quality/reference, quality/reference-r2, quality/gsm8k/base.jsonl")):
        assert step(ik, capsys, "--local", "job", kind) == (1, [f"job: {miss} missing: run `ikstep job baseline` first"])
    monkeypatch.setenv("FAKE_IK", json.dumps({"sleep": {"bench serve": 30}}))
    name, _ = run_job(ik, capsys, "screen")
    assert step(ik, capsys, "--local", "job", "screen") == (
        1, [f"job: {name} is still running: ikstep wait {name} (or ikstep stop {name})"])
    rc, out = step(ik, capsys, "--local", "wait", name, "--max-min", "0.005", *FAST)
    assert rc == 3 and len(out) == 1 and out[0].startswith(f"RUNNING {name}: 0.0 min; done: nothing yet; now: ")
    assert out[0].endswith(f". Call again: ikstep wait {name}")
    assert step(ik, capsys, "--local", "stop", "--all")[0] == 0
    assert step(ik, capsys, "--local", "job", "screen", "--candidate", "nope")[0] == 2


def test_a_server_that_dies_at_startup_ends_the_job_with_its_log(ik, capsys, tmp_path):
    fake_manifest(tmp_path, cand_flags=("--exit-early",))
    name, _ = run_job(ik, capsys, "screen")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 1 and out[0].startswith(f"SCREEN {name}: ended early")
    assert out[1] == f"SERVER DIED: fd never answered; log logs/{name}.fd.log"
    assert out[2:4] == ["  fake server fd starting on port " + str(json.loads(Path("manifest.json").read_text())[
        "engines"][1]["port"]), "  ERROR fake failure during startup"]
    assert out[4] == "NOT RUN: screen (the job ended first)"
    assert json.loads(out[-1])["evidence"] == [] and "ratio" not in json.loads(out[-1])  # nothing measured


def test_a_server_that_dies_during_a_step_stops_the_job_at_once(ik, capsys, tmp_path, monkeypatch):
    fake_manifest(tmp_path)
    baseline_done()
    monkeypatch.setenv("FAKE_IK", json.dumps({"sleep": {"quality trajectory": 60}}))
    name, _ = run_job(ik, capsys, "light")
    end = time.time() + 30
    while "== trajectory" not in Path(f"logs/{name}.log").read_text() and time.time() < end:
        time.sleep(0.1)
    os.kill(int(Path(f"logs/{name}.server").read_text().split()[0]), signal.SIGKILL)
    t0 = time.time()
    rc, out = wait_job(ik, capsys, name)
    assert rc == 1 and time.time() - t0 < 15  # not the step's 60 s
    assert out[0] == f"STOPPED {name}; GPU MiB used: no nvidia-smi" and out[1].startswith(f"LIGHT {name}: ended early")
    assert out[2] == f"SERVER DIED: fd while trajectory ran; job stopped; log logs/{name}.fd.log"
    assert "NOT RUN: trajectory, score, gsm8k (the job ended first)" in out
    assert any(x.startswith("speed PASS:") for x in out) and any(x.startswith("gates PASS:") for x in out)


def test_a_server_that_dies_while_a_short_step_ends_by_itself_fails_the_job(ik, capsys, tmp_path, monkeypatch):
    """GPU smoke test, 16 Oct: the server was killed 8 s into the screen, bench serve exited 0 a second later, the
    job ended, and wait printed a clean 1.002x screen. The job script now checks its server after every step."""
    fake_manifest(tmp_path)
    monkeypatch.setenv("FAKE_IK", json.dumps({"sleep": {"bench serve": 2}, "tok": {"screen-fd": 150.0}}))
    name, _ = run_job(ik, capsys, "screen")
    end = time.time() + 30
    while "== screen" not in Path(f"logs/{name}.log").read_text() and time.time() < end:
        time.sleep(0.05)
    os.kill(int(Path(f"logs/{name}.server").read_text().split()[0]), signal.SIGKILL)
    while not Path(f"logs/{name}.rc").exists() and time.time() < end:  # the job ends itself; wait comes after
        time.sleep(0.05)
    log = Path(f"logs/{name}.log").read_text()
    assert "== screen exit 0" in log and "== died fd during screen" in log and "== END" not in log
    rc, out = wait_job(ik, capsys, name)
    assert rc == 1 and out[0].startswith(f"SCREEN {name}: ended early")
    assert out[1] == f"SERVER DIED: fd during screen; log logs/{name}.fd.log"
    assert not any(x.startswith("FAILED") for x in out)  # the fake's screen record is whole (on a GPU: GPU smoke test)
    assert json.loads(out[-1])["checks"] == f"screen job {name}: screen only"


def test_a_screen_with_failed_requests_fails_and_gives_no_ratio(ik, capsys, tmp_path, monkeypatch):
    """GPU smoke test, 16 Oct: 13 of 16 screen requests failed and bench serve still exited 0; the 3-request median
    was printed as the screen result. Now: FAIL with the count, and no tok/s, ratio or PROGRESS speed."""
    fake_manifest(tmp_path)
    monkeypatch.setenv("FAKE_IK", json.dumps({"tok": {"base-aa1": 154.6, "base-aa2": 155.4, "screen-fd": 155.1},
                                              "failed": {"screen-fd": 13}}))
    for kind in ("baseline", "screen"):
        name, _ = run_job(ik, capsys, kind)
        rc, out = wait_job(ik, capsys, name)
    assert rc == 1 and out[0].startswith("SCREEN screen-fd: finished")
    assert out[1] == "FAILED screen: 13 requests failed; log logs/screen-fd.screen.log"
    assert any("; 13 failed" in x for x in out)  # the step's own summary, from its log tail
    assert not any(x.startswith("screen fd:") for x in out) and "tok_s" not in json.loads(out[-1])
    assert ik.tok("screen-fd") is None and ik.tok("base-aa1") == 154.6 and ik.screen_x("base") is None


def test_an_aa_screen_with_a_failed_request_fails_the_baseline_job(ik, capsys, tmp_path, monkeypatch):
    fake_manifest(tmp_path)
    monkeypatch.setenv("FAKE_IK", json.dumps({"tok": {"base-aa1": 154.6, "base-aa2": 155.4},
                                              "failed": {"base-aa2": 1}}))
    name, _ = run_job(ik, capsys, "baseline")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 1 and f"FAILED aa2: 1 requests failed; log logs/{name}.aa2.log" in out
    assert not any(x.startswith("screen A/A") for x in out)  # no noise from a partial screen


def baseline_done(n=100, wrong=()):
    """What the baseline job leaves for the light check: 8-prompt references and its GSM8K results."""
    for d in ("reference-8", "reference-r2-8", "gsm8k"):
        Path("quality", d).mkdir(parents=True)
    Path("quality/gsm8k/base.jsonl").write_text("".join(json.dumps({"idx": i, "correct": i not in wrong}) + "\n"
                                                        for i in range(n)))


def test_a_failed_step_shows_its_stderr_and_log_and_the_job_goes_on(ik, capsys, tmp_path, monkeypatch):
    fake_manifest(tmp_path)
    baseline_done(wrong=range(0, 100, 15))
    monkeypatch.setenv("FAKE_IK", json.dumps({"fail": {"gates run": 1}}))
    name, _ = run_job(ik, capsys, "light")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 1 and out[0].startswith(f"LIGHT {name}: finished")
    assert out[1:3] == [f"FAILED gates: exit 1; log logs/{name}.gates.log",
                        "  inferencekit: error: gates run failed on purpose"]
    assert json.loads(out[-1])["checks"] == f"light job {name}: speed PASS, quality PASS, gsm8k PASS"


def test_when_every_step_fails_the_verdict_still_fits_in_25_lines(ik, capsys, tmp_path, monkeypatch):
    fake_manifest(tmp_path)
    baseline_done()
    monkeypatch.setenv("FAKE_IK", json.dumps({"fail": {c: 1 for c in (
        "bench rounds", "gates run", "quality trajectory", "quality score", "quality gsm8k", "quality sampling")}}))
    name, _ = run_job(ik, capsys, "light", "--sampler-changed")  # all six kinds of step
    rc, out = wait_job(ik, capsys, name)
    assert rc == 1 and [x.split(":")[0] for x in out[1:7]] == [
        "FAILED rounds", "FAILED gates", "FAILED trajectory", "FAILED score", "FAILED gsm8k", "FAILED sampler"]
    assert out[2] == f"FAILED gates: exit 1; log logs/{name}.gates.log"  # each step names its own log
    assert out[7] == "  inferencekit: error: bench rounds failed on purpose"  # the first failing step's own stderr
    assert out[-2] == "evidence: none" and json.loads(out[-1])["checks"] == f"light job {name}: sampler NOT RUN"


def test_checks_that_fail_or_are_unproven_set_the_exit_code(ik, capsys, tmp_path, monkeypatch):
    fake_manifest(tmp_path)
    baseline_done()
    # the TRITON_ATTN build of 15 Oct: its quality score passed overall, but long prompts sat above the floor
    monkeypatch.setenv("FAKE_IK", json.dumps({"engine": "fastdec", "verdict": {"quality score": "UNPROVEN"},
                                              "rounds": {"base": 155.0, "cand": 155.1},
                                              "wrong_every": {"light-fd": 1000}}))  # 99/100: p 0.5
    name, _ = run_job(ik, capsys, "light")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 5 and out[1].startswith("speed UNPROVEN: fd 155.1 vs base 155.0 tok/s = 1.001x [")
    kl = next(x for x in out if x.startswith("  KL by bucket vs the A/A floor: "))
    assert kl.endswith("; long 0.0140 [0.0120-0.0164] vs 0.0071 ABOVE") and kl.count("ABOVE") == 1
    assert any(x.startswith("quality UNPROVEN: ") for x in out)
    monkeypatch.setenv("FAKE_IK", json.dumps({"gate_fail": ["caching"], "wrong_every": {}}))
    name, _ = run_job(ik, capsys, "light")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 1
    assert "gates FAIL: caching FAIL (repeat speedup 1.9x (limit 1.10x))" in out
    g = next(x for x in out if x.startswith("gsm8k "))  # 93/100 against a baseline that got all 100 right
    assert g.startswith("gsm8k FAIL: 93/100 correct, baseline 100/100 on the same problems, one-sided McNemar p 0.00781")


def test_a_rounds_record_with_no_ok_cell_fails_the_speed_check(ik, capsys, tmp_path, monkeypatch):
    """The frozen CLI exits 0 and saves a record when no bench rounds cell is ok (15 Oct: ev_272a31a27, 0/4 cells
    ok). Before, the light check then printed no speed line at all and exited 0 on gates, quality and GSM8K."""
    fake_manifest(tmp_path)
    baseline_done(wrong=range(0, 100, 15))  # the fake's GSM8K misses: no difference
    monkeypatch.setenv("FAKE_IK", json.dumps({"replay": {"bench rounds": "rounds-failed"}}))
    name, _ = run_job(ik, capsys, "light")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 1 and out[0].startswith(f"LIGHT {name}: finished")
    assert any(x.startswith("  bench rounds (single): 2 engines x 2 rounds, 0/4 cells ok") for x in out)  # its log
    assert "speed FAIL: no fd vs base decode ratio; baseline-recipe r1 exited_early; fastdec r1 exited_early; " \
           "fastdec r2 exited_early; baseline-recipe r2 exited_early" in out
    progress = json.loads(out[-1])
    assert progress["checks"] == f"light job {name}: speed FAIL, gates PASS, quality PASS, gsm8k PASS"
    assert "ratio" not in progress and "tok_s" not in progress


def test_a_step_that_exits_0_without_its_record_is_unproven(ik, capsys, tmp_path, monkeypatch):
    fake_manifest(tmp_path)
    baseline_done()
    monkeypatch.setenv("FAKE_IK", json.dumps({"norecord": ["gates run", "quality gsm8k", "bench rounds"]}))
    name, _ = run_job(ik, capsys, "light")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 5
    for s, check in (("rounds", "speed"), ("gates", "gates"), ("gsm8k", "gsm8k")):
        assert f"{check} UNPROVEN: exit 0, no record or file; log logs/{name}.{s}.log" in out, out
    assert json.loads(out[-1])["checks"] == (f"light job {name}: speed UNPROVEN, gates UNPROVEN, gsm8k UNPROVEN, "
                                             "quality PASS")


def test_quality_above_the_floor_is_unproven_even_when_the_score_passes(ik, capsys, tmp_path, monkeypatch):
    """15 Oct: the TRITON_ATTN build passed quality score at margin 0.01, with long prompts and the mean KL above the
    A/A floor; the task asks for agreement and KL within the baseline's own noise floor."""
    fake_manifest(tmp_path)
    baseline_done(wrong=range(0, 100, 15))  # the fake's GSM8K misses: no difference
    monkeypatch.setenv("FAKE_IK", json.dumps({"engine": "fastdec"}))
    name, _ = run_job(ik, capsys, "light")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 5
    q = next(x for x in out if x.startswith("quality "))
    assert q == ("quality UNPROVEN: agreement 97.28% (floor 97.48%), KL 0.0087 (floor 0.0064) nats, 64 prompts; KL "
                 "above the A/A floor (all, long): the next-token distribution moved")
    assert json.loads(out[-1])["checks"].endswith("quality UNPROVEN, gsm8k PASS")


def test_sampler_changed_without_a_sampling_temperature_is_not_run(ik, capsys, tmp_path):
    m = fake_manifest(tmp_path)
    m["workload"]["extra_body"] = {"seed": 0}  # greedy timing: nothing for the sampler check to test
    Path("manifest.json").write_text(json.dumps(m))
    baseline_done(wrong=range(0, 100, 15))
    name, _ = run_job(ik, capsys, "light", "--sampler-changed")
    rc, out = wait_job(ik, capsys, name)
    assert rc == 5 and "sampler NOT RUN: no temperature in workload.extra_body" in out


@pytest.mark.parametrize("prompts,lines", [
    ({"verdict": "FAIL"}, ["job: `inferencekit --json prompts verify --set dev` exit 0, verdict FAIL: no verified "
                           "prompt set", "  sha256 FAIL: file vs manifest"]),
    ({"exit": 1, "stderr": "inferencekit: error: prompt set dev: sha256 mismatch\n"},
     ["job: `inferencekit --json prompts verify --set dev` exit 1, verdict None: no verified prompt set",
      "  inferencekit: error: prompt set dev: sha256 mismatch"]),
])
def test_jobs_run_only_on_a_verified_prompt_set(ik, capsys, tmp_path, monkeypatch, prompts, lines):
    fake_manifest(tmp_path)
    monkeypatch.setenv("FAKE_IK", json.dumps({"prompts": prompts}))
    assert step(ik, capsys, "--local", "job", "screen", "--dry-run") == (1, lines)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert step(ik, capsys, "--local", "job", "screen", "--dry-run") == (1, [
        "job: `inferencekit --json prompts verify --set dev`: no inferencekit on PATH (after the prefix)"])


@pytest.mark.parametrize("name,launch", [
    ("vllm", ["vllm", "serve", "m"]), ("vllm-fastdec (custom)", ["python", "-m", "fastdec"]),
    ("sglang", ["python", "-m", "sglang.launch_server"]), ("llama.cpp", ["/opt/llama/main"]),
    ("llama-server", ["llama-server", "-m", "x.gguf"]), ("TensorRT-LLM", ["trt-llm-serve"]),
    ("tensorrt", ["python", "-m", "tensorrt_llm.serve"]), ("trtllm", ["trtllm-serve", "m"]),
    ("llama", ["python", "serve.py"]), ("custom", ["./engine", "--port", "8000"]), (None, None),
])
def test_engine_kind_is_the_wheels_rule(ik, name, launch):
    """The screen and the A/A screens pass the --engine-kind bench rounds would pick for the same engine."""
    from inferencekit.serve.servers import engine_kind
    assert ik.ekind({"name": name, "launch": launch}) == engine_kind(name, launch)


def test_mcnemar_is_one_sided_on_the_same_problems(ik, tmp_path):
    """The 15 Oct pair (fastdec vs the baseline, 1,319 problems): 16 only fastdec solved, 17 only the baseline."""
    Path("quality/gsm8k").mkdir(parents=True)
    rows = {"base": [True] * 1222 + [False] * 16 + [True] * 17 + [False] * 64,
            "cand": [True] * 1222 + [True] * 16 + [False] * 17 + [False] * 64}
    for label, ok in rows.items():
        Path(f"quality/gsm8k/{label}.jsonl").write_text("".join(json.dumps({"idx": i, "correct": c}) + "\n"
                                                                for i, c in enumerate(ok)))
    v, text = ik.gsm("cand", "base", {"data": {"correct": 1238, "n": 1319, "errors": 0}})
    assert v == "PASS" and text == ("1238/1319 correct, baseline 1239/1319 on the same problems, one-sided McNemar "
                                    "p 0.5")
    assert ik.gsm("cand", "none", {"data": {"correct": 1, "n": 2}})[0] == "UNPROVEN"


def test_speed_reads_the_ratio_and_its_ci_from_a_real_bench_rounds_record(ik):
    r = json.loads((FIX / "rounds.json").read_text())
    rows = {(x["engine"], x["bucket"]): x for x in r["data"]["rows"] if x["metric"] == "decode_tok_s"}
    q = ik.speed(r, "baseline-recipe", "fastdec")
    b = rows[("baseline-recipe", "all")]  # fastdec was best: the baseline's ratio is baseline / fastdec
    assert q["all"] == (rows[("fastdec", "all")]["median"], b["median"], 1 / b["ratio_to_best"],
                        1 / b["ratio_ci95"][1], 1 / b["ratio_ci95"][0])
    assert set(q) == {"all", "very_short", "short", "medium", "long"}
    back = ik.speed(r, "fastdec", "baseline-recipe")["all"]  # the other way round: below 1
    assert back[2] == b["ratio_to_best"] and back[3:] == tuple(b["ratio_ci95"])


# --- digest -------------------------------------------------------------------------------------------------------

def test_digest_reads_the_run_evidence_without_being_told_where(ik, capsys, tmp_path, monkeypatch):
    ev = tmp_path / "run" / "runs" / "r1" / "evidence"
    ev.mkdir(parents=True)
    ids = {}
    for f in sorted(FIX.glob("*.json")):
        if f.stem in ("manifest", "score-file", "metrics"):
            continue
        r = json.loads(f.read_text())
        ids[f.stem] = r["evidence_id"]
        if f.stem == "gates":  # the evidence store's own layout
            (ev / r["evidence_id"]).mkdir()
            (ev / r["evidence_id"] / "evidence.json").write_text(f.read_text())
        else:
            (ev / f"{r['evidence_id']}.json").write_text(f.read_text())
    monkeypatch.delenv("INFERENCEKIT_RUNS_DIR", raising=False)
    rc, out = step(ik, capsys, "digest")  # no --root, no host: ./runs/*/evidence on this machine
    assert rc == 0 and len(out) == len(ids)
    by = {x[:15]: x for x in out}
    assert all(len(x) < 700 for x in out) and set(by) == {i[:15] for i in ids.values()}
    g = by[ids["gates"][:15]]
    assert g.startswith(f"{ids['gates'][:15]} 2026-10-15T20:48 gates fastdec-fa PASS: gates PASS: caching PASS")
    r = by[ids["rounds"][:15]]
    q = ik.speed(json.loads((FIX / "rounds.json").read_text()), "baseline-recipe", "fastdec")["all"]
    assert f"; fastdec/baseline-recipe {q[2]:.3f}x [{q[3]:.3f}-{q[4]:.3f}], {q[0]:.1f} vs {q[1]:.1f} tok/s" in r
    assert "quality_task baseline-recipe FAIL: baseline-recipe: FAIL: sampled first tokens" in by[ids["sampling-baseline"][:15]]
    assert " serve_benchmark " in by[ids["rounds-failed"][:15]] and "0/4 cells ok" in by[ids["rounds-failed"][:15]]
    rc, out = step(ik, capsys, "digest", ids["score"][:10], ids["gsm8k"][3:12], "--root", str(ev))
    assert rc == 0 and [x[:15] for x in out] == [ids["score"][:15], ids["gsm8k"][:15]]
    assert "GSM8K 1238/1319" in out[1] and "McNemar p=1" in out[1]
    rc, out = step(ik, capsys, "digest", "ev_", "ev_ffff00")
    assert rc == 1 and out == [f"ev_: {len(ids)} records match in {ev.parent.relative_to(Path.cwd())}/evidence",
                               f"ev_ffff00: 0 records match in runs/r1/evidence"]
    rc, out = step(ik, capsys, "digest", "--last", "2", "--root", str(ev))
    assert rc == 0 and len(out) == 2


def test_digest_last_n_are_the_records_that_ended_last(ik, capsys, tmp_path):
    """GPU smoke test, 16 Oct: bench rounds' own record starts before the per-cell records it writes, so sorted by
    start time `digest --last 10` after a full proof left out its speed result. Sorted by end time it is there."""
    ev = tmp_path / "ev"
    ev.mkdir()
    rounds = json.loads((FIX / "rounds.json").read_text())
    for i, (kind, start, end) in enumerate((("rounds", "03:01:16", "03:17:39"), ("serve", "03:02:05", "03:03:59"),
                                           ("serve", "03:15:42", "03:17:36"), ("gates", "03:18:27", "03:18:50"))):
        r = rounds if kind == "rounds" else json.loads((FIX / f"{kind}.json").read_text())
        r = {**r, "evidence_id": f"ev_{i}{'0' * 63}", "provenance": {**r["provenance"], "started": f"2026-10-16T{start}Z",
                                                                  "ended": f"2026-10-16T{end}Z"}}
        (ev / f"{r['evidence_id']}.json").write_text(json.dumps(r))
    rc, out = step(ik, capsys, "digest", "--last", "2", "--root", str(ev))
    assert rc == 0 and [x.split()[0] for x in out] == ["ev_000000000000", "ev_300000000000"]
    assert "bench rounds" in out[0] and "2026-10-16T03:01" in out[0]  # shown with its start time


def test_digest_without_an_evidence_directory(ik, capsys):
    assert step(ik, capsys, "digest") == (2, ["digest: no evidence directory here; pass --root <run directory>/evidence"])


# --- the remote path and the PATH wrapper ------------------------------------------------------------------------

def test_a_remote_job_carries_this_checkouts_commit(ik, capsys, monkeypatch):
    seen = []

    def fake_run(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "abc1234\n" if argv[0] == "git" else "", "")
    monkeypatch.setattr(ik.subprocess, "run", fake_run)
    monkeypatch.setenv("IK_HOST", "gpu1")
    monkeypatch.setenv("IK_PREFIX", "cd ~/ik/runs/r1;")
    assert ik.main(["job", "light", "--candidate", "fd"]) == 0
    assert seen[0][:3] == ["git", "rev-parse", "--short"]
    assert seen[1][-1] == "cd ~/ik/runs/r1; python3 - --local job light --candidate fd --commit=abc1234"


def test_the_plugin_bin_wrapper_runs_the_script_from_anywhere(tmp_path):
    wrapper = ROOT / "plugin" / "bin" / "ikstep"
    assert wrapper.stat().st_mode & stat.S_IXUSR
    link = tmp_path / "ikstep"
    link.symlink_to(wrapper)  # Claude Code may reach it through a link
    for exe in (wrapper, link):
        r = subprocess.run([str(exe), "--help"], capture_output=True, text=True, cwd=tmp_path)
        assert r.returncode == 0 and "job baseline" in r.stdout and "digest" in r.stdout, r.stderr
    r = subprocess.run([str(wrapper), "digest"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 2 and r.stdout.startswith("digest: no evidence directory here")
