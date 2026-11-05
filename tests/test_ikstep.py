"""The optimizing-inference skill's host-side step tool (scripts/ikstep.py): argument parsing and the ssh line it
builds, local start/wait/stop on dummy processes, and the verdict lines. No GPU, no ssh, no network."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "plugin" / "skills" / "optimizing-inference" / "scripts" / "ikstep.py"
pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="ikstep reads process groups from /proc")
FAST = ["--poll", "0.05"]


@pytest.fixture
def ik(tmp_path, monkeypatch):
    """The script as a module, run in a fresh directory: no nvidia-smi, and a CUDA device file nothing holds."""
    spec = importlib.util.spec_from_file_location("ikstep_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "gpu_mem", lambda: None)
    monkeypatch.setattr(mod, "UVM", str(tmp_path / "no-such-device"))
    for k in ("HOST", "PORT", "USER", "KEY", "PREFIX"):
        monkeypatch.delenv(f"IK_{k}", raising=False)
    yield mod
    for f in (tmp_path / "logs").glob("*.job.json"):  # never leave a dummy job running
        try:
            os.killpg(json.loads(f.read_text())["pgid"], signal.SIGKILL)
        except OSError:
            pass


def step(ik, capsys, *argv) -> tuple[int, list[str]]:
    rc = ik.main(list(map(str, argv)))
    return rc, capsys.readouterr().out.splitlines()


def job(name: str) -> dict:
    return json.loads(Path(f"logs/{name}.job.json").read_text())


# --- the script itself ---------------------------------------------------------------------------------

def test_small_stdlib_only_python310():
    src = SCRIPT.read_text()
    assert len(src.encode()) <= 40 * 1024, len(src.encode())  # it counts toward the plugin's byte cap
    tree = ast.parse(src, feature_version=(3, 10))
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names), mods - set(sys.stdlib_module_names)


def test_usage_errors(ik, capsys):
    with pytest.raises(SystemExit) as e:
        ik.main(["--local", "start", "srv"])  # no command after --
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        ik.main(["wait", "srv"])  # neither a host nor --local
    assert e.value.code == 2
    assert "--host" in capsys.readouterr().err
    assert step(ik, capsys, "--local", "stop") == (2, ["stop: name a job, or pass --all"])
    assert step(ik, capsys, "--local", "wait", "nothing")[0] == 2


# --- the remote path: one ssh call, this file on stdin ----------------------------------------------------

class FakeRun:
    def __init__(self, *results):
        self.calls, self.results = [], list(results)

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        rc, out = self.results.pop(0) if self.results else (0, "")
        return subprocess.CompletedProcess(argv, rc, out, "")


def test_remote_step_is_one_ssh_line_after_the_prefix(ik, capsys, monkeypatch):
    for k, v in {"HOST": "gpu1", "PORT": "2222", "USER": "me", "KEY": "/k/id",
                 "PREFIX": "export PATH=$HOME/ik/bin:$PATH; cd ~/ik/runs/r1;"}.items():
        monkeypatch.setenv(f"IK_{k}", v)
    fake = FakeRun((3, ""), (255, ""))
    monkeypatch.setattr(ik.subprocess, "run", fake)
    assert ik.main(["start", "srv", "--ready-url", "http://127.0.0.1:8000/health", "--",
                    "vllm serve models/m --port 8000"]) == 3  # the host's exit code passes through
    argv, kw = fake.calls[0]
    assert argv[:2] == ["ssh", "-oBatchMode=yes"] and argv[-2] == "me@gpu1"
    assert argv[argv.index("-i") + 1] == "/k/id" and argv[argv.index("-p") + 1] == "2222"
    assert argv[-1] == ("export PATH=$HOME/ik/bin:$PATH; cd ~/ik/runs/r1; python3 - --local start srv "
                        "--ready-url http://127.0.0.1:8000/health -- 'vllm serve models/m --port 8000'")
    assert kw["stdin"].name == str(SCRIPT)
    assert ik.main(["--host", "other", "--prefix", "cd run", "wait", "srv", "--chunk", "5"]) == 1  # ssh failed
    argv, _ = fake.calls[1]
    assert argv[-2] == "me@other" and argv[-1] == "cd run; python3 - --local wait srv --chunk 5"
    assert "ssh failed" in capsys.readouterr().out


def stand_ins(tmp_path: Path, monkeypatch) -> None:
    """`ssh` that runs its last argument here, and an `nvidia-smi` that fails (no GPU is ever queried)."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for name, body in (("ssh", 'for a; do last=$a; done\nexec bash -c "$last"'), ("nvidia-smi", "exit 9")):
        (bindir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bindir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


def test_remote_path_end_to_end_through_a_stand_in_ssh(ik, capfd, monkeypatch, tmp_path):
    stand_ins(tmp_path, monkeypatch)
    run = tmp_path / "host-run"
    run.mkdir()
    monkeypatch.setenv("IK_HOST", "gpu1")
    monkeypatch.setenv("IK_PREFIX", f"cd {run};")
    try:
        t0 = time.time()
        assert ik.main(["start", "srv", "--done-regex", "MARK", "--", "echo MARK; sleep 60"]) == 0
        assert time.time() - t0 < 10  # the detached job does not hold the ssh session open
        assert ik.main(["wait", "srv", "--poll", "0.05"]) == 0
        assert ik.main(["stop", "srv"]) == 0
        out = capfd.readouterr().out.splitlines()
        assert out[0].startswith("started srv: pgid") and out[1].startswith("DONE srv:") and out[1].endswith("MARK")
        assert out[2] == "STOPPED srv; GPU MiB used: no nvidia-smi" and len(out) == 3
        assert (run / "logs" / "srv.job.json").is_file() and not Path("logs").exists()  # on the host side only
    finally:
        try:
            os.killpg(json.loads((run / "logs" / "srv.job.json").read_text())["pgid"], signal.SIGKILL)
        except (OSError, ValueError):
            pass


def test_deploy_rsyncs_into_the_run_directory_then_falls_back_to_scp(ik, capsys, monkeypatch, tmp_path):
    src = tmp_path / "engine"
    (src / ".git").mkdir(parents=True)
    (src / "serve.py").write_text("print('hi')\n")
    monkeypatch.setattr(ik.shutil, "which", lambda name: f"/usr/bin/{name}")
    fake = FakeRun((0, "/home/me/ik/runs/r1/engine\n"), (0, ""))
    monkeypatch.setattr(ik.subprocess, "run", fake)
    base = ["--host", "gpu1", "--port", "2222", "--user", "me", "--prefix", "cd ~/ik/runs/r1"]
    assert step(ik, capsys, *base, "deploy", src) == (0, [f"deployed {src} -> engine (rsync)"])
    assert fake.calls[0][0][-1] == "cd ~/ik/runs/r1; mkdir -p engine && cd engine && pwd"
    rsync = fake.calls[1][0]
    assert rsync[0] == "rsync" and "--exclude=.git" in rsync
    assert rsync[-2:] == [f"{src}/", "me@gpu1:/home/me/ik/runs/r1/engine/"]
    fake = FakeRun((0, "/home/me/ik/runs/r1/engine\n"), (12, ""), (0, ""))  # rsync missing on the host
    monkeypatch.setattr(ik.subprocess, "run", fake)
    assert step(ik, capsys, *base, "deploy", src)[1] == [f"deployed {src} -> engine (scp)"]
    scp = fake.calls[2][0]
    assert scp[:2] == ["scp", "-rpq"] and scp[scp.index("-P") + 1] == "2222" and "-p" not in scp
    assert scp[-2:] == [str(src / "serve.py"), "me@gpu1:/home/me/ik/runs/r1/engine/"]  # .git is not copied
    assert step(ik, capsys, "--local", "deploy", src)[0] == 2


# --- local jobs on dummy processes ------------------------------------------------------------------------

def test_finished_and_died_verdicts(ik, capsys):
    rc, out = step(ik, capsys, "--local", "start", "ok", "--", "echo hello")
    assert rc == 0 and len(out) == 1 and out[0].startswith("started ok: pgid ") and "log logs/ok.log" in out[0]
    rc, out = step(ik, capsys, "--local", "wait", "ok", *FAST)
    assert rc == 0 and len(out) == 1 and out[0].startswith("FINISHED ok: exit 0")
    script = "for i in $(seq 1 9); do echo \"step $i\"; echo \"RuntimeError: boom $i\" >&2; done; exit 3"
    step(ik, capsys, "--local", "start", "bad", "--", script)
    rc, out = step(ik, capsys, "--local", "wait", "bad", *FAST)
    assert rc == 1 and out[0].startswith("DIED bad: died (exit 3)")
    assert out[1:] == [f"  RuntimeError: boom {i}" for i in range(5, 10)]  # at most 5 error lines, the last ones


def test_done_regex_then_stop_kills_the_whole_group(ik, capsys):
    step(ik, capsys, "--local", "start", "srv", "--done-regex", r"Uvicorn running|READY", "--",
         "sleep 60 & echo loading; echo 'server READY on 8000'; sleep 60")
    rc, out = step(ik, capsys, "--local", "wait", "srv", *FAST)
    assert rc == 0 and out == [out[0]] and out[0].startswith("DONE srv:") and "server READY on 8000" in out[0]
    pgid = job("srv")["pgid"]
    rc, out = step(ik, capsys, "--local", "stop", "srv")
    assert rc == 0 and out[0].startswith("STOPPED srv; GPU MiB used: no nvidia-smi")
    assert not ik.alive(pgid)  # the background sleep went too


def test_wait_returns_in_chunks_and_on_timeout(ik, capsys):
    step(ik, capsys, "--local", "start", "slow", "--", "echo tick; sleep 60")
    t0 = time.time()
    rc, out = step(ik, capsys, "--local", "wait", "slow", "--chunk", "0.3", *FAST)
    assert rc == 3 and time.time() - t0 < 5 and out[0].startswith("RUNNING slow:") and out[0].endswith("tick")
    rc, out = step(ik, capsys, "--local", "wait", "slow", "--timeout", "0.01", *FAST)
    assert rc == 4 and out[0].startswith("TIMEOUT slow:") and job("slow")["timeout"] == 0.01  # kept for later waits
    assert step(ik, capsys, "--local", "start", "slow", "--", "true")[1][0].startswith("start slow: refused")
    assert step(ik, capsys, "--local", "stop", "slow")[0] == 0


def test_ready_url_and_a_server_that_exits_first(ik, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    step(ik, capsys, "--local", "start", "web", "--ready-url", url, "--",
         f"exec {sys.executable} -I -m http.server {port} --bind 127.0.0.1")
    rc, out = step(ik, capsys, "--local", "wait", "web", "--timeout", "30", *FAST)
    assert rc == 0 and out[0].startswith(f"READY web: {url} answered")
    assert step(ik, capsys, "--local", "stop", "web")[0] == 0 and not ik.alive(job("web")["pgid"])
    step(ik, capsys, "--local", "start", "early", "--ready-url", url, "--", "echo bye")
    rc, out = step(ik, capsys, "--local", "wait", "early", *FAST)
    assert rc == 1 and out[0].startswith("DIED early: exited before ready or done")


def test_stop_waits_for_gpu_memory_to_return(ik, capsys, monkeypatch):
    monkeypatch.setattr(ik, "gpu_mem", lambda: [500])  # in use before the job
    step(ik, capsys, "--local", "start", "eng", "--", "sleep 60")
    seen = iter([[9000], [9000], [600]])
    monkeypatch.setattr(ik, "gpu_mem", lambda: next(seen, [600]))
    real_sleep = time.sleep
    monkeypatch.setattr(ik.time, "sleep", lambda s: real_sleep(min(s, 0.02)))
    rc, out = step(ik, capsys, "--local", "stop", "eng")
    assert rc == 0 and out == [f"STOPPED eng; GPU MiB used: [600]"]
    step(ik, capsys, "--local", "start", "eng2", "--", "sleep 60")
    monkeypatch.setattr(ik, "gpu_mem", lambda: [9000])
    rc, out = step(ik, capsys, "--local", "stop", "eng2", "--gpu-wait", "0")
    assert rc == 1 and out[0].startswith("STOP INCOMPLETE: eng2; GPU MiB used: [9000]")


def test_a_stale_job_never_kills_a_group_that_reused_its_pgid(ik, capsys):
    other = subprocess.Popen(["sleep", "60"], start_new_session=True)  # owns pgid == its pid now
    try:
        Path("logs").mkdir()
        stale = {"name": "old", "kind": "job", "pgid": other.pid, "ticks": "1", "log": "logs/old.log",
                 "started": 0, "ready_url": None, "done_regex": None, "timeout": None, "gpu_mib_before": None}
        Path("logs/old.job.json").write_text(json.dumps(stale))
        assert not ik.alive(other.pid, "1") and ik.alive(other.pid, ik.stat(other.pid)[19])
        assert step(ik, capsys, "--local", "stop", "--all") == (0, ["STOPPED nothing was running; GPU MiB used: "
                                                                     "no nvidia-smi"])
        assert other.poll() is None  # still running
        assert step(ik, capsys, "--local", "start", "old", "--", "true")[1][0].startswith("started old")
    finally:
        other.kill()
        other.wait()


def cuda_stand_in(uvm: Path, cwd: Path, own_session: bool = True) -> subprocess.Popen:
    """A process that holds UVM open, as a CUDA process holds /dev/nvidia-uvm, started from CWD."""
    code = "import sys, time; f = open(sys.argv[1]); print('open', flush=True); time.sleep(60)"
    p = subprocess.Popen([sys.executable, "-I", "-c", code, str(uvm)], cwd=cwd, stdout=subprocess.PIPE, text=True,
                         start_new_session=own_session)
    assert p.stdout.readline() == "open\n"
    return p


@pytest.fixture
def uvm(ik, tmp_path, monkeypatch):
    dev = tmp_path / "uvm"
    dev.write_text("")
    monkeypatch.setattr(ik, "UVM", os.path.realpath(dev))
    (tmp_path / "engine").mkdir()
    (tmp_path.parent / f"{tmp_path.name}-elsewhere").mkdir()
    procs = []
    yield dev, procs
    for p in procs:
        p.kill()
        p.wait()


def test_gpu_groups_takes_only_cuda_processes_started_here(ik, uvm, tmp_path):
    dev, procs = uvm
    elsewhere = tmp_path.parent / f"{tmp_path.name}-elsewhere"
    procs += [cuda_stand_in(dev, tmp_path), cuda_stand_in(dev, tmp_path / "engine"),  # the run dir and below it
              cuda_stand_in(dev, tmp_path, own_session=False),  # in our own session (the agent's shell)
              cuda_stand_in(dev, elsewhere),  # e.g. the user's notebook
              subprocess.Popen(["sleep", "60"], start_new_session=True)]  # holds no device file: not CUDA
    here, below, own, theirs, plain = procs
    ours, rest = ik.gpu_groups()
    assert sorted(ours) == sorted([here.pid, below.pid]) and list(rest) == [theirs.pid]
    assert ours[here.pid].startswith(f"pgid {here.pid} ({sys.executable[:40]}")
    assert own.pid not in ours and os.getpgid(own.pid) not in ours and plain.pid not in ours | rest


def test_stop_all_stops_jobs_and_cuda_processes_started_here_only(ik, uvm, capsys, tmp_path):
    dev, procs = uvm
    step(ik, capsys, "--local", "start", "a", "--", "sleep 60")
    procs += [cuda_stand_in(dev, tmp_path), cuda_stand_in(dev, tmp_path.parent / f"{tmp_path.name}-elsewhere")]
    leftover, theirs = procs  # a server started by hand in the run directory; someone else's CUDA process
    rc, out = step(ik, capsys, "--local", "stop", "--all")
    assert rc == 0 and len(out) == 1 and out[0].startswith(f"STOPPED pgid {leftover.pid} (")
    assert f", a; GPU MiB used: no nvidia-smi; not ours, left running: pgid {theirs.pid} (" in out[0]
    assert leftover.wait(timeout=10) == -signal.SIGTERM and theirs.poll() is None and not ik.alive(job("a")["pgid"])


def test_start_refuses_when_the_ready_url_already_answers(ik, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    old = subprocess.Popen([sys.executable, "-I", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        end = time.time() + 20
        while not ik.url_ok(url) and time.time() < end:
            time.sleep(0.05)
        rc, out = step(ik, capsys, "--local", "start", "new", "--ready-url", url, "--", "sleep 60")
        assert rc == 1 and out == [f"start new: refused, {url} already answers; stop --all first"]
        assert not Path("logs/new.job.json").exists()
    finally:
        old.kill()
        old.wait()


def test_secrets_never_reach_the_job_record(ik, capsys):
    cmd = ("HF_TOKEN=hf_x echo hi; export MY_API_KEY='sk-a'; true --token hf_y --api-key=sk-z --hf-token hf_v "
           "-H 'Authorization: Bearer hf_w' --max-tokens=5 --tokenizer t")
    step(ik, capsys, "--local", "start", "dl", "--", cmd)
    assert step(ik, capsys, "--local", "wait", "dl", *FAST)[0] == 0
    record = Path("logs/dl.job.json").read_text()
    assert not [x for x in ("hf_x", "sk-a", "hf_y", "sk-z", "hf_v", "hf_w") if x in record]
    assert job("dl")["cmd"] == ("HF_TOKEN=*** echo hi; export MY_API_KEY=*** true --token *** --api-key=*** "
                                "--hf-token *** -H 'Authorization: Bearer *** --max-tokens=5 --tokenizer t")


# --- bench and compare: one verdict line, the full JSON in a file ------------------------------------------

def fake_inferencekit(tmp_path: Path, monkeypatch, record: dict | None, rc: int = 0) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / "inferencekit"
    body = f"cat <<'EOF'\n{json.dumps(record)}\nEOF\n" if record else "echo 'inferencekit: error: server down' >&2\n"
    exe.write_text(f'#!/bin/sh\necho "$INFERENCEKIT_OUTPUT $*" > {tmp_path}/argv.txt\n{body}exit {rc}\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return tmp_path / "argv.txt"


def test_bench_saves_the_json_and_prints_one_line(ik, capsys, monkeypatch, tmp_path):
    rec = {"evidence_id": "ev_serve_1", "summary": "decode_tok_s median 151.2 (observed)",
           "coverage": {"complete": True, "missing": []}, "data": {"rows": list(range(500))}}
    argv = fake_inferencekit(tmp_path, monkeypatch, rec)
    rc, out = step(ik, capsys, "--local", "bench", "scr", *FAST, "--", "--base-url", "http://127.0.0.1:8000",
                   "--label", "c1-screen", "--run-dir", ".")
    assert rc == 0 and len(out) == 2 and out[0].startswith("started scr")
    assert out[1] == ("OK bench scr ev_serve_1: decode_tok_s median 151.2 (observed); missing []; "
                      "full JSON logs/scr.json")
    assert json.loads(Path("logs/scr.json").read_text()) == rec
    assert argv.read_text().split() == ["json", "bench", "serve", "--base-url", "http://127.0.0.1:8000", "--label",
                                        "c1-screen", "--run-dir", "."]
    fake_inferencekit(tmp_path, monkeypatch, {**rec, "coverage": {"missing": ["gpu identity"]}})
    rc, out = step(ik, capsys, "--local", "bench", "scr2", *FAST, "--", "--label", "x")
    assert rc == 1 and out[1].startswith("INCOMPLETE bench scr2 ev_serve_1") and "['gpu identity']" in out[1]
    fake_inferencekit(tmp_path, monkeypatch, None, rc=1)
    rc, out = step(ik, capsys, "--local", "bench", "scr3", *FAST, "--", "--label", "x")
    assert rc == 1 and out[1] == "FAILED bench scr3: exit 1, no JSON record; log logs/scr3.log"
    assert out[2:] == ["  inferencekit: error: server down"]


def test_compare_prints_one_line_per_engine(ik, capsys, tmp_path):
    def row(engine, bucket, median, best, ratio=None, failed=0.0, metric="decode_tok_s"):
        return {"engine": engine, "bucket": bucket, "metric": metric, "median": median, "n": 6, "best": best,
                "ci95": [median - 1, median + 1], "ratio_to_best": ratio, "failed_requests": failed,
                "ratio_ci95": [ratio and ratio - 0.01, ratio and ratio + 0.01]}
    rows = [row("recipe", "short", 150.0, "c1", 0.95), row("c1", "short", 158.0, "c1", 1.0),
            row("recipe", "long", 140.0, "recipe", 1.0), row("c1", "long", 139.0, "recipe", 0.993, failed=1.0),
            row("c1", "long", 9.0, "c1", 1.0, metric="ttft_ms")]
    (tmp_path / "rounds").mkdir()
    (tmp_path / "rounds" / "summary.json").write_text(json.dumps({
        "plan": {"rounds": 6, "design": {"balanced": True}}, "summary": {"rows": rows},
        "resolved_config": {"c1": {"violations": ["CUDA graphs off"]}, "recipe": {"violations": []}}}))
    rc, out = step(ik, capsys, "--local", "compare")
    assert rc == 0 and len(out) == 3
    assert out[0].startswith("compare rounds/summary.json: 6 rounds, balanced True; decode_tok_s median")
    assert out[1] == "  recipe: short 150 [149-151] n=6 0.95 [0.94-0.96]x best; long 140 [139-141] n=6 best"
    assert out[2] == ("  c1 HARD-GATE: short 158 [157-159] n=6 best; "
                      "long 139 [138-140] n=6 FAILURES 0.993 [0.983-1.003]x best")
    rc, out = step(ik, capsys, "--local", "compare", "--bucket", "short", "--metric", "ttft_ms")
    assert rc == 1 and out == ["compare: no ttft_ms rows in rounds/summary.json"]
