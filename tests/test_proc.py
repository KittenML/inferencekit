"""Running subprocesses: process-group kills on timeout, and the profiler environment policy."""

import sys
import time

import pytest

from inferencekit import proc
from inferencekit.registry import OpError


def _gone(pid: int) -> bool:
    """True once the process has exited (a zombie waiting to be reaped counts as gone)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ")[-1].startswith("Z")
    except (FileNotFoundError, ProcessLookupError):  # reaped between the open and the read (ESRCH)
        return True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_timeout_kills_the_whole_process_group(tmp_path):
    # The direct child (a shell, like nsys or an engine launcher) starts a grandchild and waits.
    # subprocess.run's timeout would kill only the shell and leave the grandchild running.
    # The pid file is written via a temp name and a rename, so it is either absent or complete; the timeout
    # leaves a loaded machine (full test suite) ample time to start the grandchild and write it.
    pid_file = tmp_path / "grandchild.pid"
    r = proc.run(["sh", "-c", f"sleep 60 & echo $! > {pid_file}.tmp; mv {pid_file}.tmp {pid_file}; wait"],
                 timeout=5)
    assert r.returncode == 124 and "process group" in r.stderr
    assert pid_file.is_file(), "the shell never started the grandchild within the timeout"
    grandchild = int(pid_file.read_text())
    deadline = time.time() + 20
    while not _gone(grandchild) and time.time() < deadline:
        time.sleep(0.05)
    assert _gone(grandchild)


def test_run_still_captures_output_and_exit_codes(tmp_path):
    r = proc.run([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"])
    assert (r.returncode, r.stdout.strip(), r.stderr.strip()) == (3, "out", "err")
    out = tmp_path / "o.txt"
    r = proc.run([sys.executable, "-c", "print('to file')"], stdout_path=out)
    assert r.ok and out.read_text().strip() == "to file" == r.stdout.strip()
    assert proc.run(["/nonexistent/binary"]).returncode == 127


def test_clean_env_refuses_secret_looking_names(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_not_a_real_secret_value")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(OpError, match="HF_TOKEN"):
        proc.clean_env(["HF_HUB_OFFLINE", "HF_TOKEN"])
    env, names = proc.clean_env(["HF_HUB_OFFLINE"])
    assert names == sorted(env) and "HF_HUB_OFFLINE" in env and "HF_TOKEN" not in env


def test_dropped_engine_configuration_is_named(monkeypatch):
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.0")
    monkeypatch.setenv("VLLM_API_KEY", "not-a-real-key-123")
    env, _ = proc.clean_env(["TORCH_CUDA_ARCH_LIST"])
    dropped = proc.dropped_config_env(env)
    assert "VLLM_ATTENTION_BACKEND" in dropped  # set here, not passed: the profile may differ from bench
    assert "TORCH_CUDA_ARCH_LIST" not in dropped  # kept
    assert "VLLM_API_KEY" not in dropped  # secrets are never suggested for --keep-env
    assert "VLLM_ATTENTION_BACKEND" in proc.config_env_names()
