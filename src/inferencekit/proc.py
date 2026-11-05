"""Running external tools: nvidia-smi, nvcc, nsys, ncu, engine binaries.

Profilers run under a *clean environment*: only an allowlist of variables is passed
(PATH, HOME, CUDA_VISIBLE_DEVICES, plus names the user adds with ``--keep-env``).
This is what ``env -i PATH=... HOME=... <cmd>`` does; we pass the dict through
``subprocess`` instead of putting ``NAME=value`` pairs on a command line, so the
values never show up in ``ps`` output. Why bother: Nsight Systems and Nsight Compute
store the profiled process's command line and ``NAME=value`` for every variable it
received inside the report file, and a report shared for review must not carry
tokens. So secret-looking names are refused outright (see ``clean_env``).

Benchmarks run with the caller's full environment instead (engines need their usual
setup). The two policies meet in ``config_env_names``: both kinds of record list the
*names* of engine-configuration variables (VLLM_*, TORCH_*, ...), and a profile also
lists the ones it did not pass on, so a profile of a different configuration than the
benchmark is visible.

Every child runs in its own process group, and a timeout kills the whole group: a
profiler's or an engine's own children (the program under nsys, a vLLM engine core)
would otherwise keep running on the GPU and slow down the next measurement.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from inferencekit.evidence import looks_secret_name
from inferencekit.registry import OpError

BASE_ENV_ALLOWLIST = ("PATH", "HOME", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                      "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
# Variables that change what an engine runs (attention backend, GEMM library, compile caches,
# which libraries load). Only their NAMES are ever recorded.
CONFIG_ENV_RX = re.compile(r"^(VLLM_|SGLANG_|TORCH_|PYTORCH_|TRITON_|NCCL_|CUDA_|CUBLAS|CUDNN_|GGML_|LLAMA_|"
                           r"FLASHINFER_|ROCR_|HIP_|HSA_|GPU_DEVICE_ORDINAL$|HF_HOME$|HF_HUB_OFFLINE$|"
                           r"LD_LIBRARY_PATH$|LD_PRELOAD$|OMP_NUM_THREADS$)")


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def which(name: str) -> str | None:
    """Find a binary on PATH, then in the usual CUDA install locations."""
    found = shutil.which(name)
    if found:
        return found
    if name == "rocprofv3":
        rocm_tool = Path("/opt/rocm/bin", name)
        if rocm_tool.is_file():
            return str(rocm_tool)
    for root in [os.environ.get("CUDA_HOME"), "/usr/local/cuda"] + sorted(
            (str(p) for p in Path("/usr/local").glob("cuda-*")), reverse=True):
        if root and Path(root, "bin", name).is_file():
            return str(Path(root, "bin", name))
    for p in sorted(Path("/opt/nvidia").glob(f"nsight-*/*/bin/{name}"), reverse=True):  # nsight standalone installs
        return str(p)
    return None


def _run_in_group(argv: list[str], timeout: float | None, **popen_kw) -> tuple[int, str | None, str | None]:
    """subprocess.run, except that a timeout kills the child's whole process group.

    ``start_new_session=True`` makes the child the leader of a new process group; its own
    children (the program nsys profiles, an engine's worker processes) inherit that group.
    ``subprocess.run`` would SIGKILL only the direct child and orphan the rest, still
    holding GPU memory and bandwidth. ``os.killpg`` signals every process in the group.
    The group also no longer receives the terminal's Ctrl-C, so any exception here,
    KeyboardInterrupt included, kills the group before re-raising.
    """
    p = subprocess.Popen(argv, start_new_session=True, text=True, **popen_kw)
    try:
        out, err = p.communicate(timeout=timeout)
    except BaseException:  # TimeoutExpired, KeyboardInterrupt, ...
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:  # the group already exited
            pass
        p.communicate()  # reap the child and close its pipes
        raise
    return p.returncode, out, err


def run(argv: list[str], timeout: float | None = 120, env: dict | None = None, cwd: str | None = None,
        stdout_path: Path | None = None, stderr_path: Path | None = None) -> Result:
    """Run a command and capture output (to memory, or to files for long-running jobs)."""
    try:
        if stdout_path or stderr_path:
            with open(stdout_path or os.devnull, "w") as out, open(stderr_path or os.devnull, "w") as err:
                code, _, _ = _run_in_group(argv, timeout, stdout=out, stderr=err, env=env, cwd=cwd)
            so = Path(stdout_path).read_text(errors="replace") if stdout_path else ""
            se = Path(stderr_path).read_text(errors="replace") if stderr_path else ""
            return Result(code, so, se)
        code, so, se = _run_in_group(argv, timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                     cwd=cwd)
        return Result(code, so or "", se or "")
    except FileNotFoundError:
        return Result(127, "", f"not found: {argv[0]}")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"timed out after {timeout} s: {argv[0]} (its whole process group was killed)")


def clean_env(keep: list[str] | None = None) -> tuple[dict, list[str]]:
    """The allowlisted environment for a profiled run, and the *names* it contains.

    Only names are recorded in the evidence record. The profiler is a different matter:
    nsys and ncu write the full command line and ``NAME=value`` for every variable passed
    here into the report file. That is why secret-looking names (``*TOKEN*``, ``*KEY*``,
    ``*SECRET*``...) are refused: make the run need no secret instead (download weights
    first, set ``HF_HUB_OFFLINE=1``).
    """
    secret = [k for k in (keep or []) if looks_secret_name(k)]
    if secret:
        raise OpError(f"refusing --keep-env {', '.join(secret)}: profiler reports store NAME=value for every "
                      "variable passed to the profiled program, so a secret would end up in the report file. "
                      "Make the run need no secret (download weights first, set HF_HUB_OFFLINE=1).")
    names = list(BASE_ENV_ALLOWLIST) + [k for k in (keep or []) if k not in BASE_ENV_ALLOWLIST]
    env = {k: os.environ[k] for k in names if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return env, sorted(env)


def config_env_names(env: dict | None = None) -> list[str]:
    """Names (never values) of engine-configuration variables set in ``env`` (default: this process)."""
    return sorted(k for k in (os.environ if env is None else env) if CONFIG_ENV_RX.match(k))


def dropped_config_env(env: dict) -> list[str]:
    """Engine-configuration variables set here but not passed to a profiled run (``env``).

    A benchmark runs with the full environment, a profile with the allowlist. If, say,
    VLLM_ATTENTION_BACKEND is set here but not kept, the profile traces a different
    configuration than the benchmark did. Names only.
    """
    return [k for k in config_env_names() if k not in env and not looks_secret_name(k)]


def require(name: str, hint: str) -> str:
    path = which(name)
    if not path:
        raise OpError(f"{name} not found on PATH. {hint}")
    return path


def tool_version(name: str) -> str | None:
    """First version-looking token of ``<tool> --version`` (nsys, ncu, nvcc...)."""
    path = which(name)
    if not path:
        return None
    r = run([path, "--version"], timeout=30)
    text = r.stdout + r.stderr
    m = re.search(r"release (\d+\.\d+)", text)  # nvcc: "release 12.8, V12.8.93"
    if m and name == "nvcc":
        return m.group(1)
    m = re.search(r"(?:version|Version)\s+v?(\d+(?:\.\d+)+)", text)
    if m:
        return m.group(1)
    m = re.search(r"(\d+\.\d+(?:\.\d+)*)", text)
    return m.group(1) if m else (text.strip().splitlines() or [None])[0]
