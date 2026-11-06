"""Engine servers for round runs: launch, readiness, warm-up, /metrics, stop, GPU memory, GPU lock.

One (round, engine) cell of ``bench rounds`` does, in order:

  1. check the port is free (a leftover server on it would answer the readiness probe
     and get benchmarked in place of the engine under test);
  2. launch the server under a clean environment (only the variables the manifest
     lists, plus PATH/HOME/CUDA basics), in its own process group, stdout+stderr to a
     log file;
  3. poll ``GET <ready_check>`` until it answers 200, the process exits, or the
     startup timeout passes;
  4. warm up with a few short requests (untimed), each salted with a unique prefix tag so
     the warm-up never seeds the prefix cache for a prompt that is timed afterwards;
  5. scrape ``/metrics`` (Prometheus text) before and after the measured run, when the
     engine exposes it;
  6. stop the whole process group: SIGTERM, then SIGKILL after a grace period. Engines
     fork workers (vLLM's engine core, SGLang's scheduler and detokenizer); signalling
     only the launcher would leave them holding GPU memory;
  7. check that GPU memory returned to the level measured before the launch
     (nvidia-smi; skipped when absent), so the next engine starts on an empty GPU.

The whole run holds an exclusive lock file (``gpu_lock``), so two round runs (or a
round run and an agent's ad-hoc benchmark that also takes the lock) cannot share the GPU.

Environment values are passed to the server but never recorded: records list names only.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from inferencekit.evidence import looks_secret_name
from inferencekit.proc import run, which
from inferencekit.registry import OpError

# Passed through from the caller in addition to the manifest's own variables: enough to
# find binaries, the home directory (HF cache) and the GPU, nothing that tunes an engine.
SERVER_BASE_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "CUDA_VISIBLE_DEVICES",
                   "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES",
                   "NVIDIA_DRIVER_CAPABILITIES")


# --------------------------------------------------------------------------- environment

def server_env(listed: dict | None) -> tuple[dict, dict]:
    """The environment for one engine server, and a names-only description of it.

    ``listed`` is the manifest's ``env`` object. A string (or number) value is set as
    given; ``null`` means "inherit this variable from the caller if it is set", which is
    how a secret such as HF_TOKEN reaches the server without being written into the
    manifest. Returns ``(env, info)`` where ``info`` holds only names.
    """
    env = {k: os.environ[k] for k in SERVER_BASE_ENV if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    set_names, inherited, missing, secret_values = [], [], [], []
    for name, value in (listed or {}).items():
        name = str(name)
        if value is None:
            if name in os.environ:
                env[name] = os.environ[name]
                inherited.append(name)
            else:
                missing.append(name)
            continue
        env[name] = value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) \
            else str(value).lower() if isinstance(value, bool) else str(value)
        set_names.append(name)
        if looks_secret_name(name):
            secret_values.append(name)
    info = {"base": sorted(k for k in SERVER_BASE_ENV if k in env), "set": sorted(set_names),
            "inherited": sorted(inherited), "missing": sorted(missing),
            "secret_values_in_manifest": sorted(secret_values)}
    return env, info


# --------------------------------------------------------------------------- HTTP helpers

def http_get(url: str, timeout: float = 5.0) -> tuple[int | None, bytes | str]:
    """(status, body) or (None, error text). Never raises for network errors."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - local engine URLs only
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""
    except (urllib.error.URLError, OSError, ValueError) as e:
        return None, str(getattr(e, "reason", e))


def http_post_json(url: str, body: dict, timeout: float = 60.0) -> tuple[int | None, dict | str]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            raw = r.read()
            try:
                return r.status, json.loads(raw)
            except ValueError:
                return r.status, raw.decode(errors="replace")[:500]
    except urllib.error.HTTPError as e:
        return e.code, (e.read() if e.fp else b"").decode(errors="replace")[:500]
    except (urllib.error.URLError, OSError, ValueError) as e:
        return None, str(getattr(e, "reason", e))


def port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """True when something already accepts connections on host:port."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def served_model(base_url: str, timeout: float = 5.0) -> str | None:
    """The first model id in ``GET /v1/models`` (the name requests must use), or None."""
    status, body = http_get(base_url.rstrip("/") + "/v1/models", timeout)
    if status != 200:
        return None
    try:
        data = json.loads(body)
        return str(data["data"][0]["id"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None


# --------------------------------------------------------------------------- processes

@dataclass
class Server:
    """A launched engine server (leader of its own process group)."""

    argv: list[str]
    log_path: Path
    popen: subprocess.Popen
    started: float = field(default_factory=time.monotonic)

    @property
    def pgid(self) -> int:
        return self.popen.pid  # start_new_session=True: the child leads a new session and group


def launch(argv: list[str], env: dict, log_path: Path, cwd: str | None = None) -> Server:
    """Start ``argv`` in a new process group with stdout and stderr appended to ``log_path``.

    Raises OSError (e.g. FileNotFoundError) when the program cannot be started.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        # The child keeps its own copy of the file descriptor; ours can close right away.
        popen = subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    return Server(list(argv), log_path, popen)


def group_members(pgid: int) -> list[int] | None:
    """Live (non-zombie) pids in process group ``pgid``; None when /proc is unavailable."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    out = []
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        # fields: state ppid pgrp ...
        if len(fields) >= 3 and fields[2] == str(pgid) and fields[0] != "Z":
            out.append(int(d.name))
    return sorted(out)


def group_alive(pgid: int) -> bool:
    members = group_members(pgid)
    if members is not None:
        return bool(members)
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _wait_group_gone(server: Server, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        server.popen.poll()  # reap the leader so it does not linger as a zombie
        if not group_alive(server.pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def stop(server: Server, grace_s: float = 30.0) -> dict:
    """Stop the server's whole process group: SIGTERM, then SIGKILL after ``grace_s``.

    The group is signalled even when the leader already exited, because its children
    (engine workers) can outlive it. Returns what happened, including whether every
    process in the group is gone.
    """
    t0 = time.monotonic()
    leader_was_running = server.popen.poll() is None
    _signal_group(server.pgid, signal.SIGTERM)
    method = "sigterm" if leader_was_running else "already_exited"
    gone = _wait_group_gone(server, grace_s)
    if not gone:
        method = "sigkill"
        _signal_group(server.pgid, signal.SIGKILL)
        gone = _wait_group_gone(server, 10.0)
    with contextlib.suppress(subprocess.TimeoutExpired):
        server.popen.wait(timeout=5)
    return {"method": method, "exit_code": server.popen.returncode, "group_gone": gone,
            "seconds": round(time.monotonic() - t0, 3), "uptime_s": round(time.monotonic() - server.started, 3)}


def wait_ready(server: Server, url: str, timeout_s: float, poll_s: float = 0.5) -> dict:
    """Poll ``GET url`` until it returns 200, the server exits, or ``timeout_s`` passes.

    Returns ``{"status": "ready" | "exited" | "timeout", "seconds", "last": ...}``.
    """
    t0 = time.monotonic()
    last: str | int | None = None
    while True:
        code = server.popen.poll()
        if code is not None:
            return {"status": "exited", "seconds": round(time.monotonic() - t0, 3), "exit_code": code,
                    "last": last}
        status, body = http_get(url, timeout=min(5.0, max(0.5, poll_s * 4)))
        if status == 200:
            return {"status": "ready", "seconds": round(time.monotonic() - t0, 3), "last": 200}
        last = status if status is not None else str(body)[:200]
        if time.monotonic() - t0 >= timeout_s:
            return {"status": "timeout", "seconds": round(time.monotonic() - t0, 3), "last": last}
        time.sleep(poll_s)


def salted_copy(messages: list | None, tag: str) -> list[dict]:
    """``messages`` with ``[tag] `` put in front of the first message's text.

    The tag sits as early in the prompt as the chat template allows, so a request with the
    salted copy shares no cacheable prefix with the original (vLLM automatic prefix caching,
    the SGLang radix cache and llama-server ``cache_prompt`` all match from the start).
    """
    msgs = [dict(m) for m in (messages or []) if isinstance(m, dict)] or [{"role": "user", "content": "Hello"}]
    first = msgs[0]
    content = first.get("content")
    if isinstance(content, list):  # multi-part content: a text part in front
        first["content"] = [{"type": "text", "text": f"[{tag}] "}] + list(content)
    else:
        first["content"] = f"[{tag}] " + str(content if content is not None else "")
    return msgs


WARMUP_KIND = ("salted copies of the prompt set's first prompts (unique prefix tag per request, so no timed "
               "prompt's prefix is cached by the warm-up)")


def warmup(base_url: str, model: str | None, prompts: list[dict], n: int, max_tokens: int = 16,
           timeout: float = 300.0, salt: str = "ik-rounds-warmup", chat_template_kwargs: dict | None = None) -> dict:
    """Send ``n`` short, untimed chat requests to warm the server.

    The requests are the first prompts of the set (so kernels for those shapes are warm),
    each salted with ``[<salt>-<i>]`` in front of its first message: the warm-up must not
    fill the prefix cache for the very prompts ``bench serve`` times next, which would turn
    their TTFT into cache hits for engines with prefix caching only.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sent, ok, errors = 0, 0, []
    for i, p in enumerate((prompts or [])[:max(0, n)]):
        body = {"messages": salted_copy(p.get("messages"), f"{salt}-{i}"), "max_tokens": max_tokens,
                "stream": False, "temperature": 0}
        if model:
            body["model"] = model
        own = p.get("chat_template_kwargs")
        kw = {**(chat_template_kwargs or {}), **(own if isinstance(own, dict) else {})}
        if kw:  # the same template settings bench serve sends (the warm-up must render the same prompt shape)
            body["chat_template_kwargs"] = kw
        status, resp = http_post_json(url, body, timeout)
        sent += 1
        if status == 200:
            ok += 1
        else:
            errors.append(f"{status}: {str(resp)[:200]}")
    return {"sent": sent, "ok": ok, "errors": errors[:5], "kind": WARMUP_KIND, "salt": salt}


# --------------------------------------------------------------------------- Prometheus /metrics

def scrape_metrics(base_url: str, timeout: float = 5.0) -> str | None:
    """Raw Prometheus text from ``GET /metrics``, or None when the engine does not expose it."""
    status, body = http_get(base_url.rstrip("/") + "/metrics", timeout)
    if status != 200 or not body:
        return None
    text = body.decode(errors="replace") if isinstance(body, bytes) else str(body)
    return text if ("# TYPE" in text or "# HELP" in text or "{" in text) else None


def parse_prometheus(text: str | None) -> dict:
    """``{"types": {family: type}, "samples": {"name{labels}": value}}`` from Prometheus text format."""
    types, samples = {}, {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "TYPE":
                types[parts[2]] = parts[3]
            continue
        # name{labels} value [timestamp]; label values may contain spaces, so split at the closing brace.
        if "}" in line:
            key, _, rest = line.rpartition("}")
            key += "}"
        else:
            key, _, rest = line.partition(" ")
        fields = rest.split()
        if not fields:
            continue
        try:
            samples[key.strip()] = float(fields[0])
        except ValueError:
            continue
    return {"types": types, "samples": samples}


def _family(sample_key: str, types: dict) -> tuple[str, str | None]:
    name = sample_key.split("{", 1)[0]
    if name in types:
        return name, types[name]
    for suffix in ("_total", "_count", "_sum", "_bucket", "_created"):
        if name.endswith(suffix) and name[: -len(suffix)] in types:
            return name[: -len(suffix)], types[name[: -len(suffix)]]
    return name, None


def metrics_delta(before: str | None, after: str | None) -> dict:
    """Counter and histogram increases between two scrapes (what the measured run did).

    Gauges are reported at their ``after`` value. Series without a TYPE line are left out:
    without the type we cannot tell a counter from a gauge.
    """
    if before is None or after is None:
        return {"available": False}
    b, a = parse_prometheus(before), parse_prometheus(after)
    types = {**b["types"], **a["types"]}
    counters, gauges = {}, {}
    for key, value in a["samples"].items():
        family, kind = _family(key, types)
        if key.split("{", 1)[0].endswith("_created"):
            continue
        if kind in ("counter", "histogram", "summary"):
            diff = value - b["samples"].get(key, 0.0)
            if diff:
                counters[key] = diff
        elif kind == "gauge":
            gauges[key] = value
    return {"available": True, "counter_deltas": counters, "gauges_after": gauges}


# --------------------------------------------------------------------------- GPU memory

def gpu_memory_used_mib() -> list[int] | None:
    """Used memory per visible GPU in MiB, or None without nvidia-smi."""
    smi = which("nvidia-smi")
    if not smi:
        return None
    r = run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=30)
    if not r.ok:
        return None
    out = []
    for line in r.stdout.strip().splitlines():
        try:
            out.append(int(float(line.strip())))
        except ValueError:
            return None
    return out or None


def wait_gpu_memory(baseline: list[int] | None, tolerance_mib: int, timeout_s: float,
                    reader=gpu_memory_used_mib) -> dict:
    """Wait until every GPU is back within ``tolerance_mib`` of ``baseline`` (taken before launch)."""
    if baseline is None:
        return {"checked": False, "reason": "nvidia-smi not available"}
    t0 = time.monotonic()
    while True:
        now = reader()
        if now is None:
            return {"checked": False, "reason": "nvidia-smi stopped answering", "baseline_mib": baseline}
        excess = [max(0, n - b) for n, b in zip(now, baseline)]
        if all(x <= tolerance_mib for x in excess):
            return {"checked": True, "returned": True, "baseline_mib": baseline, "after_mib": now,
                    "seconds": round(time.monotonic() - t0, 3)}
        if time.monotonic() - t0 >= timeout_s:
            return {"checked": True, "returned": False, "baseline_mib": baseline, "after_mib": now,
                    "excess_mib": excess, "seconds": round(time.monotonic() - t0, 3)}
        time.sleep(0.5)


# --------------------------------------------------------------------------- GPU lock

def default_lock_path(resource: str = "GPU") -> Path:
    env = os.environ.get(f"INFERENCEKIT_{resource}_LOCK")
    if env:
        return Path(env)
    return Path(os.path.expanduser("~")) / ".cache" / "inferencekit" / f"{resource.lower()}.lock"


@contextlib.contextmanager
def gpu_lock(path: str | os.PathLike | None, owner: str, *, resource: str = "GPU"):
    """Hold an exclusive, non-blocking ``flock`` on ``path`` for the duration of the block.

    The kernel drops the lock if this process dies, so a crashed run never leaves a stale
    lock behind. The file holds the current holder's pid and owner string, for the error
    message another run prints.
    """
    if resource not in ("GPU", "CPU"):
        raise ValueError("resource must be GPU or CPU")
    p = Path(path) if path else default_lock_path(resource)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            holder = os.pread(fd, 4096, 0).decode(errors="replace").strip() or "unknown holder"
        finally:
            os.close(fd)
        advice = ("a different --gpu-lock only if the runs use different GPUs." if resource == "GPU" else
                  "a different --cpu-lock only if the runs use independent CPU resources.")
        raise OpError(f"{resource} lock {p} is held by {holder}: another run is using the {resource}. Wait for it, "
                      f"or pass {advice}") from None
    info = {"pid": os.getpid(), "owner": owner, "since": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        os.ftruncate(fd, 0)
        os.pwrite(fd, json.dumps(info).encode(), 0)
        yield {"path": str(p), **info}
    finally:
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# --------------------------------------------------------------------------- resolved configuration from logs
#
# What an engine *actually* ran with is what it logged, not what the launch command asked
# for: engines silently fall back (attention backend), clamp (context length, KV cache) or
# ignore settings. Each parser below reads the server log for the lines the engine prints
# about its resolved settings. Patterns were checked against the upstream sources on
# 13 Oct 2026 and kept tolerant (anywhere in the line, any prefix):
#   vLLM       vllm/entrypoints/serve/utils/api_utils.py  "non-default args: %s"
#              vllm/v1/engine/core.py                     "Initializing a V1 LLM engine (v%s) with config: %s"
#              vllm/platforms/cuda.py                     "Using %s backend." / "Using %s attention backend out of ..."
#              vllm/v1/worker/gpu_model_runner.py         "Capturing CUDA graphs (%s, %s)", "Graph capturing finished
#                                                         in %.0f secs, took %.2f GiB", "Model loading took ..."
#              vllm/v1/core/kv_cache_utils.py             "%s KV cache size: %s tokens", "Maximum concurrency for ..."
#              vllm/platforms/interface.py                "Unknown vLLM environment variable detected: %s"
#   SGLang     entrypoints/engine.py "server_args={...}" (older releases: "server_args=ServerArgs(...)"),
#              managers/scheduler.py "max_total_num_tokens=..., chunked_prefill_size=..., ...",
#              model_runner.py "Load weight end. ...", older "Capture cuda graph end. Time elapsed: ..."
#   llama.cpp  src/llama-context.cpp "<fn>: n_ctx = %u", "<fn>: flash_attn = %s" (auto/enabled/disabled; older 0/1),
#              src/llama-model.cpp "<fn>: offloaded %d/%d layers to GPU", src/llama-kv-cache.cpp
#              "... K (%s): %7.2f MiB, V (%s): ...", common "build: %d (%s) ...", "system_info: ..."
#   TRT-LLM    _torch/pyexecutor/py_executor_creator.py "LLM Args:\n%s" (pydantic repr), "[TRT-LLM] [W] ..."
# Anything else (or a format change) leaves the setting out; with nothing found the
# resolved config is UNPROVEN and the raw log stays in the run directory.
#
# A custom engine (engine kind ``generic``) can also print one line
#     {"inferencekit_resolved_config": {...}}
# Those settings are SELF_REPORTED: the process under test says so, nothing checks it. A
# self-report can reveal a hard-gate violation (e.g. ``"cuda_graphs": false``) but never
# proves there is none, and for a known engine kind (vllm, sglang, llamacpp, trtllm) it is
# kept for reference only: it never changes the status or the hard-gate flags, which come
# from the engine's own log lines.

ENGINE_KINDS = ("vllm", "sglang", "llamacpp", "trtllm", "generic")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_WARN = re.compile(r"\bWARN(?:ING)?\b|\[W\]|\[TRT-LLM\]\[WARNING\]|\bWarning:|UserWarning|DeprecationWarning|"
                   r"FutureWarning")
_ERR = re.compile(r"\bERROR\b|\bCRITICAL\b|Traceback \(most recent call last\)|\berror:|\[E\]")
MAX_LINES = 100


def engine_kind(name: str | None, launch: list[str] | None = None) -> str:
    """Map a manifest engine (``name`` and launch argv) to a log parser."""
    hay = " ".join([str(name or "")] + [str(a) for a in (launch or [])[:3]]).lower()
    if "vllm" in hay:
        return "vllm"
    if "sglang" in hay:
        return "sglang"
    if "llama" in hay and ("cpp" in hay or "server" in hay or "llama.cpp" in hay):
        return "llamacpp"
    if "trtllm" in hay or "tensorrt" in hay or "trt-llm" in hay:
        return "trtllm"
    return "generic"


def _scalar(text: str):
    """'True' -> True, '4096' -> 4096, "'auto'" -> 'auto', 'torch.bfloat16' stays a string."""
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside brackets, braces, parentheses, angle brackets and quotes."""
    parts, depth, quote, buf = [], 0, None, []
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf))
    return parts


def parse_kv_pairs(text: str, sep: str = ",") -> dict:
    """``a=1, b='x', c={...}`` -> {"a": 1, "b": "x", "c": "{...}"} (nested values kept as text)."""
    out = {}
    for part in split_top_level(text, sep):
        key, eq, value = part.strip().partition("=")
        if eq and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key.strip()):
            out[key.strip()] = _scalar(value)
    return out


def _literal_dict(text: str) -> dict | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _lines(text: str) -> list[str]:
    return [_ANSI.sub("", ln).rstrip() for ln in text.splitlines()]


def _collect(lines: list[str], rx: re.Pattern, limit: int = MAX_LINES) -> list[str]:
    seen, out = set(), []
    for ln in lines:
        if rx.search(ln):
            key = re.sub(r"^\S*\s*(INFO|WARNING|ERROR)?\s*\d\d-\d\d \d\d:\d\d:\d\d\S*\s*", "", ln)[:400]
            if key not in seen:
                seen.add(key)
                out.append(ln[:400])
                if len(out) >= limit:
                    break
    return out


def _last(rx: re.Pattern, lines: list[str]):
    m = None
    for ln in lines:
        found = rx.search(ln)
        if found:
            m = found
    return m


def _custom(lines: list[str]) -> dict | None:
    for ln in reversed(lines):
        i = ln.find('{"inferencekit_resolved_config"')
        if i >= 0:
            try:
                value = json.loads(ln[i:])["inferencekit_resolved_config"]
                if isinstance(value, dict):
                    return value
            except (ValueError, KeyError):
                continue
    return None


# CUDAGraphMode values (vllm/config/compilation.py): NONE=0, PIECEWISE=1, FULL=2, and the pairs
# FULL_DECODE_ONLY=(FULL, NONE), FULL_AND_PIECEWISE=(FULL, PIECEWISE). JSON dumps print the numbers.
_CG_NUMERIC = {"0": "NONE", "1": "PIECEWISE", "2": "FULL", "2,0": "FULL_DECODE_ONLY", "2,1": "FULL_AND_PIECEWISE"}


def _cudagraph_mode(comp: str) -> str | None:
    m = re.search(r"['\"]cudagraph_mode['\"]:\s*(?:<CUDAGraphMode\.([A-Z_]+)|['\"]?([A-Z][A-Z_]+)|"
                  r"[\[(]\s*(\d)\s*,\s*(\d)\s*[\])]|(\d))", comp)
    if not m:
        return None
    if m.group(1) or m.group(2):
        return m.group(1) or m.group(2)
    if m.group(3) is not None:
        return _CG_NUMERIC.get(f"{m.group(3)},{m.group(4)}", f"({m.group(3)}, {m.group(4)})")
    return _CG_NUMERIC.get(m.group(5), m.group(5))


def _vllm(lines: list[str], out: dict) -> None:
    s, found = out["settings"], out["matched"]
    m = _last(re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\) with config: (.*)$"), lines)
    if m:
        out["version"] = m.group(1)
        cfg = parse_kv_pairs(m.group(2))
        keep = ("model", "served_model_name", "revision", "dtype", "max_seq_len", "quantization", "enforce_eager",
                "kv_cache_dtype", "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                "enable_prefix_caching", "enable_chunked_prefill", "seed", "load_format", "speculative_config")
        s["engine_config"] = {k: cfg[k] for k in keep if k in cfg}
        comp = str(cfg.get("compilation_config") or "")
        mode = _cudagraph_mode(comp)
        if mode:
            s["cudagraph_mode"] = mode
        sizes = re.search(r"['\"]cudagraph_capture_sizes['\"]:\s*\[([^\]]*)\]", comp)
        if sizes:
            s["cudagraph_capture_sizes_count"] = len([x for x in sizes.group(1).split(",") if x.strip()])
        level = re.search(r"['\"](?:level|mode)['\"]:\s*(?:<CompilationMode\.)?([A-Z_0-9]+)", comp)
        if level:
            s["compilation_mode"] = level.group(1)
        found.append("engine_config")
    if not out.get("version"):
        v = _last(re.compile(r"vLLM (?:API server )?version:? v?([0-9][\w.+-]*)"), lines)
        if v:
            out["version"] = v.group(1)
    m = _last(re.compile(r"non-default args: (\{.*\})\s*$"), lines)
    if m:
        parsed = _literal_dict(m.group(1))
        s["non_default_args"] = parsed if parsed is not None else m.group(1)[:2000]
        found.append("non_default_args")
    from inferencekit.engines.gates import attention_backend_from_log

    backend = attention_backend_from_log("\n".join(lines))
    if backend:
        s["attention_backend"] = backend
        found.append("attention_backend")
    caps = [m.group(1, 2) for m in (re.search(r"Capturing CUDA graphs \(([^,]+), (\w+)\)", ln) for ln in lines) if m]
    if caps:
        s["cuda_graph_captures"] = sorted({f"{a} {b}" for a, b in caps})
    m = _last(re.compile(r"Graph capturing finished in ([\d.]+) secs?, took (-?[\d.]+) GiB"), lines)
    if m:
        s["cuda_graphs"] = {"captured": True, "seconds": float(m.group(1)), "gib": float(m.group(2))}
        found.append("cuda_graphs")
    m = _last(re.compile(r"(?:(\w+) )?KV cache size: ([\d,]+) tokens"), lines)
    if m:
        s["kv_cache_tokens"] = int(m.group(2).replace(",", ""))
        found.append("kv_cache")
    m = _last(re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"), lines)
    if m:
        s["max_concurrency"] = {"tokens_per_request": int(m.group(1).replace(",", "")), "x": float(m.group(2))}
    m = _last(re.compile(r"Available KV cache memory: ([\d.]+) GiB"), lines)
    if m:
        s["kv_cache_memory_gib"] = float(m.group(1))
    m = _last(re.compile(r"Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) s"), lines)
    if m:
        s["model_memory_gib"] = float(m.group(1))
    for ln in lines:
        u = re.search(r"Unknown vLLM environment variable detected: (\S+)", ln)
        if u:
            out["unknown_settings"].append(f"env {u.group(1)}")
        u = re.search(r"error: unrecognized arguments: (.*)$", ln)
        if u:
            out["unknown_settings"].append(f"args {u.group(1).strip()}")
    flags = {}
    cfg = s.get("engine_config", {})
    if "enforce_eager" in cfg:
        flags["enforce_eager"] = cfg["enforce_eager"]
    if "cudagraph_mode" in s:
        flags["cudagraph_mode"] = s["cudagraph_mode"]
    if s.get("cudagraph_capture_sizes_count") is not None:
        flags["cudagraph_capture_sizes"] = {"count": s["cudagraph_capture_sizes_count"]}
    out["_gates"] = ("vllm", flags, backend)
    core = {"engine_config", "attention_backend"}
    out["status"] = "RESOLVED" if core <= set(found) else "PARTIAL" if found else "UNPROVEN"


def _sglang(lines: list[str], out: dict) -> None:
    s, found = out["settings"], out["matched"]
    m = _last(re.compile(r"server_args=(\{.*\}|ServerArgs\(.*\))\s*$"), lines)
    if m:
        raw = m.group(1)
        if raw.startswith("{"):
            args = _literal_dict(raw)
            if args is None:  # repr of non-literal values: fall back to "'key': value" pairs
                args = {k: _scalar(v) for k, v in re.findall(r"'(\w+)':\s*('[^']*'|[^,}]+)", raw)}
        else:
            args = parse_kv_pairs(raw[len("ServerArgs("):-1])
        s["server_args"] = args
        found.append("server_args")
    m = _last(re.compile(r"(max_total_num_tokens=.*)$"), lines)
    if m:
        s["scheduler"] = {k: v for k, v in parse_kv_pairs(m.group(1)).items()}
        found.append("scheduler")
    m = _last(re.compile(r"Load weight end\.(.*)$"), lines)
    if m:
        s["load_weight"] = {k: v for k, v in parse_kv_pairs(m.group(1).rstrip(".")).items()}
    m = _last(re.compile(r"Capture (?:cuda|CUDA) graph end\. Time elapsed: ([\d.]+) s"), lines)
    if m:
        s["cuda_graphs"] = {"captured": True, "seconds": float(m.group(1))}
        found.append("cuda_graphs")
    m = _last(re.compile(r"KV Cache is allocated\. #tokens: (\d+)"), lines)
    if m:
        s["kv_cache_tokens"] = int(m.group(1))
    v = _last(re.compile(r"\bsglang(?:[ =]+version)?[:= ]+v?(\d+\.\d+\.\d+[\w.+-]*)", re.IGNORECASE), lines)
    if v:
        out["version"] = v.group(1)
    for ln in lines:
        u = re.search(r"error: unrecognized arguments: (.*)$", ln)
        if u:
            out["unknown_settings"].append(f"args {u.group(1).strip()}")
    args = s.get("server_args") or {}
    flags = {k: args[k] for k in ("disable_cuda_graph",) if k in args}
    backend = args.get("attention_backend") or args.get("decode_attention_backend")
    out["_gates"] = ("sglang", flags, str(backend) if backend else None)
    out["status"] = "RESOLVED" if "server_args" in found else "PARTIAL" if found else "UNPROVEN"


def _llamacpp(lines: list[str], out: dict) -> None:
    s, found = out["settings"], out["matched"]
    m = _last(re.compile(r"\bbuild:\s*(\d+)\s*\(([0-9a-f]+)\)"), lines)
    if m:
        out["version"] = f"b{m.group(1)} ({m.group(2)})"
        found.append("build")
    m = _last(re.compile(r"system_info:\s*(.*)$"), lines)
    if m:
        s["system_info"] = m.group(1)[:600]
        s["gpu_backend_in_build"] = bool(re.search(r"\b(CUDA|Vulkan|Metal|HIP|ROCm|SYCL|MUSA)\b", m.group(1)))
        g = re.search(r"USE_GRAPHS\s*=\s*(\d)", m.group(1))
        if g:
            s["cuda_graphs_compiled"] = g.group(1) == "1"
    ctx = {}
    keys = ("n_ctx", "n_ctx_seq", "n_ctx_per_seq", "n_batch", "n_ubatch", "n_seq_max", "flash_attn", "causal_attn",
            "kv_unified", "freq_base", "freq_scale")
    rx = re.compile(r"(?:llama_context|llama_new_context_with_model|llama_init_from_model)\s*:\s*(\w+)\s*=\s*(\S+)")
    for ln in lines:
        c = rx.search(ln)
        if c and c.group(1) in keys:
            ctx[c.group(1)] = _scalar(c.group(2))
    if ctx:
        s["context"] = ctx
        found.append("context")
    m = _last(re.compile(r"offloaded (\d+)/(\d+) layers to GPU"), lines)
    if m:
        s["offloaded_layers"] = {"gpu": int(m.group(1)), "total": int(m.group(2))}
        found.append("offload")
    m = _last(re.compile(r"K \((\w+)\):\s*[\d.]+ MiB, V \((\w+)\)"), lines) or \
        _last(re.compile(r"type_k = '(\w+)', type_v = '(\w+)'"), lines)
    if m:
        s["kv_cache_type"] = {"k": m.group(1), "v": m.group(2)}
    m = _last(re.compile(r"Device \d+: (.+?), compute capability ([\d.]+)"), lines)
    if m:
        s["device"] = {"name": m.group(1), "compute_capability": m.group(2)}
    m = _last(re.compile(r"n_slots\s*=\s*(\d+)"), lines)
    if m:
        s["n_slots"] = int(m.group(1))
    for ln in lines:
        u = re.search(r"error: (?:invalid|unknown) argument:? (\S+)", ln)
        if u:
            out["unknown_settings"].append(f"args {u.group(1)}")
    viol = []
    off = s.get("offloaded_layers")
    if off and off["gpu"] < off["total"]:
        viol.append(f"only {off['gpu']}/{off['total']} layers offloaded to the GPU")
    if s.get("gpu_backend_in_build") is False:
        viol.append("no GPU backend in this llama.cpp build (system_info)")
    out["validity"] = {"valid": (not viol) if off or "system_info" in s else None, "violations": viol,
                       "warnings": [], "checked": [c for c, k in (("gpu_offload", "offloaded_layers"),
                                                                 ("gpu_backend", "system_info")) if k in s]}
    core = {"build", "context", "offload"}
    out["status"] = "RESOLVED" if core <= set(found) else "PARTIAL" if found else "UNPROVEN"


def _trtllm(lines: list[str], out: dict) -> None:
    s, found = out["settings"], out["matched"]
    v = _last(re.compile(r"TensorRT-LLM version:? v?([0-9][\w.+-]*)"), lines)
    if v:
        out["version"] = v.group(1)
    text = "\n".join(lines)
    i = text.find("LLM Args:")
    if i >= 0:
        dump = text[i: i + 20000]
        args = {}
        for key in ("backend", "model", "dtype", "max_batch_size", "max_num_tokens", "max_seq_len", "max_input_len",
                    "max_beam_width", "attn_backend", "tensor_parallel_size", "enable_chunked_prefill",
                    "enable_attention_dp", "disable_overlap_scheduler"):
            m = re.search(rf"\b{key}=('[^']*'|[^\s,)]+)", dump)
            if m:
                args[key] = _scalar(m.group(1))
        m = re.search(r"\bcuda_graph_config=(None|CudaGraphConfig\()", dump)
        if m:
            args["cuda_graphs"] = m.group(1) != "None"
        m = re.search(r"kv_cache_config=KvCacheConfig\((.*?)\)", dump, re.DOTALL)
        if m:
            kv = {k: _scalar(v) for k, v in re.findall(r"\b(dtype|free_gpu_memory_fraction|max_tokens|"
                                                        r"enable_block_reuse)=('[^']*'|[^\s,)]+)", m.group(1))}
            if kv:
                args["kv_cache_config"] = kv
        s["llm_args"] = args
        if args:
            found.append("llm_args")
    flags = {}
    if "cuda_graphs" in (s.get("llm_args") or {}):
        flags["cuda_graphs"] = s["llm_args"]["cuda_graphs"]
    backend = (s.get("llm_args") or {}).get("attn_backend")
    out["_gates"] = ("cmd", flags, str(backend) if backend else None)
    out["status"] = "RESOLVED" if "llm_args" in found and out.get("version") else "PARTIAL" if found or \
        out.get("version") else "UNPROVEN"


_PARSERS = {"vllm": _vllm, "sglang": _sglang, "llamacpp": _llamacpp, "trtllm": _trtllm}


STATUS_RANK = {"RESOLVED": 0, "PARTIAL": 1, "SELF_REPORTED": 2, "UNPROVEN": 3}


def parse_resolved_config(kind: str, log_text: str) -> dict:
    """Settings an engine reported it resolved, from its server log.

    Returns ``{"engine_kind", "status": RESOLVED | PARTIAL | SELF_REPORTED | UNPROVEN, "version",
    "settings", "matched", "validity", "warnings", "errors", "unknown_settings", "self_reported"}``.
    RESOLVED means the engine's own configuration dump was found; PARTIAL, only some of
    the lines we look for; SELF_REPORTED, only an ``inferencekit_resolved_config`` line from a
    generic (custom) engine; UNPROVEN, nothing (an unknown engine or a changed log format).
    """
    from inferencekit.engines import gates

    lines = _lines(log_text or "")
    out: dict = {"engine_kind": kind if kind in ENGINE_KINDS else "generic", "status": "UNPROVEN", "version": None,
                 "settings": {}, "matched": [], "validity": None, "unknown_settings": [], "self_reported": []}
    parser = _PARSERS.get(out["engine_kind"])
    if parser is not None:
        parser(lines, out)
    custom = _custom(lines)
    if custom is not None:
        out["settings"]["self_reported"] = custom
        out["self_reported"] = sorted(str(k) for k in custom)
        if parser is None:
            out["matched"].append("self_reported")
            out["status"] = "SELF_REPORTED"
        else:  # a known engine: its own log lines decide; the self-report is only kept
            out["matched"].append("self_reported (ignored: the engine's own log lines decide)")
    g = out.pop("_gates", None)
    if g is not None:
        engine, flags, backend = g
        out["validity"] = gates.check(engine, flags, backend)
    elif out["validity"] is None and custom is not None and parser is None:
        v = gates.check("cmd", custom, custom.get("attention_backend"))
        v["source"] = "self_reported"
        if not v.get("violations"):
            v["valid"] = None  # a self-report can show a violation, never prove there is none
        v["warnings"] = list(v.get("warnings") or []) + ["settings self-reported by the engine process "
                                                         "(inferencekit_resolved_config line): not verified"]
        out["validity"] = v
    out["warnings"] = _collect(lines, _WARN)
    out["errors"] = _collect(lines, _ERR, 50)
    out["unknown_settings"] = sorted(set(out["unknown_settings"]))
    if out["status"] == "UNPROVEN":
        out["reason"] = (f"no resolved-configuration lines recognised for engine kind '{out['engine_kind']}'; "
                         "the raw log is kept")
    elif out["status"] == "SELF_REPORTED":
        out["reason"] = ("settings come only from the engine process's own inferencekit_resolved_config line "
                         "(self-reported, not an engine's configuration dump); the raw log is kept")
    return out
