"""doctor: what is this machine, and can it profile?

Run it first. Most "my numbers make no sense" problems are environmental: another
process on the GPU, clocks locked low, a profiler that cannot read hardware
counters, or an engine version different from the one you think you have.

Counter access is read from the driver setting and the process's capabilities, never
assumed from the user being root: root inside a container often lacks CAP_SYS_ADMIN, and
then ncu fails with ERR_NVGPUCTRPERM like any other user. ``--probe-profilers`` settles it
by profiling one tiny kernel with ncu and nsys (when they are installed).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.cpuinfo import cpu_facts, cpu_identity
from inferencekit.gpuinfo import clocks_state, driver_cuda_version, query_gpus
from inferencekit.hardware import discover_non_nvidia, profiler_route
from inferencekit.proc import run, tool_version, which
from inferencekit.registry import Arg, operation

ENGINE_PACKAGES = ["vllm", "sglang", "torch", "transformers", "triton", "flashinfer-python", "flash-attn",
                   "xformers", "tensorrt-llm", "llama-cpp-python", "diffusers", "mlx", "mlx-lm", "mlx-audio"]

# Runs inside another interpreter (an engine's venv) to read its package versions.
_VERSION_PROBE = """
import json, sys
from importlib import metadata
out = {}
for name in json.loads(sys.argv[1]):
    try:
        out[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps(out))
"""


def engine_versions(python: str | None) -> dict:
    if not python or os.path.realpath(python) == os.path.realpath(sys.executable):
        found = {}
        for name in ENGINE_PACKAGES:
            try:
                found[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        return found
    try:
        p = subprocess.run([python, "-c", _VERSION_PROBE, json.dumps(ENGINE_PACKAGES)], capture_output=True,
                           text=True, timeout=120)
        return json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return {}


def llamacpp_version() -> dict | None:
    """``llama-cli --version`` prints e.g. 'version: 4589 (1a2b3c4)' (build number and commit)."""
    for name in ("llama-cli", "llama-bench", "llama-server"):
        path = which(name)
        if not path:
            continue
        r = run([path, "--version"], timeout=30)
        m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", r.stdout + r.stderr)
        if m:
            return {"binary": name, "build": m.group(1), "commit": m.group(2)}
    return None


PARAMS_FILE = Path("/proc/driver/nvidia/params")
STATUS_FILE = Path("/proc/self/status")
CAP_SYS_ADMIN = 21  # bit of the CapEff mask in /proc/<pid>/status
PROBE_HINT = "run 'inferencekit doctor --probe-profilers' or one small 'ncu capture' to find out"


def profiling_admin_only(path: Path = PARAMS_FILE) -> int | None:
    """The driver's RmProfilingAdminOnly (1: counters for admins only, 0: for everyone), None if unreadable."""
    try:
        m = re.search(r"RmProfilingAdminOnly:\s*(\d+)", path.read_text())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def has_cap_sys_admin(path: Path = STATUS_FILE) -> bool | None:
    """Whether this process's effective capabilities include CAP_SYS_ADMIN, None if unreadable."""
    try:
        m = re.search(r"^CapEff:\s*([0-9a-fA-F]+)\s*$", path.read_text(), re.M)
    except OSError:
        return None
    return bool(int(m.group(1), 16) >> CAP_SYS_ADMIN & 1) if m else None


def counter_permission(params_path: Path = PARAMS_FILE, status_path: Path = STATUS_FILE,
                       euid: int | None = None) -> dict:
    """Can this process read GPU performance counters (needed by ncu)? Status allowed | admin_only | unknown.

    The NVIDIA driver parameter RmProfilingAdminOnly=1 (the Linux default) restricts counters to admin
    users, which the driver checks as CAP_SYS_ADMIN, not as uid 0; ncu then fails with ERR_NVGPUCTRPERM.
    So the answer follows the capability: only RmProfilingAdminOnly=0 gives ``allowed``; under
    RmProfilingAdminOnly=1 a process without CAP_SYS_ADMIN (a container's root usually) is
    ``admin_only``, and one with it (root on a VM or bare-metal host, or a non-root process granted the
    capability) is ``unknown`` until a capture is tried (``--probe-profilers``): ncu usually works there,
    but a container runtime or hypervisor can still refuse. Without the parameter the answer is
    ``unknown`` too. Nothing is profiled here.
    """
    if euid is None:
        euid = os.geteuid() if hasattr(os, "geteuid") else None
    admin_only = profiling_admin_only(params_path)
    cap = has_cap_sys_admin(status_path)
    facts = {"rm_profiling_admin_only": admin_only, "euid": euid, "cap_sys_admin": cap}
    who = "root" if euid == 0 else "non-root"
    if admin_only == 0:
        return {"status": "allowed", "reason": "RmProfilingAdminOnly=0: the driver lets every user read counters",
                **facts}
    if admin_only is not None:
        if cap:
            return {"status": "unknown", "reason": f"RmProfilingAdminOnly=1 and this {who} process has CAP_SYS_ADMIN, "
                                                   "which the driver asks for: ncu is expected to work, but a "
                                                   f"container or VM can still refuse (probe needed: {PROBE_HINT})",
                    **facts}
        if cap is False and euid == 0:
            reason = ("RmProfilingAdminOnly=1 and this root process lacks CAP_SYS_ADMIN (usual in a container): "
                      "ncu fails with ERR_NVGPUCTRPERM")
        elif cap is False:
            reason = ("RmProfilingAdminOnly=1 and this process lacks CAP_SYS_ADMIN: ncu needs CAP_SYS_ADMIN (root "
                      "on a VM or bare-metal host, or a container started with it), or the driver option "
                      "NVreg_RestrictProfilingToAdminUsers=0")
        elif euid == 0:
            return {"status": "unknown", "reason": "RmProfilingAdminOnly=1 and this process's capabilities are "
                                                   f"unreadable, so CAP_SYS_ADMIN is unknown (probe needed: "
                                                   f"{PROBE_HINT})", **facts}
        else:
            reason = ("RmProfilingAdminOnly=1, and this non-root process's capabilities are unreadable: ncu "
                      "needs CAP_SYS_ADMIN, or the driver option NVreg_RestrictProfilingToAdminUsers=0")
        return {"status": "admin_only", "reason": reason, **facts}
    if euid == 0 and cap is False:
        return {"status": "admin_only", "reason": "root without CAP_SYS_ADMIN and no readable RmProfilingAdminOnly "
                                                  "(drivers default it to 1): ncu is expected to fail with "
                                                  "ERR_NVGPUCTRPERM", **facts}
    ncu = which("ncu")
    if ncu:
        r = run([ncu, "--query-metrics"], timeout=120)
        if "ERR_NVGPUCTRPERM" in r.stdout + r.stderr:
            return {"status": "admin_only", "reason": "ncu --query-metrics reported ERR_NVGPUCTRPERM", **facts}
        if r.ok:
            return {"status": "unknown", "reason": "no readable driver parameter; ncu can enumerate metrics but "
                                                   f"counter access is untested (probe needed: {PROBE_HINT})", **facts}
    return {"status": "unknown", "reason": f"no readable /proc/driver/nvidia/params and no ncu (probe needed: "
                                           f"{PROBE_HINT})", **facts}


# --------------------------------------------------------------------------- --probe-profilers

# Programs that launch one small CUDA kernel. First choice: the standard library only. ctypes loads the
# driver (libcuda.so.1, present on every NVIDIA host), the driver compiles a PTX kernel for whatever GPU this is
# (PTX is the portable assembly that CUDA compiles to), and the kernel doubles 1M floats. Fallbacks: torch in
# the engine's interpreter, then a .cu built with nvcc.
_DRIVER_PROGRAM = r'''import ctypes, sys
PTX = b"""
.version 6.0
.target sm_50
.address_size 64
.visible .entry twice(.param .u64 x, .param .u32 n)
{
    .reg .pred p;
    .reg .b32 r<5>;
    .reg .f32 f<3>;
    .reg .b64 rd<5>;
    ld.param.u64 rd1, [x];
    ld.param.u32 r1, [n];
    mov.u32 r2, %ctaid.x;
    mov.u32 r3, %ntid.x;
    mov.u32 r4, %tid.x;
    mad.lo.s32 r2, r2, r3, r4;
    setp.ge.u32 p, r2, r1;
    @p bra DONE;
    cvta.to.global.u64 rd2, rd1;
    mul.wide.u32 rd3, r2, 4;
    add.s64 rd4, rd2, rd3;
    ld.global.f32 f1, [rd4];
    add.f32 f2, f1, f1;
    st.global.f32 [rd4], f2;
DONE:
    ret;
}
"""
cuda = ctypes.CDLL("libcuda.so.1")
def check(rc, what):
    if rc:
        name = ctypes.c_char_p()
        cuda.cuGetErrorName(rc, ctypes.byref(name))
        sys.exit(what + ": " + (name.value or b"error").decode())
n = 1 << 20
dev, ctx, mod, fn, ptr = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_uint64()
check(cuda.cuInit(0), "cuInit")
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
check(cuda.cuModuleLoadData(ctypes.byref(mod), PTX), "cuModuleLoadData")
check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"twice"), "cuModuleGetFunction")
check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(4 * n)), "cuMemAlloc")
check(cuda.cuMemsetD32_v2(ptr, ctypes.c_uint(0x3F800000), ctypes.c_size_t(n)), "cuMemsetD32")
count = ctypes.c_uint32(n)
args = (ctypes.c_void_p * 2)(ctypes.cast(ctypes.byref(ptr), ctypes.c_void_p),
                             ctypes.cast(ctypes.byref(count), ctypes.c_void_p))
u = ctypes.c_uint
check(cuda.cuLaunchKernel(fn, u((n + 255) // 256), u(1), u(1), u(256), u(1), u(1), u(0), None, args, None),
      "cuLaunchKernel")
check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
out = ctypes.c_float()
check(cuda.cuMemcpyDtoH_v2(ctypes.byref(out), ptr, ctypes.c_size_t(4)), "cuMemcpyDtoH")
check(cuda.cuMemFree_v2(ptr), "cuMemFree")
if out.value != 2.0:
    sys.exit("the kernel wrote %r, expected 2.0" % out.value)
print("ok")
'''
_TORCH_PROGRAM = ("import torch\n"
                  "x = torch.ones(1 << 20, device='cuda')\n"
                  "y = (x * 2.0).sum()\n"
                  "torch.cuda.synchronize()\n"
                  "print(float(y))\n")
_CU_PROGRAM = r"""#include <cstdio>
__global__ void twice(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 2.0f * x[i];
}
int main() {
    const int n = 1 << 20;
    float *x = nullptr;
    if (cudaMalloc(&x, n * sizeof(float)) != cudaSuccess) { printf("cudaMalloc failed\n"); return 2; }
    twice<<<(n + 255) / 256, 256>>>(x, n);
    cudaError_t e = cudaDeviceSynchronize();
    cudaFree(x);
    if (e != cudaSuccess) { printf("%s\n", cudaGetErrorString(e)); return 3; }
    printf("ok\n");
    return 0;
}
"""
NCU_METRIC = "sm__cycles_elapsed.avg"
PROBE_TIMEOUT_S = 300


def _tail(r, n: int = 300) -> str:
    lines = [ln.strip() for ln in (r.stdout + "\n" + r.stderr).splitlines() if ln.strip()]
    errs = [ln for ln in lines if "ERR_" in ln or "==ERROR==" in ln or "error" in ln.lower()]
    return ev.redact_text(" | ".join((errs or lines)[-3:]))[:n] or f"exit {r.returncode}"


def _probe_program(python: str | None, work: Path) -> tuple[list[str] | None, str]:
    """(argv of a program that runs one small CUDA kernel and exits 0, what it is); argv None when no
    candidate ran without a profiler (then the profilers are not tried: a failure would say nothing)."""
    tried = []
    driver = work / "probe_kernel.py"
    driver.write_text(_DRIVER_PROGRAM)
    argv = [sys.executable, "-I", str(driver)]
    r = run(argv, timeout=PROBE_TIMEOUT_S)
    if r.ok:
        return argv, "one PTX kernel launched through the CUDA driver API (Python standard library only)"
    tried.append(f"driver API program: {_tail(r, 160)}")
    py = python or sys.executable
    r = run([py, "-c", _TORCH_PROGRAM], timeout=PROBE_TIMEOUT_S)
    if r.ok:
        return [py, "-c", _TORCH_PROGRAM], f"torch in {Path(py).name}: one elementwise kernel and a reduction"
    tried.append(f"torch program: {_tail(r, 160)}")
    why = "; ".join(tried)
    nvcc = which("nvcc")
    if not nvcc:
        return None, f"no CUDA program could run ({why}; no nvcc to build one)"
    src, exe = work / "probe.cu", work / "probe"
    src.write_text(_CU_PROGRAM)
    b = run([nvcc, "-o", str(exe), str(src)], timeout=PROBE_TIMEOUT_S)
    if not b.ok:
        return None, f"no CUDA program could run ({why}; nvcc build failed: {_tail(b, 160)})"
    r = run([str(exe)], timeout=PROBE_TIMEOUT_S)
    if not r.ok:
        return None, f"no CUDA program could run ({why}; the nvcc-built kernel failed: {_tail(r, 160)})"
    return [str(exe)], "a one-kernel CUDA program built with nvcc"


def _probe_ncu(ncu: str, program: list[str]) -> dict:
    t0 = time.monotonic()
    r = run([ncu, "--launch-count", "1", "--metrics", NCU_METRIC, *program], timeout=PROBE_TIMEOUT_S)
    out = r.stdout + r.stderr
    res = {"ran": True, "returncode": r.returncode, "seconds": round(time.monotonic() - t0, 2)}
    if "ERR_NVGPUCTRPERM" in out:
        return {**res, "capture_ok": False, "error": "ERR_NVGPUCTRPERM: no permission to read GPU counters"}
    value = re.search(re.escape(NCU_METRIC) + r"\s+\S+\s+([\d.,]+)", out)
    if r.ok and value:
        return {**res, "capture_ok": True, "error": None, "metric": NCU_METRIC,
                "value": float(value.group(1).replace(",", ""))}
    if r.ok and NCU_METRIC in out and "No kernels were profiled" not in out:
        return {**res, "capture_ok": True, "error": None, "metric": NCU_METRIC, "value": None}
    return {**res, "capture_ok": False, "error": _tail(r)}


def _probe_nsys(nsys: str, program: list[str], work: Path) -> dict:
    t0 = time.monotonic()
    out = work / "probe"
    r = run([nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
             "-o", str(out), *program], timeout=PROBE_TIMEOUT_S)
    res = {"ran": True, "returncode": r.returncode, "seconds": round(time.monotonic() - t0, 2)}
    report = out.with_suffix(".nsys-rep")
    if r.ok and report.is_file():
        return {**res, "capture_ok": True, "error": None, "report_bytes": report.stat().st_size}
    if r.ok:
        return {**res, "capture_ok": False, "error": "nsys exited 0 but wrote no .nsys-rep report"}
    return {**res, "capture_ok": False, "error": _tail(r)}


def profiler_probe(python: str | None) -> dict:
    """Profile one tiny kernel with ncu (one counter) and nsys (a CUDA trace), when each is installed.

    ``capture_ok`` per tool: True when the capture worked, False with ``error`` when it failed, None when
    it was not tried (tool missing, or no CUDA program could run). Never raises: a probe failure is a finding.
    """
    tools = {"ncu": which("ncu"), "nsys": which("nsys")}
    out: dict = {"program": None, "ncu": None, "nsys": None}
    if not any(tools.values()):
        for name in tools:
            out[name] = {"ran": False, "capture_ok": None, "error": f"{name} not found"}
        return out
    try:
        work = ev.Store().staging()  # under the evidence store, removed below: nothing of the probe is kept
    except OSError as e:
        for name in tools:
            out[name] = {"ran": False, "capture_ok": None, "error": f"no scratch directory for the probe: {e}"[:300]}
        return out
    try:
        program, how = _probe_program(python, work)
        out["program" if program else "program_error"] = how
        for name, path in tools.items():
            if not path:
                out[name] = {"ran": False, "capture_ok": None, "error": f"{name} not found"}
            elif program is None:
                out[name] = {"ran": False, "capture_ok": None, "error": how}
            else:
                try:
                    out[name] = _probe_ncu(path, program) if name == "ncu" else _probe_nsys(path, program, work)
                except Exception as e:  # noqa: BLE001 -- the probe must never fail doctor
                    out[name] = {"ran": True, "capture_ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def gpu_processes() -> list[dict]:
    """Other compute processes on the GPU: they steal bandwidth and skew timings."""
    smi = which("nvidia-smi")
    if not smi:
        return []
    r = run([smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], timeout=30)
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            out.append({"pid": parts[0], "name": Path(parts[1]).name, "used_mib": parts[2]})
    return out


@operation("doctor", ("doctor",), "Hardware identity and profiling route (NVIDIA, AMD ROCm, Apple Metal, CPU); "
           "selected AMD/Apple OS facts; NVIDIA driver, CUDA, clocks, nsys/ncu and profiling-counter permission "
           "(driver setting and CAP_SYS_ADMIN, never assumed from root), engine versions; optionally a tiny ncu and "
           "nsys test capture.",
           [Arg("python", "string", "Interpreter of the engine's environment, to read its package versions "
                "(default: this interpreter). --probe-profilers falls back to its torch when the standard-library "
                "test program cannot run."),
            Arg("probe_profilers", "boolean", "Also profile one tiny CUDA kernel (a standard-library driver-API "
                "program; falls back to torch in --python, then nvcc) with ncu (--launch-count 1, one counter) and "
                "with nsys, when installed, and record whether each capture worked. Each step stops after "
                f"{PROBE_TIMEOUT_S // 60} minutes; a failed capture never fails doctor.", default=False)])
def doctor(call, python, probe_profilers=False):
    gpus = query_gpus()
    hardware = None if gpus else discover_non_nvidia()
    if hardware and (hardware["devices"] or hardware["platform"] == "unknown"):
        return _vendor_doctor(call, python, probe_profilers, hardware)
    if hardware and hardware["platform"] == "cpu":
        return _cpu_doctor(call, python, probe_profilers, hardware)
    limitations, missing = [], []
    if not gpus:
        missing.append("gpu")
        limitations.append("nvidia-smi not found or no GPU visible: no GPU facts recorded")
    tools = {name: tool_version(name) for name in ("nvcc", "nsys", "ncu")}
    for name, v in tools.items():
        if v is None:
            missing.append(name)
    perm = counter_permission()
    probe = profiler_probe(python) if probe_profilers else None
    if probe is not None:
        ncu = probe.get("ncu") or {}
        if ncu.get("capture_ok"):
            perm = {**perm, "status": "allowed", "reason": f"ncu read {NCU_METRIC} on a test kernel "
                                                           "(--probe-profilers)", "static_reason": perm["reason"]}
        elif ncu.get("capture_ok") is False and "ERR_NVGPUCTRPERM" in (ncu.get("error") or ""):
            perm = {**perm, "status": "admin_only", "reason": "ncu test capture failed with ERR_NVGPUCTRPERM "
                                                              "(--probe-profilers)", "static_reason": perm["reason"]}
        for name in ("ncu", "nsys"):
            r = probe.get(name) or {}
            if r.get("capture_ok") is False:
                limitations.append(f"{name} test capture failed: {r.get('error')}")
            elif r.get("capture_ok") is None and tools.get(name):
                limitations.append(f"{name} test capture not tried: {r.get('error')}")
    if perm["status"] == "admin_only":
        limitations.append(f"profiling counters are admin-only: 'ncu capture' fails without admin rights "
                           f"(CAP_SYS_ADMIN); {perm['reason']}")
    elif perm["status"] == "unknown":
        limitations.append(f"profiling counter access unknown: {perm['reason']}")
    engines = engine_versions(python)
    llama = llamacpp_version()
    procs = gpu_processes()
    if procs:
        limitations.append(f"{len(procs)} compute process(es) already on the GPU: timings will be contended")
    clocks = clocks_state(gpus[0]) if gpus else None
    if clocks == "default":
        limitations.append("clock lock state read from application clocks; locks set with 'nvidia-smi -lgc' are "
                           "not visible on every driver")
    gpu_rows = [{k: g.get(k) for k in ("index", "name", "memory.total", "compute_cap", "pstate", "persistence_mode",
                                        "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem", "power.limit",
                                        "temperature.gpu")} for g in gpus]
    ident = ev.identity(gpu={"name": gpus[0].get("name") if gpus else None, "count": len(gpus) or None,
                             "driver": gpus[0].get("driver_version") if gpus else None,
                             "cuda": driver_cuda_version(), "clocks": clocks})
    name = gpus[0]["name"] if gpus else "no GPU"
    counters = perm["status"] + (" (probe needed)" if perm["status"] == "unknown" else "")
    summary = (f"{len(gpus)}x {name}; nsys {tools['nsys'] or 'missing'}, ncu {tools['ncu'] or 'missing'}; "
               f"counters {counters}; engines: {', '.join(f'{k} {v}' for k, v in engines.items()) or 'none'}")
    if probe is not None:
        words = {True: "ok", False: "FAILED", None: "not tried"}
        summary += "; test capture: " + ", ".join(f"{t} {words[(probe.get(t) or {}).get('capture_ok')]}"
                                                  for t in ("ncu", "nsys"))
    data = {"hardware": hardware or {"platform": "nvidia", "devices": gpu_rows},
            "profiling_route": profiler_route("nvidia" if gpus else "cpu"), "gpus": gpu_rows, "tools": tools, "nvcc_toolkit": tools["nvcc"],
            "profiling_counters": perm, "engines": engines, "llamacpp": llama,
            "gpu_processes": procs,
            # names only: which relevant variables are set, never their values
            "env_set": sorted(k for k in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "HF_HOME",
                                          "VLLM_ATTENTION_BACKEND", "HF_TOKEN") if k in os.environ)}
    if probe is not None:
        data["profiler_probe"] = probe
    record = ev.build(call, kind="environment", claim_class="observed", summary=summary, identity_=ident,
                      metrics=[ev.metric("gpu_count", len(gpus), "count", "observed")], missing=missing,
                      limitations=limitations, data=data)
    return ev.Store().save(record)


def _cpu_doctor(call, python, probe_profilers, hardware):
    facts = hardware.get("cpu") or cpu_facts()
    hardware = {**hardware, "cpu": facts}
    limitations = list(hardware.get("limitations", [])) + list(facts.get("limitations", []))
    limitations += ["NVIDIA-only ceiling bw and nsys/ncu capture are NOT RUN on CPU.",
                    "CPU profiler permission is inferred from policy only; containers and security policy may still block perf.",
                    facts["library_dispatch"]["reason"]]
    engines = engine_versions(python)
    # Package metadata is a presence/version observation, never a framework import or dispatch probe.
    libraries = {}
    for name in ("numpy", "scipy", "mkl", "intel-openmp", "onednn"):
        if not python or os.path.realpath(python) == os.path.realpath(sys.executable):
            try:
                libraries[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
    facts["library_dispatch"]["package_versions"] = libraries
    data = {"hardware": hardware, "gpus": [], "tools": {"perf": tool_version("perf")},
            "profiling_route": profiler_route("cpu", facts),
            "profiling_counters": {"status": "not_applicable", "reason": "NVIDIA counters do not apply; CPU perf policy is in hardware.cpu.perf"},
            "nvcc_toolkit": None, "engines": engines, "llamacpp": llamacpp_version(), "gpu_processes": [],
            "env_set": sorted(facts.get("tuning_env", {}))}
    if probe_profilers:
        data["profiler_probe"] = {"program": None, "ncu": {"ran": False, "capture_ok": None, "error": "NVIDIA-only probe"},
                                  "nsys": {"ran": False, "capture_ok": None, "error": "NVIDIA-only probe"}}
        limitations.append("--probe-profilers is NVIDIA-only; no CPU workload or capture was launched.")
    record = ev.build(call, kind="environment", claim_class="observed",
                      summary=f"CPU {facts.get('model') or facts['architecture']}; {facts.get('physical_cores')} physical / "
                              f"{facts.get('logical_cores')} logical cores; perf policy {facts['perf']['access']}; dispatch unverified",
                      identity_=ev.identity(cpu=cpu_identity(facts)),
                      metrics=[ev.metric("cpu_logical_cores", facts.get("logical_cores"), "count", "observed")],
                      missing=["CPU runtime library dispatch", "verified CPU profiler permission"],
                      limitations=limitations, data=data)
    return ev.Store().save(record)


def _vendor_doctor(call, python, probe_profilers, hardware):
    """Report observed vendor facts without running CUDA-only permission/probe code."""
    backend, devices = hardware["platform"], hardware["devices"]
    names = ("rocprofv3", "amd-smi", "hipcc") if backend == "rocm" else ("xcrun",) if backend == "metal" else ()
    tools = {name: tool_version(name) for name in names}
    limitations = list(hardware["limitations"])
    limitations += ["NVIDIA-only ceiling bw and nsys/ncu capture are NOT RUN on this platform.",
                    "Other accelerator processes and clock state are not automatically checked here; inspect "
                    "vendor tools and record an uncontended device before timing."]
    probe = None
    if probe_profilers:
        probe = {"program": None, "ncu": {"ran": False, "capture_ok": None, "error": "NVIDIA-only probe"},
                 "nsys": {"ran": False, "capture_ok": None, "error": "NVIDIA-only probe"}}
        limitations.append("--probe-profilers is NVIDIA-only; vendor capture support is NOT RUN.")
    data = {"gpus": devices, "hardware": hardware, "tools": tools, "nvcc_toolkit": None,
            "profiling_route": profiler_route(backend),
            "profiling_counters": {"status": "not_applicable", "reason": "NVIDIA permission checks do not "
                                   "apply; vendor counter support/permission is untested"},
            "engines": engine_versions(python), "llamacpp": llamacpp_version(), "gpu_processes": [],
            "env_set": sorted(k for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                          "ROCM_PATH", "MTL_CAPTURE_ENABLED", "HF_HOME", "HF_TOKEN")
                              if k in os.environ)}
    if probe is not None:
        data["profiler_probe"] = probe
    record = ev.build(call, kind="environment", claim_class="observed",
                      summary=(f"{len(devices)}x {devices[0]['name']}" if devices else "unknown accelerator count/identity") + "; "
                              + f"platform {backend}; vendor profiling support untested",
                      identity_=ev.identity(gpu={"name": devices[0]["name"] if devices else None, "count": len(devices) or None,
                                                "driver": None, "cuda": None, "clocks": None}),
                      metrics=[ev.metric("gpu_count", len(devices), "count", "observed")] if devices else [],
                      missing=["profiling counter permission", "gpu processes", "gpu clocks"],
                      limitations=limitations, data=data)
    return ev.Store().save(record)
