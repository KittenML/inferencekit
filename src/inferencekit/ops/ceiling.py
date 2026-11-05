"""ceiling bw: measure the DRAM bandwidth this GPU actually sustains.

Why measure: rooflines built on the datasheet number overstate the ceiling, and
"% of peak" for a kernel is only meaningful against what a pure streaming kernel
achieves on the same card, same clocks, same driver.

Runner: compile ``cuda/bw_peak.cu`` with nvcc (cached by source hash and GPU arch)
when nvcc is present; otherwise run the PyTorch fallback. Both print JSON lines.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.canonical import sha256_hex
from inferencekit.gpuinfo import gpu_identity, query_gpus
from inferencekit.proc import run, tool_version, which
from inferencekit.registry import Arg, OpError, operation

CUDA_DIR = Path(__file__).resolve().parent.parent / "cuda"
SUSTAINED_MIN_BYTES = 256 << 20  # "sustained" = streams far larger than any L2


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "inferencekit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_binary(nvcc: str, compute_cap: str | None) -> tuple[Path, list[str]]:
    """Compile bw_peak.cu once per (source version, GPU architecture); reuse it afterwards."""
    src = CUDA_DIR / "bw_peak.cu"
    arch_tag = "sm" + (compute_cap or "unknown").replace(".", "")
    out = _cache_dir() / f"bw_peak-{sha256_hex(src.read_bytes())[:12]}-{arch_tag}"
    if out.is_file():
        return out, []
    notes = []
    # -arch=native compiles for the GPU in this machine (nvcc 11.6+); fall back to a generic build.
    for arch in (["-arch=native"], []):
        r = run([nvcc, "-O3", "-std=c++17", *arch, "-o", str(out), str(src)], timeout=600)
        if r.ok:
            return out, notes
        notes.append(f"nvcc {' '.join(arch) or '(default arch)'} failed: {r.stderr.strip()[-300:]}")
    raise OpError("could not compile bw_peak.cu: " + " | ".join(notes))


def parse_lines(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


@operation("measure_bandwidth", ("ceiling", "bw"), "Measured DRAM read bandwidth (CUDA microbenchmark over "
           "several buffer sizes; PyTorch fallback without nvcc).",
           [Arg("sizes_mib", "array", "Buffer sizes in MiB.", default=[8, 32, 128, 512, 1024], items="integer"),
            Arg("graph_mib", "array", "Per-launch sizes (MiB) for the graph-of-launches test.", default=[1, 4, 16],
                items="integer"),
            Arg("iters", "integer", "Timed iterations per test (median reported).", default=30),
            Arg("runner", "string", "Force a runner.", default="auto", choices=("auto", "cuda", "torch")),
            Arg("python", "string", "Interpreter with torch, for the fallback runner (default: this one).")])
def measure_bandwidth(call, sizes_mib, graph_mib, iters, runner, python):
    store = ev.Store()
    with store.staging_dir() as staging:  # removed again if the benchmark fails
        limitations, missing = [], []
        nvcc = which("nvcc")
        sizes = ",".join(str(int(s)) for s in sizes_mib)
        if runner == "cuda" or (runner == "auto" and nvcc):
            if not nvcc:
                raise OpError("runner=cuda needs nvcc on PATH")
            gpus = query_gpus()
            binary, notes = build_binary(nvcc, gpus[0].get("compute_cap") if gpus else None)
            limitations += notes
            argv = [str(binary), "--sizes-mib", sizes, "--graph-mib", ",".join(str(int(s)) for s in graph_mib),
                    "--iters", str(iters)]
            method = {"runner": "cuda", "nvcc": tool_version("nvcc"), "source": "cuda/bw_peak.cu"}
        else:
            argv = [python or sys.executable, str(CUDA_DIR / "torch_bw.py"), "--sizes-mib", sizes,
                    "--iters", str(iters)]
            method = {"runner": "torch"}
            limitations.append("PyTorch fallback: reductions read slightly below a pure load loop; no graph test")
            missing.append("graph_read")
        out_path = staging / "bw_peak.jsonl"
        r = run(argv, timeout=1800, stdout_path=out_path, stderr_path=staging / "bw_peak.stderr.txt")
        rows = parse_lines(r.stdout)
        if not r.ok or not any(row.get("test") == "read" for row in rows):
            raise OpError(f"bandwidth benchmark failed (exit {r.returncode}): {r.stderr.strip()[-500:]}")

        device = next((row for row in rows if row.get("test") == "device"), {})
        reads = [row for row in rows if row.get("test") == "read"]
        copies = [row for row in rows if row.get("test") == "copy"]
        graphs = [row for row in rows if row.get("test") == "graph_read"]
        sustained = [row for row in reads if row["bytes"] >= SUSTAINED_MIN_BYTES] or reads
        best = max(sustained, key=lambda row: row["gbs"])
        metrics = [ev.metric("dram_read_gbs", best["gbs"], "GB/s", "best_sustained", n=best.get("iters"),
                             subject=f"{best['bytes'] >> 20} MiB read")]
        # one row per size: the better of the launch configurations
        for size in sorted({row["bytes"] for row in reads}):
            row = max((x for x in reads if x["bytes"] == size), key=lambda x: x["gbs"])
            metrics.append(ev.metric("dram_read_gbs", row["gbs"], "GB/s", "median", n=row.get("iters"),
                                     subject=f"{size >> 20} MiB"))
        # Same rule as reads: only buffers well past the L2 cache measure DRAM. A small copy can sit in
        # L2 (96 MB on some consumer GPUs) and report more than the DRAM can physically deliver.
        copies = [row for row in copies if row["bytes"] >= SUSTAINED_MIN_BYTES]
        if copies:
            c = max(copies, key=lambda x: x["gbs"])
            metrics.append(ev.metric("dram_copy_gbs", c["gbs"], "GB/s", "best_median", n=c.get("iters"),
                                     subject=f"{c['bytes'] >> 20} MiB read+write"))
        theoretical = device.get("theoretical_gbs") or None
        if theoretical:
            metrics.append(ev.metric("theoretical_gbs", theoretical, "GB/s", "derived",
                                     subject="2 x memory clock x bus width"))
        ident = ev.identity(gpu=gpu_identity())
        if ident["gpu"]["clocks"] != "locked":
            limitations.append("clocks not locked: bandwidth can vary a few percent with memory clock boost")
        pct = f" ({100 * best['gbs'] / theoretical:.0f}% of theoretical {theoretical:.0f})" if theoretical else ""
        summary = f"{device.get('name', 'GPU')}: best sustained read {best['gbs']:.0f} GB/s{pct}"
        record = ev.build(call, kind="ceiling", claim_class="observed", summary=summary, identity_=ident,
                          metrics=metrics,
                          artifacts=[ev.artifact(out_path, staging, "raw_output")],
                          missing=missing, limitations=limitations,
                          data={"device": device, "method": method, "reads": reads, "copies": copies,
                                "graph_reads": graphs,
                                "definition": "best sustained = highest median GB/s over read tests of >= 256 MiB"})
        return store.save(record, staging)
