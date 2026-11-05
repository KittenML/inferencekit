"""Read-only accelerator discovery; no framework import or test kernel is launched.

NVIDIA discovery remains in gpuinfo. AMD reads Linux DRM sysfs, Apple reads
selected system_profiler/sysctl fields. Missing fields stay unknown. No host,
serial, UUID, or raw vendor output is retained.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shlex
from pathlib import Path

from inferencekit.proc import run, which


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _sysctl(key: str) -> str | None:
    result = run(["sysctl", "-n", key], timeout=15)
    return result.stdout.strip() if result.ok else None


def amd_devices(root: Path = Path("/sys/class/drm")) -> list[dict]:
    devices = []
    seen = set()
    for card in sorted(root.glob("card*")):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = card / "device"
        if _read(device / "vendor") != "0x1002":
            continue
        cls = _read(device / "class")
        if cls and not cls.startswith("0x03"):
            continue
        resolved = device.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        name = _read(device / "product_name")
        name_source = "sysfs product_name" if name else None
        pci_id = _read(device / "device")
        if not name and which("lspci"):
            result = run(["lspci", "-s", resolved.name, "-mm"], timeout=15)
            try:
                fields = shlex.split(result.stdout.strip()) if result.ok else []
            except ValueError:
                fields = []
            if len(fields) >= 4:
                name, name_source = fields[3], "lspci PCI database"
        memory = _read(device / "mem_info_vram_total")
        devices.append({"name": name or f"AMD GPU ({pci_id or 'unknown PCI device'})",
                        "name_source": name_source or "PCI id only; product unverified",
                        "backend": "rocm", "memory_bytes": int(memory) if memory and memory.isdigit() else None,
                        "memory_kind": "vram", "pci_device_id": pci_id,
                        "source": "Linux DRM sysfs"})
    return devices


def rocm_devices() -> list[dict]:
    """Fallback for WSL /dev/dxg and hosts without DRM sysfs; CPU agents do not count.

    rocminfo enumerates agents but launches no kernels. Its heap sizes are not
    labelled dedicated VRAM here: under WSL they can describe shared memory.
    """
    executable = which("rocminfo")
    if not executable:
        return []
    result = run([executable], timeout=30)
    if not result.ok:
        return []
    devices = []
    for block in re.split(r"(?m)^\s*Agent\s+\d+\s*$", result.stdout):
        name = re.search(r"(?m)^\s*Name:\s*(gfx[0-9a-f]+)\s*$", block)
        kind = re.search(r"(?m)^\s*Device Type:\s*(\w+)", block)
        if not name or not kind or kind.group(1) != "GPU":
            continue
        marketing = re.search(r"(?m)^\s*Marketing Name:\s*(.+)$", block)
        product = marketing.group(1).strip() if marketing else None
        devices.append({"name": product or f"AMD GPU ({name.group(1)})", "architecture": name.group(1),
                        "name_source": "rocminfo Marketing Name" if product else "rocminfo gfx target only",
                        "backend": "rocm", "memory_bytes": None, "memory_kind": "vram",
                        "source": "rocminfo GPU agent; dedicated VRAM not verified"})
    return devices


def apple_devices() -> list[dict]:
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        return []
    brand, memory = _sysctl("machdep.cpu.brand_string"), _sysctl("hw.memsize")
    if not brand or not brand.startswith("Apple"):
        return []
    result = run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30)
    try:
        rows = json.loads(result.stdout).get("SPDisplaysDataType", []) if result.ok else []
    except (ValueError, AttributeError):
        rows = []
    # On Apple Silicon this is one GPU sharing system memory, not dedicated VRAM.
    gpu = next((row for row in rows if str(row.get("sppci_vendor", "")).lower().startswith("sppci_vendor_apple")), {})
    return [{"name": gpu.get("sppci_model") or brand, "name_source": "system_profiler/sysctl",
             "backend": "metal", "memory_bytes": int(memory) if memory and memory.isdigit() else None,
             "memory_kind": "unified", "source": "macOS system_profiler/sysctl"}]


def discover_non_nvidia() -> dict:
    devices = apple_devices() or amd_devices() or rocm_devices()
    accelerator_interface = any(Path(p).exists() for p in ("/dev/dxg", "/dev/kfd", "/dev/dri"))
    backend = devices[0]["backend"] if devices else "unknown" if accelerator_interface else "cpu"
    return {"platform": backend, "devices": devices,
            "cpu": {"architecture": platform.machine(), "logical_cores": os.cpu_count()},
            "limitations": (["Device identification does not prove runtime or profiler support; verify the exact "
                              "GPU, OS and installed runtime against current official docs and a smoke request."]
                            if devices else ["Accelerator identity is unknown; a device interface exists but vendor "
                                             "discovery failed." if accelerator_interface else
                                             "No supported accelerator detected; a missing driver or unsupported "
                                             "device can also produce this result."])}


def profiler_route(backend: str) -> dict:
    return {
        "nvidia": {"timeline": "nsys", "kernel": "ncu", "documentation": "https://docs.nvidia.com/nsight-systems/"},
        "rocm": {"timeline": "rocprofv3", "kernel": "rocprofv3 counters (verify GPU support)",
                 "documentation": "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html"},
        "metal": {"timeline": "Instruments Metal System Trace", "kernel": "Xcode Metal debugger / MLX capture",
                  "documentation": "https://developer.apple.com/documentation/xcode/optimizing-gpu-performance"},
        "cpu": {"timeline": "perf / Instruments Time Profiler", "kernel": "platform CPU profiler",
                "documentation": "https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html"},
    }.get(backend, {"timeline": None, "kernel": None, "documentation": None})
