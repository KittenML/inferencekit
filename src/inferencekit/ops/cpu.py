"""Prepare a CPU baseline thread/placement screen; never launch it."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from inferencekit.cpuinfo import cpu_facts, cpu_identity, cpu_list
from inferencekit.ops.manifest import check_manifest, load_manifest
from inferencekit.registry import Arg, OpError, operation

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def placement_order(facts, selected, spread=False, include_smt=False):
    rows = [r for r in facts.get("topology", []) if r["cpu"] in selected]
    if len(rows) != len(selected) or any(r.get("package") is None or r.get("core") is None for r in rows):
        raise OpError("CPU topology is incomplete for the selected CPUs; pinning needs verified physical cores")
    by_core = {}
    for r in sorted(rows, key=lambda r: r["cpu"]):
        by_core.setdefault((r["package"], r["core"]), []).append(r)
    # Each physical core contributes once before SMT siblings. Spread round-robins
    # observed LLC/NUMA/core-type groups; it does not invent P/E labels.
    groups = {}
    for key, siblings in by_core.items():
        r = siblings[0]
        l3 = next((i for i, c in enumerate(facts.get("l3_groups", [])) if r["cpu"] in c["cpus"]), -1)
        node = next((n["node"] for n in facts.get("numa_nodes", []) if r["cpu"] in n["cpus"]), -1)
        group = (r["package"], node, l3, r.get("core_type") or 0)
        groups.setdefault(group, []).append(siblings)
    ordered = [groups[k] for k in sorted(groups)]
    cores = ([g[i] for i in range(max(map(len, ordered))) for g in ordered if i < len(g)]
             if spread else [core for g in ordered for core in g])
    result = [core[0]["cpu"] for core in cores]
    if include_smt:
        result += [r["cpu"] for core in cores for r in core[1:]]
    return result


def screen_manifest(manifest, *, facts, threads, pinning, thread_env=(), cpus=None, include_smt=False):
    """Pure manifest expansion. Engine knobs change only through explicit placeholders/env names."""
    if len(manifest.get("engines", [])) != 1:
        raise OpError("CPU screen needs exactly one baseline engine")
    if not threads or any(type(n) is not int or n < 1 for n in threads):
        raise OpError("threads must be positive integers")
    if not pinning or set(pinning) - {"none", "compact", "spread"}:
        raise OpError("pinning must contain none, compact, or spread")
    if set(thread_env) - set(THREAD_ENV):
        raise OpError(f"thread_env must use only {list(THREAD_ENV)}")
    baseline = manifest["engines"][0]
    argv = baseline.get("launch", [])
    if not any("{threads}" in arg for arg in argv) and not thread_env:
        raise OpError("baseline launch needs {threads}, or specify --thread-env for a supported runtime")
    if not any("{port}" in arg for arg in argv):
        raise OpError("baseline launch needs {port} so each variant can use its own port")
    if not argv or Path(argv[0]).name in ("sh", "bash", "dash", "zsh", "taskset", "numactl"):
        raise OpError("baseline launch must be direct engine argv without shell or existing placement wrapper")
    allowed = facts.get("allowed_cpus")
    if not allowed:
        raise OpError("process CPU affinity is unknown; cannot prepare a placement screen")
    selected = sorted(set(cpus if cpus is not None else allowed))
    if not selected or set(selected) - set(allowed):
        raise OpError("selected CPUs must be a nonempty subset of the process's allowed CPUs")
    counts, policies = sorted(set(threads)), list(dict.fromkeys(pinning))
    if max(counts) > len(selected):
        raise OpError("thread count exceeds selected logical CPUs")
    result = copy.deepcopy(manifest)
    result["hardware"] = {**{k: v for k, v in manifest.get("hardware", {}).items() if k != "gpu"},
                          "backend": "cpu", "cpu": cpu_identity(facts), "count": 1}
    result["engines"] = []
    plan = []
    for policy in policies:
        order = placement_order(facts, selected, policy == "spread", include_smt) if policy != "none" else selected
        for n in counts:
            if n > len(order):
                raise OpError("thread count exceeds selected physical cores; use --include-smt to screen siblings")
            port = baseline["port"] + len(plan)
            if port > 65535:
                raise OpError("variant ports exceed 65535; choose a lower baseline port")
            label = f"{baseline['label'][:40]}-t{n}-{policy}"
            engine = copy.deepcopy(baseline)
            engine.update(label=label, port=port)
            engine.pop("resolved_config", None)  # baseline's observed settings are not evidence for a new variant
            launch = [arg.replace("{threads}", str(n)).replace("{port}", str(port)) for arg in argv]
            mask = order[:n] if policy != "none" else selected
            # Even the no-binding case is contained in the selected engine pool;
            # runtime workers may migrate within it, leaving reserved client CPUs free.
            engine["launch"] = ["taskset", "-c", ",".join(map(str, mask)), *launch]
            engine["env"] = {**engine.get("env", {}), **{k: str(n) for k in thread_env},
                             "OMP_PLACES": "cores", "OMP_PROC_BIND": {"none": "false", "compact": "close", "spread": "spread"}[policy]}
            result["engines"].append(engine)
            plan.append({"label": label, "threads": n, "pinning": policy, "cpus": mask, "port": port})
    checked = check_manifest(result)
    if not checked["valid"]:
        raise OpError("invalid CPU screen manifest: " + "; ".join(checked["errors"]))
    return result, plan


@operation("cpu_screen_manifest", ("cpu", "screen-manifest"),
           "Prepare thread x pinning variants of one CPU baseline; writes a manifest, launches nothing.",
           [Arg("manifest", "string", "Single-engine manifest; launch uses {threads} and {port} placeholders.", required=True, positional=True),
            Arg("out", "string", "New JSON manifest beside the source; refuses overwrite.", required=True),
            Arg("threads", "array", "Thread counts to screen; repeat --threads for each count.", required=True, items="integer"),
            Arg("pinning", "array", "Placement policies.", default=["none", "compact", "spread"], choices=("none", "compact", "spread")),
            Arg("thread_env", "array", "Explicit runtime thread variables to set; launch {threads} is an alternative.", default=[], choices=THREAD_ENV),
            Arg("cpus", "string", "Engine CPU pool, Linux CPU list; leave client cores outside it."),
            Arg("include_smt", "boolean", "Allow sibling threads after one thread per physical core.", default=False)])
def cpu_screen_manifest(call, manifest, out, threads, pinning, thread_env, cpus, include_smt):
    target, source = Path(out), Path(manifest)
    if target.exists():
        raise OpError("output already exists; choose a new path")
    if target.resolve().parent != source.resolve().parent:
        raise OpError("write the screen manifest beside the source to preserve relative paths")
    selected = cpu_list(cpus) if cpus is not None else None
    if cpus is not None and not selected:
        raise OpError("cpus must be a valid nonempty Linux CPU list")
    result, plan = screen_manifest(load_manifest(source), facts=cpu_facts(), threads=threads, pinning=pinning,
                                   thread_env=thread_env, cpus=selected, include_smt=include_smt)
    with target.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    return {"path": str(target), "variants": plan, "ran": False,
            "limitations": ["Screen candidates only: verify actual threads, affinity and dispatch in engine logs; "
                             "select the baseline for the declared workload metric, then use independent interleaved evidence.",
                             "taskset controls CPU affinity, not NUMA memory policy; record memory placement separately."]}
