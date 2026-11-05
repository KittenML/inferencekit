#!/usr/bin/env python3
"""One host-side command per step, one verdict (references/remote-host.md, "One command per step"). Runs on --host
(IK_HOST) over ssh, this file on stdin, after --prefix (IK_PREFIX, e.g. "$IK"); --local: here. Writes logs/, jobs/.
  job baseline|screen|light|full [--candidate E] [--baseline B] [--sampler-changed] [--dry-run]: reads
      ./manifest.json, stops every server and job started here, starts jobs/<job>.sh detached; one at a time
  wait <job> [--max-min 9]: at most 25 lines: each check and why, the failing step's log tail, evidence ids and
      a JSON progress line. A step that exits 0 with no record, or KL above the A/A floor: UNPROVEN
  digest [ev_id ...] [--last N] [--root DIR]: one line per evidence record, on this machine
Exit 0 ok, 1 failed or a check FAIL, 2 usage, 3 still running (wait again), 4 timeout, 5 a check UNPROVEN."""

import argparse
import glob
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress

ERR = re.compile(r"error|exception|traceback|out of memory|\boom\b|killed|fatal|fail|assert", re.I)
EXCLUDE = (".git", "__pycache__")
SECRET = re.compile(r"((?<![-\w])\w*(?:token|key|secret|passw)\w*=|--[\w-]*(?:token|key)[= ]+|bearer +)\S+",
                    re.I)  # values masked
TOLERANCE_MIB = 1024
UVM = "/dev/nvidia-uvm"  # CUDA processes hold it open, display ones do not
WAIT_KEYS = ("ready_url", "done_regex", "timeout")
KINDS = ("baseline", "screen", "light", "full")
EV = re.compile(r'(?:evidence: \S*?|"evidence_id": ")(ev_[0-9a-f]{64})')
# Job script helpers. up: a manifest engine in a clean env, as bench rounds starts it (own process group, log
# logs/<job>.<label>.log), ready within 30 min or the job ends; down: stop it.
SH = r"""cd "$(dirname "$0")/.." || exit 9
export INFERENCEKIT_RUNS_DIR=$PWD/evidence INFERENCEKIT_OUTPUT=terse
E="PATH HOME USER LOGNAME LANG LC_ALL TMPDIR CUDA_VISIBLE_DEVICES CUDA_HOME CUDA_PATH LD_LIBRARY_PATH
 NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES"
get() { python3 -c 'import sys,json,urllib.request as u
r=u.build_opener(u.ProxyHandler({})).open(sys.argv[1],timeout=5).read()
sys.argv[2:] and print(json.loads(r)["data"][0]["id"])' "$@" 2>/dev/null; }
run() { local s=$1 t=$SECONDS; shift; echo "== $s"; "$@" > logs/$J.$s.log 2>&1; echo "== $s exit $? $((SECONDS-t))s"
  [ -z "$S" ] || live || { echo "== died $N during $s"; exit 1; }; }
live() { local x; x=$(sed 's/.*) //' /proc/$S/stat 2>/dev/null) && [ "${x%% *}" != Z ]; }
down() { [ -z "$S" ] && return; rm -f logs/$J.server; kill -TERM -- -$S 2>/dev/null
  for i in $(seq 40); do kill -0 -- -$S 2>/dev/null || break; sleep 1; done; kill -KILL -- -$S 2>/dev/null; S=; }
up() { down; local n=$1 b=$2 r=$3 e=() v t=$SECONDS; N=$1; shift 3
  for v in $E $I; do [ -n "${!v+x}" ] && e+=("$v=${!v}"); done
  echo "== up $n"; setsid env -i "${e[@]}" "$@" > logs/$J.$n.log 2>&1 < /dev/null & S=$!; echo "$S $n" > logs/$J.server
  until get $b$r; do live || { echo "== died $n"; tail -n 20 logs/$J.$n.log; exit 1; }
    [ $((SECONDS-t)) -gt 1800 ] && { echo "== timeout $n"; exit 1; }; sleep 2; done
  M=$(get $b/v1/models 1) || M=$F; echo "== ready $n $((SECONDS-t))s"; }
trap 'exit 143' TERM INT; trap down EXIT
"""


def say(line, code=0, extra=()):
    print(line[:700], *extra, sep="\n", flush=True)
    return code


def path(name, ext="job.json"):
    return f"logs/{name}.{ext}"


def load(file):
    try:
        return json.load(open(file))
    except (OSError, ValueError):
        return None


def save_job(job):
    json.dump(job, open(path(job["name"]), "w"), indent=1)


def stat(pid):  # [0] state, [2] pgrp, [19] start ticks
    return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()


def alive(pgid, ticks=None):  # a group member is not a zombie; a leader with other start TICKS: pgid reused
    with suppress(OSError):
        if ticks and stat(pgid)[19] != ticks:
            return False
    for p in filter(str.isdigit, os.listdir("/proc")):
        with suppress(OSError):
            if int((f := stat(p))[2]) == pgid and f[0] != "Z":
                return True
    return False


def gpu_mem():  # MiB per GPU, or None
    with suppress(Exception):
        return [int(x) for x in subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout.split()]


def url_ok(url):
    with suppress(Exception):  # no proxy: the server is on this host
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=5) as r:
            return 200 <= r.status < 300
    return False


def read(log, pos):  # whole lines from byte POS (< 0: from the end), and the next POS
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) + pos) if pos < 0 else pos)
            at, data = f.tell(), f.read()
    except OSError:
        return [], max(pos, 0)
    cut = max(data.rfind(b"\n"), data.rfind(b"\r")) + 1
    return data[:cut].decode(errors="replace").replace("\r", "\n").splitlines(), at + cut


def errors(log):  # the last 5 error lines in the last MB
    return ["  " + x.strip()[:200] for x in read(log, -(1 << 20))[0] if ERR.search(x)][-5:]


def start(a, kind="job", cmd="", extra=()):
    cmd = cmd or (a.cmd[0] if len(a.cmd) == 1 else shlex.join(a.cmd))
    old = load(path(a.name))
    if why := (old and alive(old["pgid"], old.get("ticks")) and f"still running (pgid {old['pgid']})" or a.ready_url
               and url_ok(a.ready_url) and f"{a.ready_url} already answers; stop --all first"):
        return say(f"start {a.name}: refused, {why}", 1)
    os.makedirs("logs", exist_ok=True)
    log, rc = path(a.name, "log"), path(a.name, "rc")
    with suppress(OSError):
        os.remove(rc)
    before = gpu_mem()
    with open(log, "wb") as out:  # own session and group (setsid); SIGHUP ignored (nohup)
        p = subprocess.Popen(
            ["bash", "-c", f"bash -c {shlex.quote(cmd)}; echo $? > {shlex.quote(rc)}"], stdin=subprocess.DEVNULL,
            stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    save_job({"name": a.name, "kind": kind, "pgid": p.pid, "ticks": stat(p.pid)[19], "log": log,
              "cmd": SECRET.sub(r"\1***", cmd), "started": time.time(), **{k: getattr(a, k) for k in WAIT_KEYS},
              "gpu_mib_before": before, **dict(extra)})
    busy = f"; GPU already holds {before} MiB" if max(before or [0]) > TOLERANCE_MIB else ""
    return say(f"started {a.name}: pgid {p.pid}, log {log}{busy}")


def ended(job, t):
    name, log, rc = job["name"], job["log"], load(path(job["name"], "rc"))  # None: killed
    if job["kind"] in KINDS:
        return report(job, t)
    if job["kind"] == "bench":
        if not (rec := load(path(name, "json"))):
            return say(f"FAILED bench {name}: exit {rc}, no JSON record; log {log}", 1, errors(log))
        miss = (rec.get("coverage") or {}).get("missing") or []
        ok = rc == 0 and not miss
        return say(f"{'OK' if ok else 'INCOMPLETE'} bench {name} {rec.get('evidence_id')}: {rec.get('summary')}; "
                   f"missing {miss}; full JSON {path(name, 'json')}", 1 - ok)
    want = job["ready_url"] or job["done_regex"]
    if rc == 0 and not want:
        return say(f"FINISHED {name}: exit 0, {t}; log {log}")
    why = "exited before ready or done" if want and rc == 0 else f"died (exit {rc})"
    return say(f"DIED {name}: {why}, {t}; log {log}", 1, errors(log))


def wait(a):
    job = load(path(a.name))
    if not job:
        return say(f"wait {a.name}: no such job here", 2)
    if new := {k: getattr(a, k) for k in WAIT_KEYS if getattr(a, k)}:
        save_job(job := {**job, **new})  # later waits keep them
    rx, t0, pos = job["done_regex"] and re.compile(job["done_regex"]), time.time(), -(1 << 26)
    proof, chunk = job["kind"] in KINDS, a.max_min * 60 if a.max_min else a.chunk
    while True:
        dead = not alive(job["pgid"], job.get("ticks"))
        lines, pos = read(job["log"], pos) if rx else ([], 0)  # then only new lines
        hit = next((x for x in lines if rx.search(x)), None)
        since = time.time() - job["started"]
        t = f"{since:.0f} s after start"
        if job["ready_url"] and not dead and url_ok(job["ready_url"]):
            return say(f"READY {a.name}: {job['ready_url']} answered {t} (pgid {job['pgid']})")
        if hit is not None:
            return say(f"DONE {a.name}: /{job['done_regex']}/ matched {t}: {hit.strip()[:160]}")
        if dead:
            return ended(job, t)
        if proof and (srv := server_died(a.name)) and not str(parse(a.name)[2]).startswith("starting"):
            stop(argparse.Namespace(names=[a.name], all=False, gpu_wait=60))
            return report(job, t, srv)  # (while starting, the job ends itself within 2 s)
        if job["timeout"] and since >= job["timeout"]:
            return say(f"TIMEOUT {a.name}: {t}, still running", 4, errors(job["log"]))
        if time.time() - t0 >= chunk:
            if proof:
                rc, _, cur = parse(a.name)
                done = ", ".join(s + f" FAILED (exit {r})" * bool(r) for s, r in rc.items()) or "nothing yet"
                return say(f"RUNNING {a.name}: {since / 60:.1f} min; done: {done}; now: {cur}. Call again: "
                           f"ikstep wait {a.name}", 3)
            last = next((x.strip() for x in reversed(read(job["log"], -65536)[0]) if x.strip()), "")
            return say(f"RUNNING {a.name}: {t}; wait again. Last line: {last[:150]}", 3)
        time.sleep(a.poll)


def bench(a):
    a.ready_url = a.done_regex = None  # ends when bench serve exits
    cmd = f"INFERENCEKIT_OUTPUT=json inferencekit bench serve {shlex.join(a.cmd)} > {path(a.name, 'json')}"
    return start(a, "bench", cmd) or wait(a)


def compare(a):
    if not (d := load(a.summary)):
        return say(f"compare: cannot read {a.summary}", 1)
    plan, res = d.get("plan") or {}, d.get("resolved_config") or {}
    rows = [r for r in d["summary"]["rows"] if r["metric"] == a.metric and r["bucket"] in (a.bucket or [r["bucket"]])]
    if not rows:
        return say(f"compare: no {a.metric} rows in {a.summary}", 1)

    def g(x, ci=None):
        return "n/a" if x is None else f"{x:.4g}" + (f" [{ci[0]:.4g}-{ci[1]:.4g}]" if ci and None not in ci else "")

    say(f"compare {a.summary}: {plan.get('rounds')} rounds, balanced {(plan.get('design') or {}).get('balanced')}; "
        f"{a.metric} median [95% CI] (observed), ratio to best [CI] (derived)")
    for e in dict.fromkeys(r["engine"] for r in rows):
        say(f"  {e}{' HARD-GATE' * bool((res.get(e) or {}).get('violations'))}: " + "; ".join(
            f"{r['bucket']} {g(r['median'], r.get('ci95'))} n={r['n']} " + "FAILURES " * bool(r.get("failed_requests"))
            + ("best" if r["best"] == e else f"{g(r['ratio_to_best'], r.get('ratio_ci95'))}x best")
            for r in rows if r["engine"] == e))
    return 0


def gpu_groups():  # CUDA process groups outside our session: started in this directory (ours), the rest
    ours, rest, here = {}, {}, os.path.join(os.getcwd(), "")
    for p in filter(str.isdigit, os.listdir("/proc")):
        with suppress(OSError):
            d, g = f"/proc/{p}/", os.getpgid(int(p))
            fds = os.listdir(d + "fd") if g > 1 and os.getsid(int(p)) != os.getsid(0) else []
            if UVM in (os.readlink(d + "fd/" + x) for x in fds):
                cmd = SECRET.sub(r"\1***", open(d + "cmdline", errors="replace").read().replace("\0", " "))
                mine = os.path.join(os.readlink(d + "cwd"), "").startswith(here)
                (ours if mine else rest).setdefault(g, f"pgid {g} ({cmd.strip()[:60]})")
    return ours, rest


def stop(a):
    if not a.names and not a.all:
        return say("stop: name a job, or pass --all", 2)
    os.makedirs("logs", exist_ok=True)
    names = a.names or [f[:-9] for f in os.listdir("logs") if f.endswith(".job.json")]
    jobs = [j for n in names if (j := load(path(n)))]
    ours, rest = gpu_groups() if a.all else ({}, {})
    groups = {**ours, **{j["pgid"]: j["name"] for j in jobs if alive(j["pgid"], j.get("ticks"))}}
    bases = [j["gpu_mib_before"] for j in jobs if j.get("gpu_mib_before") and not a.all]  # --all: GPU must empty
    base = [min(x) for x in zip(*bases)] if bases else []
    for sig, secs in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        for g in filter(alive, list(groups)):
            with suppress(OSError):
                os.killpg(g, sig)
        end = time.time() + secs
        while time.time() < end and any(map(alive, groups)):
            time.sleep(0.3)

    def held(mem):  # above the use before our jobs
        return any(m > b + TOLERANCE_MIB for m, b in zip(mem or [], base if len(base) == len(mem or []) else [0] * 64))

    end = time.time() + a.gpu_wait
    while held(mem := gpu_mem()) and time.time() < end:
        time.sleep(2)
    left = [groups[g] for g in groups if alive(g)]
    what = f"{', '.join(groups.values()) or 'nothing was running'}; GPU MiB used: {mem or 'no nvidia-smi'}"
    if others := ", ".join(v for g, v in rest.items() if alive(g)):
        what += f"; not ours, left running: {others}"
    if left or held(mem):
        return say(f"STOP INCOMPLETE: {what} after {a.gpu_wait:g} s; still alive: {left}", 1)
    return say(f"STOPPED {what}")


def lab(e):
    return e.get("label") or e.get("name")


def srv(e):  # (url, the job script's line that starts manifest engine E)
    url, env = f"http://{e.get('host') or '127.0.0.1'}:{e.get('port')}", e.get("env") or {}
    keep = " ".join(k for k, v in env.items() if v is None)  # null: inherited, as bench rounds does
    words = [f"{k}={v if isinstance(v, str) else json.dumps(v)}" for k, v in env.items() if v is not None]
    ready = "/" + str(e.get("ready_check") or "v1/models").lstrip("/")
    return url, f"I={shlex.quote(keep)} " * bool(keep) + f"up {lab(e)} {url} {ready} " + shlex.join(
        words + list(e.get("launch") or []))


def ekind(e):  # as bench rounds picks it (serve/servers.py)
    h = " ".join([str(e.get("name") or "")] + [str(x) for x in (e.get("launch") or [])[:3]]).lower()
    return "vllm" if "vllm" in h else "sglang" if "sglang" in h else "llamacpp" if "llama" in h and (
        "cpp" in h or "server" in h) else "trtllm" if re.search("trtllm|tensorrt|trt-llm", h) else "generic"


def own(e, base):  # another venv: its own FlashInfer JIT dir, or a switch rebuilds it
    p = str((e.get("launch") or [""])[0])
    return e if p == (base.get("launch") or [""])[0] else {
        **e, "env": {"FLASHINFER_WORKSPACE_BASE": "cache/" + re.sub(r"\W", "_", p), **(e.get("env") or {})}}


def script(a, J, P, m, base, cand):
    w, B, C, oc = m.get("workload") or {}, lab(base), lab(cand), own(cand, base)
    js = lambda o: shlex.quote(json.dumps(o, sort_keys=True, separators=(",", ":")))  # noqa: E731
    mt, eb, (bu, bup), (cu, cup) = w.get("max_tokens") or 256, w.get("extra_body") or {}, srv(base), srv(oc)
    ct = f" --chat-template-kwargs {js(w['chat_template_kwargs'])}" * bool(w.get("chat_template_kwargs"))
    bk, sb = "".join(f" --buckets {b}" for b in w.get("buckets") or []), {k: v for k, v in eb.items() if k != "seed"}

    def ik(s, cmd, url, more):  # step S: inferencekit CMD against the server at URL
        return f'run {s} inferencekit {cmd} --base-url {url} --model "$M"{more}{ct}'

    def serve(s, label, e, url):  # the 16-prompt screen
        return ik(s, "bench serve", url, f" --prompts $P --engine-kind {ekind(e)} --rounds 1{bk} --limit-per-bucket 4 "
                  f"--max-tokens {mt} --label {label} --run-dir .") + f" --extra-body {js(eb)}" * bool(eb)

    def traj(s, e, url, lim=""):
        k = ekind(e) if ekind(e) in ("vllm", "sglang", "llamacpp") else "openai"  # its choices
        return ik(s, "quality trajectory", url, f" --prompts $P --label {J if s == 'trajectory' else s} --run-dir . "
                  f"--engine-kind {k} --max-tokens {mt}") + f" --limit {lim}" * bool(lim)

    def gsm(label, url, more=""):
        return ik("gsm8k", "quality gsm8k", url, f" --run-dir . --label {label}") + more

    def samp(label, url):
        return ik("sampler", "quality sampling", url, f" --run-dir . --label {label}") + f" --extra-body {js(sb)}" \
            if sb.get("temperature") else "echo '== skip sampler: no temperature in workload.extra_body'"

    def rounds(n, per):
        sa = f"--max-tokens {mt}{bk}" + f" --limit-per-bucket {per}" * bool(per)
        mf = f"jobs/{J}.manifest.json" if oc is not cand else "manifest.json"
        return (f"run rounds inferencekit bench rounds --manifest {mf} --run-dir . --prompts $P --rounds {n} "
                f"--mode single --engines {B},{C} --serve-args {shlex.quote(sa)}")

    def score(ref):  # the candidate's trajectories teacher-forced on the baseline, against its A/A floor
        return (f'run score inferencekit quality score --reference-url {bu} --reference-model "$M" --trajectories '
                f"quality/{J} --baseline quality/{ref} --baseline-repeat "
                f"quality/{ref.replace('reference', 'reference-r2')} --margin-agreement 0.01 --margin-kl 0.01 "
                f"--margin-nll 0.01 --out quality/score-{J}.json")
    gates = ik("gates", "gates run", cu, f" --engine {C} --run-dir . --reference-engine {B}")
    mark = 'echo "== MARK speed and gates done"'
    steps = {
        "baseline": [bup, serve("aa1", f"{B}-aa1", base, bu),
                     ik("gates", "gates run", bu, f" --engine {B} --run-dir ."), traj("reference", base, bu),
                     traj("reference-8", base, bu, 8), gsm(B, bu), samp(B, bu), bup,
                     serve("aa2", f"{B}-aa2", base, bu), traj("reference-r2", base, bu),
                     traj("reference-r2-8", base, bu, 8)],
        "screen": [cup, serve("screen", J, cand, cu)],
        "light": [rounds(2, 8), cup, gates + " --gates caching --gates token_count --gates rare_tokens", mark,
                  traj("trajectory", cand, cu, 8), bup, score("reference-8"), cup,
                  gsm(J, cu, " --limit 100 --concurrency 0")] + [samp(J, cu)] * a.sampler_changed,
        "full": [rounds(3, 0), cup, gates, mark, traj("trajectory", cand, cu), bup, score("reference"), cup,
                 gsm(J, cu, f" --paired-with quality/gsm8k/{B}.jsonl"), samp(J, cu)]}[a.kind]
    return "\n".join([f"#!/bin/bash\n# ikstep job {a.kind}: {C} vs {B}. Each step's output: logs/{J}.<step>.log\n{SH}"
                      f"J={J} P={shlex.quote(P)} F={shlex.quote(str((m.get('model') or {}).get('id')))}", *steps,
                      'echo "== END"\n'])


def job(a):
    m = load("manifest.json") or {}
    es = [e for e in m.get("engines") or [] if isinstance(e, dict)]
    pick = lambda n: next((e for e in es if n in (e.get("label"), e.get("name"))), None)  # noqa: E731
    base = pick(a.baseline) if a.baseline else next((e for e in es if str(e.get("role")).startswith("baseline")),
                                                    es and es[0])
    cand = base if a.kind == "baseline" else pick(a.candidate) if a.candidate else next(
        (e for e in es[::-1] if e is not base), None)
    if not (base and cand) or (cand is base) != (a.kind == "baseline"):
        return say(f"job: no such engine in ./manifest.json (labels {[lab(e) for e in es]}): run in the run directory",
                   2)
    B, C = lab(base), lab(cand)
    J = n = f"{a.kind}-{C}"
    while os.path.exists(path(J)) or os.path.exists(f"jobs/{J}.sh"):
        J = f"{n}-{int(J[len(n) + 1:] or 1) + 1}"
    pv, v = "`inferencekit --json prompts verify --set dev`", {}
    try:
        r = subprocess.run(shlex.split(pv[1:-1]), capture_output=True, text=True)
    except OSError:
        return say(f"job: {pv}: no inferencekit on PATH (after the prefix)", 1)
    with suppress(ValueError):
        v = json.loads(r.stdout)
    v = v if isinstance(v, dict) else {}
    if not (P := v.get("verdict") == "PASS" and v.get("path")):
        why = [c.get("detail") for c in v.get("checks") or [] if not c.get("ok")] or r.stderr.splitlines()
        return say(f"job: {pv} exit {r.returncode}, verdict {v.get('verdict')}: no verified prompt set", 1,
                   ["  " + str(x)[:200] for x in why[-5:]])
    text = script(a, J, P, m, base, cand)
    if a.dry_run:
        return print(text, end="") or 0
    need = {"light": ["reference-8", "reference-r2-8"], "full": ["reference", "reference-r2", f"gsm8k/{B}.jsonl"]}
    if miss := [f"quality/{x}" for x in need.get(a.kind, []) if not os.path.exists(f"quality/{x}")]:
        return say(f"job: {', '.join(miss)} missing: run `ikstep job baseline` first", 1)
    os.makedirs("logs", exist_ok=True)
    if busy := [j["name"] for f in sorted(os.listdir("logs")) if f.endswith(".job.json") and (j := load(f"logs/{f}"))
                and j.get("kind") in KINDS and alive(j["pgid"], j.get("ticks"))]:
        return say(f"job: {busy[0]} is still running: ikstep wait {busy[0]} (or ikstep stop {busy[0]})", 1)
    if stop(argparse.Namespace(names=[], all=True, gpu_wait=120)):
        return 1
    os.makedirs("jobs", exist_ok=True)
    if (c := own(cand, base)) is not cand:  # for bench rounds; a re-run reads manifest.json
        json.dump({**m, "engines": [c if e is cand else e for e in m["engines"]]}, open(f"jobs/{J}.manifest.json", "w"))
    with open(f"jobs/{J}.sh", "w") as f:
        f.write(text)
    a.name, a.ready_url, a.done_regex, a.timeout = J, None, None, None
    info = {"base": B, "cand": C, "commit": a.commit or None, "sampling": "default" if (
        (m.get("workload") or {}).get("extra_body") or {}).get("temperature") else "greedy",
            "sampler_changed": a.sampler_changed, "screen_ratio": screen_x(B)}
    return start(a, a.kind, f"bash jobs/{J}.sh", info) or say(
        f"job {J}: {a.kind}, {C} vs {B}; script jobs/{J}.sh. Next: ikstep wait {J} (Bash timeout 600000)")


def parse(J):  # the job log: exit code per step, server events, what runs now
    rc, ev, cur = {}, [], None
    for w in map(str.split, read(path(J, "log"), 0)[0]):
        if w[:1] == ["=="] and len(w) > 1:
            if len(w) > 3 and w[2] == "exit":
                rc[w[1]], cur = int(w[3]), None
            elif w[1] in ("up", "died", "timeout"):
                ev.append(w[1:])
                cur = f"starting {w[2]}" if w[1] == "up" else cur
            elif len(w) == 2:
                cur = w[1]
    return rc, ev, cur


def server_died(J):  # the label of the job's current server once its process is gone
    try:
        pid, label = open(path(J, "server")).read().split()
    except (OSError, ValueError):
        return None
    try:
        return None if stat(int(pid))[0] != "Z" else label
    except OSError:
        return label


def rec(i):  # one evidence record of this run
    return i and (load(f"evidence/{i}.json") or load(f"evidence/{i}/evidence.json"))


def evid(J, s):  # the evidence id step S printed
    m = EV.search("\n".join(read(path(f"{J}.{s}", "log"), 0)[0]))
    return m and m[1]


def f3(x, f=".3f"):
    return "n/a" if x is None else format(x, f)


def order(d):
    o = ("very_short", "short", "medium", "long")
    return sorted((b for b in d if b != "all"), key=lambda b: (o.index(b) if b in o else 9, b))


def speed(r, B, C):  # {bucket: (C tok/s, B tok/s, C/B, CI low, CI high)} from a bench rounds record
    rows, out = {(x.get("engine"), x.get("bucket")): x for x in (r.get("data") or {}).get("rows") or []
                 if x.get("metric") == "decode_tok_s"}, {}
    for (e, b), c in rows.items():
        if e == C and (o := rows.get((B, b))):
            (x, (lo, hi)), y = (c.get("ratio_to_best"), c.get("ratio_ci95") or (None, None)), o.get("ratio_to_best")
            if c.get("best") == C:  # the baseline's ratio is B/C
                hi, lo = o.get("ratio_ci95") or (None, None)
                x, lo, hi = (v and 1 / v for v in (y, lo, hi))
            elif c.get("best") != B:  # a third engine is best: no CI for C/B
                x, lo, hi = x and y and x / y, None, None
            out[b] = (c.get("median"), o.get("median"), x, lo, hi)
    return out


def tok(label):  # a screen's decode tok/s median; None unless all 4 prompts per bucket answered
    with suppress(Exception):
        b = load(f"metrics/{label}.json")["speed"]["per_bucket"]
        return min(x["n_requests"] for x in b.values()) > 3 and b["all"]["decode_tok_s"]["median"] or None


def screen_x(B):  # the newest screen / the baseline's A/A screen
    js = sorted((j for f in glob.glob(path("screen-*")) if (j := load(f))), key=lambda j: j["started"])
    s, a1, a2 = js and tok(js[-1]["name"]), tok(f"{B}-aa1"), tok(f"{B}-aa2")
    return s and a1 and a2 and round(2 * s / (a1 + a2), 3) or None


def gsm(J, B, r):  # GSM8K against the baseline on the same problems
    def rows(x):
        with suppress(OSError):
            return {int(d["idx"]): bool(d.get("correct")) for d in map(json.loads, open(f"quality/gsm8k/{x}.jsonl"))}
        return {}
    d, mine, theirs = r["data"], rows(J), rows(B)
    both = [i for i in mine if i in theirs]
    w, lo = sum(mine[i] > theirs[i] for i in both), sum(mine[i] < theirs[i] for i in both)
    p = sum(math.comb(w + lo, k) for k in range(w + 1)) / 2 ** (w + lo)  # P(w wins or fewer | no difference)
    return ("FAIL" if p < 0.05 else "PASS") if both else "UNPROVEN", (
        f"{d.get('correct')}/{d.get('n')} correct, baseline {sum(theirs[i] for i in both)}/{len(both)} on the same "
        f"problems, one-sided McNemar p {p:.3g}" + f"; {d.get('errors')} request errors" * bool(d.get("errors")))


def report(job, t, died=""):
    J, kind, B, C = (job.get(k) for k in ("name", "kind", "base", "cand"))
    rc, ev, cur = parse(J)
    plan, out, vs, ids, tail, named = [], [], [], [], None, []
    pg = {"time": None, "commit": job.get("commit"), "engine": C}
    with suppress(OSError):
        plan = re.findall(r"^(?:I=\S+ )?run (\S+)", open(f"jobs/{J}.sh").read(), re.M)

    def chk(name, v, text):
        vs.append(v), named.append(f"{name} {v}"), out.append(f"{name} {v}: {text}")
    for w in [*ev, ["died", died, f"while {cur} ran; job stopped"]] if died else ev:
        if w[0] != "up":
            tail = tail or path(f"{J}.{w[1]}", "log")
            vs.append("FAIL"), out.append(f"SERVER {w[0].upper()}: {w[1]} {' '.join(w[2:]) or 'never answered'}; "
                                          f"log {tail}")
    R = lambda s: rc.get(s) == 0 and rec(evid(J, s)) or None  # noqa: E731
    got = {"score": lambda _: load(f"quality/score-{J}.json"), "screen": lambda _: tok(J),
           "aa1": lambda _: tok(f"{B}-aa1"), "aa2": lambda _: tok(f"{B}-aa2")}  # others: their evidence record
    for s in plan:
        log = path(f"{J}.{s}", "log")
        if rc.get(s) or (f := sum(x.get("n_failed", 0) for x in ((R(s) or {}).get("data") or {}).get("rounds") or [])):
            vs.append("FAIL"), out.append(f"FAILED {s}: {f'exit {rc[s]}' if rc.get(s) else f'{f} requests failed'}; "
                                          f"log {log}"), (tail := tail or log)
        elif s in rc and not got.get(s, R)(s):
            chk({"rounds": "speed", "score": "quality"}.get(s, s), "UNPROVEN", f"exit 0, no record or file; log {log}")
            tail = tail or log
        elif s in rc:
            if s == "rounds" and not speed(R(s), B, C).get("all"):
                tail = tail or log
            if i := evid(J, s):
                ids.append(f"{s} {i[:15]}")
    out += ["  " + x[:200] for x in read(tail, -8192)[0][-5:]] if tail else []  # stderr included
    if left := [s for s in plan if s not in rc]:
        vs.append("FAIL"), out.append(f"NOT RUN: {', '.join(left)} (the job ended first)")
    if r := R("rounds"):
        d, q, p = r["data"], speed(r, B, C), r["data"].get("plan") or {}
        bad = [f"{c.get('label')} r{c.get('round')} {c.get('status')}" for c in d.get("cells") or []
               if c.get("status") != "ok"] + [f"{e}: hard-gate violations" for e, v in
                                              (d.get("resolved_config") or {}).items() if v.get("violations")]
        tc, tb, x, lo, hi = q.get("all") or (None,) * 5
        chk("speed", "FAIL" if bad or x is None or (hi or 2) < 1 else "PASS" if (lo or 0) > 1 else "UNPROVEN", (
            f"{C} {f3(tc, '.1f')} vs {B} {f3(tb, '.1f')} tok/s = {f3(x)}x [{f3(lo)}-{f3(hi)}], {p.get('rounds')} "
            f"rounds, order balanced {(p.get('design') or {}).get('balanced')}" if x else f"no {C} vs {B} decode ratio")
            + "; " * bool(bad) + "; ".join(bad)[:200])
    if r and x:
        out.append("  buckets: " + "; ".join(f"{b} {f3(q[b][2])}x [{f3(q[b][3])}-{f3(q[b][4])}]" for b in order(q)))
        pg.update(tok_s=tc and round(tc, 1), baseline_tok_s=tb and round(tb, 1), ratio=x and round(x, 3),
                  ratio_ci95=[lo and round(lo, 3), hi and round(hi, 3)])
    if r := R("gates"):
        g = r["data"].get("gates") or {}
        chk("gates", r["data"].get("verdict"), "; ".join(
            f"{k} {x.get('verdict')} ({str(x.get('reason'))[:100]})" for k, x in g.items()
            if x.get("applicable") and x.get("verdict") != "PASS") or ", ".join(
            k for k, x in g.items() if x.get("verdict") == "PASS"))
    if rc.get("score") == 0 and (s := load(f"quality/score-{J}.json")):
        e, fl = (s.get("engines") or {}).get(J) or {}, s.get("floor") or {}
        v, why, kl = (s.get("verdicts") or {}).get(J, "UNPROVEN"), (e.get("verdict") or {}).get("reason"), []
        two = lambda k, f: (f"{f3(((e.get('all') or {}).get(k) or {}).get('value'), f)} (floor "  # noqa: E731
                            f"{f3(((fl.get('all') or {}).get(k) or {}).get('value'), f)})")
        dk = ((((e.get("verdict") or {}).get("checks") or {}).get("kl_mean") or {}).get("ci95") or [0])[0]
        up = ["all"] * bool(dk and dk > 0)
        for b in order(e.get("by_input_bucket") or {}):  # where it moved, even when it passes
            k, fk = (((x.get("by_input_bucket") or {}).get(b) or {}).get("kl_mean") or {} for x in (e, fl))
            (lo, hi), fh = k.get("ci95") or (None, None), (fk.get("ci95") or (None, None))[1]
            up += [b] * bool(lo and fh and lo > fh)
            kl.append(f"{b} {f3(k.get('value'), '.4f')} [{f3(lo, '.4f')}-{f3(hi, '.4f')}] vs "
                      f"{f3(fk.get('value'), '.4f')}" + " ABOVE" * (b in up))
        if up and v == "PASS":
            v, why = "UNPROVEN", f"KL above the A/A floor ({', '.join(up)}): the next-token distribution moved"
        chk("quality", v, f"agreement {two('top1_agreement', '.2%')}, KL {two('kl_mean', '.4f')} nats, "
            f"{e.get('n_scored')} prompts" + f"; {str(why)[:200]}" * (v != "PASS"))
        out.append("  KL by bucket vs the A/A floor: " + "; ".join(kl))
    if (r := R("gsm8k")) and kind != "baseline":
        chk("gsm8k", *gsm(J, B, r))
    elif r:
        out.append(f"gsm8k: {r['data'].get('correct')}/{r['data'].get('n')} correct (full proofs pair with it)")
    if r := R("sampler"):
        chk("sampler", r["data"].get("verdict"), str(r.get("summary")).split(": ", 2)[-1][:220])
    elif kind == "light" and "sampler" not in plan and not job.get("sampler_changed"):
        out.append("sampler NOT RUN: sampling code unchanged (--sampler-changed adds it)")
    elif kind != "screen" and rc.get("sampler") != 0:
        chk("sampler", "NOT RUN", "no temperature in workload.extra_body" if "sampler" not in plan else "no record")
    if kind == "baseline":
        out.append("references: " + ", ".join(f"{s} {(m or ['failed'])[0]}" for s in plan if "reference" in s for m in
                                              [re.search(r"\d+/\d+ prompts", str((R(s) or {}).get("summary")))]))
    a1, a2, s = tok(f"{B}-aa1"), tok(f"{B}-aa2"), kind == "screen" and rc.get("screen") == 0 and tok(J)
    bt = kind in ("baseline", "screen") and a1 and a2 and (a1 + a2) / 2
    if bt:
        noise = abs(a1 - a2) / bt
        out.append(f"screen A/A: {a1:.1f} then {a2:.1f} tok/s after two launches: screen noise {noise:.1%}")
        x, done = s and s / bt, [j.get("screen_ratio") or 0 for f in glob.glob(path("light-*")) if (j := load(f))]
        m = ("first >= 1.5x: light check next" if x >= 1.5 and max(done, default=0) < 1.5 else
             "first >= 1.2x: light check next" if not done else "no light check: milestone checked") if s and x >= 1.2 \
            else ""
        if s:
            out.append(f"screen {C}: {s:.1f} tok/s = {x:.3f}x the baseline screen {bt:.1f}; gain above the noise: "
                       f"{'yes' if x - 1 > noise else 'no'}" + f"; {m}" * bool(m))
    elif s:
        out.append(f"screen {C}: {s:.1f} tok/s; no baseline screen to compare (ikstep job baseline records it)")
    if s or kind == "baseline" and bt:
        pg.update(tok_s=round(s or bt, 1), baseline_tok_s=bt and round(bt, 1), ratio=bt and round((s or bt) / bt, 3))
    end = os.path.exists(path(J, "rc")) and os.path.getmtime(path(J, "rc")) or None
    pg.update(time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end)), sampling=job.get("sampling"),
              checks=f"{kind} job {J}: " + (", ".join(named) or "screen only"), evidence=ids)
    return say(f"{kind.upper()} {J}: {'finished' if cur == 'END' else 'ended early'}, {t}; log {job['log']}",
               1 if "FAIL" in vs else 5 if {"UNPROVEN", "NOT RUN"} & set(vs) else 0,
               out[:22 - bool(died)] + [f"evidence: {', '.join(ids) or 'none'}", json.dumps(pg)])  # 25 lines


def one(r):  # one line per evidence record
    d = r.get("data") if isinstance(r.get("data"), dict) else {}
    v, extra = d.get("verdict"), ""
    n = len(r.get("limitations") or []), len((r.get("coverage") or {}).get("missing") or [])
    if (p := d.get("plan")) and d.get("rows"):  # bench rounds: each engine vs the baseline
        es = [e.get("label") for e in p.get("engines") or []]
        B = next((e.get("label") for e in p["engines"] if str(e.get("role")).startswith("baseline")), es[0])
        extra = "".join(f"; {e}/{B} {f3(q[2])}x [{f3(q[3])}-{f3(q[4])}], {f3(q[0], '.1f')} vs {f3(q[1], '.1f')} tok/s"
                        for e in es if e != B and (q := speed(r, B, e).get("all")))
    return (f"{r['evidence_id'][:15]} {str((r.get('provenance') or {}).get('started'))[:16]} {r.get('kind')} "
            f"{((r.get('identity') or {}).get('engine') or {}).get('name')} "
            f"{(v.get('verdict') if isinstance(v, dict) else v) or '-'}: {str(r.get('summary'))[:200]}{extra} "
            f"({n[0]} limitations, {n[1]} missing)")


def digest(a):
    root = a.root or os.environ.get("INFERENCEKIT_RUNS_DIR") or ("evidence" if os.path.isdir("evidence") else max(
        glob.glob("runs/*/evidence"), key=os.path.getmtime, default=""))
    if not os.path.isdir(root or ""):
        return say("digest: no evidence directory here; pass --root <run directory>/evidence", 2)
    recs, rows, code = {}, [], 0
    for f in sorted(glob.glob(f"{root}/ev_*.json") + glob.glob(f"{root}/ev_*/evidence.json")):
        if isinstance(r := load(f), dict) and str(r.get("evidence_id")).startswith("ev_"):
            recs.setdefault(r["evidence_id"], r)
    for i in a.ids:
        hit = [load(i)] if os.path.isfile(i) else [r for k, r in recs.items()
                                                    if k.startswith("ev_" * (i[:3] != "ev_") + i)]
        if len(hit) == 1 and hit[0]:
            rows.append(hit[0])
        else:
            code = say(f"{i}: {len(hit)} records match in {root}", 1)
    by_time = sorted(recs.values(), key=lambda r: str((r.get("provenance") or {}).get("ended")))
    for r in rows if a.ids else by_time[-a.last:]:
        say(one(r))
    return code


def deploy(a):
    src = a.src.rstrip("/") or "/"
    if a.local or not os.path.exists(src):
        return say(f"deploy: no {src}, or --local (nothing to copy)", 2)
    isdir = os.path.isdir(src)
    dest = a.dest or (os.path.basename(src) if isdir else ".")
    q = shlex.quote(dest)  # resolved on the host, after the prefix
    ssh, tgt, line = ssh_cmd(a, f"mkdir -p {q} && cd {q} && pwd")
    r = subprocess.run([*ssh, tgt, line], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        return say(f"deploy: no {dest} on the host (ssh exit {r.returncode}) {r.stderr[-160:]}", 1)
    remote, how = f"{tgt}:{r.stdout.strip().splitlines()[-1]}/", "rsync"
    rsync = ["rsync", "-a", *[f"--exclude={x}" for x in EXCLUDE], "-e", shlex.join(ssh), src + "/" * isdir, remote]
    if not shutil.which("rsync") or subprocess.run(rsync, capture_output=True).returncode:
        how, scp = "scp", ["scp", "-rpq", *("-P" if x == "-p" else x for x in ssh[1:])]
        items = [os.path.join(src, x) for x in os.listdir(src) if x not in EXCLUDE] if isdir else [src]
        if items and subprocess.run([*scp, *items, remote]).returncode:
            return say(f"deploy: rsync and scp failed: {src} -> {dest}", 1)
    return say(f"deployed {src} -> {dest} ({how})")


def ssh_cmd(a, line):
    ssh = ["ssh", "-oBatchMode=yes", "-oStrictHostKeyChecking=accept-new", "-oServerAliveInterval=30",
           "-oConnectTimeout=20", *["-i", a.key] * bool(a.key), *["-p", a.port] * bool(a.port)]
    pre = (a.prefix or "").strip()
    return ssh, f"{a.user}@{a.host}" if a.user else a.host, pre + ";" * (pre[-1:] not in ";&") + " " + line


def parser():
    ap = argparse.ArgumentParser(prog="ikstep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true")
    for k in ("host", "port", "user", "key", "prefix"):
        ap.add_argument(f"--{k}", default=os.environ.get(f"IK_{k.upper()}"))
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("deploy")
    p.add_argument("src")
    p.add_argument("dest", nargs="?")
    for name in ("start", "wait", "bench"):  # NAME -- CMD (bench: its serve flags)
        p = sub.add_parser(name)
        p.add_argument("name")
        p.add_argument("--ready-url")  # 2xx: READY
        p.add_argument("--done-regex")  # a log line matches: DONE
        p.add_argument("--timeout", type=float)  # s after the start
        p.add_argument("--chunk", type=float, default=540)  # s per call at most (tool limit)
        p.add_argument("--max-min", type=float)  # the same, in minutes
        p.add_argument("--poll", type=float, default=3)
    p = sub.add_parser("job")
    p.add_argument("kind", choices=KINDS)
    for k in ("candidate", "baseline", "commit"):
        p.add_argument(f"--{k}")
    p.add_argument("--sampler-changed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("digest")
    p.add_argument("ids", nargs="*")
    p.add_argument("--last", type=int, default=30)
    p.add_argument("--root")
    p = sub.add_parser("compare")
    p.add_argument("--summary", default="rounds/summary.json")
    p.add_argument("--metric", default="decode_tok_s")
    p.add_argument("--bucket", action="append")
    p = sub.add_parser("stop")
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")  # every job here, and CUDA processes started here
    p.add_argument("--gpu-wait", type=float, default=120)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    ap, argv, cmd = parser(), argv[:cut], argv[cut + 1:]
    a = ap.parse_args(argv)
    a.cmd = cmd
    if a.step in ("start", "bench") and not cmd:
        ap.error(f"{a.step} needs the command after --")
    if a.step == "digest":  # evidence on this machine
        return digest(a)
    if not a.local and not a.host:
        ap.error("pass --host H (or IK_HOST) or --local")
    if a.step == "job" and not a.commit:  # the commit measured: this checkout's
        with suppress(OSError):
            a.commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                      text=True).stdout.strip()
        argv += [f"--commit={a.commit}"] * bool(a.commit)
    if not a.local and a.step != "deploy":
        step = argv[argv.index(a.step):] + ["--", *cmd] * bool(cmd)  # no global option value is a step name
        ssh, tgt, line = ssh_cmd(a, f"python3 - --local {shlex.join(step)}")
        r = subprocess.run([*ssh, tgt, line], stdin=open(__file__, "rb"))
        return say("ikstep: ssh failed (exit 255)", 1) if r.returncode == 255 else r.returncode
    return globals()[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
