"""claude -p only: a run with a stated deadline does not end before its report time. The deadline: IK_DEADLINE=<ISO
time>, else the newest runs/<run-id>/state.json (budget.run_started) plus its intake's budget.wall_clock_h, under the
session's start directory, $CLAUDE_PROJECT_DIR or cwd. Report time = deadline - max(10 min, 10% of the run), or
90% of state.json's budget.host_hours_cap when that is earlier (a GPU-hour or $ cap binds, as `intake budget` says).
SessionStart notes the start and states the times; Stop blocks before the report time. Stage 8 done: silent."""
import json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path as P


def ts(s):
    t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


def run(dirs):  # (start, wall_clock_h, run dir, hour cap) of the newest run; None: no run; False: stage 8 done
    fs = sorted({f.resolve() for d in dirs for f in P(d).glob("runs/*/state.json")}, key=os.path.getmtime)
    if not fs:
        return None
    st = json.loads(fs[-1].read_text())
    if ((st.get("stages") or {}).get("8_report") or {}).get("status") == "done":
        return False
    b = json.loads((fs[-1].parent / "intake.normalized.json").read_text())["intake"].get("budget") or {}
    return (ts(st["budget"]["run_started"]), b.get("wall_clock_h"), fs[-1].parent,
            st["budget"].get("host_hours_cap"), b.get("deadline"), b.get("report_reserve_h"))


def done(d):  # the newest full proof ended: its exit code synced back into the run's logs/
    j = sorted(d.glob("logs/full-*.job.json"), key=os.path.getmtime)[-1:] if d else []
    return any(P(str(f)[:-8] + "rc").is_file() for f in j)


def main(ev, i, now):
    e = os.environ.get
    if e("CLAUDE_CODE_SESSION_ATTENDED") == "1" or e("CLAUDE_CODE_ENTRYPOINT") == "cli":
        return
    s = re.sub(r"\W", "_", str(i.get("session_id")))
    fs = [P(d) / f"ik-start-{s}" for d in (e("CLAUDE_PLUGIN_DATA"), e("TMPDIR"), P.home() / ".cache/inferencekit") if d]
    r = next((f.read_text().split("\n") for f in fs if f.is_file()), None)
    if ev == "start" and not r:
        r = [str(now), str(P(i.get("cwd") or ".").resolve())]
        for f in fs:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("\n".join(r))
                break
            except OSError:
                pass
    dirs = [d for d in (r and r[1], e("CLAUDE_PROJECT_DIR"), i.get("cwd")) if d]
    got, dl = run(dirs), e("IK_DEADLINE")
    if got is False:
        return
    t0 = got[0] if got else r and float(r[0])
    intake_deadline = got and got[4]
    end = ts(dl or intake_deadline) if dl or intake_deadline else got and got[1] and t0 + float(got[1]) * 3600
    reserve = got[5] if got else None
    q = t0 and end and end - (float(reserve) * 3600 if reserve is not None else max(600, 0.1 * (end - t0)))
    src = "IK_DEADLINE" if dl else "intake deadline" if intake_deadline else got and f"intake: {float(got[1] or 0):g} h"
    cap = got and got[3] and float(got[3])
    cap_report = t0 + cap * 3600 - (float(reserve) * 3600 if reserve is not None else 0.1 * cap * 3600) if cap else None
    if q and cap and cap_report < q:  # a smaller budget cap binds: its reserve, as `intake budget` says
        end, q, src = t0 + cap * 3600, cap_report, f"budget cap: {cap:g} h"
    if q and now < q and reserve == 0:
        msg = (f"inferencekit: deadline {time.strftime('%H:%M UTC', time.gmtime(end))} ({src}); "
               "automatic report reserve and freeze disabled for this run. Keep optimizing and validating "
               "through the deadline, maintaining reviewable artifacts continuously. A missed target is a "
               "verdict, not a reason to stop.")
        print(json.dumps({"decision": "block", "reason": "Expected, not an error: " + msg} if ev == "stop" else
                         {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": msg}}))
        return
    if q and now < q:
        hm, go = lambda x: time.strftime("%H:%M UTC", time.gmtime(x)), now < q - 2100
        fin = not go and done(got and got[2])
        msg = (f"inferencekit: deadline {hm(end)} ({src}), {(q - now) / 60:.0f} min to the report reserve. " + (
            f"Keep improving until {hm(q - 2100)}, then freeze the best candidate and start its full proof" if go else
            "The full proof has finished: check each report number against its evidence, then keep checking"
            if fin else "Let the full proof run and draft the report") +
               f"; stage 8 (report) starts at {hm(q)}, not before. A missed target is a verdict, not a reason to stop.")
        nxt = " Next: `git log --oneline -3`, then your next change and its screen." if go else \
            " Next: `ikstep digest --last 40`, then the report draft." if fin else \
            " Next: `ikstep wait` on the full proof."
        print(json.dumps({"decision": "block", "reason": "Expected, not an error: " + msg + nxt} if ev == "stop" else
                         {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": msg}}))


try:
    main(sys.argv[1], json.loads(sys.stdin.read() or "{}"), time.time())
except Exception:
    pass
