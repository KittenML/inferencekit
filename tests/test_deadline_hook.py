"""The plugin's work-to-the-deadline hooks (plugin/hooks/): in an unattended session, a run with a stated deadline
(the intake's budget.wall_clock_h from state.json's run start, or IK_DEADLINE) is not ended before its report time;
SessionStart states the times. The script runs as Claude Code runs it: a subprocess with the hook input on stdin.
No network, no Claude Code."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parent.parent / "plugin"
SCRIPT = PLUGIN / "hooks" / "deadline.py"


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def hm(t):
    return time.strftime("%H:%M UTC", time.gmtime(t))


@pytest.fixture
def env(tmp_path):
    for d in ("data", "tmp", "home"):
        (tmp_path / d).mkdir()
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path / "home"),
            "CLAUDE_PLUGIN_DATA": str(tmp_path / "data"), "TMPDIR": str(tmp_path / "tmp")}


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    return d


def make_run(repo: Path, minutes_ago: float, wall_clock_h=2, run_id="r1", report_status="pending",
             host_hours_cap=None) -> Path:
    """A run directory as `intake validate --run-dir` writes it: state.json and intake.normalized.json."""
    rd = repo / "runs" / run_id
    rd.mkdir(parents=True, exist_ok=True)
    budget = {} if wall_clock_h is None else {"wall_clock_h": wall_clock_h}
    (rd / "intake.normalized.json").write_text(json.dumps({"intake": {"budget": {"approved": True, **budget}}}))
    cap = {} if host_hours_cap is None else {"host_hours_cap": host_hours_cap}
    state = {"run_id": run_id, "budget": {"run_started": iso(time.time() - minutes_ago * 60), **cap},
             "stages": {"8_report": {"status": report_status}}}
    (rd / "state.json").write_text(json.dumps(state))
    return rd


def started_at(rd: Path) -> float:
    s = json.loads((rd / "state.json").read_text())["budget"]["run_started"]
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def hook(event, inp, env, extra=None, raw=None, argv=None):
    """Run the hook as Claude Code does; returns its parsed stdout (None when it printed nothing)."""
    p = subprocess.run([sys.executable, "-I", str(SCRIPT), *([event] if argv is None else argv)],
                       input=raw if raw is not None else json.dumps(inp), capture_output=True, text=True,
                       env={**env, **(extra or {})}, cwd=env["HOME"], timeout=30)
    assert p.returncode == 0, p.stderr
    assert p.stderr == "", p.stderr
    return json.loads(p.stdout) if p.stdout.strip() else None


def stop_input(cwd, sid="s1", active=False, **kw):
    return {"session_id": sid, "transcript_path": "", "cwd": str(cwd), "hook_event_name": "Stop",
            "stop_hook_active": active, "background_tasks": [], **kw}


def record(env, sid):
    return Path(env["CLAUDE_PLUGIN_DATA"]) / f"ik-start-{sid}"


# --- declaration -------------------------------------------------------------------------------------

def test_hooks_json_declares_session_start_and_stop_only():
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    assert set(hooks) == {"SessionStart", "Stop"}  # never SubagentStop: subagents end when their task does
    for event, arg in (("SessionStart", "start"), ("Stop", "stop")):
        [group] = hooks[event]
        assert "matcher" not in group  # SessionStart on every source: startup, resume, compact
        [h] = group["hooks"]
        assert h["type"] == "command" and h.get("timeout", 0) <= 30
        # The placeholder in double quotes (a plugin path may hold a space), and a script that exists.
        assert h["command"] == f'python3 -I "${{CLAUDE_PLUGIN_ROOT}}/hooks/deadline.py" {arg}'
    assert SCRIPT.is_file()


def test_script_parses_as_python_3_8():
    ast.parse(SCRIPT.read_text(), feature_version=(3, 8))


# --- SessionStart -------------------------------------------------------------------------------------

def test_start_states_the_plan_for_a_run_with_a_deadline(env, repo):
    rd = make_run(repo, 0)
    t0 = started_at(rd)
    out = hook("start", {"session_id": "s1", "cwd": str(repo), "source": "startup"}, env)
    ctx = out["hookSpecificOutput"]
    assert ctx["hookEventName"] == "SessionStart"
    # 2 h: deadline t0+120 min, reserve 12 min (stage 8 at t0+108), the full proof (35 min) starts at t0+73.
    assert re.match(rf"inferencekit: deadline {hm(t0 + 7200)} \(intake: 2 h\), 10[78] min", ctx["additionalContext"])
    assert (f"Keep improving until {hm(t0 + 73 * 60)}, then freeze the best candidate and start its full proof; "
            f"stage 8 (report) starts at {hm(t0 + 108 * 60)}, not before.") in ctx["additionalContext"]


def test_start_records_the_session_once(env, repo):
    before = time.time()
    hook("start", {"session_id": "s1", "cwd": str(repo), "source": "startup"}, env)
    t0, where = record(env, "s1").read_text().split("\n")
    assert before <= float(t0) <= time.time() and where == str(repo.resolve())
    record(env, "s1").write_text(f"1000.0\n{where}")  # a compaction or resume runs SessionStart again
    hook("start", {"session_id": "s1", "cwd": str(repo), "source": "compact"}, env)
    assert record(env, "s1").read_text() == f"1000.0\n{where}"


def test_start_without_a_run_prints_nothing(env, repo):
    assert hook("start", {"session_id": "s2", "cwd": str(repo)}, env) is None


def test_start_inside_the_reserve_prints_nothing(env, repo):
    make_run(repo, 115)
    assert hook("start", {"session_id": "s4", "cwd": str(repo)}, env) is None


@pytest.mark.parametrize("unwritable", [["CLAUDE_PLUGIN_DATA"], ["CLAUDE_PLUGIN_DATA", "TMPDIR"]])
def test_start_falls_back_to_tmpdir_then_home_cache(env, repo, tmp_path, unwritable):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    for var in unwritable:
        env[var] = str(blocker / var)  # mkdir fails: its parent is a file
    hook("start", {"session_id": "s3", "cwd": str(repo)}, env)
    where = Path(env["TMPDIR"]) if len(unwritable) == 1 else Path(env["HOME"]) / ".cache" / "inferencekit"
    assert (where / "ik-start-s3").is_file()


def test_session_id_cannot_escape_the_data_dir(env, repo):
    hook("start", {"session_id": "../../x", "cwd": str(repo)}, env)
    assert [p.name for p in Path(env["CLAUDE_PLUGIN_DATA"]).iterdir()] == ["ik-start-______x"]


# --- Stop ---------------------------------------------------------------------------------------------

# Claude Code 2.1.300 keeps stop_hook_active true for the rest of a `claude -p` run once a Stop hook has
# blocked, and it lists in-flight background work in background_tasks. The hook blocks in every case: a
# headless run stops background work 10 minutes after its last turn, so a "pause" there can end the run.
@pytest.mark.parametrize("active,bg", [(False, []), (True, []), (True, [{"id": "b1", "type": "shell", "status": "running"}])])
def test_stop_blocks_before_the_reserve_also_when_stop_hook_active(env, repo, active, bg):
    t0 = started_at(make_run(repo, 30))
    out = hook("stop", stop_input(repo, active=active, background_tasks=bg), env)
    assert out["decision"] == "block"
    reason = out["reason"]
    assert re.match(rf"Expected, not an error: inferencekit: deadline {hm(t0 + 7200)} \(intake: 2 h\), 7[78] min "
                    r"to the report reserve\. Keep improving until ", reason), reason
    assert (f"until {hm(t0 + 73 * 60)}, then freeze the best candidate and start its full proof; stage 8 (report) "
            f"starts at {hm(t0 + 108 * 60)}, not before.") in reason
    # Claude Code ends a turn after 8 blocks in a row with no tool call between: each block names a tool call.
    assert reason.endswith("A missed target is a verdict, not a reason to stop. Next: `git log --oneline -3`, then "
                           "your next change and its screen.")


def test_stop_after_the_proof_start_says_to_let_the_proof_run(env, repo):
    make_run(repo, 90)
    reason = hook("stop", stop_input(repo), env)["reason"]
    assert "min to the report reserve. Let the full proof run and draft the report; stage 8" in reason
    assert "Keep improving until" not in reason and reason.endswith("Next: `ikstep wait` on the full proof.")


def test_stop_after_the_proof_has_finished_says_check_the_numbers(env, repo):
    """Once, 25 min after the proof had ended, the hook still said `ikstep wait` on it. The hook reads the run's
    logs/ as synced back: the newest full proof's exit code."""
    rd = make_run(repo, 90)
    wait = "Next: `ikstep wait` on the full proof."
    done = "The full proof has finished: check each report number against its evidence, then keep checking; stage 8"
    logs = rd / "logs"
    logs.mkdir()
    (logs / "full-c.job.json").write_text("{}")
    (logs / "full-c.rc").write_text("0\n")
    (logs / "full-c-2.job.json").write_text("{}")  # a newer proof, synced while it ran
    os.utime(logs / "full-c.job.json", (time.time() - 60,) * 2)
    assert hook("stop", stop_input(repo), env)["reason"].endswith(wait)
    (logs / "full-c-2.rc").write_text("0\n")  # its exit code synced back
    reason = hook("stop", stop_input(repo), env)["reason"]
    assert done in reason and reason.endswith("Next: `ikstep digest --last 40`, then the report draft."), reason
    make_run(repo, 30)  # before the proof start a finished proof changes nothing: keep improving
    assert "Keep improving until" in hook("stop", stop_input(repo), env)["reason"]


@pytest.mark.parametrize("wall_clock_h,reserve_min", [(2, 12), (1, 10), (1.5, 10), (3, 18), (5, 30)])
def test_stop_allows_from_the_report_reserve(env, tmp_path, wall_clock_h, reserve_min):
    edge = wall_clock_h * 60 - reserve_min
    early, late = tmp_path / "early", tmp_path / "late"
    make_run(early, edge - 0.5, wall_clock_h)
    make_run(late, edge + 0.5, wall_clock_h)
    assert hook("stop", stop_input(early, "early", active=True), env)["decision"] == "block"
    assert hook("stop", stop_input(late, "late", active=True), env) is None


def test_a_smaller_budget_cap_moves_the_report_time(env, tmp_path):
    """A GPU-hour or $ cap below the deadline: `intake budget` says reserve_only at 90% of it, so the hook never
    holds the run past it (that would spend past the approved budget)."""
    early, late = tmp_path / "early", tmp_path / "late"
    t0 = started_at(make_run(early, 30, wall_clock_h=8, host_hours_cap=1))  # report time t0 + 54 min
    make_run(late, 55, wall_clock_h=8, host_hours_cap=1)
    reason = hook("stop", stop_input(early, "early"), env)["reason"]
    assert f"deadline {hm(t0 + 3600)} (budget cap: 1 h), 2" in reason and f"starts at {hm(t0 + 54 * 60)}" in reason
    assert hook("stop", stop_input(late, "late"), env) is None
    make_run(late, 55, wall_clock_h=8, host_hours_cap=8)  # the deadline binds: unchanged
    assert hook("stop", stop_input(late, "late"), env)["decision"] == "block"


def test_without_a_deadline_nothing_blocks(env, repo):
    assert hook("start", {"session_id": "s1", "cwd": str(repo)}, env) is None
    assert hook("stop", stop_input(repo), env) is None
    make_run(repo, 1, wall_clock_h=None)  # a run capped by GPU hours or dollars only
    assert hook("stop", stop_input(repo), env) is None


def test_the_newest_run_counts(env, repo):
    old = make_run(repo, 1, run_id="r-old")
    new = make_run(repo, 115, run_id="r-new")
    os.utime(old / "state.json", (time.time() - 3600,) * 2)
    assert hook("stop", stop_input(repo), env) is None  # r-new is inside its reserve
    os.utime(new / "state.json", (time.time() - 7200,) * 2)
    assert hook("stop", stop_input(repo), env)["decision"] == "block"


def test_after_a_cd_the_run_still_counts(env, repo):
    """The Stop input's cwd follows `cd` in the Bash tool; the start record holds the session's directory."""
    hook("start", {"session_id": "s1", "cwd": str(repo), "source": "startup"}, env)
    make_run(repo, 5)  # the run directory is made after the session starts (stage 1)
    (repo / "engines" / "vllm").mkdir(parents=True)
    assert hook("stop", stop_input(repo / "engines" / "vllm", active=True), env)["decision"] == "block"
    assert hook("stop", stop_input(Path("/"), active=True), env)["decision"] == "block"


def test_the_project_dir_is_searched(env, repo, tmp_path):
    make_run(repo, 5)
    hook("start", {"session_id": "s1", "cwd": str(tmp_path / "home")}, env, {"CLAUDE_PROJECT_DIR": str(repo)})
    assert hook("stop", stop_input(tmp_path / "home"), env, {"CLAUDE_PROJECT_DIR": str(repo)})["decision"] == "block"


@pytest.mark.parametrize("extra", [{"CLAUDE_CODE_SESSION_ATTENDED": "1"}, {"CLAUDE_CODE_ENTRYPOINT": "cli"}])
def test_a_session_with_a_person_is_never_blocked(env, repo, extra):
    """An interactive session ends every turn with Stop: blocking it would take the turn away from the person."""
    make_run(repo, 5)
    assert hook("start", {"session_id": "s1", "cwd": str(repo), "source": "startup"}, env, extra) is None
    assert not record(env, "s1").exists()
    assert hook("stop", stop_input(repo, active=True), env, extra) is None
    unattended = {"CLAUDE_CODE_SESSION_ATTENDED": "0", "CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}  # claude -p, as measured
    assert hook("stop", stop_input(repo, active=True), env, unattended)["decision"] == "block"


def test_a_finished_run_is_not_reopened(env, repo):
    """A run whose stage 8 is done is a look back, not the timed run."""
    make_run(repo, 5, report_status="done")
    assert hook("start", {"session_id": "s2", "cwd": str(repo), "source": "startup"}, env) is None
    assert hook("stop", stop_input(repo, "s2", active=True), env) is None
    make_run(repo, 5, report_status="in_progress")  # a draft report does not end the run
    assert hook("stop", stop_input(repo, "s2", active=True), env)["decision"] == "block"


def test_ik_deadline_overrides(env, repo):
    make_run(repo, 1)  # the intake alone would block
    assert hook("stop", stop_input(repo), env, {"IK_DEADLINE": "2020-01-01T00:00:00Z"}) is None
    # A deadline 6 h after a run start 5 h ago: the reserve is 10% of start-to-deadline (36 min).
    rd = make_run(repo, 300)
    t0 = started_at(rd)
    end = datetime.fromtimestamp(t0 + 6 * 3600, timezone.utc)
    out = hook("stop", stop_input(repo), env, {"IK_DEADLINE": end.isoformat()})
    assert out["reason"].startswith(f"Expected, not an error: inferencekit: deadline {hm(end.timestamp())} "
                                    "(IK_DEADLINE), 2")  # 24 min
    assert f"stage 8 (report) starts at {hm(end.timestamp() - 36 * 60)}" in out["reason"]


def test_ik_deadline_without_a_run_counts_from_the_session_start(env, repo):
    end = datetime.fromtimestamp(time.time() + 3600, timezone.utc).replace(microsecond=0)
    extra = {"IK_DEADLINE": end.isoformat()}
    assert hook("stop", stop_input(repo, "none"), env, extra) is None  # no start known
    hook("start", {"session_id": "s9", "cwd": str(repo)}, env, extra)
    out = hook("stop", stop_input(repo, "s9"), env, extra)
    assert f"stage 8 (report) starts at {hm(end.timestamp() - 10 * 60)}" in out["reason"]  # the 10-minute floor


@pytest.mark.parametrize("raw,extra,argv", [
    ("not json", None, None),
    ("", None, None),
    ("[]", None, None),
    (None, {"IK_DEADLINE": "tomorrow"}, None),
    (None, None, []),  # no event argument
])
def test_any_error_allows_the_stop_quietly(env, repo, raw, extra, argv):
    make_run(repo, 1)
    assert hook("stop", stop_input(repo), env, extra, raw=raw, argv=argv) is None


def test_a_corrupt_run_allows_the_stop(env, repo):
    rd = make_run(repo, 1)
    (rd / "state.json").write_text("garbage")
    assert hook("stop", stop_input(repo), env) is None


def test_a_corrupt_start_record_allows_the_stop(env, repo):
    record(env, "s1").write_text("garbage")
    assert hook("stop", stop_input(repo), env, {"IK_DEADLINE": "2099-01-01T00:00:00Z"}) is None


def test_explicit_deadline_without_reserve_never_schedules_early_freeze(env, repo):
    rd = make_run(repo, 100, wall_clock_h=None)
    budget = {"approved": True, "spend_cap": "none", "deadline": iso(time.time() + 60),
              "stage_timeboxes": False, "report_reserve_h": 0}
    (rd / "intake.normalized.json").write_text(json.dumps({"intake": {"budget": budget}}))
    out = hook("stop", stop_input(repo), env)
    assert out["decision"] == "block"
    assert "automatic report reserve and freeze disabled" in out["reason"]
    assert "through the deadline" in out["reason"]
    assert "Let the full proof run" not in out["reason"] and "then freeze" not in out["reason"]
    budget["deadline"] = iso(time.time() - 1)
    (rd / "intake.normalized.json").write_text(json.dumps({"intake": {"budget": budget}}))
    assert hook("stop", stop_input(repo), env) is None
