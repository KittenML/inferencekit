"""The optimizing-inference flow text: route, movable ceiling, fidelity, speculation, subagent listings.

GPU-free and torch-free: these read the plugin's markdown. test_skills.py checks links, commands and flags;
this file checks that the flow's decisions stay written down where an agent reads them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugin"
FLOW = PLUGIN / "skills" / "optimizing-inference"
REFS = FLOW / "references"
SKELETON = PLUGIN / "skills" / "optimizing-kernels" / "skeleton"
AGENTS = sorted((PLUGIN / "agents").glob("*.md"))


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# --- subagents are listed everywhere a client or reader looks for them ---------------------------------

LISTINGS = [ROOT / "AGENTS.md", REFS / "parallel.md", PLUGIN / "opencode" / "sync.py",
            PLUGIN / "opencode" / "install.py", PLUGIN / "opencode" / "README.md"]


@pytest.mark.parametrize("listing", LISTINGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_agent_is_listed(listing: Path):
    text = _read(listing)
    missing = [a.stem for a in AGENTS if a.stem not in text]
    assert not missing, f"{listing.relative_to(ROOT)} does not list {missing}"


def test_parallel_md_gives_every_claude_code_agent_type():
    text = _read(REFS / "parallel.md")
    missing = [a.stem for a in AGENTS if f"`inferencekit:{a.stem}`" not in text]
    assert not missing, missing


def test_log_reader_is_read_only_and_bounded():
    text = _read(PLUGIN / "agents" / "log-reader.md")
    tools = re.search(r"^tools:\s*(.+)$", text, re.M).group(1)
    assert {t.strip() for t in tools.split(",")} == {"Read", "Grep", "Glob", "Bash"}
    assert "at most 40 lines" in text
    rule = _read(REFS / "parallel.md")
    assert "~200 lines" in rule and "`log-reader`" in rule


# --- the flow's decisions ------------------------------------------------------------------------------

def test_headroom_is_three_estimates_not_one_over_x():
    for p in (FLOW / "SKILL.md", REFS / "stop-point.md"):
        text = _read(p)
        assert "`1/x`" not in text and "neither is the default" not in text, p.name
    stop = _read(REFS / "stop-point.md")
    for heading in ("## Headroom: three estimates", "## Route: build or patch"):
        assert heading in stop
    for key in ("floor_bytes_per_token", "packable_16bit_bytes", "lm_head_share"):
        assert f"`{key}`" in _read(REFS / "ceilings.md"), key


def test_intake_conventions_are_documented():
    intake = _read(REFS / "intake.md")
    for convention in ("route: build\\|patch", "speculation: off\\|auto", "fidelity: lossless\\|lossy",
                       "after_target: stop\\|keep_climbing", "`goals.after_target`", "`quality.budget`",
                       "## Given a written brief instead of an intake"):
        assert convention in intake, convention


def test_new_references_are_linked_and_small():
    skill = _read(FLOW / "SKILL.md")
    for name, cap in (("speculative.md", 6_000), ("fidelity.md", 6_000)):
        assert (REFS / name).stat().st_size <= cap, name
        assert f"(references/{name})" in skill, f"SKILL.md does not link references/{name}"


def test_reachability_uses_the_hard_bound():
    """A target is FAIL only above the hard bound (efficiency to 1.0); the inferred cap only orders work."""
    stop = _read(REFS / "stop-point.md")
    assert '"above the hard bound' in stop and "never makes a target unreachable" in stop
    assert "says when climbing is done" not in stop and "above the combined headroom" not in stop
    assert "efficiency taken to 1.0 of the moved ceiling" in stop


def test_sampled_speculation_needs_its_own_check():
    """`quality sampling` sees only the first token, which no draft touches."""
    spec = _read(REFS / "speculative.md")
    assert "only the first token" in spec and "accept/resample" in spec
    assert "../../proving-correctness/SKILL.md#sampled-exactness" in spec
    for p in (REFS / "fidelity.md", PLUGIN / "skills" / "benchmarking" / "references" / "reward-hacks.md",
              ROOT / "AGENTS.md"):
        assert "accept/resample" in _read(p), p.name
    wording = _read(PLUGIN / "skills" / "reporting-perf-claims" / "references" / "wording-guide.md")
    assert "greedy and sampled exactness PASS" not in wording


def test_speculative_speedup_baseline_is_defined():
    """Rule 9: unless the baseline was fixed, it gets its own best speculative row."""
    for p in (REFS / "speculative.md", PLUGIN / "skills" / "benchmarking" / "references" / "engine-configs.md"):
        text = _read(p)
        assert "the task or person fixed" in text and "faster of" in text, p.name
        assert "a fixed baseline may use it" not in text, p.name


def test_task_files_and_one_full_proof():
    intake = _read(REFS / "intake.md")
    assert "per measured result, screened or verified" in intake
    assert "at every new verified best" not in intake
    skill = _read(FLOW / "SKILL.md")
    assert "any result you bank" not in skill and "the commit you hand over" in skill


def test_one_precision_rule():
    """A lower format needs fidelity: lossy AND an allowed-list entry; only a task's rule joins the main table."""
    both = re.compile(r"`fidelity: lossy` and its (entry in )?`precision_allowed`")
    for p in (REFS / "fidelity.md", PLUGIN / "skills" / "evaluating-quantization" / "SKILL.md",
              REFS / "non-interactive.md"):
        text = _read(p)
        assert both.search(text), p.name
        assert "`fidelity: lossy` or when" not in text, p.name
    for p in (ROOT / "AGENTS.md", PLUGIN / "skills" / "inference-triage" / "references" / "evidence-rules.md"):
        assert "the task or intake" not in _read(p) and "task or intake explicitly" not in _read(p), p.name


def test_build_route_points_at_the_skeleton():
    for p, link in ((FLOW / "SKILL.md", "../optimizing-kernels/skeleton/README.md"),
                    (REFS / "candidates.md", "../../optimizing-kernels/skeleton/README.md"),
                    (REFS / "stop-point.md", "../../optimizing-kernels/skeleton/README.md")):
        assert f"]({link})" in _read(p), p.name
        assert (p.parent / link).resolve() == (SKELETON / "README.md").resolve()


# --- checks, deadlines and the one review ------------------------------------------------------------

COMMAND = PLUGIN / "commands" / "optimize-inference.md"
SKILLS = PLUGIN / "skills"


def test_light_checks_are_only_at_the_milestones():
    """The light check runs at the first >= 1.2x and the first >= 1.5x (or the target) only,
    with 3 gates and an 8-prompt quality score; no all-gates or 64-prompt variant, no flags for them."""
    files = [FLOW / "SKILL.md", REFS / "state.md", REFS / "harness-commands.md", REFS / "remote-host.md", COMMAND,
             SKILLS / "optimizing-kernels" / "SKILL.md", ROOT / "AGENTS.md"]
    for p in files:
        text = _read(p)
        for bad in ("--gates-light", "--quality-limit", "by more than the screen noise", "screens above the screen noise",
                    "first screen above"):
            assert bad not in text, f"{p.name}: {bad}"
    host = _read(REFS / "remote-host.md")
    assert "2 rounds on 32 prompts; gates caching, token_count, rare_tokens; quality on 8 prompts; GSM8K 100" in host
    assert "3 rounds on 64 prompts, all gates, quality on 64, GSM8K 1,319 paired, `quality sampling`" in host
    state = _read(REFS / "state.md")
    assert "one `ikstep job` ([remote-host.md](remote-host.md), \"One command per step\")" in state
    assert "the first candidate that screens >= 1.2x and the first >= 1.5x of the baseline (or the brief's target), never otherwise" in state


def test_timed_runs_work_to_the_deadline():
    """A timed run once froze at 48 min and quit at 94 of 120, judging the target unreachable from an estimate."""
    skill = _read(FLOW / "SKILL.md")
    for gone in ("no lever with headroom above the benchmark's resolution is left",
                 "| Headroom below what the benchmark resolves", "a `claims-skeptic` panel"):
        assert gone not in skill, gone
    assert "### 8. Final report (always runs, from the report reserve, not before)" in skill
    assert '"Unreachable" and "no headroom" are verdicts, never stops under a stated deadline' in skill
    assert "measured `ceiling bw` and `roofline`" in skill
    stop = _read(REFS / "stop-point.md")
    assert "never an estimated bandwidth, and it never stops a run with a stated deadline" in stop
    assert "Otherwise, stop and say why." not in stop
    assert "**Work to the deadline:**" in _read(COMMAND) and "never a stop" in _read(COMMAND)
    kern = _read(SKILLS / "optimizing-kernels" / "SKILL.md")
    assert "A stated deadline: work to it minus the report reserve." in kern


def test_one_skeptic_at_freeze():
    """Two skeptics once cost 63 calls and ~$3.2; one runs at freeze, with the union of their checks."""
    for p in (FLOW / "SKILL.md", COMMAND, REFS / "parallel.md"):
        assert "panel" not in _read(p), p.name
    # An intake run that reaches stage 8 without a stage-7 freeze still gets its one review.
    assert "No freeze (no stage 7): one `claims-skeptic` on the draft before the host is released." in _read(
        FLOW / "SKILL.md")
    assert "(no freeze: on the draft, before the host is released)" in _read(COMMAND)
    skeptic = _read(PLUGIN / "agents" / "claims-skeptic.md")
    assert re.search(r"^maxTurns: 50$", skeptic, re.M)
    for check in ("`ikstep digest`", "--runs-dir <evidence root>", "same-session records", "void record",
                  "measured `ceiling bw` and `roofline`", "on the timed path", "one head computation",
                  "re-checked after the last config change", "VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1",
                  "rental provider", "at most about 60 lines"):
        assert check in skeptic, check
    oc = _read(PLUGIN / "opencode" / "agents" / "claims-skeptic.md")
    assert re.search(r"^steps: 50$", oc, re.M), "sync.py maps maxTurns to OpenCode's steps"


def _step_parsers() -> dict:
    """Subcommand -> option strings, from the step tool's own argparse."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_flow_ikstep", FLOW / "scripts" / "ikstep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out: dict = {}
    for action in mod.parser()._actions:
        for sub, p in (getattr(action, "choices", None) or {}).items():
            if hasattr(p, "_actions"):
                out.setdefault(sub, set()).update(o for a in p._actions for o in a.option_strings)
                out.setdefault(f"{sub}:positional", set()).update(
                    c for a in p._actions if not a.option_strings for c in (a.choices or ()))
    return out


def test_check_jobs_exist_in_the_step_tool():
    """The check jobs remote-host.md lists are ikstep's own subcommands and flags."""
    parsers = _step_parsers()
    host = _read(REFS / "remote-host.md")
    cmds = re.findall(r"`\$STEP ((?:job|wait|digest)[^`]*)`", host)
    assert len(cmds) >= 6, cmds
    for cmd in cmds:
        sub = re.split(r"[\s\\|]", cmd, maxsplit=1)[0]
        assert sub in parsers, cmd
        for flag in re.findall(r"--[\w-]+", cmd):
            assert flag in parsers[sub], f"{cmd}: {flag} not in ikstep {sub}"
    assert {"baseline", "screen", "light", "full"} <= parsers["job:positional"], parsers["job:positional"]
    for flag in ("--dry-run", "--candidate", "--sampler-changed", "--baseline"):
        assert flag in parsers["job"], flag


# --- reading material stays model-agnostic ------------------------------------------------------------

# Built at runtime so this file does not trip the same greps. Model-specific code lives in the
# skeleton's models/; skills, references, agents and commands describe techniques for any model.
_SPECIFIC = re.compile("|".join([
    r"gem" + r"ma[- ]?4", r"qwen" + r"3\.8", r"qwen" + r"3[._]5",
    r"no[- ]?" + "plugin", r"without (the )?" + "plugin", r"control[- ]" + "arm", r"plugin " + "vs",
]), re.I)
READING = sorted(p for p in PLUGIN.rglob("*.md") if SKELETON not in p.parents) + [ROOT / "AGENTS.md"]


@pytest.mark.parametrize("path", READING, ids=lambda p: str(p.relative_to(ROOT)))
def test_reading_material_is_model_agnostic(path: Path):
    hits = [ln for ln, line in enumerate(_read(path).splitlines(), 1) if _SPECIFIC.search(line)]
    assert not hits, f"{path.relative_to(ROOT)}: model-specific or comparison wording on lines {hits}"


# --- the report opens in plain English; screens use two lengths; no private run tooling --------------

def test_report_starts_with_a_plain_english_summary():
    stop = _read(REFS / "stop-point.md")
    summary = stop.index("## Summary (top of REPORT.md, before every table)")
    assert summary < stop.index("## The verdict table (REPORT.md)")
    for line in ("**What you got:**", "**How much faster vs the baseline:**", "**Still correct?:**",
                 "**How to run it:**", "**What it cost:**", "agent time", "API list price", "GPU time"):
        assert line in stop[summary:], line
    for p in (FLOW / "SKILL.md", PLUGIN / "skills" / "reporting-perf-claims" / "SKILL.md",
              PLUGIN / "commands" / "perf-report.md"):
        assert "plain-English summary" in _read(p) and '"Summary"' in _read(p), p.name


def test_screens_use_short_and_long_and_promote_secondary_wins():
    tuning, cmds = _read(REFS / "tuning.md"), _read(REFS / "harness-commands.md")
    assert "--buckets short --buckets long --rounds 1" in tuning and "--buckets medium" not in tuning
    assert "--buckets short --buckets long --rounds 1" in cmds and "--buckets medium --rounds 1" not in cmds
    assert "**Also promote**" in tuning and "e.g. TTFT" in tuning and "summary lists it" in tuning
    assert "short` and `long`" in _read(FLOW / "SKILL.md")
