"""The plugin as a plugin-only agent reads it: skills, commands and agent docs must match the real CLI.

The plugin ships alone (only ``plugin/`` is copied), so everything it links must live inside it, and every
``inferencekit`` command and flag written in a code span or code block must exist in the registry with that
exact spelling: an agent follows these files literally.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from inferencekit import registry

from conftest import needs_repo_file

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugin"
SKILLS = PLUGIN / "skills"
SKILL_DIRS = sorted(p for p in SKILLS.iterdir() if p.is_dir())
PLUGIN_MD = sorted(PLUGIN.rglob("*.md"))
PLUGIN_TEXT = sorted(p for p in PLUGIN.rglob("*") if p.is_file() and p.suffix in {".md", ".json", ".py", ".sh", ".yaml"})
# Files an agent reads as instructions: the plugin, plus the repo-level agent docs and example intakes.
# Repo-level docs agents may read that not every checkout has: checked when present.
OPTIONAL_AGENT_DOCS = [p for p in sorted((ROOT / "docs").glob("*.md")) if p.is_file()]
AGENT_MD = PLUGIN_MD + [ROOT / "AGENTS.md"] + OPTIONAL_AGENT_DOCS

MAX_SKILL_LINES = 300
MAX_MD_LINES = 400
# Includes task-quality policies and conditionally loaded AMD/Apple operational recipes.
MAX_PLUGIN_BYTES = 730_000  # context guard for skills, references and scripts
# The two video references are read only for video work (the skills link them for that alone), so they have their
# own cap and do not count toward MAX_PLUGIN_BYTES.
VIDEO_REFERENCES = (PLUGIN / "skills" / "optimizing-inference" / "references" / "video.md",
                    PLUGIN / "skills" / "benchmarking" / "references" / "video-checks.md")
MAX_VIDEO_REFERENCE_BYTES = 16_000
# Loaded only after the final report or for an explicit contribution request; keep
# publication details out of the optimization context while bounding this reference.
CONTRIBUTION_REFERENCE = PLUGIN / "skills" / "optimizing-inference" / "references" / "engine-contribution.md"
MAX_CONTRIBUTION_REFERENCE_BYTES = 7_000
# The engine skeleton is code an agent copies to its run directory and edits; it is never read whole into
# context, so it has its own cap and does not count toward MAX_PLUGIN_BYTES.
SKELETON = PLUGIN / "skills" / "optimizing-kernels" / "skeleton"
MAX_SKELETON_BYTES = 600_000


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def _frontmatter(path: Path) -> dict:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, f"{_rel(path)}: no YAML frontmatter"
    fm = yaml.safe_load(m.group(1))
    assert isinstance(fm, dict), f"{_rel(path)}: frontmatter is not a mapping"
    return fm


# --- the CLI as the registry builds it ---------------------------------------------------------------

def _cli_table() -> dict[tuple[str, ...], set[str]]:
    table: dict[tuple[str, ...], set[str]] = {}
    for op in registry.load_all().values():
        flags = {"--help"}
        for a in op.args:
            if a.positional:
                continue
            flag = "--" + a.name.replace("_", "-")
            flags.add(flag)
            if a.type == "boolean":
                flags.add("--no-" + a.name.replace("_", "-"))
        table[tuple(op.cli)] = flags
    table[("mcp", "serve")] = {"--help"}
    return table


CLI = _cli_table()
HEADS = {c[0] for c in CLI}
GROUPS = {c[0] for c in CLI if len(c) == 2}
GLOBAL_FLAGS = {"--runs-dir", "--version", "--help"}
RAW_TOOLS = {"nsys", "ncu"}  # `nsys profile`, `ncu --set full` are the profilers themselves, not inferencekit
FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
# `inferencekit [--runs-dir X] <w1> [<w2>]`; the name must not be part of a longer word or path segment.
INVOKE_RE = re.compile(r"(?<![\w-])inferencekit(?:\s+--runs-dir\s+\S+)?\s+([a-z][a-z0-9-]*)(?:[ \t]+([a-z][a-z0-9-]*))?")


def _code_snippets(path: Path) -> list[tuple[int, str]]:
    """Inline code spans and fenced-block commands (continuation lines joined, comments dropped)."""
    out: list[tuple[int, str]] = []
    in_block = False
    pending = ""
    pending_ln = 0
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_block = not in_block
            pending = ""
            continue
        if in_block:
            code = re.sub(r"(^|\s)#.*$", "", line)
            if not pending:
                pending_ln = ln
            if code.rstrip().endswith("\\"):
                pending += code.rstrip()[:-1] + " "
                continue
            out.append((pending_ln, pending + code))
            pending = ""
        else:
            out.extend((ln, m.group(1)) for m in re.finditer(r"`([^`\n]+)`", line))
    return out


def _strip_quoted_args(cmd: str) -> str:
    # --serve-args carries bench serve's flags, --extra-body / --chat-template-kwargs carry JSON, and
    # <placeholders> are prose ("<path to the inferencekit wheel>", "<exact inferencekit commands>").
    cmd = re.sub(r"""(--[a-z-]+)\s+(?:"[^"]*"|'[^']*')""", r"\1 X", cmd)
    return re.sub(r"<[^<>]*>", "<>", cmd)


def _invocations(path: Path):
    """(line, command tuple or None, words, flags) for every `inferencekit ...` in code."""
    for ln, snippet in _code_snippets(path):
        snippet = _strip_quoted_args(snippet)
        for m in INVOKE_RE.finditer(snippet):
            w1, w2 = m.group(1), m.group(2)
            if w1 in GROUPS:
                cmd = (w1, w2) if (w1, w2) in CLI else None
                if cmd is None and w2 is None:
                    continue  # `inferencekit quality <op> --help`: a placeholder subcommand
            else:
                cmd = (w1,) if (w1,) in CLI else None
            rest = snippet[m.end():]
            for stop in (" -- ", "|", ";", "&&", "inferencekit", ">", " #"):
                i = rest.find(stop)
                if i != -1:
                    rest = rest[:i]
            yield ln, cmd, (w1, w2), FLAG_RE.findall(rest)


# --- skills and commands ----------------------------------------------------------------------------

def test_skill_dirs_exist():
    names = {p.name for p in SKILL_DIRS}
    assert "optimizing-inference" in names
    assert len(names) >= 7, names


@pytest.mark.parametrize("skill", SKILL_DIRS, ids=lambda p: p.name)
def test_skill_frontmatter(skill: Path):
    md = skill / "SKILL.md"
    assert md.exists(), f"{skill.name}: no SKILL.md"
    fm = _frontmatter(md)
    assert fm.get("name") == skill.name, f"{skill.name}: frontmatter name {fm.get('name')!r} != directory"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", skill.name) and len(skill.name) <= 64
    desc = fm.get("description")
    assert isinstance(desc, str) and desc.strip(), f"{skill.name}: no description"
    assert len(desc) <= 1024, f"{skill.name}: description is {len(desc)} chars (max 1024)"
    assert "disable-model-invocation" not in fm, f"{skill.name}: disable-model-invocation hides it from the model"
    lines = len(md.read_text().splitlines())
    assert lines <= MAX_SKILL_LINES, f"{skill.name}/SKILL.md: {lines} lines > {MAX_SKILL_LINES}; move depth to references/"


@pytest.mark.parametrize("cmd", sorted((PLUGIN / "commands").glob("*.md")), ids=lambda p: p.name)
def test_command_frontmatter(cmd: Path):
    fm = _frontmatter(cmd)
    assert isinstance(fm.get("description"), str) and fm["description"].strip()
    assert "disable-model-invocation" not in fm
    for skill_path in re.findall(r"\$\{CLAUDE_PLUGIN_ROOT\}/([\w./-]+)", cmd.read_text()):
        assert (PLUGIN / skill_path).exists(), f"{cmd.name}: ${{CLAUDE_PLUGIN_ROOT}}/{skill_path} does not exist"


def test_file_sizes_bounded():
    big = [f"{_rel(p)}: {n} lines" for p in PLUGIN_MD if (n := len(p.read_text().splitlines())) > MAX_MD_LINES]
    assert not big, big
    # __pycache__ is git-ignored and never ships (test_opencode.py creates one when it imports install.py).
    files = [p for p in PLUGIN.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    skel = sum(p.stat().st_size for p in files if SKELETON in p.parents)
    video = sum(p.stat().st_size for p in VIDEO_REFERENCES)
    contribution = CONTRIBUTION_REFERENCE.stat().st_size
    total = sum(p.stat().st_size for p in files) - skel - video - contribution
    assert total <= MAX_PLUGIN_BYTES, f"plugin/ (without the skeleton and separately capped references) is {total} bytes"
    assert skel <= MAX_SKELETON_BYTES, f"the engine skeleton is {skel} bytes"
    assert video <= MAX_VIDEO_REFERENCE_BYTES, f"the video references are {video} bytes"
    assert contribution <= MAX_CONTRIBUTION_REFERENCE_BYTES, f"the contribution reference is {contribution} bytes"


# --- subagents (plugin/agents/) ------------------------------------------------------------------------

AGENTS = sorted((PLUGIN / "agents").glob("*.md"))
# Claude Code ignores these in plugin agents: setting one means the file relies on behaviour it will not get.
IGNORED_PLUGIN_AGENT_KEYS = {"permissionMode", "hooks", "mcpServers", "initialPrompt"}


def test_agents_shipped_and_used():
    names = {p.stem for p in AGENTS}
    assert {"engine-scout", "claims-skeptic", "profile-analyst"} <= names, names
    flow = (SKILLS / "optimizing-inference" / "SKILL.md").read_text()
    missing = [n for n in names if f"`{n}`" not in flow]
    assert not missing, f"optimizing-inference/SKILL.md never names these agents: {missing}"


@pytest.mark.parametrize("agent", AGENTS, ids=lambda p: p.name)
def test_agent_frontmatter(agent: Path):
    fm = _frontmatter(agent)
    assert fm.get("name") == agent.stem, f"{agent.name}: frontmatter name {fm.get('name')!r} != file name"
    desc = fm.get("description")
    assert isinstance(desc, str) and desc.strip(), f"{agent.name}: no description"
    assert not IGNORED_PLUGIN_AGENT_KEYS & set(fm), f"{agent.name}: {IGNORED_PLUGIN_AGENT_KEYS & set(fm)} are ignored"
    tools = fm.get("tools")
    assert isinstance(tools, str) and tools.strip(), f"{agent.name}: give an explicit tools allowlist"
    # The agents only read, search and analyse: none of them may edit files.
    assert not {"Edit", "Write", "NotebookEdit"} & {t.strip() for t in tools.split(",")}, f"{agent.name}: {tools}"


# --- descriptions load on every request ----------------------------------------------------------------

# Claude Code lists every skill and command as `- name: description[ - when_to_use]` and every agent with its
# description, in every request. An earlier version had 10,870 chars of them; the cut to ~6,100 keeps the trigger
# phrases and drops method detail, which lives in the bodies.
MAX_DESCRIPTION_CHARS = 6_500


def test_descriptions_stay_short():
    files = [s / "SKILL.md" for s in SKILL_DIRS] + AGENTS + sorted((PLUGIN / "commands").glob("*.md"))
    sizes = {_rel(p): sum(len(str(fm.get(k) or "")) for k in ("description", "when_to_use"))
             for p in files for fm in [_frontmatter(p)]}
    total = sum(sizes.values())
    biggest = sorted(sizes.items(), key=lambda kv: -kv[1])[:3]
    assert total <= MAX_DESCRIPTION_CHARS, f"descriptions total {total} chars > {MAX_DESCRIPTION_CHARS}; biggest {biggest}"


# --- the plugin ships alone ---------------------------------------------------------------------------

def test_relative_links_stay_inside_the_plugin():
    bad = []
    plugin = PLUGIN.resolve()
    for md in PLUGIN_MD:
        for link in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", md.read_text()):
            if re.match(r"^[a-z]+:", link):
                continue  # http:, https:, mailto:
            target = (md.parent / link).resolve()
            if not target.exists():
                bad.append(f"{_rel(md)}: broken link {link}")
            elif plugin != target and plugin not in target.parents:
                bad.append(f"{_rel(md)}: link {link} leaves plugin/ (the plugin is installed alone)")
    assert not bad, "\n".join(bad)


def test_skill_local_paths_exist():
    """`references/x.md` and `scripts/x.py` written in a skill's code spans resolve inside that skill."""
    bad = []
    for skill in SKILL_DIRS:
        for md in skill.rglob("*.md"):
            for ln, snippet in _code_snippets(md):
                for m in re.finditer(r"(?<![\w/.-])((?:references|scripts)/[\w.-]+\.(?:md|py|sh))", snippet):
                    if not (skill / m.group(1)).exists():
                        bad.append(f"{_rel(md)}:{ln}: {m.group(1)} is not in {skill.name}/")
    assert not bad, "\n".join(bad)


def test_no_repo_paths_an_installed_plugin_cannot_open():
    """A plugin-only agent has no repository checkout: do not send it to repo-root directories."""
    bad = []
    for md in PLUGIN_MD:
        for ln, snippet in _code_snippets(md):
            if re.match(r"^(\.\./)*(docs|examples|schemas|hero|evals|engines|benchmarks)/", snippet.strip()):
                bad.append(f"{_rel(md)}:{ln}: `{snippet.strip()}`")
    assert not bad, "\n".join(bad)


# --- commands and flags match the CLI ------------------------------------------------------------------

def test_cli_table_has_the_contract_commands():
    for cmd in [("bench", "serve"), ("bench", "rounds"), ("bench", "decode"), ("gates", "run"), ("prompts", "verify"),
                ("quality", "trajectory"), ("quality", "score"), ("review", "build"), ("ledger", "trailers"),
                ("history",), ("intake", "validate"), ("intake", "budget"), ("manifest", "validate"), ("roofline",),
                ("ceiling", "bw"), ("nsys", "capture"), ("ncu", "summarize"), ("compare",), ("correctness",),
                ("report",), ("doctor",)]:
        assert cmd in CLI, cmd


@pytest.mark.parametrize("path", AGENT_MD, ids=_rel)
def test_documented_commands_and_flags_exist(path: Path):
    bad = []
    for ln, cmd, words, flags in _invocations(path):
        if cmd is None:
            bad.append(f"{_rel(path)}:{ln}: no command `inferencekit {' '.join(w for w in words if w)}`")
            continue
        for flag in flags:
            if flag not in CLI[cmd] and flag not in GLOBAL_FLAGS:
                bad.append(f"{_rel(path)}:{ln}: `inferencekit {' '.join(cmd)}` has no {flag}")
    assert not bad, "\n".join(bad)


def test_serve_args_are_bench_serve_flags():
    """`bench rounds --serve-args "..."` is appended to every bench serve call: its flags must be bench serve's."""
    bad = []
    for md in AGENT_MD:
        for ln, snippet in _code_snippets(md):
            for m in re.finditer(r"""--serve-args\s+(?:"((?:[^"\\]|\\.)*)"|'([^']*)')""", snippet):
                inner = re.sub(r"""(--[a-z-]+)\s+(?:\\"[^"]*\\"|'[^']*')""", r"\1 X", m.group(1) or m.group(2) or "")
                bad += [f"{_rel(md)}:{ln}: --serve-args has {f}, which bench serve does not take"
                        for f in FLAG_RE.findall(inner) if f not in CLI[("bench", "serve")]]
    assert not bad, "\n".join(bad)


def test_unprefixed_command_spans_exist():
    """`bench serve`, `quality trajectory`, ... written without the `inferencekit` prefix are real commands."""
    bad = []
    for md in AGENT_MD:
        for ln, snippet in _code_snippets(md):
            m = re.match(r"^(?:\$IK\s+)?([a-z]+)[ \t]+([a-z][a-z0-9-]*)(\s.*)?$", _strip_quoted_args(snippet.strip()))
            if not m or m.group(1) not in GROUPS - RAW_TOOLS:
                continue
            cmd = (m.group(1), m.group(2))
            if cmd not in CLI:
                bad.append(f"{_rel(md)}:{ln}: no command `{' '.join(cmd)}`")
                continue
            rest = re.split(r" -- |\||;|&&| #", m.group(3) or "")[0]
            bad += [f"{_rel(md)}:{ln}: `{' '.join(cmd)}` has no {f}" for f in FLAG_RE.findall(rest)
                    if f not in CLI[cmd] and f not in GLOBAL_FLAGS]
    assert not bad, "\n".join(bad)


def _repeatable() -> dict[tuple[str, ...], set[str]]:
    out: dict[tuple[str, ...], set[str]] = {}
    for op in registry.load_all().values():
        out[tuple(op.cli)] = {"--" + a.name.replace("_", "-") for a in op.args if a.type == "array" and not a.positional}
    return out


def test_repeatable_flags_take_one_value_each():
    """argparse `append` flags take one value per flag: `--trajectories a b` is an error, `... a --trajectories b` is not."""
    rep = _repeatable()
    bad = []
    for md in PLUGIN_MD:
        for ln, snippet in _code_snippets(md):
            snippet = _strip_quoted_args(snippet)
            for m in INVOKE_RE.finditer(snippet):
                key = (m.group(1), m.group(2)) if (m.group(1), m.group(2)) in rep else (m.group(1),)
                for flag in rep.get(key, ()):
                    for hit in re.finditer(re.escape(flag) + r"\s+(\S+)\s+([^\s\\]+)", snippet[m.end():]):
                        nxt = hit.group(2)
                        if not nxt.startswith(("-", "#", "|", ";", "&", ">", "[", "]", "(")):
                            bad.append(f"{_rel(md)}:{ln}: `{flag} {hit.group(1)} {nxt}`: repeat {flag} per value")
    assert not bad, "\n".join(bad)


def test_mcp_tool_names_exist():
    """Tool names listed on an "MCP" line (`bench_rounds`, `run_gates`, ...) are registered operations."""
    names = set(registry.load_all())
    bad = [f"{_rel(md)}:{ln}: no MCP tool `{t}`"
           for md in AGENT_MD for ln, line in enumerate(md.read_text().splitlines(), 1) if "MCP" in line
           for t in re.findall(r"`([a-z]+(?:_[a-z0-9]+)+)`", line) if t not in names]
    assert not bad, "\n".join(bad)


EXAMPLE_INTAKES = ROOT / "examples" / "intake"  # not in the plugin repository: checked when present
# The end-to-end test's copy of the Qwen example intake (tests/test_e2e_fake.py checks that they match).
FIXTURE_INTAKES = ROOT / "tests" / "fixtures" / "intake"


@needs_repo_file(EXAMPLE_INTAKES, "examples/intake/")
def test_example_intakes_validate():
    examples = sorted(EXAMPLE_INTAKES.glob("*.yaml"))
    assert examples, f"no example intakes in {_rel(EXAMPLE_INTAKES)}"
    for p in examples:
        out = registry.call("intake_validate", {"file": str(p), "allow_placeholders": True})
        assert out["ok"], (p.name, out)


# --- names that never ship ----------------------------------------------------------------------------

# Built at runtime so this file never contains the name itself.
_PROVIDER = "".join(["v", "a", "s", "t"])
PROVIDER_RE = re.compile(r"(?i)(?<![a-z])" + _PROVIDER + r"(?:[\s._-]*ai)?(?![a-z])")


@pytest.mark.parametrize("path", PLUGIN_TEXT + [ROOT / "AGENTS.md"] + OPTIONAL_AGENT_DOCS
                         + sorted(EXAMPLE_INTAKES.glob("*")) + sorted(FIXTURE_INTAKES.glob("*")), ids=_rel)
def test_no_gpu_provider_name(path: Path):
    hits = [ln for ln, line in enumerate(path.read_text().splitlines(), 1) if PROVIDER_RE.search(line)]
    assert not hits, f"{_rel(path)}: GPU rental provider named on lines {hits}; say 'a rented GPU host'"
