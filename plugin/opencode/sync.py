#!/usr/bin/env python3
"""Build the OpenCode copies of this plugin's commands and agents from the Claude Code originals.

    python3 plugin/opencode/sync.py           # rewrite the copies
    python3 plugin/opencode/sync.py --check   # exit 1 if they are stale (the tests run it)

Edit plugin/commands/ and plugin/agents/ only, then run this. For OpenCode (checked against v1.19.0), commands
name skills instead of ${CLAUDE_PLUGIN_ROOT} paths and get "OpenCode notes"; agents lose `name`, `tools` (OpenCode
refuses a string) and `model`, and get `steps`, `mode: subagent` and a `permission` block from their tools.

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent
GENERATED = "# Generated: edit plugin/{kind}/{name}"

COMMAND_NOTES = """
## OpenCode notes

- **Authorization:** reuse the approved spending/scope; ask only for new decisions. Plans are opt-in. Respect the client's current execution permissions.
- **Subagents** (engine-scout, quant-scout, profile-analyst, claims-skeptic, log-reader) are read-only. Independent kernel writing may use implementation-capable workers with isolated files. Serialize integration and device measurements.
- **Long steps** (`bench rounds`, `ceiling bw`, quality runs, captures) **use the CLI, detached with a log:** OpenCode ends an MCP call at its `timeout`; this server sends no progress.
"""


def split_frontmatter(text: str, source: Path) -> tuple[list[str], str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise SystemExit(f"{source}: no YAML frontmatter")
    return m.group(1).splitlines(), text[m.end():]


def top_level(lines: list[str]) -> dict[str, str]:
    """Top-level `key: value` lines, with the value exactly as written (quoting kept)."""
    out = {}
    for line in lines:
        m = re.match(r"^([A-Za-z][\w-]*):\s?(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def command(src: Path) -> str:
    fm, body = split_frontmatter(src.read_text(encoding="utf-8"), src)
    keep = [line for line in fm if not line.startswith("argument-hint:")]
    # `${CLAUDE_PLUGIN_ROOT}/skills/<s>/SKILL.md` (in a checkout of this repo: `plugin/skills/<s>/SKILL.md`)
    body = re.sub(r"`\$\{CLAUDE_PLUGIN_ROOT\}/skills/([a-z0-9-]+)/SKILL\.md`"
                  r"(?: \(in a checkout of this repo: `plugin/skills/\1/SKILL\.md`\))?",
                  r"the `\1` skill (load it with the skill tool, which also prints its base directory)", body)
    body = re.sub(r"`\$\{CLAUDE_PLUGIN_ROOT\}/skills/([a-z0-9-]+)/([\w./-]+)`", r"the \1 skill's `\2`", body)
    body = re.sub(r"^- \*\*Plan mode is optional:\*\* do not enter it yourself; the person may review the plan in it "
                  r"\(Shift\+Tab, or `claude --permission-mode plan`\)\.$",
                  "- **Approval in OpenCode: the same direct question** (notes below); its Plan agent does not guard "
                  "GPU spend.", body, flags=re.M)
    if "Shift+Tab" in body or "permission-mode plan" in body:
        raise SystemExit(f"{src}: a Claude Code plan-mode instruction sync.py does not know how to rewrite")
    if "CLAUDE_PLUGIN_ROOT" in body:
        raise SystemExit(f"{src}: a ${{CLAUDE_PLUGIN_ROOT}} path sync.py does not know how to rewrite")
    # OpenCode expands !`cmd` (runs it) and @path (inlines the file) in a command template.
    if re.search(r"!`", body) or re.search(r"(?<![\w`])@", body):
        raise SystemExit(f"{src}: contains !` or @, which OpenCode would run or expand in a command template")
    header = GENERATED.format(kind="commands", name=src.name)
    return "---\n" + header + "\n" + "\n".join(keep) + "\n---\n" + body.lstrip("\n").rstrip("\n") + "\n" + COMMAND_NOTES


def agent(src: Path) -> str:
    fm, body = split_frontmatter(src.read_text(encoding="utf-8"), src)
    keys = top_level(fm)
    if "description" not in keys:
        raise SystemExit(f"{src}: no description")
    tools = {t.strip() for t in keys.get("tools", "").split(",") if t.strip()}

    def allow(tool: str) -> str:
        return "allow" if tool in tools else "deny"

    out = [
        "---",
        GENERATED.format(kind="agents", name=src.name),
        f"description: {keys['description']}",
        "mode: subagent",
        *([f"steps: {keys['maxTurns']}"] if "maxTurns" in keys else []),
        "permission:",
        "  edit: deny",
        f"  bash: {allow('Bash')}",
        f"  webfetch: {allow('WebFetch')}",
        f"  websearch: {allow('WebSearch')}",
        "  task: deny",
        "---",
    ]
    return "\n".join(out) + "\n" + body.rstrip("\n") + "\n"


def build() -> dict[Path, str]:
    files = {}
    for src in sorted((PLUGIN / "commands").glob("*.md")):
        files[HERE / "commands" / src.name] = command(src)
    for src in sorted((PLUGIN / "agents").glob("*.md")):
        files[HERE / "agents" / src.name] = agent(src)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the OpenCode files are out of date")
    args = ap.parse_args(argv)
    want = build()
    stale = sorted(p for kind in ("commands", "agents") for p in (HERE / kind).glob("*.md") if p not in want)
    differ = [p for p, text in want.items() if not p.is_file() or p.read_text(encoding="utf-8") != text]
    if args.check:
        for p in differ:
            print(f"out of date: {p.relative_to(PLUGIN.parent)}", file=sys.stderr)
        for p in stale:
            print(f"no source any more: {p.relative_to(PLUGIN.parent)}", file=sys.stderr)
        if differ or stale:
            print("run: python3 plugin/opencode/sync.py", file=sys.stderr)
            return 1
        return 0
    for p, text in want.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    for p in stale:
        p.unlink()
    print(f"wrote {len(want)} files, removed {len(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
