#!/usr/bin/env python3
"""Check that every place that states inferencekit's version agrees.

Sources (all must name the same version):
  - pyproject.toml              [project] version
  - src/inferencekit/__init__.py __version__
  - plugin/.claude-plugin/plugin.json      version
  - .claude-plugin/marketplace.json        metadata.version and the inferencekit plugin entry's version
  - plugin/.mcp.json            the ``inferencekit-cli[mcp]==X`` pin the MCP launcher installs
  - plugin/skills/*/SKILL.md    each skill's frontmatter ``metadata.version``
  - plugin/skills/optimizing-inference/references/manifest.md   the example's ``"inferencekit_version"``
  - any ``inferencekit-cli...==X`` pin written in README.md, AGENTS.md, docs/ or plugin/ text files

The skills and the manifest example are checked when they exist (a partial copy of the sources has
neither). A SKILL.md with frontmatter but no metadata.version, or a manifest.md without
inferencekit_version, is an error.

With ``--release`` it also fails while the placeholder organisation ORG is still in place: a
``github.com/ORG/...`` URL (links and image sources alike) or an ``ORG/inferencekit...`` repo slug
(``claude plugin marketplace add ORG/inferencekit``) in pyproject.toml, README.md, AGENTS.md, docs/ or
plugin/ text files. README.md is also the PyPI long description, so a missed ORG would ship twice.

Standard library only (Python 3.10+), so CI can run it before anything is installed:
    python scripts/release/check_versions.py [--root DIR] [--release] [--json]
Exit code 0 when everything agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PIN_RX = re.compile(r"inferencekit-cli(?:\[[a-z0-9,_-]+\])?==([0-9][0-9A-Za-z.+!-]*)")  # a concrete version, not ==X
PLACEHOLDER_RX = re.compile(r"github\.com/ORG/|\bORG/inferencekit")  # URL, raw link or repo slug
TEXT_SUFFIXES = {".md", ".json", ".txt", ".toml", ".yaml", ".yml", ".sh"}
PIN_DIRS = ("docs", "plugin")
PIN_FILES = ("README.md", "AGENTS.md")
SKILLS_DIR = "plugin/skills"
MANIFEST_EXAMPLE = "plugin/skills/optimizing-inference/references/manifest.md"
MANIFEST_VERSION_RX = re.compile(r'"inferencekit_version"\s*:\s*"([^"]*)"')
SKILL_VERSION_RX = re.compile(r"""\s+version\s*:\s*["']?([^"'\s#]+)["']?\s*(?:#.*)?""")


def _pyproject_version(text: str) -> str | None:
    """[project] version, without tomllib (absent on Python 3.10)."""
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]").strip()
            continue
        if section == "project":
            m = re.match(r'version\s*=\s*"([^"]+)"', stripped)
            if m:
                return m.group(1)
    return None


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _skill_version(text: str) -> str | None:
    """``metadata.version`` from a SKILL.md's YAML frontmatter, without a YAML parser."""
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    in_metadata = False
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():  # a top-level key starts or ends the metadata block
            in_metadata = line.split(":", 1)[0].strip() == "metadata"
            continue
        if in_metadata:
            v = SKILL_VERSION_RX.fullmatch(line)
            if v:
                return v.group(1)
    return None


def collect(root: Path) -> tuple[dict[str, str | None], list[str]]:
    """Return ({source: version or None}, [problems reading the sources])."""
    root = Path(root)
    found: dict[str, str | None] = {}
    problems: list[str] = []

    def read(rel: str, fn):
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
            return
        try:
            fn(path)
        except (ValueError, KeyError, TypeError) as e:
            problems.append(f"{rel}: unreadable ({e})")

    read("pyproject.toml", lambda p: found.__setitem__("pyproject.toml [project].version",
                                                         _pyproject_version(p.read_text(encoding="utf-8"))))

    def init(p: Path):
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', p.read_text(encoding="utf-8"), re.M)
        found["src/inferencekit/__init__.py __version__"] = m.group(1) if m else None

    read("src/inferencekit/__init__.py", init)
    read("plugin/.claude-plugin/plugin.json",
         lambda p: found.__setitem__("plugin/.claude-plugin/plugin.json version", _json(p).get("version")))

    def marketplace(p: Path):
        d = _json(p)
        found[".claude-plugin/marketplace.json metadata.version"] = (d.get("metadata") or {}).get("version")
        entries = [e for e in d.get("plugins", []) if e.get("name") == "inferencekit"]
        if not entries:
            problems.append(".claude-plugin/marketplace.json: the plugins list has no entry named inferencekit")
        for e in entries:
            found[".claude-plugin/marketplace.json plugins[inferencekit].version"] = e.get("version")

    read(".claude-plugin/marketplace.json", marketplace)

    def mcp(p: Path):
        server = (_json(p).get("mcpServers") or {}).get("inferencekit")
        if not server:
            problems.append("plugin/.mcp.json: no mcpServers.inferencekit entry")
            return
        pins = [m.group(1) for a in server.get("args", []) for m in [PIN_RX.fullmatch(a)] if m]
        if len(pins) != 1:
            problems.append(f"plugin/.mcp.json: expected exactly one inferencekit-cli[mcp]==X pin in args, got {pins}")
        found["plugin/.mcp.json uvx pin"] = pins[0] if pins else None

    read("plugin/.mcp.json", mcp)

    # Each skill states the version it was written for; the manifest example states the harness version.
    skills = root / SKILLS_DIR
    for md in sorted(skills.glob("*/SKILL.md")) if skills.is_dir() else []:
        text = md.read_text(encoding="utf-8")
        if text.startswith("---\n"):  # no frontmatter at all is tests/test_skills.py's failure, not a version's
            found[f"{md.relative_to(root)} metadata.version"] = _skill_version(text)
    if (root / MANIFEST_EXAMPLE).is_file():
        versions = MANIFEST_VERSION_RX.findall((root / MANIFEST_EXAMPLE).read_text(encoding="utf-8"))
        for i, version in enumerate(versions or [None]):
            suffix = f" #{i + 1}" if len(versions) > 1 else ""
            found[f"{MANIFEST_EXAMPLE} inferencekit_version{suffix}"] = version

    # Pins written in prose (install instructions) must not drift either.
    found.update(_written_pins(root))
    return found, problems


def _written_pins(root: Path) -> dict[str, str]:
    """{"path:line pin": version} for every inferencekit-cli pin written in the text files."""
    checked: dict[str, str] = {}
    for path in _text_files(root):
        if path.name == ".mcp.json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for m in PIN_RX.finditer(line):
                checked[f"{path.relative_to(root)}:{i} pin"] = m.group(1)
    return checked


def _text_files(root: Path) -> list[Path]:
    """README.md, AGENTS.md and the text files under docs/ and plugin/ (the files that state pins or links)."""
    paths = [root / f for f in PIN_FILES] + [p for d in PIN_DIRS for p in sorted((root / d).rglob("*"))]
    return [p for p in paths if p.is_file() and p.suffix in TEXT_SUFFIXES]


def placeholders(root: Path) -> list[tuple[str, str]]:
    """Every line that still names the placeholder organisation ORG, as (``path:line``, text)."""
    root = Path(root)
    hits: list[tuple[str, str]] = []
    for path in [root / "pyproject.toml"] + _text_files(root):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if PLACEHOLDER_RX.search(line):
                hits.append((f"{path.relative_to(root)}:{i}", line.strip()[:160]))
    return hits


def check(root: Path, release: bool = False) -> tuple[dict[str, str | None], list[str]]:
    """Return (versions found, errors). No errors means every source agrees."""
    found, errors = collect(root)
    for source, version in found.items():
        if version is None:
            errors.append(f"{source}: no version found")
    expected = found.get("pyproject.toml [project].version")
    if expected:
        for source, version in found.items():
            if version is not None and version != expected:
                errors.append(f"{source} = {version}, but pyproject.toml says {expected}")
    if release:
        for where, text in placeholders(root):
            errors.append(f"{where}: placeholder repository URL (ORG) still in place: {text}")
    return found, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]), help="repository root")
    ap.add_argument("--release", action="store_true",
                    help="also fail while the placeholder organisation ORG is still in place")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    found, errors = check(Path(args.root), release=args.release)
    if args.json:
        print(json.dumps({"ok": not errors, "versions": found, "errors": errors},
                         indent=2))
    else:
        width = max((len(s) for s in found), default=0)
        for source, version in found.items():
            print(f"{source:<{width}}  {version}")
        sys.stdout.flush()
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print("versions agree" if not errors else f"{len(errors)} problem(s)", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
