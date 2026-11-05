"""The OpenCode port (plugin/opencode/): generated files in sync, shapes OpenCode accepts, a safe installer.

OpenCode itself is not needed: the shapes checked here are the ones OpenCode v1.19.0 parses (command
frontmatter keys, agent `permission` values, the `mcp` local-server entry). sync.py and install.py are
standard-library scripts, so they are run with this interpreter.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import inferencekit

ROOT = Path(__file__).resolve().parents[1]
OC = ROOT / "plugin" / "opencode"
COMMAND_KEYS = {"description", "agent", "model", "variant", "subtask"}  # OpenCode's command schema, minus template
AGENT_KEYS = {"description", "mode", "permission", "model", "temperature", "steps", "hidden", "color"}
ACTIONS = {"allow", "deny", "ask"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"opencode_{name}", OC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


install = _load("install")


def _frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, f"{path.name}: no frontmatter"
    return yaml.safe_load(m.group(1)), text[m.end():]


def test_generated_files_are_in_sync():
    r = subprocess.run([sys.executable, "-I", str(OC / "sync.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_one_opencode_file_per_claude_file():
    for kind in ("commands", "agents"):
        claude = sorted(p.name for p in (ROOT / "plugin" / kind).glob("*.md"))
        assert claude and claude == sorted(p.name for p in (OC / kind).glob("*.md")), kind


@pytest.mark.parametrize("path", sorted((OC / "commands").glob("*.md")), ids=lambda p: p.name)
def test_commands_fit_opencode(path: Path):
    fm, body = _frontmatter(path)
    assert set(fm) <= COMMAND_KEYS, set(fm) - COMMAND_KEYS
    assert isinstance(fm.get("description"), str) and fm["description"].strip()
    assert "CLAUDE_PLUGIN_ROOT" not in body  # only Claude Code substitutes it
    assert "$ARGUMENTS" in body
    assert "!`" not in body and not re.search(r"(?<![\w`])@", body)  # OpenCode runs / inlines these
    for skill in re.findall(r"the `([a-z0-9-]+)` skill", body):
        assert (ROOT / "plugin" / "skills" / skill / "SKILL.md").is_file(), skill


@pytest.mark.parametrize("path", sorted((OC / "agents").glob("*.md")), ids=lambda p: p.name)
def test_agents_fit_opencode(path: Path):
    fm, _ = _frontmatter(path)
    assert set(fm) <= AGENT_KEYS, set(fm) - AGENT_KEYS  # `tools` as a string would make OpenCode refuse the file
    assert fm["mode"] == "subagent"
    perm = fm["permission"]
    assert set(perm.values()) <= ACTIONS
    assert perm["edit"] == "deny" and perm["task"] == "deny"
    claude = yaml.safe_load(re.match(r"^---\n(.*?)\n---\n", (ROOT / "plugin" / "agents" / path.name).read_text(),
                                     re.S).group(1))
    tools = {t.strip() for t in claude["tools"].split(",")}
    assert (perm["bash"] == "allow") == ("Bash" in tools)
    assert (perm["webfetch"] == "allow") == ("WebFetch" in tools)


def test_mcp_entry_matches_the_claude_launcher():
    entry = install.mcp_entry()
    assert entry["type"] == "local" and entry["enabled"] is True and entry["timeout"] >= 60_000
    assert entry["command"] == ["uvx", "--from", f"inferencekit-cli[mcp]=={inferencekit.__version__}",
                                "inferencekit", "mcp", "serve"]


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(OC / "install.py"), *map(str, args)],
                          capture_output=True, text=True)


def test_install_is_idempotent_keeps_user_keys_and_uninstalls(tmp_path):
    cfg_dir = tmp_path / "opencode"
    cfg_dir.mkdir()
    cfg = cfg_dir / "opencode.json"
    cfg.write_text(json.dumps({"model": "p/m", "skills": {"paths": ["/mine"]}, "mcp": {"other": {"type": "remote",
                                                                                              "url": "https://x"}}}))
    mine = cfg_dir / "commands" / "bench.md"
    mine.parent.mkdir()
    mine.write_text("---\ndescription: mine\n---\nmine\n")

    r = _run("--config-dir", cfg_dir)
    assert r.returncode == 0, r.stderr
    assert "skipped" in r.stderr and mine.read_text().endswith("mine\n")  # never replaces a user's file
    data = json.loads(cfg.read_text())
    assert data["model"] == "p/m" and "other" in data["mcp"]
    assert data["skills"]["paths"] == ["/mine", str(ROOT / "plugin" / "skills")]
    assert data["mcp"]["inferencekit"] == install.mcp_entry()
    for kind in ("commands", "agents"):
        for src in (OC / kind).glob("*.md"):
            link = cfg_dir / kind / src.name
            if link != mine:
                assert link.is_symlink() and link.resolve() == src.resolve()

    before = cfg.read_text()
    r = _run("--config-dir", cfg_dir)
    assert r.returncode == 0 and "already installed" in r.stdout and cfg.read_text() == before

    r = _run("--uninstall", "--config-dir", cfg_dir)
    assert r.returncode == 0, r.stderr
    data = json.loads(cfg.read_text())
    assert data == {"model": "p/m", "skills": {"paths": ["/mine"]},
                    "mcp": {"other": {"type": "remote", "url": "https://x"}}}
    assert mine.is_file() and not any(p.is_symlink() for p in cfg_dir.rglob("*.md"))


def test_install_refuses_to_edit_a_config_with_comments(tmp_path):
    cfg = tmp_path / "opencode.json"
    text = '{\n  // my settings\n  "model": "p/m",\n}\n'
    cfg.write_text(text)
    r = _run("--config-dir", tmp_path)
    assert r.returncode == 2 and '"inferencekit"' in r.stderr
    assert cfg.read_text() == text and not (tmp_path / "commands").exists()


def test_dry_run_writes_nothing(tmp_path):
    r = _run("--config-dir", tmp_path / "oc", "--dry-run")
    assert r.returncode == 0 and "would set" in r.stdout
    assert not (tmp_path / "oc").exists()
