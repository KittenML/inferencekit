#!/usr/bin/env python3
"""Install (or remove) inferencekit for OpenCode from this checkout (README.md has the details).

    python3 plugin/opencode/install.py                       # all your projects (~/.config/opencode)
    python3 plugin/opencode/install.py --config-dir ./.opencode   # one project only
    python3 plugin/opencode/install.py --uninstall [--config-dir DIR]
    python3 plugin/opencode/install.py --dry-run              # print what it would change

commands/ and agents/ (engine-scout, quant-scout, profile-analyst, claims-skeptic, log-reader) get symlinks.
Rerun after a pull that changes plugin/.mcp.json. Standard library only (Python 3.8+).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # plugin/opencode
PLUGIN = HERE.parent                            # plugin
SKILLS = PLUGIN / "skills"
SERVER = "inferencekit"
SCHEMA = "https://opencode.ai/config.json"
# A cold `uvx` start downloads the CLI first; OpenCode also applies this limit to every MCP tool call.
MCP_TIMEOUT_MS = 120_000


def default_config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "opencode"


def mcp_entry() -> dict:
    server = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"][SERVER]
    return {"type": "local", "command": [server["command"], *server["args"]], "enabled": True,
            "timeout": MCP_TIMEOUT_MS}


def links(config_dir: Path) -> list[tuple[Path, Path]]:
    """(link in the config dir, file in this checkout) for every command and agent."""
    out = []
    for kind in ("commands", "agents"):
        for src in sorted((HERE / kind).glob("*.md")):
            out.append((config_dir / kind / src.name, src))
    return out


def is_ours(link: Path, src: Path) -> bool:
    if link.is_symlink():
        try:
            return link.resolve() == src.resolve()
        except OSError:
            return False
    # a copy made where symlinks are not allowed
    return link.is_file() and src.is_file() and link.read_bytes() == src.read_bytes()


def load_config(path: Path) -> dict | None:
    """The config as a dict; None if it cannot be edited safely (comments, trailing commas, not an object)."""
    if not path.exists():
        return {"$schema": SCHEMA}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {"$schema": SCHEMA}
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def write_config(path: Path, data: dict) -> None:
    new = json.dumps(data, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == new:
            return
        shutil.copy2(path, path.with_name(path.name + ".bak-inferencekit"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-inferencekit")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, path)


def manual_snippet() -> str:
    return json.dumps({"$schema": SCHEMA, "skills": {"paths": [str(SKILLS)]}, "mcp": {SERVER: mcp_entry()}},
                      indent=2)


def install(config_dir: Path, dry_run: bool) -> int:
    cfg_path = config_dir / "opencode.json"
    cfg = load_config(cfg_path)
    if cfg is None:
        print(f"{cfg_path} has comments or is not a JSON object, so it is not edited automatically.\n"
              f"Merge these keys into it by hand, then run this installer again:\n{manual_snippet()}",
              file=sys.stderr)
        return 2
    changes = []
    skills = cfg.setdefault("skills", {})
    paths = skills.setdefault("paths", [])
    if str(SKILLS) not in paths:
        paths.append(str(SKILLS))
        changes.append(f"skills.paths += {SKILLS}")
    want = mcp_entry()
    mcp = cfg.setdefault("mcp", {})
    if mcp.get(SERVER) != want:
        changes.append(f"mcp.{SERVER} = {' '.join(want['command'])} (timeout {MCP_TIMEOUT_MS} ms)")
        mcp[SERVER] = want
    config_changed = bool(changes)

    skipped = []
    todo = []
    for link, src in links(config_dir):
        if is_ours(link, src):
            continue
        if link.exists() or link.is_symlink():
            skipped.append(link)
            continue
        todo.append((link, src))
        changes.append(f"link {link.parent.name}/{link.name} -> {src.relative_to(PLUGIN.parent)}")

    for c in changes:
        print(("would " if dry_run else "") + "set: " + c)
    for link in skipped:
        print(f"skipped: {link} already exists and is not from this checkout; rename it to get "
              f"/{link.stem}", file=sys.stderr)
    if dry_run:
        return 0
    if config_changed:
        write_config(cfg_path, cfg)
    for link, src in todo:
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(src)
        except OSError:
            shutil.copy2(src, link)  # no symlinks here: a copy, refreshed by re-running the installer
    jsonc = config_dir / "opencode.jsonc"
    if jsonc.exists():
        print(f"note: {jsonc} is read after opencode.json; if it sets skills or mcp.{SERVER}, it wins",
              file=sys.stderr)
    print(f"inferencekit is installed for OpenCode in {config_dir}" if changes else
          f"inferencekit was already installed for OpenCode in {config_dir}")
    print("Check: `opencode mcp list` shows inferencekit connected (the first start downloads the CLI), and "
          "/optimize-inference is in the TUI's command list. Restart a running OpenCode to load it.")
    return 0


def uninstall(config_dir: Path, dry_run: bool) -> int:
    cfg_path = config_dir / "opencode.json"
    changes = []
    for link, src in links(config_dir):
        if is_ours(link, src):
            changes.append(f"remove {link.parent.name}/{link.name}")
            if not dry_run:
                link.unlink()
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    if cfg_path.exists() and cfg is None:
        print(f"{cfg_path} has comments: remove skills.paths entry {SKILLS} and mcp.{SERVER} by hand",
              file=sys.stderr)
    elif cfg is not None:
        paths = (cfg.get("skills") or {}).get("paths") or []
        if str(SKILLS) in paths:
            paths.remove(str(SKILLS))
            changes.append(f"skills.paths -= {SKILLS}")
            if not paths:
                cfg["skills"].pop("paths")
            if not cfg["skills"]:
                cfg.pop("skills")
        if SERVER in (cfg.get("mcp") or {}):
            cfg["mcp"].pop(SERVER)
            changes.append(f"remove mcp.{SERVER}")
            if not cfg["mcp"]:
                cfg.pop("mcp")
        if changes and not dry_run:
            write_config(cfg_path, cfg)
    for c in changes:
        print(("would " if dry_run else "") + c)
    if not changes:
        print(f"nothing of inferencekit's found in {config_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config-dir", type=Path, default=None,
                    help="OpenCode config directory (default: $XDG_CONFIG_HOME/opencode or ~/.config/opencode; "
                         "<project>/.opencode installs for one project)")
    ap.add_argument("--uninstall", action="store_true", help="remove what this installer added")
    ap.add_argument("--dry-run", action="store_true", help="print the changes, write nothing")
    args = ap.parse_args(argv)
    config_dir = (args.config_dir or default_config_dir()).expanduser().absolute()
    if not (SKILLS.is_dir() and (PLUGIN / ".mcp.json").is_file()):
        print(f"run this from an inferencekit checkout: {PLUGIN} has no skills/ or .mcp.json", file=sys.stderr)
        return 2
    return (uninstall if args.uninstall else install)(config_dir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
