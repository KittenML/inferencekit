# inferencekit for OpenCode

The same plugin as in Claude Code and Codex, set up for [OpenCode](https://opencode.ai) (checked against OpenCode 1.19.0):
- the 9 skills, loaded by OpenCode's `skill` tool (each one is also a `/<skill-name>` command);
- the commands `/optimize-inference`, `/bench`, `/profile` and `/perf-report`;
- the read-only subagents `engine-scout`, `quant-scout`, `profile-analyst`, `claims-skeptic` and `log-reader`;
- the inferencekit MCP server, launched with `uvx` exactly as in Claude Code.

**Status:** unit-tested, and a CI job checks that OpenCode loads everything; no full optimisation run has used OpenCode yet.

## Install

Needs git, Python 3.8+ for the installer, and [uv](https://docs.astral.sh/uv/) for the CLI and the MCP server.

```bash
git clone --depth 1 https://github.com/kittenml/inferencekit ~/.local/share/inferencekit
python3 ~/.local/share/inferencekit/plugin/opencode/install.py
```

- **What it changes,** in `~/.config/opencode` (or `$XDG_CONFIG_HOME/opencode`): `opencode.json` gets this checkout's skills folder in `skills.paths` and the `inferencekit` MCP server (other keys kept; the old file saved as `opencode.json.bak-inferencekit`); `commands/` and `agents/` get symlinks into this checkout, never replacing your files.
- **One project only:** add `--config-dir <project>/.opencode`.
- **Preview:** `--dry-run` prints the changes and writes nothing.
- **Update:** `git -C ~/.local/share/inferencekit/ pull`, then run the installer again (it changes only what moved).
- **Remove:** `python3 ~/.local/share/inferencekit/plugin/opencode/install.py --uninstall`, then delete the clone.
- **Check:** `opencode mcp list` shows `inferencekit` connected; the first start downloads the CLI. Restart a running OpenCode to load the changes.
- **Always-on rules (optional):** `"instructions": ["~/.local/share/inferencekit/AGENTS.md"]` in a project's `opencode.json`; the skills carry the rules they need without it.

An `opencode.json` with comments is never edited; the installer prints these keys to merge by hand:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": { "paths": ["~/.local/share/inferencekit/plugin/skills"] },
  "mcp": {
    "inferencekit": {
      "type": "local",
      "command": ["uvx", "--from", "inferencekit-cli[mcp]==0.1.1", "inferencekit", "mcp", "serve"],
      "enabled": true,
      "timeout": 120000
    }
  }
}
```

With a manual setup, link the commands and agents too: `ln -s ~/.local/share/inferencekit/plugin/opencode/commands/*.md ~/.config/opencode/commands/`, and the same for `agents/`.

## How it differs from Claude Code

| | Claude Code | OpenCode |
|---|---|---|
| Commands | `/inferencekit:optimize-inference` | `/optimize-inference` (OpenCode does not namespace commands) |
| Start | simple preferences; plans only on request | the same; respect the current client's execution permissions |
| Subagents | the Task tool | the task tool, or `@engine-scout` in a message |
| Long MCP calls | run until done | stopped at the server's `timeout` (120 s here): long steps use the CLI, detached with a log |
| `${CLAUDE_PLUGIN_ROOT}` | substituted | not substituted: commands name skills instead of paths |
| Hooks | in `claude -p`, a stated deadline holds the stop until the report time | none: the flow's deadline rules apply |
| `ikstep` | on PATH (the plugin's `bin/`) | `python3 <plugin>/skills/optimizing-inference/scripts/ikstep.py` |

## For maintainers

The files in `commands/` and `agents/` are generated from `plugin/commands/` and `plugin/agents/`: edit those, then run `python3 plugin/opencode/sync.py`. The test suite fails while they are out of date (`sync.py --check`).
