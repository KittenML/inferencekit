# Installing inferencekit

The plugin is the same in every harness: 9 skills, 4 commands, 5 read-only subagents and the inferencekit MCP server. Pick the harness you use.

**Requirements:** [uv](https://docs.astral.sh/uv/), because the MCP server and the CLI run through `uvx`, and Python 3.10 or newer for the CLI. Install the engine runtime and profiling tools for your hardware; see [What has been tested](#what-has-been-tested).

## Claude Code

```bash
claude plugin marketplace add kittenml/inferencekit
claude plugin install inferencekit@inferencekit
```

- **Commands:** `/inferencekit:optimize-inference`, `/inferencekit:bench`, `/inferencekit:profile`, `/inferencekit:perf-report`. Skills load by themselves when a request matches them.
- **Check:** `claude plugin list` shows inferencekit enabled, and `/mcp` inside a session lists the inferencekit server.
- **Update:** `claude plugin update inferencekit@inferencekit`. **Remove:** `claude plugin uninstall inferencekit@inferencekit`.

## Codex

```bash
codex plugin marketplace add kittenml/inferencekit
codex plugin add inferencekit@inferencekit
```

- **Skills:** `$inferencekit:optimizing-inference`, `$inferencekit:benchmarking`, `$inferencekit:profiling`, `$inferencekit:reporting-perf-claims`. Type `$` in Codex to select an inferencekit skill, then describe the task.
- **Check:** `codex plugin list` shows `inferencekit@inferencekit` as `installed, enabled`, and `codex mcp list` shows the server launched with `uvx --from inferencekit-cli[mcp]==0.1.0 inferencekit mcp serve`.
- **Update:** `codex plugin marketplace upgrade` refreshes the marketplace; if the new version is not picked up, remove the plugin and add it again. **Remove:** `codex plugin remove inferencekit@inferencekit`.
- Independent work can use Codex's native subagents; without delegation, the same tasks run sequentially. See [parallel execution and device isolation](../plugin/skills/optimizing-inference/references/parallel.md).

## OpenCode

```bash
git clone --depth 1 https://github.com/kittenml/inferencekit ~/.local/share/inferencekit
python3 ~/.local/share/inferencekit/plugin/opencode/install.py
```

- **What the installer changes:** in `~/.config/opencode/opencode.json` it adds the skills folder (`skills.paths`) and the MCP server, and keeps a backup of the previous file. It also symlinks the commands and subagents into `~/.config/opencode/commands/` and `agents/`, and never replaces a file of yours.
- **Commands:** `/optimize-inference`, `/bench`, `/profile`, `/perf-report` (OpenCode does not namespace them), and each skill is also a `/<skill-name>` command.
- **Check:** `opencode mcp list` shows inferencekit connected.
- **One project only:** `--config-dir <project>/.opencode`. **Preview:** `--dry-run`. **Remove:** `--uninstall`, then delete the clone.
- **Update:** `git -C ~/.local/share/inferencekit/ pull`, then run the installer again.
- **Differences from Claude Code** (hooks, command names and long-running tools): [plugin/opencode/README.md](../plugin/opencode/README.md).

## The CLI or the MCP server alone

- **CLI:** `uvx --from 'inferencekit-cli==0.1.0' inferencekit doctor`, or `uv tool install 'inferencekit-cli==0.1.0'`.
- **MCP server, for any MCP client:** `uvx --from 'inferencekit-cli[mcp]==0.1.0' inferencekit mcp serve` (stdio). Every CLI command is also a tool with the same arguments.
- **Unreleased source changes:** the plugin's MCP configuration uses the published CLI version above, not the current checkout. New operations require a release or an MCP server installed from the matching source; check the advertised tool list before relying on them.
- **If the MCP server does not start** (no `uvx`, no network, a failed install): install the CLI pinned to the plugin's version and let the agent call it from the shell. The skills say how.
  ```bash
  uv tool install "inferencekit-cli==0.1.0"
  inferencekit doctor
  ```

## The always-on rules (AGENTS.md)

The skills carry the rules they need. The repository's `AGENTS.md` lists all of them in one place, for agents that read only that file. To make them always-on in one project:
- **Codex:** it reads the `AGENTS.md` of the project you work in. Copy the rules into yours.
- **OpenCode:** add `"instructions": ["~/.local/share/inferencekit/AGENTS.md"]` to the project's `opencode.json`.

## What has been tested

See [supported hardware and agents](../README.md#supported-hardware-and-agents) for end-to-end coverage and [results](../README.md#results) for the measured configurations. OpenCode has install checks but no end-to-end result. The Codex and OpenCode install CI jobs are advisory: a failure there does not block a merge.

### Other hardware

Tool coverage on AMD ROCm GPUs, x86 CPUs and Apple Silicon depends on the workload and installed runtime:

| Available where the runtime supports the workload | NVIDIA-only (the skills describe alternatives) |
|---|---|
| `bench rounds`, `bench serve`, `gates run` and LLM quality checks including GSM8K (compatible HTTP servers); media benchmarks; `roofline` with a bandwidth number; intake, manifest, ledger, review and report; AMD `rocprof capture` and `rocprof analyze` when supported | `ceiling bw`, `nsys` and `ncu` are NVIDIA-only; on other hardware use the platform profiler and a vendor or STREAM-style bandwidth test |

- **Unverified:** ARM CPU-only inference. MLX and Ollama have no dedicated exact-length request profiles.
- Details per platform: [the hardware guide](../plugin/skills/inference-triage/references/hardware.md).

## FAQ

**I have no NVIDIA GPU. Can I use it?**
Yes, with limits. The skills guide the work and the CLI includes AMD rocprofv3 trace capture and analysis when supported by the installed tool and GPU. `ceiling bw`, Nsight Systems and Nsight Compute are NVIDIA-only; see [What has been tested](#what-has-been-tested). Platform support and tracing permissions must be checked on the actual host.

**`doctor` says the GPU counters are blocked. What now?**
Many cloud GPU containers block the hardware counters that Nsight Compute (`ncu`) needs, whatever the GPU. `inferencekit doctor` reports it as `profiling_counters.status: admin_only`. Benchmarks and Nsight Systems (`nsys`) timelines still work. For kernel-level counters, capture a `.ncu-rep` on a machine that allows it and hand the file to the profile command.

**What does a run cost?**
Two parts: the AI agent's tokens and hardware use. Published agent costs are API list-price estimates; most reported tokens are cached conversation reads. See [results](../README.md#results) for each run's scope, cost and accounting. Hardware rental charges are separate.

The intake sets spending and time limits. An explicitly uncapped run uses `budget.spend_cap: none` with a finite deadline; see [budget settings](../plugin/skills/optimizing-inference/references/intake.md).

**Does it spend money on hardware without asking?**
No. Hardware use needs spending authorization; the agent reuses authorization you already gave. A plan is optional. For unattended runs, set `non_interactive: true` and `budget.approved: true` with a cap in the intake: that is your approval.

**Does it run sub-agents in parallel?**
Yes: research, review and independent kernel implementation can run in parallel. Measurements can run concurrently on separately authorized devices, with one owner/lock per device and no shared-host contention. Warmup, quality checks, profiling and timing stay serialized on each device. The five shipped subagents remain read-only; implementation uses workers with write access. See [parallel execution](../plugin/skills/optimizing-inference/references/parallel.md).
