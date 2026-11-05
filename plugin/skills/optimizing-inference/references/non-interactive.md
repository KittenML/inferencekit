# Running unattended (`non_interactive: true`)

Nobody is watching. **Never block on input:** every question gets a conservative answer inside the approved budget, logged where the user will see it, and the run continues.

**Preconditions (validate exits 1 without them):** `budget.approved: true` and a spend cap (`gpu_hours`, or `usd` plus `hardware.price_usd_h`). They are the pre-approved budget: no plan, no wait; record their source in `state.decisions` after `--run-dir`; no plan file is required ([plan.md](plan.md)).

## Never block
- **ssh:** `-o BatchMode=yes` (fail instead of prompting for a password), `-o StrictHostKeyChecking=accept-new`, `-o ServerAliveInterval=30`.
- **Package managers:** `-y` / `--yes`; `DEBIAN_FRONTEND=noninteractive` for apt; `HF_HUB_DISABLE_PROGRESS_BARS=1`.
- **Timeouts on every command:** `timeout 1800 <cmd>` for installs and builds; readiness polls end after about 20 min.
- **Long jobs run detached with a log** (remote-host.md), and **one blocking wait per job:** `scripts/ikstep.py wait` (under the shell's 10-minute limit; one verdict line). Never `tail -f` or `cat` a whole log; logs over ~200 lines go to `log-reader` ([parallel.md](parallel.md)).
- **No remote publication by default:** never `git push`, open PRs or create remote repos except for a completed [engine contribution](engine-contribution.md) explicitly authorized by the user for its destination, public contents and any needed fork. Without that opt-in, leave the contribution awaiting permission in `QUESTIONS.md`. Never start a second paid GPU host.

## QUESTIONS.md
Seeded by `intake validate --run-dir`; append one entry per new question:

```markdown
3. **Engine X needs a nightly build to load this checkpoint. Use it?**
   - Chosen: yes, as the one documented fix; recorded as nightly <version/commit> in the manifest
   - Why: a release build failed with "<error>"; the nightly is the documented route for this model
   - Stage: 3_candidates (12:41Z)
```

Mirror each entry into `state.decisions` with `"asked_user": false`.

## Conservative choices

| Situation | Choice |
|---|---|
| No target | Continue while budget remains, keeping the report reserve. |
| Every target met | Keep climbing unless `after_target: stop` ([state.md](state.md), best-speed log). |
| A written brief, no intake | Write the intake from it ([intake.md](intake.md), "Given a written brief"): the brief's rules win. |
| Route not stated | `single_user` and nothing needs a general-purpose server: build; else patch. Log it. |
| Fidelity / speculation not stated | `lossless` (lossy levers reported, never run); speculation `auto`. |
| A precision change would help (KV fp8, a different quant) | Only with `fidelity: lossy` and its `precision_allowed` / `kv_cache_dtype_allowed` entry, or a task's precision rule ([fidelity.md](fidelity.md)). Report it as a separate-track opportunity. |
| An engine needs a nightly, an RC or a source build to work | Allowed as the one documented fix, recorded as such. |
| An engine fails after one fix | NOT SUPPORTED. Move on. |
| The time box for a stage expires | Finish the current measurement, mark the rest NOT RUN, move on. |
| `intake budget` says `reserve_only` or `over_budget` | Stop GPU work, go to stage 8. |
| A gate fails on the best row | The row is unranked. Report it. Do not tune around the gate. |
| Quality is outside the noise floor | The row is not kept, however fast it is. |
| A constraint is ambiguous | Take the stricter reading, and log it. |
| Ledger repo missing | No repo is created; the ledger is NOT RUN. |
| Push | No optimization-branch push. Only an explicitly authorized post-run [engine contribution](engine-contribution.md) may publish its separate branch and PR. |
| The model's revision is a branch | Resolve it to the sha at stage 1. Every engine uses that sha. |
| ncu counters blocked | nsys only (or the PyTorch profiler), ncu NOT RUN ([profiling](../../profiling/SKILL.md), "Fallbacks"). A `QUESTIONS.md` entry asks for profiles captured where counters are allowed. Continue. |
| A workflow or subagent tool is available | Use it for independent research, review and implementation with isolated files and resource ownership ([parallel.md](parallel.md)). Never run two timed measurements on one GPU at once. |
| The intake leaves out workload numbers (lengths, concurrency, prefix share, peak rps) | Use the intake as given; never change the lead goal or buckets after `--run-dir`. Log each missing number in `QUESTIONS.md` with the value assumed, and add it to the "Workload: real vs measured" gap table ([workload.md](workload.md)). |
| No `goals.traffic.arrival_rate_rps` | The replica plan is NOT RUN. Still report $ per request and $ per 1M tokens ([capacity.md](capacity.md), section 7). |
| The SLO holds at the top of the load grid | Extend the grid upward inside the stage box ([capacity.md](capacity.md), section 2). If the box runs out, N\* is a lower bound: say so. |
| An autotuner proposes a config | Screening only: it goes through knob proof, `bench serve` and `bench rounds` like any other ([tuning.md](tuning.md), "Vendor autotuners"). |
| A high shared-prefix share | The warm-prefix track is NOT RUN for now. Never use `--no-cache-salt` in a timed run. |
| `hardware.count` is 1 | Multi-GPU is NOT RUN ([parallelism.md](parallelism.md)). |

## Heartbeat
- **State:** update `state.json` (`updated`, `next_action`) after every step. A resumed or replacement agent must be able to continue from it alone.
- **On a crash** (the context limit included): resume from `state.json`. Never re-run a done stage whose evidence exists.
