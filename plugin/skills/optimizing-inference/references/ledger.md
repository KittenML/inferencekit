# Commit ledger

Every **kept** change is a revertible git commit with its numbers in git trailers. Failed or slower attempts are not committed. They live in `experiments.jsonl` with their numbers.

## Repository (from `intake.repo`)
- **`repo.path` given:** work on branch `repo.branch` (default `perf/<model>-<gpu>`). Create it from the current HEAD: `git switch -c <branch>`. Never commit to the user's main branch.
- **No repo given, interactive:** ask whether to use an existing repo or create a new one (`git init`, optionally on GitHub via `gh` after a yes).
- **No repo given, non_interactive:** create no repo and push nothing. The ledger is **NOT RUN**: say so in the report, with the would-be commits listed from `experiments.jsonl`.
- **Push:** `push: ask` means ask before **each** push; `never` means never. A non_interactive run never pushes, whatever the setting.

For a separate post-run [engine contribution](engine-contribution.md), an explicit opt-in covers its reviewed branch push and PR, including unattended. It does not permit pushing this ledger; overriding `push: never` requires a later explicit user decision recorded without editing the pinned intake.

## What a commit contains
Things a reviewer needs to rebuild the configuration:
- launch scripts or config files (`extra.yaml`, flags files);
- engine patches (`patches/<engine>-<n>.patch`);
- custom engine source;
- `manifest.json`;
- `REPORT.md`.

**Not** model weights, `runs/` raw outputs, logs with host details, or secrets.

`runs/` is usually git-ignored. Add the run's own files by path, e.g. `git add -f runs/<run-id>/manifest.json runs/<run-id>/REPORT.md runs/<run-id>/reproduce.sh runs/<run-id>/patches/`. Never run `git add -A` or `git add runs/`: evidence, outputs and logs stay out of the history. Scan what you stage for host names, IPs and user names first.

## Trailers
Generate them from evidence, never typed by hand: `inferencekit ledger trailers <ev_speed> <ev_quality> [<ev_capacity> ...]`. Message format and an example (`perf(<engine>): <change>`, a body with what changed and the evidence, then the `Perf-*` trailers): [commit-ledger.md](../../optimizing-kernels/references/commit-ledger.md).

- **Capacity goal active:** also `Perf-Capacity-N: 24 (SLO ttft_p95<=500ms, tpot_p95<=50ms, 0 fail)` and `Perf-Goodput: 3.1 req/s`.
- **No AI attribution lines.** Use the user's configured git identity.
- **History:** `inferencekit history --repo <path>` shows the metric trend. Tag the best commit per headline metric with `perf/best-<metric>`.
- **Revert:** plain git (`git switch -c restore <commit>`).
