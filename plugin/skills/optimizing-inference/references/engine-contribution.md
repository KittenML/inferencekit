# Optional engine contribution after a completed run

Use after the final report or on request. Default destination: [KittenML/inferencekit-engines](https://github.com/KittenML/inferencekit-engines). The coding agent executes this workflow through GitHub tools or `gh`; it is not a background hook or CLI/MCP operation.

## Decide whether the result qualifies

Read the destination's current contribution instructions and existing model/hardware entry. Reuse the final report and evidence; publication does not authorize another benchmark, a longer optimization run or additional hardware spending. Release rented hardware as already authorized.

Check these conditions during local preparation; all must hold before offering or submitting the PR:

- **A measured improvement in the agreed objective.** Compare against the tuned, supported baseline on the declared workload: lower complete-request latency, higher throughput/capacity, lower peak memory, or lower serving cost while meeting the other agreed constraints. Timing gains need the registered balanced comparison and its confidence interval; cost claims need their measured performance and stated price inputs. A screen, profile, estimated ceiling or attractive absolute number alone is insufficient.
- **Applicable correctness and quality requirements pass for the exact candidate.** Preserve the agreed precision, quality policy, completion checks and workload. Disclose permitted numerical/model changes and quality losses. An accepted lossy policy is not a waiver for failed or missing required checks, and a later acceptance must not rewrite an earlier FAIL as PASS.
- **A reproducible, redistributable package.** Source, pinned dependencies, launch settings, model revision and reproduction commands identify the measured candidate. Complete the destination's required installation/smoke checks within authorized resources. If packaging changes execution, existing measurements alone do not prove the new package; missing reproduction stays NOT RUN and blocks automatic submission.
- **The destination accepts this contribution.** Follow its current rules for baseline, metric, quality and packaging. For example, if it currently requires a speed win, a memory-only improvement is not automatically eligible. Explain the mismatch and keep the proposal local instead of silently relaxing the repository's rules.

If any condition fails, record the reason and stop the contribution step. Keep the optimization result and report available; do not open a speculative PR to stand in for missing evidence.

## Prepare the exact proposal before asking

Use a separate local checkout/worktree from the destination's default branch. Inspect the existing `<model>_<hardware>` entry before adding or updating it. Copy only the selected source, scripts and shareable evidence; never push the optimization workspace's private history.

Follow the destination's package format. The README identifies the agent/plugin, hardware, baseline, metric and quality scope, explains the optimizations, and links evidence and limits. A configuration/adapter contribution must also meet the destination's rules; do not imply custom-kernel work that was not done.

Review the full diff and PR title/body. Exclude weights, credentials, private prompts/outputs, host/provider details, raw profiles and unrelated files. Evidence must be publicly inspectable; local IDs alone are not public links. Preserve source/model licenses and attribution. Resolve uncertain rights or disclosure scope before publication.

Save the diff/body locally and progress in `runs/<run-id>/contribution.json` (alongside the report for a standalone run): status, target/base, package path, candidate commit, evidence, head repository/branch, authorization source/scope and PR URL when available. Statuses: `ineligible`, `prepared`, `awaiting_permission`, `submitted`, `blocked`. This agent-maintained record is not an intake schema field; never edit the pinned intake.

## Publication is opt-in

Optimization, spending, local commits and permission to push the user's working repository do not imply permission to publish to the engines repository. Honor a current no-publication instruction or `repo.push: never`; only a later explicit user decision that covers this contribution can change that restriction.

- **Already authorized:** reuse the user's explicit approval to publish this run's qualifying engine, source and evidence to this destination. Record its source; do not ask again. It covers the initial branch push and PR, and a fork only if included in the authorization.
- **Interactive, no approval:** show the local diff/package, result, destination and PR body. Ask once: “Publish this engine's source and evidence to `<owner/repo>` as a PR? This will push the proposed branch [and create a fork if needed].” Include the fork clause only if needed. A decline leaves it local; do not re-ask.
- **Unattended:** publish only with that explicit approval. Otherwise mark `awaiting_permission`, append the proposal to `QUESTIONS.md` and finish without blocking or remote mutation. A saved state label or the agent's own choice is not user approval.

This narrowly scoped post-run publication is the exception referenced by the ledger and unattended-run rules. It does not authorize unrelated pushes, repository creation, reviewer requests, merges or releases.

## Open or resume one PR

Use the client's GitHub connector when it supports the required operations; otherwise use installed GitHub CLI commands. Authentication must already be available for unattended publication. Never expose tokens or broaden access on the user's behalf.

1. Confirm the destination/default branch and authorized remote/fork. Check the handoff record and remote PR state for this head repository/branch before any mutation. Reuse an existing open PR; if closed or merged, report its state without automatically recreating it.
2. Review the commit range against the destination base; stage only intended paths and confirm the commit matches the reviewed package/evidence. Push the named branch to the approved remote, without force-pushing or writing to the default branch.
3. Open the reviewed PR with an explicit target, base and head. With `gh`, use `--repo`, `--base`, `--head`, `--title` and `--body-file`; explicit `--head` prevents implicit pushing/forking. Never use `gh pr create --dry-run` as a local preview: it can still push. Draft PRs need the same publication approval.
4. After a timeout, inspect remote branch/PR state before retrying. On auth/policy failure or uncertain remote state, retain the package and record `blocked` with the reason; do not loop on login prompts or create duplicate PRs.
5. Save and report the verified PR URL and published commit. Do not merge or claim maintainer acceptance. Further changes or another destination require authorization covering that scope.
