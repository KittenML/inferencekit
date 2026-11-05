# Running on an ssh GPU host (`hardware.access.kind: ssh`)

These recipes target Linux/NVIDIA hosts. `ikstep` process tracking uses `/proc`, and its GPU-memory checks use `nvidia-smi`; those checks do not validate AMD or Apple memory cleanup.

The GPU host does all GPU work: inferencekit, every engine, and the benchmark client. Your machine orchestrates, keeps `runs/<run-id>/` in sync, and holds the ledger repo. Host, user, port and key come from the original intake file. They never go into the manifest, reports, commits, evidence, `QUESTIONS.md` or any file under `runs/` (the run directory's intake copies are redacted).

## Connect
```bash
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p <port> -i <key> <user>@<host>"
$SSH 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h ~ | tail -1; nproc; grep MemAvailable /proc/meminfo; cat /sys/fs/cgroup/memory.max'
$SSH 'mkdir -p ~/ik/runs/<run-id>/evidence ~/ik/models ~/ik/envs ~/ik/bin ~/ik/cache \
      && ln -sfn ~/ik/models ~/ik/runs/<run-id>/models && ln -sfn ~/ik/cache ~/ik/runs/<run-id>/cache'
```
Record the meter start in `state.budget.meter` the first time this succeeds.

**Keep `SSH` and `IK` in a file, not in your shell.** Agent shells often lose variables between commands, so write both definitions once to a session file and start every command with `. <file> &&`.
- Put the file outside the run directory and outside the repo, e.g. `~/.ik-session.sh`. It holds the host, port, user and key, so it must never be committed.
- Example: `. ~/.ik-session.sh && $SSH "$IK inferencekit doctor"`.
- Put the step tool's settings in the same file, after `IK` (next section, then "One command per step"):
  ```bash
  export IK_HOST=<host> IK_PORT=<port> IK_USER=<user> IK_KEY=<key> IK_PREFIX="$IK"
  STEP=ikstep   # Claude Code: the plugin's bin/; elsewhere python3 <this skill's directory>/scripts/ikstep.py
  ```

## One evidence root, on both machines
Every inferencekit call, on the host and locally, writes and reads evidence in the run directory's `evidence/`:
```bash
IK='export PATH=$HOME/ik/bin:$HOME/ik/envs/ik/bin:$HOME/.local/bin:$PATH INFERENCEKIT_RUNS_DIR=$HOME/ik/runs/<run-id>/evidence; cd ~/ik/runs/<run-id>;'
$SSH "$IK inferencekit doctor"                          # host: records land in ~/ik/runs/<run-id>/evidence/
export INFERENCEKIT_RUNS_DIR="$PWD/runs/<run-id>/evidence"  # local: the same layout, filled by the sync below
```
- Keep the single quotes around `IK`, so that `$HOME` expands on the host, not locally.
- Prefix **every** host-side inferencekit command with `$IK`, including detached jobs. Without it, records go to `~/ik/runs/ev_*`, outside the synced directory.
- **`$IK` works from the run directory** (`~/ik/runs/<run-id>`), the same place a re-run works from. So on the host the run directory is `.`: use `--run-dir .`, `--manifest manifest.json` and `quality/<label>`. Locally it is `runs/<run-id>`.
- After a sync, `inferencekit evidence show <id>`, `ledger trailers <ev...>` and `review build` work locally on the same ids.
- The variable is `INFERENCEKIT_RUNS_DIR` (the CLI's `--runs-dir` does the same for one call).

## Workspace on the host
```
~/ik/
  envs/ik/           # inferencekit (the harness)
  envs/<engine>/     # one venv per engine (or its container)
  bin/               # one wrapper per engine program (see "Engine programs")
  models/<dir>/      # the checkpoint, downloaded once at the resolved sha
  cache/<venv>/      # compile caches, one per distinct engine venv (see "JIT builds and caches")
  runs/<run-id>/     # the run directory: $IK's working directory (synced back)
    evidence/        # INFERENCEKIT_RUNS_DIR: every evidence record of this run
    models -> ~/ik/models   # symlink, so manifest paths models/<dir> resolve here as on a re-run host
    cache -> ~/ik/cache     # symlink, so manifest env paths cache/<venv>/... resolve the same way
    logs/            # job logs, job files and saved JSON (ikstep writes them here)
    jobs/            # scripts for commands with JSON or nested quotes, written locally and deployed
    manifest.json    # copied from your machine after every change
```

## Install the harness
**Already there?** If `$SSH 'inferencekit --version'` prints the same version as `inferencekit --version` here, skip the install, but still record the wheel's sha256 for the manifest, from:
- the intake's `harness.wheel_sha256`;
- a wheel file or `SHA256SUMS` next to either install (`readlink -f "$(command -v inferencekit)"` shows where the install lives);
- a run record in your run repository that lists it.

If none of these has it, write a `QUESTIONS.md` entry: a re-run refuses a manifest without the wheel hash.

Otherwise, install it:
```bash
scp -P <port> -i <key> <wheel> <user>@<host>:~/ik/
$SSH 'command -v uv || ls ~/.local/bin/uv || curl -LsSf https://astral.sh/uv/install.sh | sh'
$SSH 'cd ~/ik && export PATH=$HOME/.local/bin:$PATH && sha256sum <wheel-name> \
      && uv venv envs/ik && uv pip install --python envs/ik/bin/python ./<wheel-name>'
$SSH "$IK inferencekit --version && inferencekit doctor"
```
`uv venv` environments have no `pip`: always `uv pip install --python <env>/bin/python ...` (a system `pip install` may be refused as "externally managed", so uv comes from its installer).
- The sha256 must equal `harness.wheel_sha256` when the intake gives it. Record it either way: it goes into `manifest.json`.
- If the intake gives `harness.requirements` instead, run `uv pip install --python envs/ik/bin/python -r <requirements>`.
- If neither is given and the host has no inferencekit, install the same `inferencekit-cli` version you run locally (`uv pip install --python envs/ik/bin/python "inferencekit-cli==<version>"`). If the package index does not have that version, find the wheel you run locally and copy it.

## Download the model once
```bash
$SSH 'cd ~/ik && export PATH=$HOME/.local/bin:$PATH && { command -v hf || uv tool install "huggingface_hub[cli]"; } \
      && hf download <model id> --revision <sha> --local-dir models/<dir>'
```
Older `huggingface_hub` versions name the command `huggingface-cli download` with the same flags. For a large checkpoint run it as a job: `$STEP start download -- '<the command>'`, then `$STEP wait download` (below). Record the download's file list and sizes. The sum is the sanity check for bytes per token (ceilings.md).
Gated models need `HF_TOKEN` on the host. Keep every secret (the HF token, an API key) in a file on the host that only the job reads, never inline in a command: write it once from your machine's environment, without printing it: `printf 'export HF_TOKEN=%s\n' "$HF_TOKEN" | $SSH 'umask 077; cat > ~/.ik-secrets'`. Then `$STEP start download -- '. ~/.ik-secrets && hf download ...'`. `~/.ik-secrets` is outside the run directory, so it is never synced back or committed. `start` masks `NAME=value` secrets and `--token`/`--api-key` values in `logs/<name>.job.json`, but the log keeps whatever the command prints.

## One command per step: `scripts/ikstep.py`
Run each host-side step as one `scripts/ikstep.py` command. It runs here, sends itself to the host over ssh (stdin, after `IK_PREFIX`, so from the run directory) and prints **one verdict line**; full output stays in the host's `logs/`. Stdlib Python 3.10+, nothing to install; `--local` runs a step here instead (`access: local`, from the run directory).

| Step | Command | Prints |
|---|---|---|
| copy files into the run directory | `$STEP deploy <local dir or file> [<dest>]` | `deployed ... (rsync)`; scp when rsync is missing |
| start a long job, detached | `$STEP start <name> [--ready-url <url>] [--done-regex <re>] [--timeout <s>] -- '<command>'` | `started <name>: pgid <n>, log logs/<name>.log` |
| wait for it | `$STEP wait <name>` | one verdict, below |
| screen one config | `$STEP bench <name> -- <bench serve flags>` | `OK bench <name> <evidence id>: <summary>; ...; full JSON logs/<name>.json` |
| read a `bench rounds` result | `$STEP compare [--metric decode_tok_s] [--bucket <b>]` | one line per engine: median [95% CI], ratio to best [CI], `FAILURES`, `HARD-GATE` |
| stop | `$STEP stop <name>`, or `$STEP stop --all` | `STOPPED ...` once the GPU memory is back |
| baseline checks, once | `$STEP job baseline` | gates, two greedy trajectories (the A/A floor), GSM8K 1,319, `quality sampling`, an A/A screen (its noise) |
| screen a candidate | `$STEP job screen --candidate E` | one 16-prompt `bench serve`: tok/s, ratio, vs the noise |
| light check | `$STEP job light --candidate E [--sampler-changed]` | 2 rounds on 32 prompts; gates caching, token_count, rare_tokens; quality on 8 prompts; GSM8K 100 |
| full proof | `$STEP job full --candidate E` | 3 rounds on 64 prompts, all gates, quality on 64, GSM8K 1,319 paired, `quality sampling` |
| a job's verdict | `$STEP wait <job> [--max-min 9]` | verdicts, KL per bucket vs the floor, evidence ids; exit 3: call again |
| evidence lines | `$STEP digest [ids \| --last N]` | one line per evidence record |

- **`start`** puts the command in its own session and process group (setsid, SIGHUP ignored): a dropped ssh connection never kills it. `logs/<name>.job.json` records the group, the log and the GPU memory in use. It refuses a name still running, and a `--ready-url` that already answers (an old server holds the port; `wait` would say `READY`): `$STEP stop --all` first.
- **`job`** reads `./manifest.json`, first stops every job, server and build started in the run directory, then starts `jobs/<job>.sh` detached. Do GPU-free work, then one `wait` (shell timeout 600000 ms); no polling in between.
- **`bench`** runs `inferencekit bench serve` with `INFERENCEKIT_OUTPUT=json`, keeps the full record in `logs/<name>.json` (and the evidence store) and waits like `wait`.
- **Quoting:** one layer only, for your own shell. A command with JSON or nested quotes goes into a script in `runs/<run-id>/jobs/` here: `$STEP deploy runs/<run-id>/jobs/<step>.sh jobs`, then `$STEP start <step> -- 'bash jobs/<step>.sh'`.
- One-off reads (`nvidia-smi`, `df`, a file listing) still go through `$SSH "$IK ..."`.

## One blocking wait per job
After `start`, the next command for that job is `$STEP wait <name>`, nothing else until it returns. It polls the ready URL, the log and the process group, and ends on the first of:

| Verdict (exit code) | Meaning |
|---|---|
| `READY` (0) | `--ready-url` answered 2xx while the job was alive |
| `DONE` (0) | a new log line matched `--done-regex`; the line is printed |
| `FINISHED` (0) | no URL or regex given; the job exited 0 |
| `DIED` (1) | the process group is gone (or exited before ready): exit code and up to 5 error lines from the log's end |
| `RUNNING` (3) | still running after `--chunk` seconds (default 540): run the same `wait` again |
| `TIMEOUT` (4) | `--timeout` seconds after the start, still running: stop it, or wait with a larger `--timeout` |

- **Give the call room to block:** `wait` up to `--chunk` seconds, `bench` as long as the benchmark, `stop` up to 150 s (more with `--gpu-wait`). A cut-off call prints no verdict, and shell tools often stop at 2 minutes: run `$STEP wait`, `bench` and `stop` with the shell timeout at its maximum (600000 ms under a 10-minute limit), or pass `--chunk` about 20 s below it.
- **Never** `sleep` loops, `tail -f`, or `cat`/`tail -n` of a whole log: grep it on the host with a bound (`$SSH "$IK grep -n -m 20 -i error logs/<name>.log"`); a log you need more of goes to a subagent, which returns a summary.
- Check `intake budget` between waits.

## Engine programs and paths (what `bench rounds` and a re-run launch)
- **Working directory: the run directory, on both hosts.** `bench rounds` starts each engine from the directory you call it in: with `$IK`, that is `~/ik/runs/<run-id>`. A re-run starts each engine from its copy of your run directory.
  - So every relative path in `launch` resolves the same way on both hosts. That covers `models/<dir>` (through the `models` symlink here; a re-run downloads `model.local_dir: "models/<dir>"`), an engine config file (`extra.yaml`), a custom engine's source and patches.
  - Keep those files in the run directory, locally and on the host. Never write an absolute path.
  - Start engines by hand (gates, quality, smoke tests) from the same directory, with the same argv.
- **`launch[0]` is a bare program name** (`vllm`, `trtllm-serve`, `llama-server`, `lmdeploy`).
  - `bench rounds` finds it on the PATH it was started with.
  - A re-run replaces it with the same name from that engine's own venv.
  - So give each engine a wrapper in `~/ik/bin`, which `$IK` puts first on PATH:
    ```bash
    $SSH 'printf "#!/bin/sh\nexec $HOME/ik/envs/vllm/bin/vllm \"\$@\"\n" > ~/ik/bin/vllm && chmod +x ~/ik/bin/vllm'
    ```
- **Engines started as `python -m ...`** (e.g. SGLang): `python` would collide between venvs, so give it a unique name on both sides.
  - Here: a wrapper `~/ik/bin/python-sglang` that execs `~/ik/envs/sglang/bin/python`.
  - In the manifest: `launch[0] = "python-sglang"`, with a last `install` step `ln -sf python "$VIRTUAL_ENV/bin/python-sglang"`. A re-run runs `install` inside that engine's activated venv.
- **Getting the manifest to the host.** You write `manifest.json` locally, but `bench rounds` reads it on the host. Copy it after every change:
  ```bash
  $STEP deploy runs/<run-id>/manifest.json          # into the host's run directory
  $STEP deploy runs/<run-id>/<engine dir> <engine dir>   # a directory, to the same relative path
  ```
  Copy every other file the manifest references (patches, `extra.yaml`, engine source) the same way, into the same relative path.

## JIT builds and caches
A kernel compile inside a timed launch costs minutes (sometimes over an hour) and can hit the startup timeout. Build once, keep the result, and share it only where that is safe.
- **`MAX_JOBS`:** about one per 8 GB of min(`MemAvailable`, cgroup `memory.max`), not `nproc` (a container may show the whole machine); too many JIT `nvcc` jobs get OOM-killed.
- **Prebuilt first.** If the engine's kernel library publishes prebuilt kernels for your CUDA version and GPU (for example FlashInfer's `flashinfer-cubin` and `flashinfer-jit-cache` packages, at the exact FlashInfer version the engine pins), add them to that engine's `install` so a re-run gets them too.
- **Otherwise prebuild at install,** detached, before any timed launch: start the engine once (`$STEP start <label>-build --ready-url <url> -- '<launch line>'`, then `$STEP wait`, then `$STEP stop`), or run the library's ahead-of-time build command, with `MAX_JOBS` sized as above. Nothing else CPU-heavy runs meanwhile.
- **One cache directory per distinct venv, shared by identical venvs** (same engine, torch and kernel-library versions), never one per label: FlashInfer `FLASHINFER_WORKSPACE_BASE`, Triton `TRITON_CACHE_DIR`, TorchInductor `TORCHINDUCTOR_CACHE_DIR` (its default is under `/tmp`, which a restart can wipe), vLLM `VLLM_CACHE_ROOT`. Two different venvs on one FlashInfer workspace can load kernels built for the other one.
- **Server env:** `bench rounds` passes only PATH, HOME, CUDA basics and the manifest's `env`, so the cache variables, `HF_HOME` and `MAX_JOBS` go there. Write cache paths relative to the run directory, which is the server's working directory on both hosts, through the `cache` symlink: `"TRITON_CACHE_DIR": "cache/<venv>/triton"`. Never an absolute path (it names the user). Check in the first launch's log that the cache landed there. If a tool needs an absolute path, leave it out of the manifest and set it in that engine's wrapper in `~/ik/bin` (`export TRITON_CACHE_DIR=$HOME/ik/cache/<venv>/triton` before the `exec`); a re-run then builds its own. Download with the same `HF_HOME`.

## GPU hygiene
- **Stop leftover servers before every sweep:** before each `bench rounds` call, screening series, gates session or quality run, run `$STEP stop --all`. It stops every job in `logs/` and every other CUDA process started from the run directory (`bench rounds` engines, servers started by hand with `$IK`), then waits until the GPU memory is back. A CUDA process started anywhere else is never killed: the verdict lists it as `not ours, left running`. Start nothing after `STOP INCOMPLETE`: find what still holds the GPU first.
- **One engine on the GPU at a time.** `start` prints `GPU already holds ... MiB` when something else is still there.
- **Stop servers by process group:** `$STEP stop <name>` (SIGTERM to the group, SIGKILL after 20 s, then it waits for the GPU memory).
- **No other CPU-heavy work on the host during timed runs:** no builds, no downloads, no compiles. They slow the client and the engine's scheduler.
- **Keep the startup log** of every launch: it is the proof of the resolved config.

## Sync back
```bash
rsync -a --exclude /models --exclude /cache -e "ssh -p <port> -i <key>" <user>@<host>:~/ik/runs/<run-id>/ runs/<run-id>/
```
- This brings back `evidence/`, `outputs/`, `inputs/`, `metrics/`, `rounds/`, `quality/`, `jobs/` and `logs/`. Never use `--delete`: `state.json`, `experiments.jsonl` and the intake copies live only on your machine.
- Sync after every stage, so a lost host loses at most one stage.
- Logs copied back are scanned for the host name and user before anything is committed.
