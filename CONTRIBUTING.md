# Contributing to inferencekit

Thanks for helping. Bug reports, engine settings, new skills and fixes are all welcome.

## Before you start

For anything bigger than a small fix, open an issue first and say what you want to change. It saves you building something we can't merge.

## Set up

You need [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer. The tests need no GPU.

```bash
uv sync --all-extras
uv run --all-extras pytest -q
```

## What a pull request needs

- **The problem:** what was wrong or missing.
- **What changed:** a short description of the change.
- **Proof it works:** the test output, or for a speed change, a before/after measurement taken with inferencekit (for example `inferencekit bench rounds`), with the hardware, engine version and model.

Keep pull requests small and focused, and add tests for new code.

Pull requests written with an AI agent are welcome, as long as a person has read and checked the whole diff before opening it.

## Commit messages

Use `<area>: <short description>`, for example `gates: catch replayed prompts with changed whitespace` or `docs: fix the OpenCode update command`.

## Engines

Custom engines go in [inferencekit-engines](https://github.com/kittenml/inferencekit-engines), one folder per model and hardware. Read its CONTRIBUTING guide before you open a pull request there.

If a result uses your engine with settings that are not its best, open an issue here with the better settings and the engine version they apply to.

## Safety

- Never commit profiles (`.nsys-rep`, `.ncu-rep`, `.sqlite`), engine logs or `runs/` folders. They can hold secrets, host names and paths.
- Never put API keys or tokens in code, tests or commit messages.
- Don't submit NVIDIA vs non-NVIDIA comparisons built from Nsight data; NVIDIA's licence terms can restrict that. Use end-to-end measurements instead.
- Report security problems privately: see [SECURITY.md](SECURITY.md).

By contributing, you agree that your work is licensed under the Apache License 2.0 (section 5 of [LICENSE](LICENSE)).
