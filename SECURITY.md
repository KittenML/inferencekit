# Security policy

## Reporting a vulnerability

Please report security issues **privately**, through GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue, pull request or discussion for a security problem.

Include what you found, how to reproduce it, the inferencekit version (`inferencekit --version`) and the impact you expect. We aim to acknowledge reports within 5 working days and to agree a disclosure date with you once a fix is ready.

## Supported versions

Only the latest released version receives security fixes.

## Sensitive data in profiles and logs

Performance work produces files that can leak secrets. Treat them as sensitive.

- **Nsight Systems (`.nsys-rep`, and the `.sqlite` exported from it) and Nsight Compute (`.ncu-rep`) files can embed the profiled process's environment variables**, command lines, host names, user names and file paths. An environment variable can hold an API key or an access token (for example a Hugging Face token).
- **Engine logs** (vLLM, SGLang, llama.cpp, TensorRT-LLM and custom engines) can print their configuration, environment, command line, request contents and paths.
- **Evidence records** under `runs/` store command lines and the resolved engine configuration.

What inferencekit does about it:
- `inferencekit nsys capture` and `inferencekit ncu capture` run the profiled program under `env -i`, with an allowlist of environment variables (`PATH`, `HOME`, `CUDA_VISIBLE_DEVICES`, plus the names you pass with `--keep-env`). They refuse to pass names that look like secrets (names containing `TOKEN`, `SECRET`, `PASSW`, `KEY`, `CREDENTIAL`, `AUTH`, `COOKIE` and similar), and record only the **names** of the variables passed, never their values.
- `runs/`, `*.nsys-rep`, `*.ncu-rep` and `*.sqlite` are listed in `.gitignore`.

What you should do:
- Profile only through `inferencekit nsys capture` / `inferencekit ncu capture`, not by calling `nsys` or `ncu` directly with your full environment.
- **Never publish raw profiles or engine logs without scanning them first** for tokens, keys, host names, IP addresses, user names and private paths. Even a capture made under `env -i` still contains command lines, host names and paths.
- Share derived summaries (`inferencekit nsys analyze`, `inferencekit ncu summarize`, evidence records) instead of raw profile files where you can, and review those too before publishing.
- Never pass a secret on a command line; read it from an environment variable by name.
