"""Benchmark adapters: one per engine.

vLLM and SGLang run in a *worker script* executed by the engine's own interpreter
(``--python /path/to/venv/bin/python``), so inferencekit never has to share a virtual
environment with an engine and the engine's GPU memory is freed when the worker
exits. llama.cpp and ``cmd`` are external binaries/commands.
"""
