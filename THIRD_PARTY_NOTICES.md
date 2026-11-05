# Third-party notices

inferencekit (`inferencekit-cli` on PyPI) is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`). This file lists the software it depends on, and the software it calls but never includes.

The wheel contains inferencekit's code and the attributed data described below. Python dependencies are installed separately by your package manager, under their own licenses; the external tools are not bundled.

## Python runtime dependencies

Versions are those pinned in `uv.lock` for this release. License identifiers are taken from each package's published metadata; check the package itself for the authoritative text.

### Always installed

| Package | Version | License |
|---|---|---|
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| referencing | 0.37.0 | MIT |
| rpds-py | 2026.9.1 (0.30.0 on Python 3.10) | MIT |
| attrs | 26.1.0 | MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| PyYAML | 6.0.3 | MIT |

### Installed with the `mcp` extra (`inferencekit-cli[mcp]`)

| Package | Version | License |
|---|---|---|
| mcp | 2.3.0 | MIT |
| mcp-types | 2.3.0 | MIT |
| anyio | 4.15.1 | MIT |
| annotated-types | 0.8.0 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause |
| cffi | 2.1.1 | MIT-0 |
| pycparser | 3.11 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| httpcore2 | 2.13.1 | BSD-3-Clause |
| httpx2 | 2.13.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| opentelemetry-api | 1.45.1 | Apache-2.0 |
| pydantic | 2.14.1 | MIT |
| pydantic-core | 2.50.1 | MIT |
| PyJWT | 2.15.1 | MIT |
| python-multipart | 0.0.32 | Apache-2.0 |
| sse-starlette | 3.5.0 | BSD-3-Clause |
| starlette | 1.7.0 | BSD-3-Clause |
| truststore | 0.10.5 | MIT |
| typing-inspection | 0.4.4 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| exceptiongroup | 1.3.1 (Python 3.10 only) | MIT |

Platform-only entries in the lock file that are not installed on Linux: `pywin32` (Windows) and `httpx2-jsfetch` (Emscripten).

To regenerate this list for another version: `uv export --no-dev --all-extras --no-hashes --no-emit-project`, then read each package's license metadata.

## Tools that inferencekit calls but never redistributes

inferencekit runs these as separate programs or in separate interpreters that **you** install. They are not dependencies of the package, are not included in it, and stay under their own licenses and terms. Using them through inferencekit does not change those terms.

| Tool | Used for | License / terms |
|---|---|---|
| NVIDIA Nsight Systems (`nsys`) | timeline captures (`nsys capture`, `nsys analyze`) | NVIDIA's license for Nsight Systems |
| NVIDIA Nsight Compute (`ncu`) | kernel captures (`ncu capture`, `ncu summarize`) | NVIDIA's license for Nsight Compute |
| NVIDIA CUDA Toolkit (`nvcc`), NVIDIA driver (`nvidia-smi`) | building the bandwidth microbenchmark from source; GPU inventory in `doctor` | NVIDIA's CUDA EULA and driver license |
| vLLM | engine under test | Apache-2.0 |
| SGLang | engine under test | Apache-2.0 |
| llama.cpp (`llama-server`, `llama-cli`, `llama-bench`) | engine under test | MIT |
| TensorRT-LLM (`trtllm-serve`) | engine under test | Apache-2.0 |
| PyTorch | optional bandwidth probe (`torch_bw.py`) and engines | BSD-3-Clause |
| Hugging Face Transformers / tokenizers | optional re-tokenization when a server does not return token ids | Apache-2.0 |

Engine and tool licenses can change between versions: check the license of the exact version you install. Model weights and datasets you benchmark carry their own licenses and use policies, which you are responsible for following.

The source of the bandwidth microbenchmark (`src/inferencekit/cuda/bw_peak.cu`) is inferencekit's own code, licensed under Apache-2.0; it is compiled on your machine with your CUDA toolkit.

## Data shipped in the package

The development prompt set under `src/inferencekit/prompts/` is built from public datasets. Every prompt item records its origin and license in its `source` field (`dataset/license`), and the set's manifest records its sha256. Those per-item licenses apply to the prompt texts.
