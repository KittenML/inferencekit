"""Shared fixtures. Every test gets its own evidence store under a temp dir."""

from pathlib import Path

import pytest

from inferencekit import evidence as ev
from inferencekit.registry import Call, Operation

REPO = Path(__file__).resolve().parents[1]
# Only the source repository has the release tools. The public plugin repository ships without the files that
# document the published runs (the example intakes, the worked examples, the method).
SOURCE_REPO = (REPO / "scripts" / "release" / "split.sh").is_file()


def needs_repo_file(path: Path, what: str):
    """Skip a test that reads path in the public plugin repository, which does not have it. The source
    repository never skips it: there a missing path (moved or renamed) fails the test."""
    return pytest.mark.skipif(not SOURCE_REPO and not path.exists(),
                              reason=f"{what} is not in this repository (the source repository runs this test)")


@pytest.fixture(autouse=True)
def _cli_prints_json(monkeypatch):
    """Tests parse the CLI's stdout as JSON: pin the full-JSON output (the default is the terse view).
    Subprocesses inherit it. tests/test_cli_terse.py sets the terse mode itself."""
    monkeypatch.setenv("INFERENCEKIT_OUTPUT", "json")


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "runs"
    ev.set_default_root(root)
    yield ev.Store(root)
    ev.set_default_root(None)


def fake_call(name: str = "bench_decode", cli=("bench", "decode"), args=None) -> Call:
    """A Call for building records in tests without running an operation."""
    op = Operation(name, tuple(cli), "test op", (), lambda call, **kw: {})
    return Call(op, dict(args or {}), started="2026-01-01T00:00:00.000Z")


def make_benchmark(store: ev.Store, values, t0=0.0, step=10.0, gpu="Test GPU", prompt_len=128, valid=True):
    """Store a synthetic benchmark record whose rounds start at t0, t0+step, ..."""
    rounds = [{"index": i, "t_start": t0 + i * step, "t_end": t0 + i * step + 1.0, "decode_tok_s": v,
               "ttft_ms": 1000.0 / v, "tpot_ms": 1000.0 / v, "gen_tokens": 128} for i, v in enumerate(values)]
    validity = {"valid": valid, "violations": [] if valid else ["CUDA graphs off (enforce_eager=True)"],
                "warnings": [], "checked": ["cuda_graphs"]}
    rec = ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="synthetic benchmark",
                   identity_=ev.identity(gpu={"name": gpu}, engine={"name": "synthetic"},
                                         workload={"prompt_len": prompt_len, "gen_len": 128, "batch": 1, "seed": 0}),
                   metrics=[ev.metric("decode_tok_s", sorted(values)[len(values) // 2], "tok/s", "median",
                                      n=len(values))],
                   limitations=[] if valid else ["HARD GATE: CUDA graphs off (enforce_eager=True)"],
                   data={"rounds": rounds, "claim_validity": validity, "tokens": [1, 2, 3]})
    return store.save(rec)
