"""The CLI and MCP front ends are generated from one registry; check they agree with the contract."""

import json

import jsonschema
import pytest

from inferencekit import cli, registry
from inferencekit.evidence import set_default_root
from inferencekit.mcp_server import fit
from inferencekit.registry import OpError

CONTRACT_TOOLS = {"doctor", "roofline", "measure_bandwidth", "bench_decode", "capture_nsys", "analyze_nsys",
                  "capture_ncu", "summarize_ncu", "compare", "check_correctness", "list_evidence",
                  "inspect_evidence", "report"}
CONTRACT_TOOLS |= {"intake_template", "intake_validate", "intake_budget", "intake_resume"}  # intake
CONTRACT_TOOLS |= {"build_review", "ledger_trailers", "history"}  # review page and ledger
CONTRACT_TOOLS |= {"quality_trajectory", "quality_score", "quality_gsm8k", "quality_niah"}  # quality checks
CONTRACT_TOOLS |= {"quality_sampling"}  # first-token sampler check (draws vs the server's own logprobs)
CONTRACT_TOOLS |= {"bench_serve"}  # client-side timer
CONTRACT_TOOLS |= {"bench_rounds"}  # interleaved rounds
CONTRACT_TOOLS |= {"validate_manifest"}  # run manifest
CONTRACT_TOOLS |= {"prompts_build", "prompts_seal", "prompts_verify", "prompts_show", "run_gates",
                   "gates_spec_exact"}  # prompt sets and gates
CONTRACT_TOOLS |= {"bench_clip"}  # video (bench clip)
CONTRACT_TOOLS |= {"bench_image", "bench_audio"}  # client-timed image and speech outputs
CONTRACT_TOOLS |= {"gates_clip", "quality_clip", "quality_clip_steps"}  # video (gates, quality, per-step)
CONTRACT_TOOLS |= {"capture_rocprof", "analyze_rocprof"}  # AMD profiling
CONTRACT_TOOLS |= {"cpu_screen_manifest", "quality_audio_policy", "quality_audio"}


def test_registry_matches_contract_tool_names():
    assert set(registry.load_all()) == CONTRACT_TOOLS


def test_every_input_schema_is_valid_json_schema():
    for op in registry.load_all().values():
        jsonschema.Draft202012Validator.check_schema(op.input_schema())


def test_validation_fills_defaults_and_rejects_unknown_args():
    op = registry.get("bench_decode")
    args = op.validate({"engine": "vllm", "model": "m", "prompt_len": "64"})
    assert args["prompt_len"] == 64 and args["rounds"] == 5
    with pytest.raises(OpError):
        op.validate({"engine": "vllm", "bogus": 1})
    with pytest.raises(OpError):
        op.validate({"engine": "not-an-engine"})


def test_cli_parses_contract_commands():
    p = cli.build_parser()
    ns = p.parse_args(["compare", "ev_aaa", "ev_bbb", "--threshold", "0.05"])
    assert (ns._op, ns.baseline, ns.candidate, ns.threshold) == ("compare", "ev_aaa", "ev_bbb", 0.05)
    ns = p.parse_args(["ncu", "capture", "--kernel-regex", "gemv", "--launch-skip", "3", "--count", "2"])
    assert (ns._op, ns.kernel_regex, ns.launch_skip, ns.count) == ("capture_ncu", "gemv", 3, 2)
    ns = p.parse_args(["rocprof", "capture", "--timeout", "60"])
    assert (ns._op, ns.timeout) == ("capture_rocprof", 60)
    ns = p.parse_args(["rocprof", "analyze", "trace.csv"])
    assert (ns._op, ns.source) == ("analyze_rocprof", "trace.csv")
    ns = p.parse_args(["roofline", "--model", "cfg.json", "--dtype", "fp8", "--gpu-bw", "measure"])
    assert ns._op == "roofline" and ns.gpu_bw == "measure"
    ns = p.parse_args(["report", "ev_a", "ev_b", "--format", "json"])
    assert ns.ids == ["ev_a", "ev_b"]


def test_cli_runs_an_operation(tmp_path, capsys):
    try:
        code = cli.main(["--runs-dir", str(tmp_path / "runs"), "evidence", "list"])
    finally:
        set_default_root(None)  # --runs-dir sets process-wide state
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 0 and out["items"] == []


def test_cli_requires_command_after_double_dash(tmp_path):
    try:
        with pytest.raises(SystemExit):
            cli.main(["--runs-dir", str(tmp_path), "nsys", "capture"])
    finally:
        set_default_root(None)


def test_mcp_results_are_trimmed_with_paging_hints():
    record = {"evidence_id": "ev_" + "0" * 64, "summary": "s",
              "data": {"top_kernels": [{"name": f"kernel_{i}", "pad": "x" * 200} for i in range(500)]}}
    view = fit({"record": record}, max_chars=8000)
    assert len(json.dumps(view)) <= 8000
    cut = view["record"]["data"]["top_kernels"]
    assert cut["truncated"] and cut["total"] == 500 and "path='data.top_kernels'" in cut["more"]


def test_mcp_trims_a_page_without_skipping_items():
    # inspect_evidence(path='data.kernels', offset=50, limit=50) on 120 large kernel entries:
    # the page is shortened, and both next_offset and the hint continue at the first item cut.
    from inferencekit.ops.store import page

    kernels = [{"index": i, "pad": "x" * 900} for i in range(120)]
    result = {"evidence_id": "ev_abc", "path": "data.kernels", "integrity": {"id_ok": True},
              **page(kernels, 50, 50)}
    view = fit(result, max_chars=20_000)
    assert len(json.dumps(view)) <= 20_000
    k = len(view["items"])
    assert 1 <= k < 50 and view["items"][0]["index"] == 50
    assert view["next_offset"] == 50 + k
    assert f"path='data.kernels', offset={50 + k}, limit={k}" in view["more"]
    assert view["total"] == 120


def test_mcp_trims_a_list_evidence_page():
    from inferencekit.ops.store import page

    rows = [{"evidence_id": f"ev_{i:064d}", "summary": "s" * 600} for i in range(100)]
    view = fit({"root": "runs", **page(rows, 0, 100)}, max_chars=20_000)
    k = len(view["items"])
    assert len(json.dumps(view)) <= 20_000 and view["next_offset"] == k
    assert f"offset={k}" in view["more"]


def test_mcp_hint_inside_a_page_item_points_at_the_real_path():
    from inferencekit.ops.store import page

    huge = [{"index": 0, "stalls": [{"reason": f"r{i}", "pad": "y" * 200} for i in range(400)]}]
    result = {"evidence_id": "ev_abc", "path": "data.kernels", "integrity": {}, **page(huge * 8, 7, 1)}
    view = fit(result, max_chars=8_000)
    cut = view["items"][0]["stalls"]
    assert cut["truncated"] and "path='data.kernels.7.stalls'" in cut["more"]


def test_mcp_preserves_operation_and_source_arguments_when_shortening_a_page():
    from inferencekit.ops.store import page

    args = {"run_dir": "runs/experiment", "file": "alternate-intake.yaml", "now": "2026-11-05T21:00:00Z"}
    rows = [{"name": f"row_{i}", "detail": "x" * 600} for i in range(100)]
    source = {"pagination": {"tool": "intake_resume", "args": args}, **page(rows, 10, 50)}
    view = fit(source, max_chars=6000)
    k = len(view["items"])
    assert len(json.dumps(view)) <= 6000 and 1 <= k < 50
    assert view["next_call"] == {"tool": "intake_resume", "arguments": {**args, "offset": 10 + k, "limit": k}}
    assert view["next_offset"] == 10 + k
    assert source["pagination"]["args"] == args and "next_call" not in source


@pytest.mark.parametrize("width,max_chars", [(600, 9802), (823, 32000)])
def test_resume_page_hint_overhead_cannot_skip_unshown_items(width, max_chars):
    from inferencekit.ops.store import page

    args = {"run_dir": "runs/experiment", "file": "original.json", "now": "2026-11-05T21:00:00Z"}
    rows = [{"index": i, "summary": "x" * width} for i in range(100)]
    offset, limit, seen = 0, 50, []
    while offset is not None:
        source = {"pagination": {"tool": "intake_resume", "args": args}, **page(rows, offset, limit)}
        if source["next_offset"] is not None:
            source["next_call"] = {"tool": "intake_resume", "arguments": {
                **args, "offset": source["next_offset"], "limit": limit}}
        before = json.dumps(source)
        shown = fit(source, max_chars=max_chars)
        assert json.dumps(source) == before
        assert len(json.dumps(shown)) <= max_chars
        assert isinstance(shown["items"], list) and shown["items"]
        assert shown["items"][0]["index"] == offset
        seen.extend(row["index"] for row in shown["items"])
        offset = shown["next_offset"]
        if offset is not None:
            assert offset == shown["items"][-1]["index"] + 1
            call = shown["next_call"]
            assert call["tool"] == "intake_resume"
            assert {k: call["arguments"][k] for k in args} == args
            assert call["arguments"]["offset"] == offset
            limit = call["arguments"]["limit"]
    assert seen == list(range(100))
