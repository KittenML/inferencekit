"""ROCm tracing integration without a GPU or profiler installation."""
import csv
from pathlib import Path

import pytest
from inferencekit import registry
from inferencekit.ops import rocprof
from inferencekit.proc import Result
from inferencekit.registry import OpError


def test_capture_uses_allowlist_and_records_traces(store, tmp_path, monkeypatch):
    monkeypatch.setattr(rocprof, 'require', lambda *_: '/usr/bin/rocprofv3')
    monkeypatch.setattr(rocprof, 'tool_version', lambda _: '7.2')
    monkeypatch.setenv('SECRET_TOKEN', 'never-record-this')
    monkeypatch.setenv('ROCR_VISIBLE_DEVICES', '1')
    def fake_run(argv, timeout, env, stdout_path, stderr_path):
        assert '--kernel-trace' in argv and '--hip-trace' in argv
        assert env['ROCR_VISIBLE_DEVICES'] == '1' and 'SECRET_TOKEN' not in env
        assert argv[-1] == 'model.py'
        out = Path(argv[argv.index('-d') + 1])
        (out / 'trace_kernel_trace.csv').write_text('Kernel_Name,Start_Timestamp,End_Timestamp\nfoo,1,9\n')
        (out / 'trace_hip_api_trace.csv').write_text('Domain,Function\nHIP,launch\n')
        stdout_path.write_text('ok')
        stderr_path.write_text('')
        return Result(0, 'ok', '')
    monkeypatch.setattr(rocprof, 'run', fake_run)
    rec = registry.call('capture_rocprof', {'argv': ['python', 'model.py'], 'gpu_name': 'RX 9070 XT'})
    assert rec['kind'] == 'rocprof_capture' and rec['coverage']['complete']
    assert {a['role'] for a in rec['artifacts']} >= {'kernel_trace', 'hip_trace'}
    assert 'never-record-this' not in str(rec)
    trace = store.run_dir(rec['evidence_id']) / next(a['path'] for a in rec['artifacts'] if a['role'] == 'kernel_trace')
    summary = registry.call('analyze_rocprof', {'source': str(trace)})
    assert summary['data']['kernels'] == [{'kernel': 'foo', 'dispatches': 1, 'duration_ns': 8}]
    assert any('overlap' in limit for limit in summary['limitations'])


def test_missing_trace_is_incomplete(store, monkeypatch):
    monkeypatch.setattr(rocprof, 'require', lambda *_: '/usr/bin/rocprofv3')
    monkeypatch.setattr(rocprof, 'tool_version', lambda _: None)
    def fake_run(argv, timeout, env, stdout_path, stderr_path):
        stdout_path.write_text('')
        stderr_path.write_text('failed')
        return Result(1, '', 'failed')
    monkeypatch.setattr(rocprof, 'run', fake_run)
    rec = registry.call('capture_rocprof', {'argv': ['python', 'model.py']})
    assert not rec['coverage']['complete'] and 'kernel trace' in rec['coverage']['missing']


def test_bad_trace_is_rejected(tmp_path, store):
    path = tmp_path / 'bad.csv'
    path.write_text('Kernel_Name,Start_Timestamp,End_Timestamp\nfoo,10,2\n')
    with pytest.raises(OpError, match='invalid'):
        registry.call('analyze_rocprof', {'source': str(path)})
    path.write_text('wrong,format\n1,2\n')
    with pytest.raises(OpError, match='not a rocprofv3'):
        registry.call('analyze_rocprof', {'source': str(path)})
