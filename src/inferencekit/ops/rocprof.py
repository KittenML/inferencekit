"""ROCm tracing via rocprofv3. Raw traces are sensitive and never benchmark proof."""
from __future__ import annotations

import csv
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.proc import clean_env, dropped_config_env, require, run, tool_version
from inferencekit.registry import Arg, OpError, operation


@operation('capture_rocprof', ('rocprof', 'capture'),
           'Capture AMD kernel and HIP API traces with rocprofv3 under an allowlisted environment.',
           [Arg('keep_env', 'array', 'Additional environment variable names (never values).', default=[]),
            Arg('gpu_name', 'string', 'Verified GPU name, if known.'),
            Arg('timeout', 'integer', 'Time limit in seconds for the profiled process group.', default=3600)],
           passthrough=True)
def capture_rocprof(call, keep_env, gpu_name, timeout, argv):
    if timeout < 1:
        raise OpError('timeout must be positive')
    profiler = require('rocprofv3', 'Install ROCprofiler-SDK for the installed ROCm release.')
    env, names = clean_env(keep_env)
    dropped = dropped_config_env(env)
    store = ev.Store()
    with store.staging_dir() as staging:
        # CSV supports these two trace types on both current and older ROCm versions.
        # Current ROCm recommends rocpd for other trace features.
        cmd = [profiler, '--kernel-trace', '--hip-trace', '--output-format', 'csv',
               '-d', str(staging), '-o', 'trace', '--', *argv]
        result = run(cmd, timeout=timeout, env=env, stdout_path=staging / 'stdout.txt',
                     stderr_path=staging / 'stderr.txt')
        for name in ('stdout.txt', 'stderr.txt'):
            path = staging / name
            path.write_text(ev.redact_text(path.read_text(errors='replace')))
        kernels = sorted(staging.glob('*kernel_trace.csv'))
        hip = sorted(staging.glob('*hip_api_trace.csv'))
        missing = ([] if kernels else ['kernel trace']) + ([] if result.ok else ['clean exit'])
        limits = ['Raw profiler traces may contain sensitive names and paths; inspect before sharing.',
                  'Tracing can add overhead; compare unprofiled end-to-end timings separately.',
                  'No kernel counters are collected by this operation.']
        if dropped:
            limits.append('Configuration variables not passed: ' + ', '.join(dropped))
        artifacts = [ev.artifact(p, staging, 'kernel_trace') for p in kernels]
        artifacts += [ev.artifact(p, staging, 'hip_trace') for p in hip]
        artifacts += [ev.artifact(staging / n, staging, n[:-4]) for n in ('stdout.txt', 'stderr.txt')]
        record = ev.build(call, kind='rocprof_capture', claim_class='observed',
                          summary=f'rocprofv3: {len(kernels)} kernel trace(s), exit {result.returncode}',
                          identity_=ev.identity(gpu={'name': gpu_name}), artifacts=artifacts,
                          missing=missing, limitations=limits,
                          data={'rocprof_version': tool_version('rocprofv3'), 'exit_code': result.returncode,
                                'env_names': names, 'env_dropped': dropped, 'output_format': 'csv'})
        return store.save(record, staging)


@operation('analyze_rocprof', ('rocprof', 'analyze'),
           'Summarize kernel dispatch counts and durations from a rocprofv3 CSV trace.',
           [Arg('source', 'string', 'Kernel trace CSV path.', required=True, positional=True, sensitive=True)])
def analyze_rocprof(call, source):
    path = Path(source)
    if not path.is_file():
        raise OpError('kernel trace CSV not found')
    totals = {}
    try:
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not {'Kernel_Name', 'Start_Timestamp', 'End_Timestamp'} <= set(reader.fieldnames or []):
                raise OpError('not a rocprofv3 kernel trace CSV')
            for row in reader:
                name = row['Kernel_Name']
                start, end = int(row['Start_Timestamp']), int(row['End_Timestamp'])
                if not name or end < start:
                    raise OpError('invalid kernel trace row')
                item = totals.setdefault(name, {'kernel': name, 'dispatches': 0, 'duration_ns': 0})
                item['dispatches'] += 1
                item['duration_ns'] += end - start
    except (ValueError, UnicodeError, csv.Error, KeyError) as exc:
        raise OpError('invalid kernel trace CSV') from exc
    rows = sorted(totals.values(), key=lambda x: x['duration_ns'], reverse=True)
    store = ev.Store()
    with store.staging_dir() as staging:
        # Store only the raw file's hash, not a second sensitive copy.
        source_hash, _ = ev.file_sha256(path)
        record = ev.build(call, kind='rocprof_summary', claim_class='observed',
                          summary=f'rocprofv3 kernel trace: {sum(x["dispatches"] for x in rows)} dispatches',
                          missing=[] if rows else ['kernel dispatches'],
                          limitations=['Summed kernel time can overlap and is not critical-path or end-to-end time.',
                                       'Source trace may contain sensitive data.'],
                          data={'source_sha256': source_hash, 'kernels': rows})
        return store.save(record, staging)
