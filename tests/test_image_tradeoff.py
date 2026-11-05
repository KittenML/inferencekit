"""Image tradeoffs retain task identity and never prove visual quality from timing."""
import copy

import pytest

from inferencekit import registry
from inferencekit.registry import OpError
from inferencekit.ops.bench_image import image_task_set
from inferencekit.ops.report import claims_of
from test_compare_media import media_record


def task(rows):
    return image_task_set(rows, 'model', 'rev', 'per_round', 10, None)['sha256']


def test_task_identity_allows_steps_and_formats_only():
    row = {'id': 'a', 'prompt': 'a fox', 'width': 4, 'height': 3, 'steps': 50,
           'guidance_scale': 3, 'settings': {'dtype': 'bf16', 'scheduler': 'euler'}}
    same = {**row, 'steps': 25, 'settings': {'dtype': 'fp8', 'scheduler': 'euler'}}
    assert task([row]) == task([same])
    for changed in ({**same, 'prompt': 'a cat'}, {**same, 'width': 5},
                    {**same, 'settings': {'dtype': 'fp8', 'scheduler': 'other'}}):
        assert task([row]) != task([changed])
    assert task([row]) != image_task_set([row], 'model', 'rev', 'per_round', 11, None)['sha256']
    assert task([row]) != image_task_set([row], 'model', 'other', 'per_round', 10, None)['sha256']


def with_task(store, record, digest):
    # Rebuild the synthetic evidence with the task identity included in its content hash.
    from inferencekit import evidence as ev
    from conftest import fake_call
    identity = record['identity']
    identity['workload']['image_task_set'] = {'sha256': digest, 'n': 1}
    return store.save(ev.build(fake_call(), kind=record['kind'], claim_class='observed',
                               summary='synthetic', identity_=identity, data=record['data']))


def run(a, b, **kwargs):
    return registry.call('compare', {'baseline': a['evidence_id'], 'candidate': b['evidence_id'],
                                     'metric': 'image_ms', 'resamples': 200, **kwargs})


def test_tradeoff_is_opt_in_and_quality_stays_unproven(store):
    a = with_task(store, media_record(store), '1' * 64)
    b = with_task(store, media_record(store, offset=3, scale=.5, request_hash='b' * 64, quant='fp8'), '1' * 64)
    assert run(a, b)['data']['verdict'] == 'inconclusive'
    result = run(a, b, image_tradeoff=True)
    assert result['data']['verdict'] == 'improved'
    assert result['data']['comparison_scope'] == 'image_format_work_tradeoff'
    assert 'tradeoff' in result['summary']
    assert any(c['claim'] == 'media quality' and c['verdict'] == 'UNPROVEN'
               for c in claims_of(result, 'improved', {'id_ok': True}))


def test_tradeoff_rejects_changed_or_missing_task_and_failed_images(store):
    a = with_task(store, media_record(store), '1' * 64)
    for b in (with_task(store, media_record(store, offset=3, scale=.5), '2' * 64),
              media_record(store, offset=3, scale=.5),
              with_task(store, media_record(store, offset=3, scale=.5, failed=1), '1' * 64),
              with_task(store, media_record(store, offset=3, scale=.5, revision='other'), '1' * 64)):
        assert run(a, b, image_tradeoff=True)['data']['verdict'] == 'inconclusive'


def test_end_to_end_tradeoff_uses_real_image_evidence(tmp_path, store):
    import json
    from test_bench_image import bench
    from test_image_client import image_server
    row = {'id': 'fox', 'prompt': 'a fox', 'width': 4, 'height': 3, 'steps': 50,
           'guidance_scale': 3.0, 'settings': {}}
    baseline, candidate = tmp_path / 'baseline.jsonl', tmp_path / 'candidate.jsonl'
    baseline.write_text(json.dumps(row) + '\n')
    candidate.write_text(json.dumps({**row, 'steps': 25}) + '\n')
    a = b = None
    with image_server() as (url, _):
        for _ in range(3):
            a = bench(url, baseline, rounds=1, warmup=0, label='base',
                      **({'extend': a['evidence_id']} if a else {}))
            b = bench(url, candidate, rounds=1, warmup=0, label='candidate', dtype='fp8',
                      **({'extend': b['evidence_id']} if b else {}))
    assert a['identity']['workload']['image_task_set'] == b['identity']['workload']['image_task_set']
    assert a['identity']['workload']['request_set'] != b['identity']['workload']['request_set']
    assert run(a, b)['data']['verdict'] == 'inconclusive'
    comparison = run(a, b, image_tradeoff=True)
    assert comparison['data']['interleaved'] and comparison['data']['pairs'] == 3
    assert 'comparable inputs' not in comparison['coverage']['missing']
    assert comparison['data']['required_proofs']['media quality'] == 'UNPROVEN'


@pytest.mark.parametrize('kind', ['benchmark', 'serve_benchmark', 'clip_benchmark', 'audio_benchmark'])
def test_image_tradeoff_cannot_relax_other_modalities(store, kind):
    a = media_record(store, kind=kind)
    b = media_record(store, kind=kind, offset=3, scale=.5)
    with pytest.raises(OpError, match='only available for two image benchmark records'):
        run(a, b, image_tradeoff=True)


@pytest.mark.parametrize('change', [
    {'overlap': True}, {'count': 2}, {'offset': 100}, {'valid': False}, {'failed': 1},
])
def test_tradeoff_retains_timing_and_structural_checks(store, change):
    a = with_task(store, media_record(store), '1' * 64)
    b = with_task(store, media_record(store, **{'offset': 3, 'scale': .5, **change}), '1' * 64)
    assert run(a, b, image_tradeoff=True)['data']['verdict'] == 'inconclusive'


@pytest.mark.parametrize('field', ['prompt', 'width', 'height', 'guidance_scale', 'scheduler', 'cache'])
def test_tradeoff_retains_task_settings(field):
    row = {'id': 'a', 'prompt': 'a fox', 'width': 4, 'height': 3, 'steps': 50,
           'guidance_scale': 3, 'settings': {'scheduler': 'euler'}}
    changed = copy.deepcopy(row)
    if field == 'cache':
        assert task([row]) != image_task_set([row], 'model', 'rev', 'per_round', 10, None,
                                             allow_cache='text_encoder')['sha256']
        return
    if field == 'scheduler':
        changed['settings']['scheduler'] = 'other'
    else:
        changed[field] = 'a cat' if field == 'prompt' else row[field] + 1
    assert task([row]) != task([changed])


def test_legacy_python_compare_call_keeps_strict_default(store):
    from conftest import fake_call, make_benchmark
    from inferencekit.ops.compare import compare
    a = make_benchmark(store, [100.] * 4)
    b = make_benchmark(store, [150.] * 4, t0=5., prompt_len=512)
    result = compare(fake_call('compare'), a['evidence_id'], b['evidence_id'], 'decode_tok_s', .02, 200, 0)
    assert result['data']['verdict'] == 'inconclusive'
    assert 'workloads differ' in result['data']['decision']['reason']
