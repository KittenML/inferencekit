"""Media speed comparisons reject incomplete, unequal or concurrent experiments."""
from inferencekit import evidence as ev, registry
from conftest import fake_call


def media_record(store, *, offset=0, scale=1, request_hash='a' * 64, quant='bf16', failed=0,
                 kind='image_benchmark', count=4, valid=True, overlap=False, revision='test-sha',
                 timing=None):
    metric = 'image_ms' if kind == 'image_benchmark' else 'audio_ms'
    rows = [{'index': i, 'seed': 100 + i, 't_start': 100 + 10 * i + offset,
             't_end': 100 + 10 * i + offset + (8 if overlap else 1),
             'n_failed': failed, metric: 100 * scale} for i in range(count)]
    rec = ev.build(fake_call(), kind=kind, claim_class='observed', summary='synthetic media',
                   identity_=ev.identity(gpu={'name': 'test device'}, engine={'version': 'test-version'},
                                         model={'id': 'test/model', 'quant': quant, 'dtype': 'bf16', 'revision': revision},
                                         workload={'mode': 'single', 'request_set': {'sha256': request_hash}}),
                   data={'rounds': rows, 'timing_verdict': timing,
                         'claim_validity': {'valid': valid, 'violations': []}})
    return store.save(rec)


def compare(a, b):
    return registry.call('compare', {'baseline': a['evidence_id'], 'candidate': b['evidence_id'],
                                     'metric': 'image_ms', 'resamples': 200})


def test_media_matched_interleaved_timing_can_improve(store):
    a = media_record(store)
    b = media_record(store, offset=3, scale=.5)
    result = compare(a, b)
    assert result['data']['verdict'] == 'improved'
    assert any('quality' in s for s in result['limitations'])


def test_media_changed_requests_or_quantization_cannot_win(store):
    a = media_record(store)
    for kwargs in ({'request_hash': 'b' * 64}, {'quant': 'q3'}):
        result = compare(a, media_record(store, offset=3, scale=.5, **kwargs))
        assert result['data']['verdict'] == 'inconclusive'


def test_media_failed_requests_and_invalid_outputs_cannot_win(store):
    a = media_record(store)
    for kwargs in ({'failed': 1}, {'valid': False}):
        result = compare(a, media_record(store, offset=3, scale=.5, **kwargs))
        assert result['data']['verdict'] == 'inconclusive'


def test_media_overlap_and_incomplete_rounds_cannot_win(store):
    a = media_record(store)
    for kwargs in ({'overlap': True}, {'count': 2}, {'offset': 100}):
        result = compare(a, media_record(store, **{'offset': 3, 'scale': .5, **kwargs}))
        assert result['data']['verdict'] == 'inconclusive'


def test_media_quality_never_passes_from_structural_checks(store):
    from inferencekit.ops.report import claims_of
    rec = media_record(store)
    claims = claims_of(rec, 'improved', {'id_ok': True})
    assert [claim['verdict'] for claim in claims] == ['PASS', 'UNPROVEN']


def test_media_unknown_revision_is_not_pinned_provenance(store):
    a = media_record(store, revision=None)
    b = media_record(store, offset=3, scale=.5, revision=None)
    result = compare(a, b)
    assert result['data']['verdict'] == 'inconclusive'
    assert 'missing model revision' in result['data']['decision']['reason']


def test_media_comparison_alone_preserves_missing_quality_and_lifecycle(store):
    from inferencekit.ops.report import claims_of
    result = compare(media_record(store), media_record(store, offset=3, scale=.5))
    claims = claims_of(result, 'improved', {'id_ok': True})
    assert claims[0]['verdict'] == 'PASS'
    assert {c['claim'] for c in claims if c['verdict'] == 'UNPROVEN'} == {
        'media quality', 'fresh-process lifecycle', 'repeated-run reliability'}


def test_speech_report_preserves_failed_absolute_slo(store):
    from inferencekit.ops.report import claims_of
    rec = media_record(store, kind='audio_benchmark', timing='FAIL')
    claims = claims_of(rec, 'improved', {'id_ok': True})
    assert next(c for c in claims if c['claim'] == 'speech latency and streaming SLOs')['verdict'] == 'FAIL'
