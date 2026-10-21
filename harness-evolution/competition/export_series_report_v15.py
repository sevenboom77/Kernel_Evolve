#!/usr/bin/env python3
"""Export a public, full-denominator summary of the original numeric reader."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

TASKS = [f'K{i:02}' for i in range(1, 21)]
READER_SHA = '49e6d69a05959fa4a794ce48d914d3038be7f5c176d2aa14c63faeafc8f82de7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('public metric must be finite and nonnegative')
    return value


def public_summary(raw, digest):
    if (raw.get('schema_version') != 1 or raw['arm_count'] != 8
            or raw['task_position_denominator'] != 160 or len(raw['arms']) != 8
            or raw['source_patch_author'] != 'Codex' or raw['source_effect_claimed'] is not False
            or raw['incumbent_modified'] is not False or not re.fullmatch('[0-9a-f]{64}', digest)):
        raise ValueError('original audit identity or claim boundary differs')
    arms = []
    seen = set()
    for arm in raw['arms']:
        if arm['batch_id'] in seen:
            raise ValueError('duplicate campaign arm')
        seen.add(arm['batch_id'])
        row = {k: arm[k] for k in ('batch_id', 'pair', 'kind', 'arm', 'execution_status')}
        if arm['execution_status'] == 'CLOSED':
            values = arm['rows']
            if len(values) != 20 or sorted(x['task_id'] for x in values) != TASKS:
                raise ValueError('twenty unique task positions required')
            for x in values:
                if any(type(x[k]) is not bool for k in ('terminal', 'evaluation_complete', 'environment_error', 'correct', 'verified_gain')):
                    raise ValueError('task qualification flags must be boolean')
                if finite(x['deployment_speedup']) <= 0 or (not x['verified_gain'] and x['deployment_speedup'] != 1):
                    raise ValueError('unverified task contribution must remain one')
                if x['verified_gain'] and not x['correct']:
                    raise ValueError('gain has no correct source')
            row.update(source_commit=arm['source_commit'], task_denominator=20,
                terminal_count=sum(x['terminal'] for x in values),
                evaluation_complete_count=sum(x['evaluation_complete'] for x in values),
                environment_error_count=sum(x['environment_error'] for x in values),
                correct_count=sum(x['correct'] for x in values),
                verified_gain_count=sum(x['verified_gain'] for x in values),
                deployment_geomean=math.exp(sum(math.log(x['deployment_speedup']) for x in values) / 20),
                strict_eligible=arm['all20_eligible'] is True and all(x['terminal'] and x['evaluation_complete'] and not x['environment_error'] for x in values),
                task_contributions=[{'task_id': x['task_id'], 'correct': x['correct'], 'verified_gain': x['verified_gain'],
                                     'environment_error': x['environment_error'], 'deployment_speedup': x['deployment_speedup']} for x in values],
                resource_coverage_complete=arm['usage'].get('coverage_complete') is True)
        else:
            row.update(task_denominator=20, strict_eligible=False)
        arms.append(row)
    closed = sum(x['execution_status'] == 'CLOSED' for x in arms)
    if closed != raw['closed_arm_count']:
        raise ValueError('closed arm count differs from actual positions')
    promotion = raw['promotion']
    if promotion['promotion_eligible'] is not False or promotion['rsi_established'] is not False:
        # A promoted generation requires a separate review of pointer/authoring
        # evidence; this exporter covers the present rejected series only.
        raise ValueError('unexpected promotion needs separate provenance review')
    metric_keys = ('deployment_geomean_ratio', 'paired_ci_low', 'paired_ci_high')
    present = [key in promotion for key in metric_keys]
    if any(present) and not all(present):
        raise ValueError('partial original aggregate metrics are not allowed')
    ratio, lo, hi = [finite(promotion[key]) for key in metric_keys] if all(present) else (None, None, None)
    if closed == 8 and not all(present):
        raise ValueError('complete series needs original aggregate metrics')
    if (lo is not None and lo > hi) or any(type(v) is not bool for v in promotion['gates'].values()):
        raise ValueError('original promotion interval or gate schema differs')
    return {'schema_version': 1, 'kind': 'source-bound-original-reader-campaign-summary',
        'status': 'ALL_EIGHT_ARMS_CLOSED' if closed == 8 else 'PARTIAL_CLOSED_SNAPSHOT',
        'original_audit_sha256': digest, 'original_reader_sha256': READER_SHA,
        'arm_count': 8, 'closed_arm_count': closed, 'task_position_denominator': 160,
        'source_patch_author': 'Codex', 'rsi_established': False, 'promotion_eligible': False,
        'arms': arms, 'original_promotion_gates': promotion['gates'],
        'development_geomean_ratio': ratio, 'development_paired_ci_low': lo, 'development_paired_ci_high': hi,
        'summary': (f'G005系列原评分门晋级拒绝。三DEV全20几何均值比{ratio:.4f}，配对95%区间[{lo:.4f}, {hi:.4f}]。保留全部160任务位置，未证明搜索性能持续改善。' if closed == 8 else f'G005系列仍在实测：{closed}/8臂严格闭账，完整160位置均保留；尚未取得完整DEV与独立留出结果，不提前判定性能收益。'),
        'cost_scope': 'V15 primary-only intervals; the original reader uses bound before/after global counters. Historical unknown reservations and prior application/auxiliary totals remain in cumulative accounting.',
        'raw_private_requests_exported': False, 'incumbent_modified': False, 'independent_GPU_reproduction_claimed': False}


def export(audit, digest, out, checkpoint=None, checkpoint_sha256=None):
    if sha(audit) != digest:
        raise ValueError('original audit digest changed')
    raw = json.loads(Path(audit).read_bytes())
    # The audit was produced by this exact original reader. Check its complete
    # immutable evidence closure, rather than trusting a count copied to a slide.
    reader = Path(__file__).resolve().parents[2] / 'harness-evolution/codex-rsi/audit_measurement_series_campaign_v15.py'
    if sha(reader) != READER_SHA:
        raise ValueError('original numeric reader source changed')
    references = raw['evidence_references']
    if not references or any(sha(r['path']) != r['sha256'] for r in references):
        raise ValueError('original audit evidence binding changed')
    summary = public_summary(raw, digest)
    summary['original_bound_evidence_reference_count'] = len(references)
    series_ref = raw['series']
    if series_ref not in references or sha(series_ref['path']) != series_ref['sha256']:
        raise ValueError('original campaign declaration is not bound by the audit')
    series = json.loads(Path(series_ref['path']).read_bytes())
    if (series.get('reliability_revision') != 15
            or series['parent_source_commit'] != '54b05f2a959b5f6328ac2bf572fbd981bf567094'
            or series['candidate_source_commit'] != '8195f2debdff45e9702d4187b2eac698482c8585'
            or series['gpu_indices'] != [0, 1, 3, 4]
            or series['task_concurrency'] != 20 or series['evaluator_concurrency'] != 4):
        raise ValueError('the frozen V15 source or resource declaration differs')
    summary['comparison'] = {'campaign_revision': 15, 'parent_generation': 'G000', 'candidate_generation': 'G005',
        **{key: series[key] for key in ('parent_source_commit', 'candidate_source_commit', 'gpu_indices',
                                        'task_concurrency', 'evaluator_concurrency')},
        'series_declaration_sha256': series_ref['sha256'], 'cost_scope': series['cost_intervals']}
    for arm in summary['arms']:
        if arm['execution_status'] == 'CLOSED' and arm['source_commit'] != series[arm['arm'] + '_source_commit']:
            raise ValueError('an arm source differs from its declared comparison role')
    if checkpoint is not None:
        reference = {'path': str(checkpoint), 'sha256': checkpoint_sha256}
        if reference not in references or sha(checkpoint) != checkpoint_sha256:
            raise ValueError('closing checkpoint not bound by the original audit')
        value = json.loads(Path(checkpoint).read_bytes())
        if (value['active_http_count'] != 0 or value['gpu_active_span_count'] != 0
                or value['coverage_complete'] is not True or value['should_stop'] is not False
                or any(value[k] is not None for k in ('max_tokens', 'max_usd', 'max_gpu_seconds', 'deadline_at'))):
            raise ValueError('closing global checkpoint is not eligible')
        summary['accounting'] = {'unknown_requests': finite(value['unknown_request_count']),
            'gpu_seconds': finite(value['gpu_seconds']), 'usd_upper_bound': finite(value['usd_upper_bound']),
            'reported_token_lower_bound': finite(value['reported_tokens']),
            'token_upper_bound': finite(value['token_upper_bound']), 'confirmed_bill': False,
            'global_limits': dict.fromkeys(['USD', 'tokens', 'GPU_seconds', 'deadline']),
            'checkpoint_sha256': checkpoint_sha256,
            'asof': 'Actual closed checkpoint at the last closed arm; cumulative counters retain inherited history. V15 primary intervals run without new application/auxiliary admissions.'}
    with Path(out).open('x') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return {'sha256': sha(out), 'status': summary['status'], 'closed_arm_count': summary['closed_arm_count'], 'RSI_established': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--audit-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resource-checkpoint', type=Path)
    parser.add_argument('--resource-checkpoint-sha256')
    a = parser.parse_args()
    if bool(a.resource_checkpoint) != bool(a.resource_checkpoint_sha256):
        parser.error('checkpoint and its SHA256 are required together')
    print(json.dumps(export(a.audit, a.audit_sha256, a.out, a.resource_checkpoint, a.resource_checkpoint_sha256)))
