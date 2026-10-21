#!/usr/bin/env python3
"""Export recorded, source-bound results without private trace contents."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import re
import random
import shutil
import statistics
from pathlib import Path


NAMES = {
    'K04': 'masked_softmax', 'K05': 'row_logsumexp', 'K06': 'silu_mul',
    'K07': 'bias_gelu', 'K09': 'transpose_scale_cast', 'K10': 'RoPE adjacent pairs',
}
EXTRA_NAMES = {'K12': 'small_m_gemm', 'K13': 'softmax_cross_entropy',
               'K18': 'depthwise_conv2d_bias_relu'}
TOOLS = frozenset({'read', 'edit', 'write', 'shell', 'kernel_get_task',
    'kernel_submit_source', 'kernel_submit_candidate', 'kernel_get_feedback',
    'kernel_get_job', 'kernel_freeze_current', 'kernel_get_current',
    'kernel_submit', 'kernel_read_feedback', 'kernel_get_task_status'})
TIMESTAMP = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_file(reference):
    path = Path(reference['path'])
    if not path.is_file() or sha(path) != reference['sha256']:
        raise ValueError('evidence source hash mismatch')
    return path


def verify_references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and re.fullmatch(
                r'[0-9a-f]{64}', str(value.get('sha256', ''))):
            bound_file(value)
        for child in value.values():
            verify_references(child)
    elif isinstance(value, list):
        for child in value:
            verify_references(child)


def finite_number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('expected nonnegative finite metric')
    return value


def integer(value):
    finite_number(value)
    if type(value) is not int:
        raise ValueError('expected integer metric')
    return value


def operators_from_manifest(directory, manifest):
    operators = []
    if {row['task_id'] for row in manifest['operators']} != set(NAMES):
        raise ValueError('operator delivery set changed')
    for row in manifest['operators']:
        task = row['task_id']
        source = Path(directory) / task / 'kernel.py'
        if sha(source) != row['source_file_sha256']:
            raise ValueError('operator source hash mismatch')
        operators.append({'task_id': task, 'name': NAMES[task],
            'verified_speedup': finite_number(row['verified_speedup']),
            'ci_low': finite_number(row['verified_ci_low']),
            'ci_high': finite_number(row['verified_ci_high']),
            'shape_speedups': {k: finite_number(row['shape_speedup_medians'][k])
                               for k in ('t0', 't1', 't2', 'r0', 'r1')},
            'strict_environment_error_preserved': row['strict_environment_error_preserved'] is True,
            'source_sha256': row['source_file_sha256'],
            'source': f'operators/{task}/kernel.py'})
    return operators


def additional_operator_result(directory, manifest):
    """Deliver three new types without granting this failed primary arm credit."""
    if (manifest.get('kind') != 'source-bound-verified-kernel-delivery'
            or manifest.get('denominator') != 20
            or manifest.get('all20_eligible') is not False
            or manifest.get('harness_performance_causality_proven') is not False
            or manifest.get('no_inheritance_for_future_search') is not True
            or manifest.get('primary_environment_error_count') != 1
            or manifest.get('verified_gain_count') != 8):
        raise ValueError('additional operator whole-arm qualification changed')
    directory = Path(directory).resolve()
    rows = manifest['operators']
    if len(rows) != 8 or len({r['task_id'] for r in rows}) != 8:
        raise ValueError('additional source set duplicated or incomplete')
    selected = []
    for row in rows:
        task = row['task_id']
        source = (directory / row['source']).resolve()
        if not source.is_relative_to(directory) or sha(source) != row['source_file_sha256']:
            raise ValueError('additional operator source changed')
        if (row.get('contaminated') is not False or row.get('independent_verification_rounds') != 5
                or integer(row['original_correctness_case_count']) < 20 or not row.get('verify_job_id')
                or finite_number(row['verified_ci_low']) <= 1):
            raise ValueError('additional individual operator verification incomplete')
        if task in EXTRA_NAMES:
            selected.append({'task_id': task, 'name': EXTRA_NAMES[task],
                'verified_speedup': finite_number(row['verified_speedup']),
                'ci_low': finite_number(row['verified_ci_low']), 'ci_high': finite_number(row['verified_ci_high']),
                'shape_speedups': {k: finite_number(row['shape_speedup_medians'][k])
                                   for k in ('t0', 't1', 't2', 'r0', 'r1')},
                'strict_environment_error_preserved': True,
                'source_sha256': row['source_file_sha256'], 'source_generation': 'g003-v13-dev1',
                'source': f'operators/{task}/kernel.py'})
    if {r['task_id'] for r in selected} != set(EXTRA_NAMES):
        raise ValueError('three additional operator types required')
    return selected


def normalization_operator_result(directory, manifest):
    names = {'K01': 'residual_rmsnorm', 'K02': 'residual_layernorm'}
    if (manifest.get('kind') != 'source-bound-verified-kernel-delivery'
            or manifest.get('agh_source_commit') != '54b05f2a959b5f6328ac2bf572fbd981bf567094'
            or manifest.get('batch_id') != 'v14-d1-parent' or manifest.get('denominator') != 20
            or manifest.get('all20_eligible') is not False
            or manifest.get('primary_environment_error_count') != 1
            or manifest.get('verified_gain_count') != 6
            or manifest.get('harness_performance_causality_proven') is not False
            or manifest.get('no_inheritance_for_future_search') is not True):
        raise ValueError('normalization original H0 whole-batch identity differs')
    rows = manifest['operators']
    if len(rows) != 6 or {r['task_id'] for r in rows} != {'K01', 'K02', 'K04', 'K05', 'K06', 'K10'}:
        raise ValueError('normalization original verified set differs')
    directory = Path(directory).resolve()
    result = []
    for row in rows:
        task = row['task_id']
        source = (directory / row['source']).resolve()
        shapes = row['shape_speedup_medians']
        if (not source.is_relative_to(directory) or sha(source) != row['source_file_sha256']
                or set(shapes) != {'t0', 't1', 't2', 'r0', 'r1'}
                or row['independent_verification_rounds'] != 5
                or row['original_correctness_case_count'] != 21
                or row['contaminated'] is not False
                or finite_number(row['verified_ci_low']) <= 1.05
                or min(finite_number(x) for x in shapes.values()) < .95
                or not row['verify_job_id']):
            raise ValueError('normalization original five-shape verification differs')
        if task in names:
            result.append({'task_id': task, 'name': names[task],
                'verified_speedup': finite_number(row['verified_speedup']),
                'ci_low': finite_number(row['verified_ci_low']), 'ci_high': finite_number(row['verified_ci_high']),
                'shape_speedups': shapes, 'strict_environment_error_preserved': True,
                'source_sha256': row['source_file_sha256'], 'source_generation': 'g000-v14-dev1',
                'source': f'operators/{task}/kernel.py', 'source_harness_generation': 'H0/G000'})
    return result


def gemm_variant_result(directory, manifest):
    """Accept the recorded H0 variant while preserving the older application kernel."""
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()
    if digest != '0bca16d40815d1f8ec737dfe3195185d6231ee9a5f86dac5cb2069c23751e63f':
        raise ValueError('original H0 variant manifest changed')
    directory = Path(directory).resolve()
    for row in manifest['operators']:
        source = (directory / row['source']).resolve()
        if not source.is_relative_to(directory) or sha(source) != row['source_file_sha256']:
            raise ValueError('original variant executable source changed')
    row = next(r for r in manifest['operators'] if r['task_id'] == 'K12')
    return {'task_id': 'K12', 'name': row['name'],
        'verified_speedup': row['verified_speedup'],
        'ci_low': row['verified_ci_low'], 'ci_high': row['verified_ci_high'],
        'shape_speedups': row['shape_speedup_medians'],
        'candidate_us_medians': row['candidate_us_medians'],
        'paired_baseline_us_medians': row['paired_baseline_us_medians'],
        'original_correctness_case_count': row['original_correctness_case_count'],
        'independent_verification_rounds': row['independent_verification_rounds'],
        'verify_job_id': row['verify_job_id'],
        'source_sha256': row['source_file_sha256'],
        'source_generation': 'g000-v15-dev2', 'source_harness_generation': 'H0/G000 · V15 DEV2',
        'strict_environment_error_preserved': False,
        'harness_performance_causality_proven': False,
        'source': 'operator-variants/g000-v15-dev2/K12/kernel.py'}


def public_job_metadata(value):
    """Allow only matching terminal state and hex source identity, never text."""
    if not isinstance(value, dict) or value.get('has_result') is not True:
        return {}
    job, result = value.get('job'), value.get('result')
    if not isinstance(job, dict) or not isinstance(result, dict):
        return {}
    failed = {'compile_fail', 'incorrect', 'timeout', 'oom', 'crashed'}
    status, source = value.get('status'), job.get('source_hash')
    if (status not in failed | {'complete'} or job.get('status') != status
            or result.get('status') != status or not isinstance(source, str)
            or not re.fullmatch(r'[a-f0-9]{64}', source)
            or not isinstance(job.get('job_id'), str) or not 1 <= len(job['job_id']) <= 128
            or ('source_hash' in result and result['source_hash'] != source)
            or ('job_id' in result and result['job_id'] != job['job_id'])):
        return {}
    return {'job_status': status, 'job_failed': status in failed, 'source_hash': source}


def public_trace(rows):
    """Keep execution structure; never retain arguments, text, IDs or headers."""
    calls = {}
    result = []
    for row in rows:
        seq, ts, kind = row.get('seq'), row.get('ts'), row.get('type')
        data = row.get('data', {})
        if type(seq) is not int or not isinstance(data, dict):
            continue
        public = {'seq': seq}
        if isinstance(ts, str) and TIMESTAMP.fullmatch(ts):
            public['timestamp'] = ts
        if kind == 'tool/call' and data.get('name') in TOOLS:
            tool = data['name']
            calls[data.get('toolUseId')] = tool
            public.update(type='call', tool=tool)
        elif kind == 'tool/result' and data.get('toolUseId') in calls:
            public.update(type='result', tool=calls[data.get('toolUseId')],
                          is_error=data.get('isError') is True)
            if public['tool'] == 'kernel_get_job':
                public.update(public_job_metadata(data.get('structured')))
        elif kind in ('turn/start', 'turn/end'):
            public['type'] = kind
        else:
            continue
        result.append(public)
    return result


def load(path):
    document = json.loads(Path(path).read_text())
    verify_references(document)
    return document


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def application_result(repo, path, expected_hash):
    """Recompute public performance claims from source-bound paired samples."""
    document = json.loads(bound_file({'path': str(path), 'sha256': expected_hash}).read_text())
    if document.get('status') != 'MEASURED_CLEAN':
        raise ValueError('application must have a clean completed measurement')
    package, protocol = document['source_package'], document['protocol']
    if (canonical_hash(package) != document['source_package_sha256'] or
            canonical_hash(protocol) != document['protocol_sha256'] or
            package['protocol_sha256'] != document['protocol_sha256']):
        raise ValueError('application package/protocol hash mismatch')
    for name, digest in package['application_sources'].items():
        source = (Path(repo) / name).resolve()
        if not source.is_relative_to(Path(repo).resolve()):
            raise ValueError('application source must be inside the repository')
        bound_file({'path': str(source), 'sha256': digest})
    manifest = Path(repo) / 'harness-evolution/codex-rsi/operators/g001/manifest.json'
    bound_file({'path': str(manifest), 'sha256': package['manifest_sha256']})
    for task, digest in package['operator_sources'].items():
        if task not in ('K04', 'K06', 'K10'):
            raise ValueError('unexpected application operator')
        bound_file({'path': str(manifest.parent / task / 'kernel.py'), 'sha256': digest})
    labels, rounds = protocol['case_labels'], integer(protocol['timing']['paired_rounds'])
    if len(labels) != 4 or len(protocol['cases']) != 4 or set(labels) != set(document['measurement_cases']) or rounds != 30:
        raise ValueError('application declared case/sample denominator changed')
    clean = document['clean_measurement']
    if clean['all_snapshots_available_and_foreign_processes_zero'] is not True:
        raise ValueError('application clean measurement failed')
    optimized = document.get('optimized_backend','candidate')
    if optimized not in ('candidate','compiled_candidate'):
        raise ValueError('application optimized backend is not declared')
    cases = []
    gpu_samples = 0
    for label, config in zip(labels, protocol['cases']):
        row = document['measurement_cases'][label]
        if row['status'] != 'MEASURED_CLEAN' or row['config'] != config:
            raise ValueError('application case is incomplete')
        samples = row['samples']
        expected_backends = {'eager','compiled','candidate'}
        if optimized == 'compiled_candidate': expected_backends.add(optimized)
        if set(samples) != expected_backends:
            raise ValueError('strong compiled baseline and all measured backends required')
        medians = {}
        for name, sample in samples.items():
            for field in ('gpu_us_samples', 'host_us_samples'):
                values = sample[field]
                if len(values) != rounds or any(finite_number(x) <= 0 for x in values):
                    raise ValueError('invalid application sample denominator')
                stored = finite_number(sample[field.replace('samples', 'median')])
                if not math.isclose(statistics.median(values), stored, rel_tol=1e-12):
                    raise ValueError('application sample median mismatch')
            medians[name] = sample['gpu_us_median']
            if len(sample['batch_label_order']) != rounds or any(
                    sorted(order) != sorted(samples) for order in sample['batch_label_order']):
                raise ValueError('application paired order incomplete')
        if any(samples[name]['batch_label_order'] != samples[optimized]['batch_label_order']
               for name in samples):
            raise ValueError('application backends are not paired')
        public = {'label': label, 'config': config, 'gpu_us_medians': medians,
                  'paired_rounds': rounds,
                  'max_abs_error': finite_number(row['correctness']['full_block_max_abs_error'])}
        if row['correctness']['full_block_tolerance'] != protocol['full_block_output_tolerance']:
            raise ValueError('application numerical contract changed')
        for baseline in ('eager', 'compiled'):
            field = f'against_{baseline}'
            metrics = row['compiled_candidate_speedups'] if optimized == 'compiled_candidate' else row['speedups']
            metric = metrics[field]
            ratios = [x/y for x, y in zip(samples[baseline]['gpu_us_samples'],
                                         samples[optimized]['gpu_us_samples'])]
            if metric['paired_ratio_samples'] != ratios:
                raise ValueError('application paired ratio samples mismatch')
            median = statistics.median(ratios)
            rng = random.Random(metric['bootstrap_seed'])
            if metric['bootstrap_resamples'] != 2000:
                raise ValueError('application bootstrap contract changed')
            boot = sorted(statistics.median(rng.choices(ratios, k=rounds)) for _ in range(2000))
            interval = [boot[49], boot[1949]]
            if metric['median'] != median or metric['bootstrap_95_percent_interval'] != interval:
                raise ValueError('application paired median/interval mismatch')
            public[field] = {'median': median, 'ci_low': interval[0], 'ci_high': interval[1]}
        cases.append(public)
        gpu_samples += rounds * len(samples)
    eager = [c['against_eager']['median'] for c in cases]
    compiled = [c['against_compiled']['median'] for c in cases]
    beats_compiled = all(c['against_compiled']['ci_low'] > 1 for c in cases)
    if document['conclusions']['beats_compiled_baseline'] is not beats_compiled:
        raise ValueError('application strong baseline conclusion inconsistent')
    if document['raw_gpu_latency_samples'] != gpu_samples or document['raw_host_latency_samples'] != gpu_samples:
        raise ValueError('application raw sample count mismatch')
    return {'status': 'MEASURED_CLEAN', 'cases': cases, 'raw_gpu_samples': gpu_samples,
            'optimized_backend': optimized,
            'raw_host_samples': gpu_samples, 'beats_compiled_baseline': beats_compiled,
            'pretrained_model': document['pretrained_model'] is True,
            'text_quality_measured': document['text_quality_measured'] is True,
            'source_package_sha256': document['source_package_sha256'],
            'raw_result_sha256': document['raw_result_sha256'],
            'report_sha256': expected_hash, 'report': 'application/' + Path(path).name,
            'summary': f'四种形状完整块输出通过；相对 eager {min(eager):.2f}–{max(eager):.2f}×，'
                       f'相对 torch.compile {min(compiled):.2f}–{max(compiled):.2f}×。',
            'limits': ['固定随机权重的完整decoder计算；没有预训练文本质量测量。',
                       '本版候选在四种形状均慢于编译基线。',
                       '30轮配对计时；独立核查是只读重算，尚无第二次GPU复现。']}


def closed_failure(path, expected_hash):
    document = load(bound_file({'path': str(path), 'sha256': expected_hash}))
    if (document['kind'] != 'source-bound-official-first-primary-closed-failure' or
            document['status'] != 'CLOSED_INFRASTRUCTURE_FAILURE_NOT_ELIGIBLE' or
            document['denominator'] != 20 or document['official_score']['all20_eligible'] is not False):
        raise ValueError('closed failure status/denominator inconsistent')
    raw = document['controller_terminal_statuses']
    if raw != {'GLOBAL_BUDGET_EXHAUSTED': 20}:
        raise ValueError('closed failure full denominator changed')
    reason = document['failure_evidence']['safe_final_CLI_error']
    if reason != 'successful exit0 stop snapshot: gpu_coverage_stale; coverage false':
        raise ValueError('unrecognized closed failure diagnostic')
    score, accounting = document['official_score'], document['accounting']
    latest = {'id': document['batch_id'], 'status': 'CLOSED',
              'raw_terminal_counts': raw,
              'verified_gain_count': integer(score['counts']['verified_gain']),
              'full20_geomean': finite_number(score['all20_deployment_geomean']),
              'eligible': False, 'stop_diagnostic': 'gpu_coverage_stale',
              'historical_stop_cause': 'UNRESOLVED'}
    public_accounting = {'unknown_requests': integer(accounting['unknown_all_sources']),
        'gpu_seconds': finite_number(accounting['GPU_seconds_all_sources']),
        'usd_upper_bound': finite_number(accounting['USD_conservative_upper_bound']),
        'confirmed_bill': False, 'global_limits': accounting['global_limits']}
    if public_accounting['global_limits'] != dict.fromkeys(('USD', 'tokens', 'GPU_seconds', 'deadline')):
        raise ValueError('current campaign global limits changed')
    return latest, public_accounting


def cached_application_result(repo, path, expected_hash):
    """Validate paired public samples before displaying the strongest baseline."""
    report_path = bound_file({'path': str(path), 'sha256': expected_hash})
    raw = json.loads(report_path.read_bytes())
    helper_path = Path(__file__).parent / 'export_cached_application.py'
    spec = importlib.util.spec_from_file_location('cached_evidence_math', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if (raw.get('kind') != 'source-bound-cached-decoder-gpu-measurement'
            or raw.get('status') != 'MEASURED_CLEAN'
            or raw['clean_measurement']['all_snapshots_available_and_foreign_processes_zero'] is not True
            or raw['accounting']['official_queue_job_status'] != 'complete'
            or raw['accounting']['queue_report_sha256_matches'] is not True
            or raw['protocol']['cases'] != helper.CONFIGS
            or raw['protocol']['full_block_output_tolerance'] != {'rtol': .005, 'atol': .005}
            or raw['protocol']['compiler_specialization_limit'] != 32
            or helper.canonical_hash(raw['protocol']) != raw['protocol_sha256']
            or helper.canonical_hash(raw['source_package']) != raw['source_package_sha256']):
        raise ValueError('cached application protocol, source or accounting differs')
    for name, expected in raw['source_package']['application_sources'].items():
        source = (Path(repo) / name).resolve()
        if not source.is_relative_to(Path(repo).resolve()) or sha(source) != expected:
            raise ValueError('cached application source hash mismatch')
    cases = []
    if set(raw['measurement_cases']) != set(raw['protocol']['case_labels']):
        raise ValueError('cached application case denominator differs')
    for label, config in zip(raw['protocol']['case_labels'], helper.CONFIGS):
        row = raw['measurement_cases'][label]
        samples = row['samples']
        if (row['config'] != config or row['status'] != 'MEASURED_CLEAN'
                or set(samples) != set(helper.BACKENDS)
                or any(row['backend_qualification'][name]['status'] != 'AVAILABLE' for name in helper.BACKENDS)):
            raise ValueError('cached application required backend missing')
        helper.validate_samples(samples, 30)
        metric = helper.strong_metric(samples, 'compiled_candidate')
        if row['candidate_gains']['compiled_candidate']['against_strong_library'] != metric:
            raise ValueError('cached application strong paired gain differs')
        errors = row['candidate_correctness']['full_decode_max_abs_errors']
        cases.append({'label': label, 'config': config, 'paired_rounds': 30,
            'gpu_us_medians': {name: finite_number(samples[name]['gpu_us_median']) for name in helper.BACKENDS},
            'against_strong_library': {'median': metric['median'],
                'ci_low': metric['bootstrap_95_percent_interval'][0],
                'ci_high': metric['bootstrap_95_percent_interval'][1]},
            'max_abs_error': max(finite_number(value) for value in errors.values())})
    victory = all(row['against_strong_library']['ci_low'] > 1 for row in cases)
    if (raw['raw_gpu_latency_samples'] != 720 or raw['raw_host_latency_samples'] != 720
            or raw['conclusions']['beats_all_strong_library_cases']['compiled_candidate'] is not victory):
        raise ValueError('cached application sample or victory claim differs')
    ratios = [c['against_strong_library']['median'] for c in cases]
    return {'status': 'MEASURED_CLEAN', 'cases': cases, 'raw_gpu_samples': 720,
        'raw_host_samples': 720, 'beats_strong_library': victory,
        'summary': f'单token与KV缓存完整解码：四形状、六后端、30配对轮。整块编译候选相对每轮最快四种库基线为{min(ratios):.3f}–{max(ratios):.3f}倍，未超过强基线。',
        'report': 'application/' + report_path.name, 'report_sha256': expected_hash,
        'pretrained_model': False, 'harness_performance_causality_proven': False,
        'limits': raw['limits']}


def packed_application_result(repo, path, expected_hash):
    """Recompute ordinary and six-library paired ratios, preserving losses."""
    report_path = bound_file({'path': str(path), 'sha256': expected_hash})
    raw = json.loads(report_path.read_bytes())
    helper_path = Path(__file__).parent / 'export_packed_application.py'
    spec = importlib.util.spec_from_file_location('packed_evidence_math', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    protocol, package = raw['protocol'], raw['source_package']
    if (raw.get('kind') != 'source-bound-packed-cached-decoder-gpu-measurement'
            or raw.get('status') != 'MEASURED_CLEAN'
            or raw['clean_measurement']['all_snapshots_available_and_foreign_processes_zero'] is not True
            or raw['accounting']['official_queue_job_status'] != 'complete'
            or raw['accounting']['queue_report_sha256_matches'] is not True
            or protocol['cases'] != helper.CONFIGS
            or protocol['requested_backends'] != list(helper.BACKENDS)
            or protocol['strong_baseline_required'] != list(helper.LIBRARY)
            or protocol['logical_projection_count'] != 7 or protocol['candidate_physical_projection_count'] != 4
            or protocol['timing']['paired_rounds'] != 30 or protocol['max_wall_seconds'] != 900
            or protocol['full_block_output_tolerance'] != {'rtol': .005, 'atol': .005}
            or helper.canonical_hash(protocol) != raw['protocol_sha256']
            or helper.canonical_hash(package) != raw['source_package_sha256']
            or package['protocol_sha256'] != raw['protocol_sha256']):
        raise ValueError('packed application protocol/source/accounting differs')
    for name, expected in package['application_sources'].items():
        source = (Path(repo) / name).resolve()
        if not source.is_relative_to(Path(repo).resolve()) or sha(source) != expected:
            raise ValueError('packed application source hash mismatch')
    manifests = [Path(repo) / 'harness-evolution/codex-rsi/operators/g001/manifest.json',
                 Path(repo) / 'harness-evolution/codex-rsi/operators/g003-v13-dev1/manifest.json']
    if [sha(p) for p in manifests] != package['operator_manifest_sha256s'] or set(package['operator_sources']) != {'K04', 'K06', 'K10', 'K12'}:
        raise ValueError('packed application original operator binding differs')
    for task, expected in package['operator_sources'].items():
        source = manifests[task == 'K12'].parent / task / 'kernel.py'
        if sha(source) != expected:
            raise ValueError('packed application original kernel changed')
    if len(protocol['case_labels']) != 4 or set(raw['measurement_cases']) != set(protocol['case_labels']):
        raise ValueError('packed application full case denominator required')
    cases = []
    for label, config in zip(protocol['case_labels'], helper.CONFIGS):
        row = raw['measurement_cases'][label]
        samples = row['samples']
        if (row['config'] != config or row['status'] != 'MEASURED_CLEAN'
                or set(samples) != set(helper.BACKENDS)
                or row['strong_baseline_complete'] is not True
                or any(row['backend_qualification'][name]['status'] != 'AVAILABLE' for name in helper.BACKENDS)):
            raise ValueError('packed application all eight qualified backends required')
        helper.validate_samples(samples, 30)
        strong = helper.strong_metric(samples, 'compiled_candidate')
        ordinary = helper.pair_metric(samples, 'fp16_library', 'compiled_candidate')
        if (row['candidate_gains']['compiled_candidate']['against_strong_library'] != strong
                or row['candidate_gains']['compiled_candidate']['against_fp16_library'] != ordinary
                or row['candidate_correctness']['full_block_tolerance'] != protocol['full_block_output_tolerance']):
            raise ValueError('packed application paired gain or correctness differs')
        def compact(metric):
            return {'median': metric['median'], 'ci_low': metric['bootstrap_95_percent_interval'][0],
                    'ci_high': metric['bootstrap_95_percent_interval'][1]}
        cases.append({'label': label, 'config': config, 'paired_rounds': 30,
                      'gpu_us_medians': {name: finite_number(samples[name]['gpu_us_median']) for name in helper.BACKENDS},
                      'against_fp16_library': compact(ordinary), 'against_strong_library': compact(strong),
                      'max_abs_error': max(finite_number(x) for x in row['candidate_correctness']['full_decode_max_abs_errors'].values())})
    victory = all(c['against_strong_library']['ci_low'] > 1 for c in cases)
    if (raw['raw_gpu_latency_samples'] != 960 or raw['raw_host_latency_samples'] != 960
            or raw['conclusions']['beats_all_strong_library_cases']['compiled_candidate'] is not victory):
        raise ValueError('packed application sample or victory claim differs')
    ordinary = [c['against_fp16_library']['median'] for c in cases]
    strong = [c['against_strong_library']['median'] for c in cases]
    return {'status': 'MEASURED_CLEAN', 'cases': cases, 'raw_gpu_samples': 960, 'raw_host_samples': 960,
            'beats_strong_library': victory,
            'summary': f'合并投影缓存解码：四形状、八后端、30配对轮，七次逻辑投影合并为四次。相对普通FP16库{min(ordinary):.2f}–{max(ordinary):.2f}倍；相对每轮最快六库{min(strong):.3f}–{max(strong):.3f}倍，尚未证明全形状强基线胜出。',
            'report': 'application/' + report_path.name, 'report_sha256': expected_hash,
            'pretrained_model': False, 'harness_performance_causality_proven': False, 'limits': raw['limits']}


def packed_variant_application_result(repo, path, expected_hash):
    """Verify both actual reports with the unchanged sample/source validator."""
    report_path = bound_file({'path': str(path), 'sha256': expected_hash})
    combined = json.loads(report_path.read_bytes())
    if (combined.get('kind') != 'two-replica-packed-GEMM-source-bound-GPU-verification'
            or combined.get('status') != 'TWO_FULL_REPLICAS_CLOSED_ORIGINAL_EXPORT_AND_INPUT_CONSISTENCY_PASS'
            or combined.get('owned_GPU_groups_RPC_socket_owner_closed') is not True
            or combined.get('independent_fresh_GPU_reproduction') is not True
            or combined.get('matching_input_weights_cache_and_source_protocol') is not True
            or combined.get('all_original_four_shapes_six_libraries_eight_backends_30_rounds_tolerances_and_900_seconds_preserved') is not True
            or combined.get('observer_errors') != [] or combined.get('new_model_API_calls') != 0
            or combined.get('foreign_processes_signaled') != 0
            or any(combined.get(key) is not False for key in
                   ('harness_performance_causality_proven', 'pretrained_model', 'text_quality_measured', 'RSI_proven'))
            or combined.get('source_patch_author') != 'Codex outer loop'
            or combined.get('operator_source_generation') != 'original H0 G000 V15 DEV2'):
        raise ValueError('two-replica application closure or provenance differs')
    previous = combined.get('previous_r1_budget_failures_retained', {})
    if (previous.get('status') != 'CLOSED_PACKED_GEMM_FAILURES_PRESERVED'
            or previous.get('partition_exit_codes') != [2, 2]
            or previous.get('partial_samples_and_GPU_cost_not_refunded') is not True):
        raise ValueError('prior application failures or cost missing')
    replicas = combined.get('replicas', [])
    if len(replicas) != 2 or {row['physical_gpu'] for row in replicas} != {0, 3}:
        raise ValueError('two independent physical GPU replicas required')
    spec = importlib.util.spec_from_file_location('packed_variant_evidence_validator',
        Path(__file__).parent / 'export_packed_gemm_variant.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    _, plan = helper.expected_plan()
    repo = Path(repo).resolve()
    checked, fingerprints, raw_hashes = [], [], []
    for replica in replicas:
        reference = replica['report']
        relative = Path(reference['path'])
        source = (repo / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(repo):
            raise ValueError('application report path escapes repository')
        bound_file({'path': str(source), 'sha256': reference['sha256']})
        raw = json.loads(source.read_bytes())
        if (raw.get('kind') != 'source-bound-packed-gemm-variant-gpu-measurement'
                or raw['device']['physical_gpu'] != replica['physical_gpu']
                or raw['device']['tf32'] is not False
                or raw['device']['fp16_reduced_precision_reduction'] is not False
                or raw['clean_measurement']['all_snapshots_available_and_foreign_processes_zero'] is not True
                or integer(raw['clean_measurement']['snapshot_count']) <= 0
                or raw['accounting']['official_queue_job_status'] != 'complete'
                or raw['accounting']['queue_report_sha256_matches'] is not True
                or raw['raw_result_sha256'] != replica['raw_result_sha256']
                or raw['original_numeric_validator_sha256'] != helper.FROZEN_VALIDATOR_SHA
                or any(raw.get(key) is not False for key in
                       ('pretrained_model', 'text_quality_measured', 'harness_performance_causality_proven'))):
            raise ValueError('actual replica measurement or original accounting differs')
        projection = {**raw, 'kind': plan['kind'], 'measurements': raw['measurement_cases'],
            'all_strong_baselines_complete': raw['conclusions']['all_strong_baselines_complete']}
        result = helper.validate_partition(projection)
        if (result['raw_gpu_latency_samples'] != raw['raw_gpu_latency_samples']
                or result['raw_host_latency_samples'] != raw['raw_host_latency_samples']
                or result['raw_gpu_latency_samples'] != 960
                or result['beats_all_strong_library_cases'] != raw['conclusions']['beats_all_strong_library_cases']
                or raw['conclusions']['all_four_cases_measured'] is not True
                or set(replica['cases']) != set(result['measurement_cases'])):
            raise ValueError('full application sample denominator or victory differs')
        cases, inputs = [], {}
        for label, row in result['measurement_cases'].items():
            statuses = {name: item['status'] for name, item in row['backend_qualification'].items()}
            summary_gains = {name: {'median': metric['median'],
                'bootstrap_95_percent_interval': metric['bootstrap_95_percent_interval'],
                'victory_proven': metric.get('victory_proven')}
                for name, metric in row['candidate_gains']['compiled_candidate'].items()}
            if (len(statuses) != 8 or any(value != 'AVAILABLE' for value in statuses.values())
                    or replica['cases'][label] != {'all_backends': statuses,
                                                  'compiled_candidate': summary_gains}):
                raise ValueError('combined summary differs from original qualified samples')
            def compact(metric):
                return {'median': finite_number(metric['median']),
                        'ci_low': finite_number(metric['bootstrap_95_percent_interval'][0]),
                        'ci_high': finite_number(metric['bootstrap_95_percent_interval'][1])}
            gains = row['candidate_gains']['compiled_candidate']
            cases.append({'label': label, 'config': row['config'], 'paired_rounds': 30,
                'gpu_us_medians': {name: finite_number(sample['gpu_us_median']) for name, sample in row['samples'].items()},
                'against_fp16_library': compact(gains['against_fp16_library']),
                'against_strong_library': compact(gains['against_strong_library']),
                'max_abs_error': max(row['candidate_correctness']['full_decode_max_abs_errors'].values())})
            inputs[label] = row['input_weights_cache_sha256']
        checked.append({'physical_gpu': replica['physical_gpu'], 'cases': cases,
            'report': 'application/' + source.name, 'report_sha256': reference['sha256']})
        fingerprints.append((raw['protocol_sha256'], raw['source_package_sha256'], inputs))
        raw_hashes.append(raw['raw_result_sha256'])
    if fingerprints[0] != fingerprints[1] or len(set(raw_hashes)) != 2:
        raise ValueError('replica source, protocol or independent input identity differs')
    victory = all(row['against_strong_library']['ci_low'] > 1 for rep in checked for row in rep['cases'])
    if combined.get('all_strong_library_victory_proven') is not victory:
        raise ValueError('two-replica strong baseline conclusion differs')
    ordinary = [row['against_fp16_library']['median'] for rep in checked for row in rep['cases']]
    strong = [row['against_strong_library']['median'] for rep in checked for row in rep['cases']]
    return {'status': 'TWO_FULL_REPLICAS_CLOSED', 'replicas': checked, 'physical_gpu_indices': [0, 3],
        'raw_gpu_samples': 1920, 'raw_host_samples': 1920, 'beats_strong_library': victory,
        'summary': f'新H0 GEMM版本：GPU0/3两独立副本，各四形状、八后端、30轮。相对普通FP16库{min(ordinary):.2f}–{max(ordinary):.2f}倍；相对每轮最快六库{min(strong):.3f}–{max(strong):.3f}倍，边界形状更慢。',
        'previous_r1_budget_failures_retained': previous,
        'report': 'application/' + report_path.name, 'report_sha256': expected_hash,
        'operator_source_generation': combined['operator_source_generation'],
        'pretrained_model': False, 'harness_performance_causality_proven': False,
        'viewer_rechecks_original_public_samples_not_fresh_GPU_execution': True}


def pretrained_application_result(repo, path, expected_hash, precision_diagnostic=False):
    """Recompute all four pretrained cases, including unavailable timings."""
    report_path = bound_file({'path': str(path), 'sha256': expected_hash})
    raw = json.loads(report_path.read_bytes())
    helper_path = Path(__file__).parent / ('export_pretrained_precision.py' if precision_diagnostic
                                         else 'export_pretrained_application.py')
    spec = importlib.util.spec_from_file_location('pretrained_evidence_math', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if precision_diagnostic:
        diagnostic = raw.get('precision_diagnostic', {})
        if (diagnostic.get('emulate_precision_casts') is not True
                or diagnostic.get('original_tolerances_preserved') is not True
                or diagnostic.get('prior_failed_results_superseded') is not False
                or diagnostic.get('fresh_GPU_reproduction') is not False
                or diagnostic.get('original_validator_sha256') != helper.ORIGINAL_EXPORT_SHA
                or diagnostic.get('exporter_source_sha256') != sha(helper_path)):
            raise ValueError('precision diagnostic source or preserved failures differ')
    package, protocol = raw['source_package'], raw['protocol']
    if (raw.get('kind') != 'source-bound-pretrained-gpt2-gpu-measurement'
            or canonical_hash(package) != raw['source_package_sha256']
            or canonical_hash(protocol) != raw['protocol_sha256']
            or package['protocol_sha256'] != raw['protocol_sha256']
            or raw['conclusions']['all_four_cases_measured'] is not True):
        raise ValueError('pretrained source/protocol or whole-case identity differs')
    repo = Path(repo).resolve()
    for name, expected in package['application_sources'].items():
        source = (repo / name).resolve()
        if not source.is_relative_to(repo) or sha(source) != expected:
            raise ValueError('pretrained measured application source changed')
    model_spec = importlib.util.spec_from_file_location('private_pretrained_view_model',
        Path(__file__).parent / 'application/pretrained_gpt2.py')
    model_app = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_app)
    manifest = repo / 'harness-evolution/codex-rsi/operators/g000-v14-dev1/manifest.json'
    binding = model_app.verify_kernel(manifest, package['operator_manifest_sha256'])
    if (binding['source']['sha256'] != package['operator_kernel_sha256']
            or binding['candidate_id'] != package['candidate_id']
            or binding['verify_job_id'] != package['verify_job_id']
            or package['model']['model'] != 'gpt2'
            or package['model']['revision'] != model_app.MODEL_REVISION
            or set(package['model']['files']) != set(model_app.MODEL_FILE_SHA256S)
            or any(package['model']['files'][name]['sha256'] != value
                   for name, value in model_app.MODEL_FILE_SHA256S.items())):
        raise ValueError('pretrained weights or original verified K02 identity differs')
    partitions = raw['partitions']
    if len(partitions) != 2 or {row['device']['physical_gpu'] for row in partitions} != {0, 3}:
        raise ValueError('pretrained original two-GPU partition coverage differs')
    for row in partitions:
        account, device = row['accounting'], row['device']
        if (account['official_queue_job_status'] != 'complete'
                or account['queue_job_kind'] != 'application'
                or account['queue_report_sha256_matches'] is not True
                or integer(row['clean_snapshot_count']) <= 0
                or device['tf32'] is not False
                or device['fp16_reduced_precision_reduction'] is not False):
            raise ValueError('pretrained original terminal accounting or precision differs')
    # This is a read-only schema projection of public samples. It makes no new
    # assertion about private queues, device execution or a fresh reproduction.
    projection = {**raw, 'kind': 'source-bound-pretrained-gpt2-application',
                  'execution_partition': 'all', 'selected_case_labels': protocol['case_labels'],
                  'measurements': raw['measurement_cases'],
                  'all_strong_baselines_complete': raw['conclusions']['all_strong_baselines_complete']}
    checked = helper.validate_partition(projection)
    if (checked['raw_gpu_latency_samples'] != raw['raw_gpu_latency_samples']
            or checked['raw_host_latency_samples'] != raw['raw_host_latency_samples']
            or checked['beats_all_strong_library_cases'] != raw['conclusions']['beats_all_strong_library_cases']):
        raise ValueError('pretrained sample denominator or victory differs')
    cases = []
    def compact(metric):
        return None if metric is None else {
            'median': metric['median'], 'ci_low': metric['bootstrap_95_percent_interval'][0],
            'ci_high': metric['bootstrap_95_percent_interval'][1]}
    for label in protocol['case_labels']:
        row = checked['measurement_cases'][label]
        gains = row['candidate_gains']['compiled_candidate'] or {}
        cases.append({'label': label, 'config': row['config'], 'paired_rounds': 30,
            'backend_status': {key: value['status'] for key, value in row['backend_qualification'].items()},
            'gpu_us_medians': {key: finite_number(value['gpu_us_median']) for key, value in row['samples'].items()},
            'against_fp16_library': compact(gains.get('against_fp16_library')),
            'against_strong_library': compact(gains.get('against_strong_library'))})
    qualified = sum(row['backend_status']['compiled_candidate'] == 'AVAILABLE' for row in cases)
    ordinary = cases[0]['against_fp16_library']
    short_gain = f"{ordinary['median']:.2f}倍" if ordinary is not None else '无有效加速比'
    return {'status': 'MEASURED_CLEAN', 'pretrained_model': True, 'parameter_count': raw['parameter_count'],
            'cases': cases, 'raw_gpu_samples': checked['raw_gpu_latency_samples'],
            'raw_host_samples': checked['raw_host_latency_samples'],
            'beats_strong_library': checked['beats_all_strong_library_cases']['compiled_candidate'],
            'summary': f'真实预训练GPT-2：124,439,808参数、12层、四种序列形状、双卡实测。编译候选{qualified}/4形状通过；短序列相对普通FP16库约{short_gain}。长序列候选及编译库基线存在数值失败，未证明全形状或强基线收益。',
            'report': 'application/' + report_path.name, 'report_sha256': expected_hash,
            'text_quality_measured': False, 'harness_performance_causality_proven': False,
            'limits': raw['limits'],
            **({'precision_diagnostic': raw['precision_diagnostic']} if precision_diagnostic else {})}


def pretrained_precision_application_result(repo, path, expected_hash):
    return pretrained_application_result(repo, path, expected_hash, precision_diagnostic=True)


def independent_cached_result(repo, path, expected_hash, original, original_hash):
    result = cached_application_result(repo, path, expected_hash)
    previous = json.loads(bound_file({'path': str(original), 'sha256': original_hash}).read_bytes())
    cached_application_result(repo, original, original_hash)
    raw = json.loads(Path(path).read_bytes())
    proof = raw['independent_gpu_reproduction']
    if (proof['status'] != 'COMPLETE_CLEAN' or proof['original_public_report_sha256'] != original_hash
            or proof['original_raw_result_sha256'] != previous['raw_result_sha256']
            or type(proof['original_physical_gpu']) is not int or type(proof['new_physical_gpu']) is not int
            or not 0 <= proof['new_physical_gpu'] < 8 or proof['new_physical_gpu'] == proof['original_physical_gpu']
            or proof['complete_fresh_gpu_samples'] != 720 or proof['complete_fresh_host_samples'] != 720
            or proof['all_58_queue_observer_errors'] != 0
            or any(proof[key] is not True for key in ('identical_frozen_source_and_protocol',
                'identical_all_four_input_weight_cache_hashes', 'fresh_official_queue_job_after_original_finish', 'not_a_pooled_measurement'))
            or any(raw[key] != previous[key] for key in ('source_package', 'source_package_sha256', 'protocol', 'protocol_sha256'))
            or raw['accounting']['claim_at'] <= previous['accounting']['finish_at']
            or raw['raw_result_sha256'] == previous['raw_result_sha256']):
        raise ValueError('independent reproduction identity or fresh execution differs')
    for label in raw['protocol']['case_labels']:
        if raw['measurement_cases'][label]['input_weights_cache_sha256'] != previous['measurement_cases'][label]['input_weights_cache_sha256']:
            raise ValueError('independent input/weight/cache hash changed')
    result['independent_gpu_reproduction'] = proof
    result['summary'] = f"独立GPU复测：原GPU{proof['original_physical_gpu']}→GPU{proof['new_physical_gpu']}，原源码、协议及四形状输入完全一致；新720个GPU和720个主机样本。支持可复现性，强基线全形状胜出仍未成立。"
    return result


def g004_source_result(path, expected_hash):
    public = json.loads(bound_file({'path': str(path), 'sha256': expected_hash}).read_bytes())
    if (expected_hash != 'ac4bd4a2b64b3739d160f1c38c80317aad9b1a92fc285798f61576b96f183c4a'
            or public['kind'] != 'source-bound-g004-engineering-qualification'
            or public['status'] != 'ACTUAL_NEW_RUNTIME_CPU_PASS'
            or public['source_commit'] != '271487a50514cbd4208309e884c6c9ef3e3c31af'
            or public['parent_source_commit'] != '7e9aaa163b1a46eb6fd632ea0bf267202449cd0a'
            or public['original_CPU_receipt_sha256'] != '9ca69eb522e4b249d6c9df518fe6608e5f4c93d6ea4e55b825a0eeb631a6c2ec'
            or public['runtime_file_count'] != 175 or public['actual_localhost_request_bodies'] != 39
            or public['model_authorship_verified'] is not False
            or public['formal_source_performance_promoted'] is not False
            or public['external_model_calls'] != 0 or public['GPU_calls'] != 0):
        raise ValueError('G004 engineering source qualification differs')
    return {'generation': 'G004', 'commit': public['source_commit'],
            'title': '工作区路径恢复提示',
            'detail': '路径拒绝提示显示当前工作区，并建议使用相对路径。权限与ACL保持原样。229项相关测试、551项guard通过，保留2项原跳过；新运行包和39次localhost请求已资格验证。',
            'paid_improvement_measured': False, 'model_authorship_verified': False,
            'source_author': public['source_author'], 'runtime_file_count': 175,
            'actual_localhost_request_bodies': 39, 'qualification_sha256': expected_hash,
            'qualification_report': 'source/' + Path(path).name}


def g005_source_result(path, expected_hash):
    raw = json.loads(bound_file({'path': str(path), 'sha256': expected_hash}).read_bytes())
    if (expected_hash != '1786481de9d034ba23cc123902cd0c3bf1889c9cf21f688a87a010ad8a317193'
            or raw['kind'] != 'source-bound-g005-engineering-qualification'
            or raw['status'] != 'ACTUAL_NEW_RUNTIME_CPU_PASS'
            or raw['source_commit'] != '8195f2debdff45e9702d4187b2eac698482c8585'
            or raw['parent_source_commit'] != '271487a50514cbd4208309e884c6c9ef3e3c31af'
            or raw['model_authorship_verified'] is not False
            or raw['formal_source_performance_promoted'] is not False
            or raw['runtime_file_count'] != 175 or raw['actual_localhost_request_bodies'] != 39
            or raw['external_model_calls'] != 0 or raw['GPU_calls'] != 0):
        raise ValueError('G005 engineering qualification identity differs')
    return {'generation': 'G005', 'commit': raw['source_commit'],
            'title': '拒绝操作与失败作业的恢复提示',
            'detail': '拒绝操作后先确认当前状态；明确失败的计算作业提醒先诊断、修改再重试。保留权限和原结果。189项相关回归、551项guard及39次新运行包localhost请求通过；搜索收益由新V15对照验证。',
            'paid_improvement_measured': False, 'model_authorship_verified': False,
            'source_author': raw['source_author'], 'runtime_file_count': 175,
            'actual_localhost_request_bodies': 39, 'qualification_sha256': expected_hash,
            'qualification_report': 'source/' + Path(path).name}


def g006_source_result(path, expected_hash):
    raw = json.loads(bound_file({'path': str(path), 'sha256': expected_hash}).read_bytes())
    context = raw.get('actual_context_regression', {})
    if (expected_hash != '33996b1fb03d7d8548e826720f6607b56dec45207bacb019f0d320ae472d2264'
            or raw['kind'] != 'source-runtime-small-window-compaction-CPU-qualification'
            or raw['status'] != 'ACTUAL_NEW_RUNTIME_CPU_PASS'
            or raw['source_commit'] != '79329a1e36b9e2c9785bae1e29d5b788d7149e26'
            or raw['parent_source_commit'] != '8195f2debdff45e9702d4187b2eac698482c8585'
            or raw['model_authorship_verified'] is not False
            or raw['performance_promotion'] is not False or raw['rsi_established'] is not False
            or raw['runtime_file_count'] != 175 or raw['actual_localhost_request_bodies'] != 39
            or raw['external_model_calls'] != 0 or raw['GPU_calls'] != 0
            or context['actual_legal_HTTP_requests'] != 1 or context['overflow_HTTP_requests'] != 0
            or context['effective_reserve_tokens'] != 8192 or context['output_tokens_limit'] != 4096
            or context['exact_68000_byte_input_preserved_in_actual_body'] is not True):
        raise ValueError('G006 actual context/runtime qualification identity differs')
    return {'generation': 'G006', 'commit': raw['source_commit'],
            'title': '按实际窗口适配压缩预留',
            'detail': '32K窗口预留适配为8K。旧运行包提前停止的合法大输入，在新包完整发送；真正超限输入仍零请求拒绝。221项相关回归、551项guard、独立175文件构建和39次localhost资格通过；原24请求、四候选及权限保持。搜索性能等待新对照。',
            'paid_improvement_measured': False, 'model_authorship_verified': False,
            'source_author': raw['source_author'], 'runtime_file_count': 175,
            'actual_localhost_request_bodies': 39, 'actual_context_requests': 1,
            'qualification_sha256': expected_hash, 'qualification_report': 'source/' + Path(path).name}


def g007_source_result(path, expected_hash):
    raw = json.loads(bound_file({'path': str(path), 'sha256': expected_hash}).read_bytes())
    required = {'generation': 'G007', 'status': 'SOURCE_FROZEN_FRESH_BUILD_ACTUAL_SDK_CPU_QUALIFIED',
                'source_commit': 'c2de21af179b3fb24f1a67147ba871322f94493a',
                'engineering_parent_commit': '79329a1e36b9e2c9785bae1e29d5b788d7149e26',
                'runtime_file_count': 175, 'original_canonical_G006_unchanged': True,
                'model_authorship_verified': False, 'external_model_calls': 0, 'GPU_calls': 0,
                'RSI_proven': False, 'quality_promotion': False}
    if (expected_hash != '5e8471506e8743a53b39a936a16ffff22c274d2ee335061ab1e4c35b45f555bf'
            or any(raw.get(k) != v for k, v in required.items())):
        raise ValueError('G007 actual source/runtime qualification identity differs')
    verify_references(raw)
    sdk = load(bound_file(raw['actual_sdk_qualification']))
    if sdk.get('actual_body_count') != 39 or sdk.get('owned_daemon_stopped') is not True:
        raise ValueError('G007 actual SDK request count differs')
    return {'generation': 'G007', 'commit': raw['source_commit'],
            'title': '结构化工具 JSON 的模型视图压缩',
            'detail': '仅压缩与结构化结果匹配的完整 JSON 字符串外空白；保留数字、重复键、字符串、原始事件和权限封套。364项相关回归、551项guard及独立175文件新运行包/39次实际SDK请求通过。搜索收益由V17同任务对照验证。',
            'paid_improvement_measured': False, 'model_authorship_verified': False,
            'source_author': raw['source_author'], 'runtime_file_count': 175,
            'actual_localhost_request_bodies': 39, 'qualification_sha256': expected_hash,
            'qualification_report': 'source/' + Path(path).name}


def pretrained_numerical_diagnosis_result(path, expected_hash):
    raw = json.loads(bound_file({'path': str(path), 'sha256': expected_hash}).read_bytes())
    required = {'kind': 'source-bound-original-K02-pretrained-GPU-numerical-diagnosis',
                'status': 'DIAGNOSIS_EXECUTION_CLOSED_CANDIDATE_STILL_INCORRECT',
                'parameter_count': 124439808, 'local_kernel_checks_passed': 48,
                'reference_glue_full_model_bitwise_equal_cases': 4,
                'original_K02_full_model_correct_cases': 1, 'performance_measured': False,
                'strong_baseline_victory_proven': False, 'RSI_proven': False,
                'model_API_calls': 0, 'original_input_tokens_exported': False,
                'owned_processes_and_RPC_closed': True}
    if (expected_hash != '4c6a0a597a76593ab060301d6bac9bc785e0a5fc5cdc7fc25d47189202e074be'
            or any(raw.get(k) != v for k, v in required.items())):
        raise ValueError('original K02 actual GPU numerical diagnosis identity differs')
    verify_references(raw)
    labels = ['B1_S33', 'B1_S128', 'B1_S512', 'B2_S1024']
    if (set(raw['cases']) != set(labels)
            or raw['protocol']['logit_tolerance'] != {'atol': .05, 'rtol': .005}
            or raw['protocol']['kernel_tolerance'] != {'atol': .01, 'rtol': .002}):
        raise ValueError('original four cases or numerical tolerances differ')
    cases = []
    for label in labels:
        row = raw['cases'][label]
        local = row['original_k02_checks']
        layers = row['layer_output_errors']
        candidate = row['backend_qualification']['original_K02']
        if (len(local) != 12 or len(layers['original_K02']) != 12
                or len(layers['reference_glue']) != 12
                or any(x['status'] != 'AVAILABLE' or x['reference_residual_bitwise_equal'] is not True for x in local)
                or any(x['bitwise_equal'] is not True for x in layers['reference_glue'])
                or candidate['status'] != ('AVAILABLE' if label == 'B1_S33' else 'INCORRECT')
                or row['timing_measured'] is not False):
            raise ValueError('local passes must retain failed complete-model outputs')
        cases.append({'label': label, 'candidate_status': candidate['status'],
            'local_checks_passed': len(local), 'reference_glue_bitwise_equal': True,
            'logit_max_abs_error': finite_number(candidate['correctness']['max_absolute_error']),
            'local_max_abs_error': max(finite_number(x['errors'][0]['max_absolute_error']) for x in local),
            'layer_max_abs_errors': [finite_number(x['max_absolute_error']) for x in layers['original_K02']],
            'timing': None})
    return {'status': raw['status'], 'parameter_count': raw['parameter_count'],
            'source_harness_generation': raw['operator_sources']['source_harness_generation'],
            'local_kernel_checks_passed': 48, 'reference_glue_bitwise_equal_cases': 4,
            'full_model_correct_cases': 1, 'performance_measured': False, 'beats_strong_library': False,
            'cases': cases, 'logit_tolerance': raw['protocol']['logit_tolerance'],
            'kernel_tolerance': raw['protocol']['kernel_tolerance'],
            'summary': '实际两GPU定位：参考接线四形状逐位相等；原K02局部48项全部通过，完整GPT-2输出仅1/4形状通过。微小归一化差异逐层累积，后三形状仍不合格；本诊断未测性能。',
            'report': 'application/' + Path(path).name, 'report_sha256': expected_hash}


def measurement_series_result(path, expected_hash):
    raw = json.loads(bound_file({'path': str(path), 'sha256': expected_hash}).read_bytes())
    readers = {'eca2beddab178940454bbc70dbc81256f76b624482b03914f52d92f91116f9a1': 13,
               '94aea2947f3e00f61da4771004f01b3475fd2d6d86bd9ec1c49d8367c8859e91': 14,
               '49e6d69a05959fa4a794ce48d914d3038be7f5c176d2aa14c63faeafc8f82de7': 15,
               'a36381226052accd2cac1dbd592df2e25b67fa13717f46d0435767b3d52de481': 16,
               '17d6bd95e718a7832b6c12055de60aae40cc5f88e7f4670a715711c3730a7f46': 17,
               'b65b5e7ceaffb87d9745cc4f22e479288efff4f699930fdf076998d42d83d1c0': 18}
    revision = readers.get(raw.get('original_reader_sha256'))
    if (raw.get('kind') != 'source-bound-original-reader-campaign-summary'
            or raw['arm_count'] != 8 or len(raw['arms']) != 8 or raw['task_position_denominator'] != 160
            or raw['rsi_established'] is not False or raw['promotion_eligible'] is not False
            or revision is None):
        raise ValueError('original campaign summary identity differs')
    if revision in (14, 15, 16, 17, 18):
        comparison = raw.get('comparison', {})
        required = {'campaign_revision': revision, 'parent_generation': 'G000',
                    'candidate_generation': {14: 'G004', 15: 'G005', 16: 'G006', 17: 'G007', 18: 'G007'}[revision],
            'parent_source_commit': '54b05f2a959b5f6328ac2bf572fbd981bf567094',
            'candidate_source_commit': {14: '271487a50514cbd4208309e884c6c9ef3e3c31af',
                                        15: '8195f2debdff45e9702d4187b2eac698482c8585',
                                        16: '79329a1e36b9e2c9785bae1e29d5b788d7149e26',
                                        17: 'c2de21af179b3fb24f1a67147ba871322f94493a',
                                        18: 'c2de21af179b3fb24f1a67147ba871322f94493a'}[revision],
            'gpu_indices': [0, 1, 3, 4], 'task_concurrency': 20, 'evaluator_concurrency': 4}
        if any(comparison.get(key) != value for key, value in required.items()):
            raise ValueError('campaign comparison source or resource identity differs')
    seen = set()
    for row in raw['arms']:
        if row['batch_id'] in seen or row['task_denominator'] != 20:
            raise ValueError('campaign arm denominator differs')
        seen.add(row['batch_id'])
        if row['execution_status'] != 'CLOSED':
            continue
        if revision in (14, 15, 16, 17, 18) and row['source_commit'] != comparison[row['arm'] + '_source_commit']:
            raise ValueError('closed arm source differs from its comparison role')
        tasks = row['task_contributions']
        if len(tasks) != 20 or sorted(x['task_id'] for x in tasks) != [f'K{i:02}' for i in range(1, 21)]:
            raise ValueError('campaign task denominator differs')
        for field in ('correct', 'verified_gain', 'environment_error'):
            if any(type(x[field]) is not bool for x in tasks) or sum(x[field] for x in tasks) != row[field + '_count']:
                raise ValueError('campaign count differs from original tasks')
        values = [finite_number(x['deployment_speedup']) for x in tasks]
        if any(v <= 0 for v in values) or not math.isclose(math.exp(sum(math.log(v) for v in values) / 20), row['deployment_geomean'], rel_tol=1e-12):
            raise ValueError('campaign geomean differs from all twenty tasks')
        if row['strict_eligible'] and (row['environment_error_count'] or row['terminal_count'] != 20 or row['evaluation_complete_count'] != 20):
            raise ValueError('campaign strict qualification conflicts with failures')
    closed = sum(r['execution_status'] == 'CLOSED' for r in raw['arms'])
    if closed != raw['closed_arm_count']:
        raise ValueError('campaign closed count differs')
    metrics = [raw.get(key) for key in ('development_geomean_ratio', 'development_paired_ci_low', 'development_paired_ci_high')]
    if any(value is None for value in metrics):
        if closed == 8 or any(value is not None for value in metrics):
            raise ValueError('complete series cannot omit original aggregate metrics')
    elif (any(finite_number(value) < 0 for value in metrics) or metrics[1] > metrics[2]):
        raise ValueError('campaign aggregate metrics differ')
    return {**raw, 'report': 'series/' + Path(path).name, 'report_sha256': expected_hash}


def build(repo, trace=None, trace_sha256=None, application=None, application_sha256=None,
          closed_batch=None, closed_batch_sha256=None,
          additional_operators=None, additional_operators_sha256=None,
          cached_application=None, cached_application_sha256=None,
          measurement_series=None, measurement_series_sha256=None,
          packed_application=None, packed_application_sha256=None,
          independent_cached=None, independent_cached_sha256=None,
          g004_qualification=None, g004_qualification_sha256=None,
          normalization_operators=None, normalization_operators_sha256=None,
          pretrained_application=None, pretrained_application_sha256=None,
          g005_qualification=None, g005_qualification_sha256=None,
          gemm_variant=None, gemm_variant_sha256=None,
          g006_qualification=None, g006_qualification_sha256=None,
          packed_variant_application=None, packed_variant_application_sha256=None,
          pretrained_precision_application=None, pretrained_precision_application_sha256=None,
          g007_qualification=None, g007_qualification_sha256=None,
          pretrained_numerical_diagnosis=None, pretrained_numerical_diagnosis_sha256=None):
    root = Path(repo) / 'harness-evolution/codex-rsi'
    manifest_path = root / 'operators/g001/manifest.json'
    manifest = load(manifest_path)
    gpu_path = root / 'reports/all-eight-gpus-actually-used.json'
    gpu = load(gpu_path)
    batch_path = root / 'reports/g003-first-eight-gpu-real-run-v9.json'
    batch = load(batch_path)
    engineering_path = root / 'reports/g003-source-runtime-cpu-verification-v9.json'
    engineering = load(engineering_path)
    budget = json.loads(bound_file(batch['budget_after']).read_text())
    replay = []
    if trace is not None:
        path = bound_file({'path': str(trace), 'sha256': trace_sha256})
        replay = public_trace(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    data = {'schema_version': 1, 'mode': 'recorded',
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model': 'mimo-v2.6-pro', 'rsi_performance_established': False,
        'operators': operators_from_manifest(manifest_path.parent, manifest),
        'all_eight_gpu_runs': {
            'closed_batches': integer(gpu['actual_closed_batches']),
            'task_positions': integer(gpu['actual_task_positions']),
            'job_claim_count': integer(gpu['actual_job_claim_count']),
            'claims_per_gpu': {str(i): integer(gpu['actual_job_claims_per_gpu_index'][str(i)])
                              for i in range(8)}},
        'latest_batch': {'id': 'g003-b020', 'status': 'CLOSED',
            'raw_terminal_counts': {'GLOBAL_BUDGET_EXHAUSTED': integer(
                batch['original_raw_terminal_counts']['GLOBAL_BUDGET_EXHAUSTED'])},
            'verified_gain_count': integer(batch['official_counts']['verified_gain']),
            'full20_geomean': finite_number(batch['official_all20_geomean']),
            'eligible': batch['all20_eligible'] is True,
            'historical_stop_cause': 'UNRESOLVED'},
        'accounting': {'unknown_requests': integer(budget['unknown_request_count']),
            'gpu_seconds': finite_number(budget['gpu_seconds']),
            'usd_upper_bound': finite_number(budget['usd_upper_bound']),
            'confirmed_bill': False},
        'source_changes': [
            {'generation': 'G001', 'commit': manifest['agh_source_commit'],
             'title': '编辑匹配失败反馈', 'detail': '明确整批编辑未应用，重试前读取文件。'},
            {'generation': 'G002', 'commit': engineering['parent_source_commit'],
             'title': '审批拒绝反馈', 'detail': '明确调用未执行，重试仍需授权。'},
            {'generation': 'G003', 'commit': engineering['source_commit'],
             'title': '上下文压缩分段',
             'detail': '保留runtime_context类型，避免误开用户回合。',
             'cpu_summary_calls_before': integer(engineering['source_mechanism'][
                 'default_first_turn_summary_requests_before']),
             'cpu_summary_calls_after': integer(engineering['source_mechanism'][
                 'default_first_turn_summary_requests_after']),
             'paid_improvement_measured': False}],
        'application': {'status': 'NOT_MEASURED'}, 'trace_replay': replay,
        'evidence': [{'file': p.name, 'sha256': sha(p)}
                     for p in (manifest_path, gpu_path, batch_path, engineering_path)],
        'limitations': ['算子收益来自此前独立GPU复验，相对项目冻结基线。',
            '八卡执行记录不能代替干净的AGH性能对照。',
            'G003总结请求2→1是CPU生产路径复现，未测得付费搜索收益。',
            '费用为保守上界，未知请求预留未清除。']}
    if application is not None:
        data['application'] = application_result(repo, application, application_sha256)
        data['evidence'].append({'file': Path(application).name, 'sha256': application_sha256})
    if additional_operators is not None:
        path = bound_file({'path': str(additional_operators), 'sha256': additional_operators_sha256})
        extra = json.loads(path.read_bytes())
        data['operators'].extend(additional_operator_result(path.parent, extra))
        data['additional_operator_batch'] = {
            'id': extra['batch_id'], 'all20_eligible': False,
            'environment_error_count': extra['primary_environment_error_count'],
            'verified_gain_count': extra['verified_gain_count'],
            'full20_geomean': finite_number(extra['all20_deployment_geomean']),
            'harness_performance_causality_proven': False,
            'manifest_sha256': additional_operators_sha256}
        data['evidence'].append({'file': 'g003-v13-dev1-manifest.json', 'sha256': additional_operators_sha256})
    if closed_batch is not None:
        data['historical_batch'] = data['latest_batch']
        data['latest_batch'], data['accounting'] = closed_failure(closed_batch, closed_batch_sha256)
        data['evidence'].append({'file': Path(closed_batch).name, 'sha256': closed_batch_sha256})
        data['campaign'] = {'status': 'REPAIR_IN_PROGRESS', 'promotion_proven': False,
            'summary': '首臂20任务因GPU观测过期停止，严格闭批；0正式增益。修复验证后重新开始三DEV配对与独立留出。'}
    if cached_application is not None:
        data['cached_application'] = cached_application_result(repo, cached_application, cached_application_sha256)
        data['evidence'].append({'file': Path(cached_application).name, 'sha256': cached_application_sha256})
    if packed_application is not None:
        data['packed_application'] = packed_application_result(repo, packed_application, packed_application_sha256)
        data['evidence'].append({'file': Path(packed_application).name, 'sha256': packed_application_sha256})
    if pretrained_application is not None:
        data['pretrained_application'] = pretrained_application_result(repo, pretrained_application, pretrained_application_sha256)
        data['evidence'].append({'file': Path(pretrained_application).name, 'sha256': pretrained_application_sha256})
    if packed_variant_application is not None:
        data['packed_variant_application'] = packed_variant_application_result(repo, packed_variant_application,
                                                                               packed_variant_application_sha256)
        data['evidence'].append({'file': Path(packed_variant_application).name, 'sha256': packed_variant_application_sha256})
    if pretrained_precision_application is not None:
        data['pretrained_precision_application'] = pretrained_precision_application_result(
            repo, pretrained_precision_application, pretrained_precision_application_sha256)
        data['evidence'].append({'file': Path(pretrained_precision_application).name,
                                 'sha256': pretrained_precision_application_sha256})
    if independent_cached is not None:
        if cached_application is None:
            raise ValueError('independent cached reproduction requires the original report')
        data['independent_cached'] = independent_cached_result(repo, independent_cached, independent_cached_sha256,
                                                              cached_application, cached_application_sha256)
        data['evidence'].append({'file': Path(independent_cached).name, 'sha256': independent_cached_sha256})
    if g004_qualification is not None:
        data['source_changes'].append(g004_source_result(g004_qualification, g004_qualification_sha256))
        data['evidence'].append({'file': Path(g004_qualification).name, 'sha256': g004_qualification_sha256})
    if g005_qualification is not None:
        data['source_changes'].append(g005_source_result(g005_qualification, g005_qualification_sha256))
        data['evidence'].append({'file': Path(g005_qualification).name, 'sha256': g005_qualification_sha256})
    if g006_qualification is not None:
        data['source_changes'].append(g006_source_result(g006_qualification, g006_qualification_sha256))
        data['evidence'].append({'file': Path(g006_qualification).name, 'sha256': g006_qualification_sha256})
    if g007_qualification is not None:
        data['source_changes'].append(g007_source_result(g007_qualification, g007_qualification_sha256))
        data['evidence'].append({'file': Path(g007_qualification).name, 'sha256': g007_qualification_sha256})
    if pretrained_numerical_diagnosis is not None:
        data['pretrained_numerical_diagnosis'] = pretrained_numerical_diagnosis_result(
            pretrained_numerical_diagnosis, pretrained_numerical_diagnosis_sha256)
        data['evidence'].append({'file': Path(pretrained_numerical_diagnosis).name,
                                 'sha256': pretrained_numerical_diagnosis_sha256})
    if normalization_operators is not None:
        path = bound_file({'path': str(normalization_operators), 'sha256': normalization_operators_sha256})
        data['operators'].extend(normalization_operator_result(path.parent, json.loads(path.read_bytes())))
        data['evidence'].append({'file': 'g000-v14-dev1/' + path.name, 'sha256': normalization_operators_sha256})
    if gemm_variant is not None:
        if not any(r['task_id'] == 'K12' for r in data['operators']):
            raise ValueError('additional variant requires the original K12 delivery')
        path = bound_file({'path': str(gemm_variant), 'sha256': gemm_variant_sha256})
        data['operator_variants'] = [gemm_variant_result(path.parent, json.loads(path.read_bytes()))]
        data['evidence'].append({'file': 'operator-variants/g000-v15-dev2/manifest.json',
                                 'sha256': gemm_variant_sha256})
        data['limitations'].append('新增K12版本来自原H0；旧应用保持原K12，新版本应用需读取其独立实测，不能直接迁移算子加速比。')
    if measurement_series is not None:
        series = measurement_series_result(measurement_series, measurement_series_sha256)
        data['measurement_series'] = series
        data['historical_batch'] = data['latest_batch']
        last = [row for row in series['arms'] if row['execution_status'] == 'CLOSED'][-1]
        data['latest_batch'] = {'id': last['batch_id'], 'status': 'CLOSED',
            'verified_gain_count': last['verified_gain_count'], 'full20_geomean': last['deployment_geomean'],
            'eligible': last['strict_eligible'], 'evaluation_complete_count': last['evaluation_complete_count'],
            'correct_count': last['correct_count'], 'environment_error_count': last['environment_error_count'],
            'diagnostic_detail': ('最新严格闭批结果。完整系列评分门未通过；此前停止和未知记录保留。' if series['closed_arm_count'] == 8
                                  else ('旧系列因共同控制器错误停止，后续六臂未启动；新V18另行重测，原失败和未知记录保留。' if series.get('stopped') else '最新严格闭批结果。完整系列仍在测量，不提前宣称性能晋级；此前停止和未知记录保留。'))}
        if 'accounting' in series:
            data['accounting'] = series['accounting']
        data['campaign'] = {'status': series['status'], 'promotion_proven': False, 'summary': series['summary']}
        data['evidence'].append({'file': Path(measurement_series).name, 'sha256': measurement_series_sha256})
        if series.get('comparison', {}).get('campaign_revision') == 14:
            previous = root / 'reports/hackathon-v13-eight-arm-summary.json'
            prior = measurement_series_result(previous, '590e1c63a759c1060dcafc0835e9555ae633babb1bb5a1040db6aab078fb12bb')
            data['series_history'] = [prior]
        if series.get('comparison', {}).get('campaign_revision') in (17,18):
            prior_specs = [('g006-v16-original-series-full-eight.json',
                'dd825b54b216b7f4c066d0eecd7498a5635c55fae42c877df7130e5ce758120c')]
            if series['comparison']['campaign_revision']==18:
                prior_specs.append(('g007-v17-stopped-two-closed-six-not-started.json',
                    'd940ac4fcfc06961353e55cd8935052ff1ff55878a1b6bc7056bf38479a7ba97'))
            data['series_history'] = [measurement_series_result(root/'reports'/name, digest)
                                      for name,digest in prior_specs]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--trace-sha256')
    parser.add_argument('--application', type=Path)
    parser.add_argument('--application-sha256')
    parser.add_argument('--closed-batch', type=Path)
    parser.add_argument('--closed-batch-sha256')
    parser.add_argument('--additional-operators', type=Path)
    parser.add_argument('--additional-operators-sha256')
    parser.add_argument('--cached-application', type=Path)
    parser.add_argument('--cached-application-sha256')
    parser.add_argument('--measurement-series', type=Path)
    parser.add_argument('--measurement-series-sha256')
    parser.add_argument('--packed-application', type=Path)
    parser.add_argument('--packed-application-sha256')
    parser.add_argument('--independent-cached', type=Path)
    parser.add_argument('--independent-cached-sha256')
    parser.add_argument('--g004-qualification', type=Path)
    parser.add_argument('--g004-qualification-sha256')
    parser.add_argument('--normalization-operators', type=Path)
    parser.add_argument('--normalization-operators-sha256')
    parser.add_argument('--pretrained-application', type=Path)
    parser.add_argument('--pretrained-application-sha256')
    parser.add_argument('--g005-qualification', type=Path)
    parser.add_argument('--g005-qualification-sha256')
    parser.add_argument('--g006-qualification', type=Path)
    parser.add_argument('--g006-qualification-sha256')
    parser.add_argument('--gemm-variant', type=Path)
    parser.add_argument('--gemm-variant-sha256')
    parser.add_argument('--packed-variant-application', type=Path)
    parser.add_argument('--packed-variant-application-sha256')
    parser.add_argument('--pretrained-precision-application', type=Path)
    parser.add_argument('--pretrained-precision-application-sha256')
    parser.add_argument('--g007-qualification', type=Path)
    parser.add_argument('--g007-qualification-sha256')
    parser.add_argument('--pretrained-numerical-diagnosis', type=Path)
    parser.add_argument('--pretrained-numerical-diagnosis-sha256')
    args = parser.parse_args()
    if bool(args.trace) != bool(args.trace_sha256):
        parser.error('trace path and expected SHA256 are required together')
    if bool(args.application) != bool(args.application_sha256):
        parser.error('application path and expected SHA256 are required together')
    if bool(args.closed_batch) != bool(args.closed_batch_sha256):
        parser.error('closed batch path and expected SHA256 are required together')
    if bool(args.additional_operators) != bool(args.additional_operators_sha256):
        parser.error('additional operators and expected SHA256 are required together')
    if bool(args.cached_application) != bool(args.cached_application_sha256):
        parser.error('cached application and expected SHA256 are required together')
    if bool(args.measurement_series) != bool(args.measurement_series_sha256):
        parser.error('measurement series and expected SHA256 are required together')
    for key in ('packed_application', 'independent_cached', 'g004_qualification', 'normalization_operators', 'pretrained_application', 'g005_qualification', 'gemm_variant', 'g006_qualification', 'packed_variant_application', 'pretrained_precision_application', 'g007_qualification', 'pretrained_numerical_diagnosis'):
        if bool(getattr(args, key)) != bool(getattr(args, key + '_sha256')):
            parser.error(key + ' and expected SHA256 are required together')
    data = build(args.repo, args.trace, args.trace_sha256, args.application, args.application_sha256,
                 args.closed_batch, args.closed_batch_sha256,
                 args.additional_operators, args.additional_operators_sha256,
                 args.cached_application, args.cached_application_sha256,
                 args.measurement_series, args.measurement_series_sha256,
                 args.packed_application, args.packed_application_sha256,
                 args.independent_cached, args.independent_cached_sha256,
                 args.g004_qualification, args.g004_qualification_sha256,
                 args.normalization_operators, args.normalization_operators_sha256,
                 args.pretrained_application, args.pretrained_application_sha256,
                 args.g005_qualification, args.g005_qualification_sha256,
                 args.gemm_variant, args.gemm_variant_sha256,
                 args.g006_qualification, args.g006_qualification_sha256,
                 args.packed_variant_application, args.packed_variant_application_sha256,
                 args.pretrained_precision_application, args.pretrained_precision_application_sha256,
                 args.g007_qualification, args.g007_qualification_sha256,
                 args.pretrained_numerical_diagnosis, args.pretrained_numerical_diagnosis_sha256)
    args.out.mkdir(parents=True, exist_ok=True)
    for asset in (Path(__file__).parent / 'demo').iterdir():
        if asset.is_file():
            shutil.copyfile(asset, args.out / asset.name)
    (args.out / 'data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                                allow_nan=False) + '\n')
    (args.out / 'evidence-data.js').write_text('window.EVIDENCE = ' + json.dumps(
        data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c') + ';\n')
    if args.application:
        target = args.out / data['application']['report']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.application, target)
    if args.cached_application:
        target = args.out / data['cached_application']['report']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.cached_application, target)
    if args.measurement_series:
        target = args.out / data['measurement_series']['report']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.measurement_series, target)
    for prior in data.get('series_history', []):
        target = args.out / prior['report']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.repo / 'harness-evolution/codex-rsi/reports' / Path(prior['report']).name, target)
    for key in ('packed_application', 'independent_cached', 'pretrained_application', 'packed_variant_application', 'pretrained_precision_application'):
        if getattr(args, key):
            target = args.out / data[key]['report']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(getattr(args, key), target)
    if args.packed_variant_application:
        combined = json.loads(args.packed_variant_application.read_bytes())
        for replica in combined['replicas']:
            target = args.out / 'application' / Path(replica['report']['path']).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.repo / replica['report']['path'], target)
    for key, generation in [('g004_qualification', 'G004'), ('g005_qualification', 'G005'), ('g006_qualification', 'G006'), ('g007_qualification', 'G007')]:
        if getattr(args, key):
            change = next(row for row in data['source_changes'] if row['generation'] == generation)
            target = args.out / change['qualification_report']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(getattr(args, key), target)
    if args.pretrained_numerical_diagnosis:
        target = args.out / data['pretrained_numerical_diagnosis']['report']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.pretrained_numerical_diagnosis, target)
    for operator in data['operators']:
        target = args.out / operator['source']
        target.parent.mkdir(parents=True, exist_ok=True)
        generation = operator.get('source_generation', 'g001')
        original = args.repo / 'harness-evolution/codex-rsi/operators' / generation / operator['task_id'] / 'kernel.py'
        shutil.copyfile(original, target)
    if args.gemm_variant:
        original = json.loads(args.gemm_variant.read_bytes())
        target = args.out / 'operator-variants/g000-v15-dev2'
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.gemm_variant, target / 'manifest.json')
        for row in original['operators']:
            destination = target / row['source']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.gemm_variant.parent / row['source'], destination)
    print(json.dumps({'out': str(args.out), 'mode': data['mode'],
                      'verified_operator_count': len(data['operators'])}))


if __name__ == '__main__':
    main()
