"""Check the new GEMM's source, four-case samples and official application span."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN_VALIDATOR_SHA = 'dc2dbb1279e3a082da28731b2803a3bbba386991e016d52b57f720c4ffd49f15'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_validator():
    path = HERE / 'export_packed_application.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_VALIDATOR_SHA:
        raise ValueError('original packed sample/accounting validator changed')
    return load_module('private_original_packed_validator', path)


def expected_plan():
    profile = load_module('private_packed_gemm_profile', HERE / 'application/benchmark_packed_gemm_variant.py')
    base = profile.adapt_private_base(profile.load_frozen_base())
    return base, base.make_plan(base.verify_sources(), base.make_protocol())


def validate_correctness(diagnostic, protocol):
    tasks = {**{name: 'K12' for name in ('q', 'k', 'v', 'o', 'gate', 'up', 'down')},
             'rope_q': 'K10', 'rope_k': 'K10', 'softmax': 'K04', 'silu_mul': 'K06'}
    if (not isinstance(diagnostic, dict)
            or diagnostic.get('full_block_tolerance') != protocol['full_block_output_tolerance']
            or set(diagnostic.get('operators', {})) != set(tasks)
            or set(diagnostic.get('full_decode_max_abs_errors', {})) != {'output', 'cache_k', 'cache_v'}):
        raise ValueError('original output/cache and operator correctness records required')
    for name, task in tasks.items():
        row = diagnostic['operators'][name]
        if row.get('task_id') != task or [row.get('rtol'), row.get('atol')] != protocol['operator_tolerances'][task]:
            raise ValueError('original local operator numerical tolerance changed')
    errors = [row.get('max_abs_error') for row in diagnostic['operators'].values()]
    errors += list(diagnostic['full_decode_max_abs_errors'].values())
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in errors):
        raise ValueError('finite complete output/cache error records required')


def validate_partition(raw):
    old = load_validator()
    base, expected = expected_plan()
    if raw.get('status') != 'MEASURED_CLEAN' or raw.get('kind') != expected['kind']:
        raise ValueError('complete new GEMM packed application required')
    protocol = raw['protocol']
    if protocol != expected['protocol'] or raw['source_package'] != expected['source_package']:
        raise ValueError('source-bound new GEMM profile or original protocol changed')
    for key in ('source_package', 'protocol'):
        if old.canonical_hash(raw[key]) != raw[key + '_sha256']:
            raise ValueError('source/protocol canonical binding changed')
    if set(raw['measurements']) != set(protocol['case_labels']):
        raise ValueError('four-case denominator changed')
    cases, count = {}, 0
    for label, config in zip(protocol['case_labels'], protocol['cases']):
        row = raw['measurements'][label]
        if row['status'] != 'MEASURED_CLEAN' or row['config'] != config:
            raise ValueError('measured case incomplete or relabeled')
        case = base.make_case(base.CachedConfig(**config), protocol['seed'])
        if row['input_weights_cache_sha256'] != case.fingerprint():
            raise ValueError('fixed input/weights/KV prefix changed')
        samples, statuses = row['samples'], row['backend_qualification']
        old.validate_samples(samples, 30)
        if (set(statuses) != set(old.BACKENDS)
                or any(statuses[key]['status'] not in ('AVAILABLE', 'INCORRECT', 'UNAVAILABLE')
                       or ((statuses[key]['status'] == 'AVAILABLE') != (key in samples)) for key in statuses)
                or 'fp32_reference' not in samples):
            raise ValueError('backend qualification and paired samples disagree')
        for compiled, source in (('compiled_fp32', 'fp32_reference'), ('compiled_fp16', 'fp16_library'),
                                 ('compiled_packed_fp16', 'packed_fp16_library'), ('compiled_candidate', 'candidate')):
            if compiled in samples and source not in samples:
                raise ValueError('compiled backend lacks its qualified source')
        for name in ('fp32_reference', 'fp16_library', 'packed_fp16_library', 'candidate'):
            field = 'reference_correctness' if name == 'fp32_reference' else name + '_correctness'
            if name in samples:
                validate_correctness(row.get(field), protocol)
        complete = all(key in samples for key in old.LIBRARY)
        if row['strong_baseline_complete'] is not complete:
            raise ValueError('strong library completeness claim changed')
        for candidate in ('candidate', 'compiled_candidate'):
            gains = row['candidate_gains'][candidate]
            if candidate not in samples:
                if gains is not None:
                    raise ValueError('unqualified candidate has a gain claim')
                continue
            recomputed = {
                'against_fp32_reference': old.pair_metric(samples, 'fp32_reference', candidate),
                'against_fp16_library': old.pair_metric(samples, 'fp16_library', candidate) if 'fp16_library' in samples else None,
                'against_strong_library': old.strong_metric(samples, candidate)}
            if gains != recomputed:
                raise ValueError('stored metric differs from original independent sample math')
        safe = {key: copy.deepcopy(row[key]) for key in ('status', 'config', 'input_weights_cache_sha256',
                 'samples', 'candidate_gains', 'strong_baseline_complete')}
        safe['backend_qualification'] = {name: {k: v for k, v in data.items() if k in
            ('status', 'error_type', 'mode', 'fullgraph', 'dynamic', 'compile_warmup_seconds')}
            for name, data in statuses.items()}
        for name in ('reference_correctness', 'candidate_correctness', 'fp16_library_correctness', 'packed_fp16_library_correctness'):
            if name in row:
                validate_correctness(row[name], protocol)
                safe[name] = copy.deepcopy(row[name])
        cases[label] = safe
        count += len(samples) * 30
    complete = all(row['strong_baseline_complete'] for row in cases.values())
    if raw['all_strong_baselines_complete'] is not complete:
        raise ValueError('whole application library completeness changed')
    victory = {}
    for candidate in ('candidate', 'compiled_candidate'):
        metrics = [(row['candidate_gains'][candidate] or {}).get('against_strong_library') for row in cases.values()]
        victory[candidate] = complete and all(m is not None and m['victory_proven'] for m in metrics)
    return {'case_count': len(cases), 'measurement_cases': cases,
            'raw_gpu_latency_samples': count, 'raw_host_latency_samples': count,
            'all_strong_baselines_complete': complete, 'beats_all_strong_library_cases': victory}


def make_report(raw_file, digest, queue_file, repo=ROOT):
    old = load_validator()
    if Path(repo).resolve() != ROOT or old.sha(raw_file) != digest:
        raise ValueError('repository or original raw report binding changed')
    raw = json.loads(Path(raw_file).read_bytes())
    result = validate_partition(raw)
    snapshots = raw['gpu_snapshots']
    if not snapshots or not all(row.get('available') is True and row.get('foreign_processes') == 0 for row in snapshots):
        raise ValueError('actual GPU snapshots not all clean')
    env = raw['environment']
    if env['tf32'] is not False or env['fp16_reduced_precision_reduction'] is not False:
        raise ValueError('original library precision protocol changed')
    account = old.accounting_receipt(queue_file, raw, digest)
    return {'schema_version': 1, 'kind': 'source-bound-packed-gemm-variant-gpu-measurement',
            'status': 'MEASURED_CLEAN', **result,
            'source_package': raw['source_package'], 'source_package_sha256': raw['source_package_sha256'],
            'protocol': raw['protocol'], 'protocol_sha256': raw['protocol_sha256'],
            'raw_result_sha256': digest, 'accounting': account,
            'device': {k: env[k] for k in ('torch', 'cuda', 'python', 'device_name', 'device_total_memory',
                                         'tf32', 'fp16_reduced_precision_reduction', 'physical_gpu')},
            'clean_measurement': {'snapshot_count': len(snapshots), 'all_snapshots_available_and_foreign_processes_zero': True},
            'conclusions': {'all_four_cases_measured': True, 'all_strong_baselines_complete': result['all_strong_baselines_complete'],
                            'beats_all_strong_library_cases': result['beats_all_strong_library_cases']},
            'pretrained_model': False, 'text_quality_measured': False, 'harness_performance_causality_proven': False,
            'original_numeric_validator_sha256': FROZEN_VALIDATOR_SHA,
            'independent_consistency_scope': 'Read-only original sources/samples and actual official Queue; not fresh GPU reproduction.',
            'limits': raw['limits']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--raw-sha256', required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = make_report(args.raw, args.raw_sha256, args.queue, args.repo)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'sha256': load_validator().sha(args.out), 'sample_count': report['raw_gpu_latency_samples'],
                      'conclusions': report['conclusions']}))


if __name__ == '__main__':
    main()
