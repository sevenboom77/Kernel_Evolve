"""Recompute pretrained GPT-2 samples and bind their original official jobs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'application'))
from export_packed_application import accounting_receipt, canonical_hash, pair_metric, sha

LIBRARIES = ('fp16_library', 'compiled_fp16')
BACKENDS = (*LIBRARIES, 'candidate', 'compiled_candidate')


def finite(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('finite nonnegative measurement required')
    return value


def strong_metric(samples, candidate):
    if any(key not in samples for key in (*LIBRARIES, candidate)):
        return None
    minima = [min(a, b) for a, b in zip(*(samples[key]['gpu_us_samples'] for key in LIBRARIES))]
    ratios = [a / b for a, b in zip(minima, samples[candidate]['gpu_us_samples'])]
    rng = random.Random(20261006)
    boot = sorted(statistics.median(rng.choices(ratios, k=30)) for _ in range(2000))
    return {'required_library_backends': list(LIBRARIES), 'paired_rounds': 30,
        'strong_baseline_us_samples': minima, 'paired_ratio_samples': ratios,
        'median': statistics.median(ratios), 'bootstrap_95_percent_interval': [boot[49], boot[1949]],
        'bootstrap_seed': 20261006, 'bootstrap_resamples': 2000, 'victory_proven': boot[49] > 1.}


def validate_partition(raw):
    import benchmark_pretrained_gpt2 as benchmark
    protocol = benchmark.make_protocol()
    if (raw.get('status') != 'MEASURED_CLEAN' or raw.get('kind') != 'source-bound-pretrained-gpt2-application'
            or raw['protocol'] != protocol or raw['pretrained_model'] is not True
            or raw['text_quality_measured'] is not False or raw['harness_performance_causality_proven'] is not False
            or raw['parameter_count'] != 124439808 or raw['all_pretrained_weights_preserved'] is not True
            or raw['actual_K02_Triton_executed'] is not True):
        raise ValueError('original full pretrained inference protocol or completion differs')
    partition = raw['execution_partition']
    selected = protocol['case_labels'] if partition == 'all' else (
        protocol['case_labels'][int(partition) * 2:int(partition) * 2 + 2] if partition in ('0', '1') else [])
    if not selected or raw['selected_case_labels'] != selected or set(raw['measurements']) != set(selected):
        raise ValueError('predeclared partition or case denominator differs')
    cases, count = {}, 0
    for label in selected:
        index = protocol['case_labels'].index(label)
        config = protocol['cases'][index]
        row = raw['measurements'][label]
        if (row['status'] != 'MEASURED_CLEAN' or row['config'] != config
                or row['input_token_sha256'] != raw['source_package']['input_token_sha256'][label]
                or row['parameter_sha256'] != raw['parameter_sha256']):
            raise ValueError('actual token, parameter or case identity changed')
        checks = row['original_k02_checks']
        if len(checks) != 12 or [r['block_index'] for r in checks] != list(range(12)):
            raise ValueError('all twelve original K02 calls must be checked')
        for check in checks:
            if (check['shape'] != [config['batch'], config['sequence'], 768]
                    or check['epsilon'] != 1e-5 or check['kernel_tolerance'] != protocol['kernel_tolerance']
                    or len(check['max_absolute_errors']) != 2):
                raise ValueError('original normalization domain or tolerance differs')
            for error in check['max_absolute_errors']: finite(error)
        samples, qualified = row['samples'], row['backend_qualification']
        if not samples or not set(samples).issubset(BACKENDS) or set(qualified) != set(BACKENDS):
            raise ValueError('predeclared backend denominator differs')
        orders = next(iter(samples.values()))['batch_label_order']
        for name, receipt in qualified.items():
            if receipt['status'] not in ('AVAILABLE', 'UNAVAILABLE', 'INCORRECT') or (receipt['status'] == 'AVAILABLE') != (name in samples):
                raise ValueError('backend qualification does not match timed samples')
            if name not in samples: continue
            diagnostic = receipt['correctness']
            if (diagnostic['shape'] != [config['batch'], config['sequence'], 50257]
                    or diagnostic['tolerance'] != protocol['logit_tolerance']):
                raise ValueError('full pretrained logits correctness differs')
            for key in ('max_absolute_error', 'mean_squared_error', 'last_token_argmax_agreement', 'all_token_argmax_agreement'):
                finite(diagnostic[key])
            if any(diagnostic[k] > 1 for k in ('last_token_argmax_agreement', 'all_token_argmax_agreement')):
                raise ValueError('logit argmax agreement outside [0,1]')
            sample = samples[name]
            for field in ('gpu_us_samples', 'host_us_samples'):
                values = sample[field]
                if len(values) != 30: raise ValueError('thirty complete paired rounds required')
                for value in values: finite(value, positive=True)
                if statistics.median(values) != sample[field.replace('samples', 'median')]:
                    raise ValueError('stored median differs from raw samples')
            if (len(orders) != 30 or sample['batch_label_order'] != orders
                    or any(len(order) != len(samples) or set(order) != set(samples) for order in orders)):
                raise ValueError('backend samples are not paired')
        complete = all(key in samples for key in LIBRARIES)
        if row['strong_baseline_complete'] is not complete:
            raise ValueError('strong-baseline completeness differs')
        for candidate in ('candidate', 'compiled_candidate'):
            expected = None if candidate not in samples else {
                'against_fp16_library': pair_metric(samples, 'fp16_library', candidate) if 'fp16_library' in samples else None,
                'against_strong_library': strong_metric(samples, candidate)}
            if row['candidate_gains'][candidate] != expected:
                raise ValueError('stored gain differs from independent paired recomputation')
        safe = {key: copy.deepcopy(row[key]) for key in ('status', 'config', 'input_token_sha256',
            'parameter_sha256', 'samples', 'candidate_gains', 'strong_baseline_complete')}
        safe['original_k02_checks'] = [{key: r[key] for key in ('block_index', 'shape', 'epsilon',
                                      'kernel_tolerance', 'max_absolute_errors')} for r in checks]
        safe['backend_qualification'] = {name: {key: value for key, value in receipt.items()
            if key in ('status', 'error_type', 'compiled', 'fullgraph', 'mode', 'correctness', 'qualification_seconds')}
            for name, receipt in qualified.items()}
        cases[label] = safe
        count += len(samples) * 30
    complete = all(row['strong_baseline_complete'] for row in cases.values())
    if raw['all_strong_baselines_complete'] is not complete:
        raise ValueError('whole partition strong completeness differs')
    victory = {name: all((row['candidate_gains'][name] or {}).get('against_strong_library', {}).get('victory_proven') is True
                        for row in cases.values()) if complete else False
               for name in ('candidate', 'compiled_candidate')}
    return {'measurement_cases': cases, 'case_count': len(cases), 'raw_gpu_latency_samples': count,
            'raw_host_latency_samples': count, 'all_strong_baselines_complete': complete,
            'beats_all_strong_library_cases': victory}


def make_report(entries, repo):
    import benchmark_pretrained_gpt2 as benchmark
    from transformers.models.gpt2 import modeling_gpt2
    repo = Path(repo).resolve()
    raw_rows, validated, partitions = [], [], []
    for raw_file, digest, queue in entries:
        if sha(raw_file) != digest: raise ValueError('raw report digest changed')
        raw = json.loads(Path(raw_file).read_bytes())
        package, protocol = raw['source_package'], raw['protocol']
        if (canonical_hash(package) != raw['source_package_sha256'] or canonical_hash(protocol) != raw['protocol_sha256']
                or package['protocol_sha256'] != raw['protocol_sha256']):
            raise ValueError('original source/protocol hash binding changed')
        for name, expected in package['application_sources'].items():
            path = (repo / name).resolve()
            if not path.is_relative_to(repo) or sha(path) != expected:
                raise ValueError('original measured application source changed')
        manifest = repo / 'harness-evolution/codex-rsi/operators/g000-v14-dev1/manifest.json'
        binding = benchmark.model_app.verify_kernel(manifest, package['operator_manifest_sha256'])
        if (binding['source']['sha256'] != package['operator_kernel_sha256']
                or binding['candidate_id'] != package['candidate_id'] or binding['verify_job_id'] != package['verify_job_id']
                or package['model'] != {k: raw['model_receipt'][k] for k in ('model', 'revision', 'files')}
                or any(package['model']['files'][name]['sha256'] != expected
                       for name, expected in benchmark.model_app.MODEL_FILE_SHA256S.items())
                or package['transformers'] != importlib.metadata.version('transformers')
                or package['transformers_gpt2_source_sha256'] != sha(modeling_gpt2.__file__)):
            raise ValueError('original pretrained implementation/weight/operator identity differs')
        snapshots = raw['gpu_snapshots']
        env = raw['environment']
        if (not snapshots or not all(x.get('available') is True and x.get('foreign_processes') == 0 for x in snapshots)
                or env['tf32'] is not False or env['fp16_reduced_precision_reduction'] is not False
                or env['compiler'] != protocol['compile']):
            raise ValueError('clean device or numerical/compiler precision differs')
        result = validate_partition(raw)
        account = accounting_receipt(queue, raw, digest)
        result['accounting'] = account
        result['device'] = {k: env[k] for k in ('torch', 'cuda', 'python', 'device_name', 'physical_gpu',
                                              'tf32', 'fp16_reduced_precision_reduction')}
        result['raw_result_sha256'] = digest
        result['clean_snapshot_count'] = len(snapshots)
        partitions.append(raw['execution_partition']); raw_rows.append(raw); validated.append(result)
    if not ((len(entries) == 1 and partitions == ['all']) or (len(entries) == 2 and set(partitions) == {'0', '1'})):
        raise ValueError('exact full four-case partition coverage required')
    first = raw_rows[0]
    if any(row['source_package'] != first['source_package'] or row['parameter_sha256'] != first['parameter_sha256'] for row in raw_rows):
        raise ValueError('partition sources/inputs/weights differ')
    cases = {}
    for result in validated:
        if set(cases) & set(result['measurement_cases']): raise ValueError('duplicate case partition')
        cases.update(result['measurement_cases'])
    if set(cases) != set(first['protocol']['case_labels']): raise ValueError('four fixed pretrained cases required')
    complete = all(v['all_strong_baselines_complete'] for v in validated)
    victory = {name: all(v['beats_all_strong_library_cases'][name] for v in validated) for name in ('candidate', 'compiled_candidate')}
    return {'schema_version': 1, 'kind': 'source-bound-pretrained-gpt2-gpu-measurement', 'status': 'MEASURED_CLEAN',
        'source_package': first['source_package'], 'source_package_sha256': first['source_package_sha256'],
        'protocol': first['protocol'], 'protocol_sha256': first['protocol_sha256'], 'parameter_sha256': first['parameter_sha256'],
        'parameter_count': 124439808, 'measurement_cases': cases,
        'partitions': [{k: v[k] for k in ('accounting', 'device', 'raw_result_sha256', 'clean_snapshot_count')} for v in validated],
        'raw_gpu_latency_samples': sum(v['raw_gpu_latency_samples'] for v in validated),
        'raw_host_latency_samples': sum(v['raw_host_latency_samples'] for v in validated),
        'all_pretrained_weights_preserved': True, 'actual_K02_Triton_executed': True,
        'pretrained_model': True, 'text_quality_measured': False, 'harness_performance_causality_proven': False,
        'conclusions': {'all_four_cases_measured': True, 'all_strong_baselines_complete': complete,
                        'beats_all_strong_library_cases': victory},
        'independent_consistency_scope': 'Read-only original samples, sources and terminal official queues; not a fresh GPU reproduction.',
        'limits': first['limits']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition-result', nargs=3, action='append', required=True, metavar=('RAW', 'SHA256', 'QUEUE'))
    p.add_argument('--repo', type=Path, default=HERE.parents[1])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = make_report(a.partition_result, a.repo)
    with a.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'sha256': sha(a.out), 'GPU_samples': report['raw_gpu_latency_samples'], 'conclusions': report['conclusions']}))


if __name__ == '__main__': main()
