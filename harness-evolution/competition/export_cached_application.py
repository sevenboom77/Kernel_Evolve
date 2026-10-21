"""Independent source/sample/accounting consistency check for cached decoding."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics

LIBRARY = ("fp32_reference", "fp16_library", "compiled_fp32", "compiled_fp16")
BACKENDS = (*LIBRARY, "candidate", "compiled_candidate")
CONFIGS = [{"batch": 1, "context": 64, "heads": 8, "head_dim": 64, "mlp_dim": 1024},
           {"batch": 1, "context": 128, "heads": 8, "head_dim": 64, "mlp_dim": 1024},
           {"batch": 1, "context": 33, "heads": 8, "head_dim": 64, "mlp_dim": 512},
           {"batch": 2, "context": 128, "heads": 8, "head_dim": 128, "mlp_dim": 1024}]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_samples(samples, rounds):
    if not samples or not set(samples).issubset(BACKENDS):
        raise ValueError('unrecognized or missing samples')
    first = next(iter(samples.values()))['batch_label_order']
    for row in samples.values():
        for field in ('gpu_us_samples', 'host_us_samples'):
            values = row[field]
            if len(values) != rounds or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError('full finite paired sample denominator required')
            if statistics.median(values) != row[field.replace('samples', 'median')]:
                raise ValueError('stored sample median inconsistent')
        orders = row['batch_label_order']
        if len(orders) != rounds or orders != first or any(len(order) != len(samples) or set(order) != set(samples) for order in orders):
            raise ValueError('sample ordering is not paired')


def bootstrap(ratios):
    rng = random.Random(20261005)
    values = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return [values[49], values[1949]]


def pair_metric(samples, baseline, candidate):
    ratios = [a / b for a, b in zip(samples[baseline]['gpu_us_samples'], samples[candidate]['gpu_us_samples'])]
    return {'metric': 'median of within-round full-block baseline/candidate latency ratios',
            'paired_ratio_samples': ratios, 'median': statistics.median(ratios),
            'bootstrap_95_percent_interval': bootstrap(ratios), 'bootstrap_seed': 20261005, 'bootstrap_resamples': 2000}


def strong_metric(samples, candidate):
    if any(key not in samples for key in (*LIBRARY, candidate)):
        return None
    vectors = [samples[key]['gpu_us_samples'] for key in LIBRARY]
    count = len(samples[candidate]['gpu_us_samples'])
    if count != 30 or any(len(v) != count for v in vectors):
        raise ValueError('strong comparison requires thirty complete paired rounds')
    minima = [min(v) for v in zip(*vectors)]
    ratios = [a / b for a, b in zip(minima, samples[candidate]['gpu_us_samples'])]
    interval = bootstrap(ratios)
    return {'metric': 'median of within-round strongest-library/candidate latency ratios',
            'required_library_backends': list(LIBRARY), 'paired_rounds': count,
            'strong_baseline_us_samples': minima, 'paired_ratio_samples': ratios,
            'median': statistics.median(ratios), 'bootstrap_95_percent_interval': interval,
            'bootstrap_seed': 20261005, 'bootstrap_resamples': 2000, 'victory_proven': interval[0] > 1.0}


def accounting_receipt(queue_file, raw, digest):
    entry = raw['accounting']
    with sqlite3.connect(Path(queue_file).resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        job = db.execute('SELECT * FROM jobs WHERE run_id=? AND job_id=?', (entry['run_id'], entry['job_id'])).fetchone()
        result = db.execute('SELECT * FROM results WHERE job_id=?', (entry['job_id'],)).fetchone()
        events = db.execute('SELECT kind,ts,payload_json FROM events WHERE run_id=? ORDER BY id', (entry['run_id'],)).fetchall()
    if (job is None or result is None or job['kind'] != 'application' or job['task_id'] != 'APP_DECODER'
            or job['status'] != 'complete' or result['status'] != 'complete'
            or job['source_hash'] != raw['source_package_sha256']
            or json.loads(result['payload_json']).get('report_sha256') != digest):
        raise ValueError('actual terminal official application job binding failed')
    times = {'job_claimed': [], 'job_finished': []}
    for event in events:
        if event['kind'] in times and json.loads(event['payload_json']).get('job_id') == entry['job_id']:
            times[event['kind']].append(event['ts'])
    if any(len(v) != 1 for v in times.values()):
        raise ValueError('one actual complete accounting span required')
    start, finish = times['job_claimed'][0], times['job_finished'][0]
    seconds = (datetime.fromisoformat(finish.replace('Z', '+00:00')) - datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()
    if seconds < 0:
        raise ValueError('accounting span clock rollback')
    return {'queue_job_kind': 'application', 'official_queue_job_status': 'complete', 'queue_report_sha256_matches': True,
            'claim_at': start, 'finish_at': finish, 'conservative_claim_to_finish_seconds': seconds,
            'note': 'Includes compilation/host work; not CUDA elapsed time.'}


def make_report(raw_file, digest, queue_file, repo):
    repo = Path(repo).resolve()
    if sha(raw_file) != digest:
        raise ValueError('raw report digest changed')
    raw = json.loads(Path(raw_file).read_bytes())
    if raw.get('status') != 'MEASURED_CLEAN' or raw.get('kind') != 'source-bound-cached-decoder-application':
        raise ValueError('complete clean cached application required')
    package, protocol = raw['source_package'], raw['protocol']
    if canonical_hash(package) != raw['source_package_sha256'] or canonical_hash(protocol) != raw['protocol_sha256'] or package['protocol_sha256'] != raw['protocol_sha256']:
        raise ValueError('application source/protocol hash binding failed')
    for name, expected in package['application_sources'].items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or sha(path) != expected:
            raise ValueError('application source changed')
    paths = [repo / 'harness-evolution/codex-rsi/operators/g001/manifest.json',
             repo / 'harness-evolution/codex-rsi/operators/g003-v13-dev1/manifest.json']
    if [sha(path) for path in paths] != package['operator_manifest_sha256s']:
        raise ValueError('original operator manifests changed')
    if set(package['operator_sources']) != {'K04', 'K06', 'K10', 'K12'}:
        raise ValueError('declared operator set changed')
    for key, expected in package['operator_sources'].items():
        path = paths[1 if key == 'K12' else 0].parent / key / 'kernel.py'
        if sha(path) != expected:
            raise ValueError('original verified kernel changed')
    if (protocol['cases'] != CONFIGS or len(protocol['case_labels']) != 4 or protocol['timing']['paired_rounds'] != 30
            or protocol['strong_baseline_required'] != list(LIBRARY)
            or protocol['full_block_output_tolerance'] != {'rtol': .005, 'atol': .005}
            or protocol['operator_tolerances'] != {'K04': [.0001, .00001], 'K06': [.002, .01], 'K10': [.002, .01], 'K12': [.02, .02]}
            or protocol['compile_dynamic'] is not False or protocol['compile_fullgraph'] is not True):
        raise ValueError('predeclared numerical/backend/case protocol changed')
    snapshots = raw['gpu_snapshots']
    if not snapshots or not all(row.get('available') is True and row.get('foreign_processes') == 0 for row in snapshots):
        raise ValueError('actual measurement snapshots not all clean')
    env = raw['environment']
    if env['tf32'] is not False or env['fp16_reduced_precision_reduction'] is not False:
        raise ValueError('library precision protocol changed')
    account = accounting_receipt(queue_file, raw, digest)
    if set(raw['measurements']) != set(protocol['case_labels']):
        raise ValueError('case denominator changed')
    cases, count = {}, 0
    for label, config in zip(protocol['case_labels'], CONFIGS):
        row = raw['measurements'][label]
        if row['status'] != 'MEASURED_CLEAN' or row['config'] != config:
            raise ValueError('measured case incomplete or relabeled')
        samples = row['samples']
        validate_samples(samples, 30)
        statuses = row['backend_qualification']
        if set(statuses) != set(BACKENDS) or any((statuses[key]['status'] == 'AVAILABLE') != (key in samples) for key in statuses):
            raise ValueError('backend qualification does not match timing')
        complete = all(key in samples for key in LIBRARY)
        if row['strong_baseline_complete'] is not complete:
            raise ValueError('strong baseline completeness claim inconsistent')
        for candidate in ('candidate', 'compiled_candidate'):
            gains = row['candidate_gains'][candidate]
            if candidate not in samples:
                if gains is not None:
                    raise ValueError('unmeasured candidate has a gain claim')
                continue
            expected = {'against_fp32_reference': pair_metric(samples, 'fp32_reference', candidate),
                        'against_fp16_library': pair_metric(samples, 'fp16_library', candidate) if 'fp16_library' in samples else None,
                        'against_strong_library': strong_metric(samples, candidate)}
            if gains != expected:
                raise ValueError('stored paired metric differs from independent raw recomputation')
        safe = {key: copy.deepcopy(row[key]) for key in ('status', 'config', 'input_weights_cache_sha256', 'samples', 'candidate_gains', 'strong_baseline_complete')}
        safe['backend_qualification'] = {name: {k: v for k, v in data.items() if k in
            ('status', 'error_type', 'mode', 'fullgraph', 'dynamic', 'compile_warmup_seconds')}
            for name, data in statuses.items()}
        for name in ('reference_correctness', 'candidate_correctness', 'fp16_library_correctness'):
            if name in row:
                diagnostic = row[name]
                if diagnostic['full_block_tolerance'] != protocol['full_block_output_tolerance']:
                    raise ValueError('full decode tolerance changed')
                safe[name] = copy.deepcopy(diagnostic)
        cases[label] = safe
        count += len(samples) * 30
    complete = all(row['strong_baseline_complete'] for row in cases.values())
    if raw['all_strong_baselines_complete'] is not complete:
        raise ValueError('whole application strong-baseline completeness inconsistent')
    victory = {}
    for candidate in ('candidate', 'compiled_candidate'):
        metrics = [(row['candidate_gains'][candidate] or {}).get('against_strong_library') for row in cases.values()]
        victory[candidate] = complete and all(m is not None and m['victory_proven'] for m in metrics)
    return {'schema_version': 1, 'kind': 'source-bound-cached-decoder-gpu-measurement', 'status': 'MEASURED_CLEAN',
            'source_package': package, 'source_package_sha256': raw['source_package_sha256'],
            'protocol': protocol, 'protocol_sha256': raw['protocol_sha256'], 'raw_result_sha256': digest,
            'measurement_cases': cases, 'raw_gpu_latency_samples': count, 'raw_host_latency_samples': count,
            'device': {k: env[k] for k in ('torch', 'cuda', 'python', 'device_name', 'device_total_memory', 'tf32', 'fp16_reduced_precision_reduction', 'physical_gpu')},
            'clean_measurement': {'snapshot_count': len(snapshots), 'all_snapshots_available_and_foreign_processes_zero': True},
            'accounting': account, 'conclusions': {'all_four_cases_measured': True, 'all_strong_baselines_complete': complete,
                                                 'beats_all_strong_library_cases': victory},
            'pretrained_model': False, 'text_quality_measured': False, 'harness_performance_causality_proven': False,
            'independent_consistency_scope': 'Read-only original samples, sources and official queue; not fresh GPU reproduction.',
            'limits': raw['limits']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--raw-sha256', required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    report = make_report(a.raw, a.raw_sha256, a.queue, a.repo)
    with a.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'sha256': sha(a.out), 'sample_count': report['raw_gpu_latency_samples'], 'conclusions': report['conclusions']}))


if __name__ == '__main__':
    main()
