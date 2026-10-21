"""Publish every closed graph timing sample without exporting private paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

SUMMARY_SHA256 = '67eccbadda2471354b1d6f286e05bc4ba68ae9cb25716ee5f54a2227fae2b25b'
ARCHIVE_SHA256 = '49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276'
CANDIDATE_SHA256 = '045663fab1a66b8b24944234babb65d27dc5827149debb4899d08081097fa937'
CASES = ('arxiv-full', 'arxiv-first-2048', 'boundary-fixture')
LIBRARIES = ('segment_reduce', 'index_add', 'sparse_mm', 'compiled_segment_reduce', 'compiled_index_add')
BACKENDS = (*LIBRARIES, 'candidate')
TOLERANCE = {'rtol': .0003, 'atol': .00001}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(ref):
    need(sha(ref['path']) == ref['sha256'], 'original graph evidence binding changed')
    return json.loads(Path(ref['path']).read_text(), parse_constant=lambda _: need(False, 'nonfinite JSON'))


def finite(value, *, positive=False):
    need(type(value) in (int, float) and math.isfinite(value) and
         (value > 0 if positive else value >= 0), 'nonfinite or invalid original metric')
    return value


def qualification(value):
    need(value['status'] == 'AVAILABLE' and value['timing_admitted'] is True and
         value['input_bytes_preserved'] is True and value['independent_output_storage'] is True and
         value['input_mutation_detected'] is False and value['fallback_used'] is False and
         value['tolerance'] == TOLERANCE, 'unqualified original graph output or aggregation')
    return {key: value[key] for key in ('status', 'timing_admitted', 'input_bytes_preserved',
        'independent_output_storage', 'input_mutation_detected', 'fallback_used', 'tolerance')} | {
        'max_absolute_error': finite(value['max_absolute_error'])}


def recompute_gain(backends):
    minima = [min(backends[name]['gpu_us_samples'][i] for name in LIBRARIES) for i in range(30)]
    ratios = [minima[i] / backends['candidate']['gpu_us_samples'][i] for i in range(30)]
    rng = random.Random(20261006)
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=30)) for _ in range(2000))
    return {'required_library_backends': list(LIBRARIES), 'paired_rounds': 30,
        'strong_baseline_samples': minima, 'paired_ratio_samples': ratios,
        'median': statistics.median(ratios),
        'bootstrap_95_percent_interval': [bootstrap[49], bootstrap[1949]],
        'bootstrap_seed': 20261006, 'bootstrap_resamples': 2000, 'victory_proven': bootstrap[49] > 1}


def public_replica(raw, gpu):
    need(raw['status'] == 'MEASURED_CLEAN' and raw['performance_measured'] is True and
         raw['archive_sha256'] == ARCHIVE_SHA256 and raw['operator_sha256'] == CANDIDATE_SHA256 and
         raw['candidate_source_author'] == 'Codex' and raw['selected_case'] == 'all' and
         raw['tolerance'] == TOLERANCE and raw['max_wall_seconds'] == 900 and
         raw['strong_baselines_required'] == list(LIBRARIES) and
         raw['computation']['trained_weights'] is False and raw['model_api_calls'] == 0 and
         raw['rsi_promotion'] is False, 'original graph protocol or claim boundary changed')
    need(set(raw['measurements']) == set(CASES), 'all three fixed cases required')
    physical = raw['actual_physical_gpu_binding']
    need(type(gpu) is int and physical['physical_gpu'] == gpu and
         isinstance(physical['uuid'], str) and physical['uuid'].startswith('GPU-'), 'physical device binding changed')
    cases = []
    for label in CASES:
        row = raw['measurements'][label]
        need(row['status'] == 'MEASURED_CLEAN' and row['case_id'] == label and
             row['strong_baseline_complete'] is True and set(row['samples']) == set(BACKENDS) and
             set(row['backend_qualification']) == set(BACKENDS), 'missing original library or complete case')
        backends = {}
        for name in BACKENDS:
            checked = row['backend_qualification'][name]
            whole = qualification(checked)
            aggregation = qualification(checked['aggregation_check'])
            prepared = checked['prepared']
            need(prepared['fallback_used'] is False and prepared['compiled_requested'] is
                 (name.startswith('compiled_') or name == 'candidate'), 'compiled backend identity changed')
            if name == 'candidate':
                need(prepared['candidate_source_sha256'] == CANDIDATE_SHA256 and
                     prepared['source_author'] == 'Codex', 'candidate source provenance changed')
            samples = row['samples'][name]
            exported = {}
            for kind in ('gpu', 'host'):
                values = samples[kind+'_us_samples']
                need(len(values) == 30, 'all thirty original GPU and host samples required')
                exported[kind+'_us_samples'] = [finite(v, positive=True) for v in values]
                median = statistics.median(values)
                need(median == samples[kind+'_us_median'], 'original sample median changed')
                exported[kind+'_us_median'] = median
            halves = [statistics.median(exported['gpu_us_samples'][:15]),
                      statistics.median(exported['gpu_us_samples'][15:])]
            state_ratio = max(halves) / max(min(halves), 1e-9)
            state = {'half_medians': halves, 'state_shift_ratio': state_ratio,
                     'state_shift_flag': state_ratio > 1.25}
            need(samples['timing_backend'] == 'cuda_events' and samples['gpu_state_split'] == state and
                 state['state_shift_flag'] is False and
                 samples['warmup'] == 20 and type(samples['iters_per_batch']) is int and
                 samples['iters_per_batch'] > 0, 'original timing protocol changed')
            exported.update(qualification=whole, aggregation_qualification=aggregation,
                compiled_requested=prepared['compiled_requested'], fallback_used=False,
                timing_backend='cuda_events', gpu_state_split=state, warmup=20,
                iters_per_batch=samples['iters_per_batch'],
                calibration_batch_ms=finite(samples['calibration_batch_ms'], positive=True),
                host_time_note='Submission-loop cost; final device wait excluded. Not synchronized end-to-end wall time.')
            backends[name] = exported
        gain = recompute_gain(backends)
        need(gain == row['strong_gain'] and gain['victory_proven'] is True, 'original strong aggregate changed')
        cases.append({'case_id': label, 'input_sha256': row['input_sha256'],
                      'backends': backends, 'strong_gain': gain})
    return {'physical_gpu': gpu, 'physical_gpu_uuid': physical['uuid'], 'cases': cases}


def export(summary, expected_sha256, repo, out):
    need(expected_sha256 == SUMMARY_SHA256, 'exact original closed graph summary required')
    original = read_bound({'path': str(summary), 'sha256': expected_sha256})
    need(original['status'] == 'ALL_THREE_FIXED_CASES_TWO_DEVICES_STRONG_VICTORIES_PROVEN' and
         len(original['replicas']) == 2 and original['model_API_calls'] == 0 and
         original['foreign_processes_signaled'] == 0 and original['RSI_proven'] is False,
         'closed original graph summary differs')
    execution = read_bound(original['execution'])
    need(execution['status'] == 'ALL_REAL_CITATION_GRAPH_REPLICAS_CLOSED' and execution['exit_code'] == 0 and
         execution['owned_RPC_formally_closed'] is True, 'actual original application lifetimes not closed')
    frozen = read_bound(original['source_freeze'])
    for binding in frozen['sources']:
        need(sha(binding['path']) == binding['sha256'], 'frozen original graph source changed')
    replicas = []
    sources = None
    metadata = None
    for replica in original['replicas']:
        raw = read_bound(replica['report'])
        for name, digest in raw['source_package'].items():
            path = Path(name)
            need(not path.is_absolute() and '..' not in path.parts and sha(Path(repo)/path) == digest,
                 'original public source package changed')
        need(hashlib.sha256(json.dumps(raw['source_package'], sort_keys=True).encode()).hexdigest() ==
             raw['source_package_sha256'], 'original source package hash changed')
        protocol = {key: raw[key] for key in ('protocol', 'cases', 'computation', 'tolerance',
                    'strong_baselines_required', 'timing', 'max_wall_seconds')}
        need(hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == raw['protocol_sha256'],
             'original graph protocol hash changed')
        if sources is None:
            sources = raw['source_package']
            metadata = protocol
        need(raw['source_package'] == sources and protocol == metadata, 'replica protocol/source mismatch')
        saved = public_replica(raw, replica['gpu'])
        saved['original_raw_report_sha256'] = replica['report']['sha256']
        replicas.append(saved)
    need(len({x['physical_gpu_uuid'] for x in replicas}) == 2, 'two distinct actual devices required')
    result = {'schema_version': 1, 'kind': 'complete-original-citation-graph-public-samples',
        'status': original['status'], 'source_author': 'Codex', 'archive_sha256': ARCHIVE_SHA256,
        'dataset': 'ogbn-arxiv', 'dataset_license': 'ODC-BY', 'protocol': metadata,
        'source_package': sources, 'original_closed_summary_sha256': expected_sha256,
        'original_execution_sha256': original['execution']['sha256'], 'replicas': replicas,
        'GPU_sample_count': 1080, 'host_sample_count': 1080,
        'fresh_GPU_execution_performed_by_export': False, 'model_API_calls': 0, 'RSI_proven': False,
        'classification_accuracy_claim': False, 'compile_inclusive_job_speedup_claim': False,
        'private_paths_exported': False,
        'limits': ['Fixed seeded untrained two-layer FP32 computation on three declared cases.',
            'Strong baseline is the per-round fastest of all five qualified libraries.',
            'Schedule construction and compilation excluded from paired inference timing.',
            'CUDA graphs disabled; host samples exclude final synchronization.',
            'Application measurements do not prove AGH/RSI source causality.']}
    with Path(out).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {'status': result['status'], 'sha256': sha(out), 'replica_count': 2,
            'case_count': 6, 'GPU_samples': 1080, 'host_samples': 1080, 'new_GPU_model_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.summary, args.summary_sha256, args.repo, args.out)))
