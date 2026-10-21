"""Prepare or run an independently accounted, offline GPT-2 head-layout trial."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent / 'application'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_application import local_status, reference, validate_fresh_output, write_new

ALIGNED_BENCHMARK_SHA256 = 'd2a521ec708e1cec85c660e644400ed6660d388503f672a01c7b27d95b28a63f'


@contextmanager
def CPU_import_scope(enabled):
    if not enabled:
        yield
        return
    import torch
    available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = available


def load_aligned_benchmark():
    path = APP / 'benchmark_pretrained_aligned_head.py'
    if reference(path)['sha256'] != ALIGNED_BENCHMARK_SHA256:
        raise ValueError('frozen aligned-head benchmark source changed')
    spec = importlib.util.spec_from_file_location('independent_local_aligned_head_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(8), default=0)
    parser.add_argument('--operator-manifest', type=Path)
    parser.add_argument('--operator-manifest-sha256')
    parser.add_argument('--local-status', type=Path)
    parser.add_argument('--local-status-sha256')
    args = parser.parse_args(argv)
    # Keep the real local status subprocess free of model/GPU imports, so the
    # original budget checker can use its unchanged freshness limit.
    if args.local_status:
        try:
            if reference(args.local_status)['sha256'] != args.local_status_sha256:
                raise ValueError('local policy identity changed')
            print(json.dumps(local_status(json.loads(args.local_status.read_bytes())), allow_nan=False))
            return 0
        except (OSError, ValueError, sqlite3.Error, KeyError):
            print(json.dumps({'coverage_complete': False, 'should_stop': True,
                              'error_code': 'LOCAL_ACCOUNTING_REJECTED'}))
            return 2
    if args.out is None or args.model_dir is None:
        parser.error('--out and --model-dir are required; weights are not downloaded')
    if bool(args.operator_manifest) != bool(args.operator_manifest_sha256):
        parser.error('an explicit operator manifest requires its SHA256')
    out = validate_fresh_output(args.out)
    os.environ['HF_HUB_OFFLINE'] = '1'
    with CPU_import_scope(not args.execute):
        aligned = load_aligned_benchmark()
        benchmark = aligned.adapt(aligned.load_frozen_base())
        app = benchmark.model_app
        manifest = args.operator_manifest or app.DEFAULT_MANIFEST
        manifest_sha = args.operator_manifest_sha256 or app.DEFAULT_MANIFEST_SHA256
        binding = app.verify_kernel(manifest, manifest_sha)
        receipt = app.model_receipt(args.model_dir)
        tokens = app.input_ids(args.model_dir)
        report = benchmark.make_plan(binding, receipt, tokens, 'all')
    report.update(kind='source-bound-pretrained-gpt2-aligned-head-independent-application',
                  performance_measured=False, RSI_proven=False, actual_K02_Triton_executed=False)
    out.mkdir(parents=True, mode=0o700)
    write_new(out / 'benchmark-plan.json', report)
    write_new(out / 'reproduction-plan.json', {
        'status': 'EXECUTE_REQUESTED' if args.execute else 'CPU_PLAN_ONLY',
        'scope': 'INDEPENDENT_LOCAL_ALIGNED_HEAD_APPLICATION', 'executor': reference(__file__),
        'aligned_benchmark': reference(aligned.__file__), 'original_benchmark': reference(benchmark.__file__),
        'gpu': args.gpu, 'operator_manifest_reference': reference(manifest),
        'historical_K02_is_provenance_only': True, 'model_requests': 0,
        'production_ledger_modified': False, 'primary_series_qualification_claimed': False,
        'weights_downloaded': False, 'weights_redistributed': False,
        'performance_measured': False, 'RSI_proven': False,
        'CPU_availability_probe_disabled_during_import': not args.execute,
        'original_numerical_protocol': report['protocol']})
    if not args.execute:
        return 0
    from kernel_lab.queue import Queue
    queue = out / 'queue.sqlite'
    instance = Queue(queue)
    instance.close()
    run_id = 'local-pretrained-aligned-head-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    policy = write_new(out / 'local-policy.json', {'kind': 'independent-local-app-reproduction',
        'run_id': run_id, 'queue': str(queue), 'gpu_queue_paths': [str(queue)], 'model_requests': 0})
    budget = write_new(out / 'budget-argv.json', [sys.executable, str(Path(__file__).resolve()),
        '--local-status', policy['path'], '--local-status-sha256', policy['sha256']])
    command = ['--execute', '--model-dir', str(args.model_dir), '--gpu', str(args.gpu),
        '--run-id', run_id, '--operator-manifest', str(manifest),
        '--operator-manifest-sha256', manifest_sha, '--out', str(out / 'application-result.json'),
        '--accounting-queue', str(queue), '--registry-reference', policy['path'],
        '--registry-sha256', policy['sha256'], '--budget-argv-json', budget['path'],
        '--budget-argv-sha256', budget['sha256']]
    code = benchmark.main(command)
    result = out / 'application-result.json'
    measured = result.exists() and json.loads(result.read_text()).get('status') == 'MEASURED_CLEAN'
    write_new(out / 'execution.json', {'exit_code': code, 'executor': reference(__file__),
        'scope': 'INDEPENDENT_LOCAL_ALIGNED_HEAD_APPLICATION',
        'final_local_accounting': local_status(json.loads(Path(policy['path']).read_bytes())),
        'result': reference(result) if result.exists() else None,
        'performance_measured': code == 0 and measured,
        'source_gain_proven': False, 'primary_series_qualification_claimed': False, 'RSI_proven': False})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
