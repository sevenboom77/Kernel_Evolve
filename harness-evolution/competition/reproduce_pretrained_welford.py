"""Prepare or run an independently accounted local Welford numerical diagnosis."""
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

DIAGNOSTIC_SHA256 = 'e014aaadc10f19df106f9e60e3aba18270f12cfa487febe25b178da80da3ba76'
ORIGINAL_REPRODUCTION_SHA256 = 'd986fac266d1ff1f028d2f721b03763a7adfc93716688c012ecc671edf323ced'


@contextmanager
def CPU_import_scope(enabled):
    """A host-only plan needs no availability probe from Transformers imports."""
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


def load_diagnostic():
    path = APP / 'diagnose_pretrained_gpt2_welford.py'
    if reference(path)['sha256'] != DIAGNOSTIC_SHA256:
        raise ValueError('frozen Welford diagnostic source changed')
    spec = importlib.util.spec_from_file_location('independent_local_welford_diagnostic', path)
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
    if reference(Path(__file__).with_name('reproduce_pretrained_application.py'))['sha256'] != ORIGINAL_REPRODUCTION_SHA256:
        raise ValueError('original independent reproduction contract changed')
    os.environ['HF_HUB_OFFLINE'] = '1'
    with CPU_import_scope(not args.execute):
        diagnostic = load_diagnostic()
        # A local reproduction has its own Queue and no author primary-series
        # evidence. The production diagnostic entry keeps its full V18 gate.
        benchmark = diagnostic.adapt_private_base(diagnostic.load_frozen_base(), closure=None)
        app = benchmark.model_app
        manifest = args.operator_manifest or app.DEFAULT_MANIFEST
        manifest_sha = args.operator_manifest_sha256 or app.DEFAULT_MANIFEST_SHA256
        binding = app.verify_kernel(manifest, manifest_sha)
        receipt = app.model_receipt(args.model_dir)
        tokens = app.input_ids(args.model_dir)
        report = benchmark.make_plan(binding, receipt, tokens, 'all')
    out.mkdir(parents=True, mode=0o700)
    write_new(out / 'benchmark-plan.json', report)
    write_new(out / 'reproduction-plan.json', {
        'status': 'EXECUTE_REQUESTED' if args.execute else 'CPU_PLAN_ONLY',
        'scope': 'INDEPENDENT_LOCAL_NUMERICAL_DIAGNOSIS', 'executor': reference(__file__),
        'diagnostic': reference(diagnostic.__file__), 'original_benchmark': reference(benchmark.__file__),
        'gpu': args.gpu, 'operator_manifest_reference': reference(manifest),
        'model_requests': 0, 'production_ledger_modified': False,
        'primary_series_qualification_claimed': False, 'weights_downloaded': False,
        'weights_redistributed': False, 'performance_measured': False, 'RSI_proven': False,
        'CPU_availability_probe_disabled_during_import': not args.execute,
        'original_numerical_protocol': report['protocol']})
    if not args.execute:
        return 0
    from kernel_lab.queue import Queue
    queue = out / 'queue.sqlite'
    instance = Queue(queue)
    instance.close()
    run_id = 'local-pretrained-welford-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
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
    write_new(out / 'execution.json', {'exit_code': code, 'executor': reference(__file__),
        'scope': 'INDEPENDENT_LOCAL_NUMERICAL_DIAGNOSIS',
        'final_local_accounting': local_status(json.loads(Path(policy['path']).read_bytes())),
        'result': reference(out / 'application-result.json') if (out / 'application-result.json').exists() else None,
        'performance_measured': False, 'primary_series_qualification_claimed': False, 'RSI_proven': False})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
