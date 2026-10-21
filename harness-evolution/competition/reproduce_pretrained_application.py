"""Reproduce the full pretrained GPT-2 application with local frozen weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent / 'application'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_application import local_status, reference, validate_fresh_output, write_new


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
    # The accounting subprocess completes a real observation without importing
    # Torch/Transformers and consuming the original two-second freshness window.
    if args.local_status:
        try:
            if reference(args.local_status)['sha256'] != args.local_status_sha256:
                raise ValueError('local policy identity changed')
            print(json.dumps(local_status(json.loads(args.local_status.read_bytes())), allow_nan=False))
            return 0
        except (OSError, ValueError, sqlite3.Error):
            print(json.dumps({'coverage_complete': False, 'should_stop': True,
                              'error_code': 'LOCAL_ACCOUNTING_REJECTED'}))
            return 2
    if args.out is None or args.model_dir is None:
        parser.error('--out and --model-dir are required; model weights are not downloaded')
    if bool(args.operator_manifest) != bool(args.operator_manifest_sha256):
        parser.error('an explicit operator manifest requires its SHA256')
    out = validate_fresh_output(args.out)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import benchmark_pretrained_gpt2 as benchmark
    app = benchmark.model_app
    manifest = args.operator_manifest or app.DEFAULT_MANIFEST
    manifest_sha = args.operator_manifest_sha256 or app.DEFAULT_MANIFEST_SHA256
    binding = app.verify_kernel(manifest, manifest_sha)
    receipt = app.model_receipt(args.model_dir)
    tokens = app.input_ids(args.model_dir)
    report = benchmark.make_plan(binding, receipt, tokens, 'all')
    out.mkdir(parents=True, mode=0o700)
    write_new(out / 'benchmark-plan.json', report)
    write_new(out / 'reproduction-plan.json', {'status': 'EXECUTE_REQUESTED' if args.execute else 'CPU_PLAN_ONLY',
        'executor': reference(__file__), 'benchmark': reference(benchmark.__file__), 'model_requests': 0,
        'production_ledger_modified': False, 'gpu': args.gpu, 'operator_manifest': reference(manifest),
        'pretrained_model': True, 'weights_downloaded': False, 'weights_redistributed': False,
        'original_numerical_protocol': report['protocol']})
    if not args.execute:
        return 0
    from kernel_lab.queue import Queue
    queue = out / 'queue.sqlite'
    instance = Queue(queue); instance.close()
    run_id = 'local-pretrained-gpt2-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    policy = write_new(out / 'local-policy.json', {'kind': 'independent-local-app-reproduction',
        'run_id': run_id, 'queue': str(queue), 'gpu_queue_paths': [str(queue)], 'model_requests': 0})
    budget = write_new(out / 'budget-argv.json', [sys.executable, str(Path(__file__).resolve()),
        '--local-status', policy['path'], '--local-status-sha256', policy['sha256']])
    command = [sys.executable, str(APP / 'benchmark_pretrained_gpt2.py'), '--execute',
        '--model-dir', str(args.model_dir), '--gpu', str(args.gpu), '--run-id', run_id,
        '--operator-manifest', str(manifest), '--operator-manifest-sha256', manifest_sha,
        '--out', str(out / 'application-result.json'), '--accounting-queue', str(queue),
        '--registry-reference', policy['path'], '--registry-sha256', policy['sha256'],
        '--budget-argv-json', budget['path'], '--budget-argv-sha256', budget['sha256']]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'HF_HUB_OFFLINE': '1'})
    write_new(out / 'execution.json', {'exit_code': result.returncode, 'executor': reference(__file__),
        'final_local_accounting': local_status(json.loads(Path(policy['path']).read_bytes())),
        'result': reference(out / 'application-result.json') if (out / 'application-result.json').exists() else None})
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
