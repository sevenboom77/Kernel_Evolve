"""Independently account the unchanged packed decoder on a user's clean GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent / 'application'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reproduce_application import local_status, reference, validate_fresh_output, write_new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(8), default=0)
    parser.add_argument('--operator-manifest', type=Path)
    parser.add_argument('--operator-manifest-sha256')
    parser.add_argument('--k12-manifest', type=Path)
    parser.add_argument('--k12-manifest-sha256')
    parser.add_argument('--local-status', type=Path)
    parser.add_argument('--local-status-sha256')
    args = parser.parse_args(argv)
    # Budget subprocesses must finish the real local observation before its
    # original two-second window expires. This path imports no Torch module.
    if args.local_status:
        try:
            if reference(args.local_status)['sha256'] != args.local_status_sha256:
                raise ValueError('local policy changed')
            print(json.dumps(local_status(json.loads(args.local_status.read_bytes())), allow_nan=False))
            return 0
        except Exception:
            print(json.dumps({'coverage_complete': False, 'should_stop': True,
                              'error_code': 'LOCAL_ACCOUNTING_REJECTED'}))
            return 2
    if args.out is None:
        parser.error('--out required')
    for flag in ('operator', 'k12'):
        if bool(getattr(args, flag + '_manifest')) != bool(getattr(args, flag + '_manifest_sha256')):
            parser.error('each explicit manifest requires its SHA256')
    out = validate_fresh_output(args.out)
    out.mkdir(parents=True, mode=0o700)
    import benchmark_packed_cached_decoder as benchmark
    import cached_decoder
    bindings = cached_decoder.verify_sources(
        args.operator_manifest or cached_decoder.base.DEFAULT_MANIFEST,
        args.operator_manifest_sha256 or cached_decoder.base.DEFAULT_MANIFEST_SHA256,
        args.k12_manifest or cached_decoder.K12_MANIFEST,
        args.k12_manifest_sha256 or cached_decoder.K12_MANIFEST_SHA256)
    report = benchmark.make_plan(bindings, benchmark.make_protocol())
    write_new(out / 'benchmark-plan.json', report)
    write_new(out / 'reproduction-plan.json', {
        'status': 'EXECUTE_REQUESTED' if args.execute else 'CPU_PLAN_ONLY',
        'executor': reference(__file__), 'benchmark': reference(APP / 'benchmark_packed_cached_decoder.py'),
        'gpu': args.gpu, 'operator_manifests': list(bindings.manifests), 'model_requests': 0,
        'production_ledger_modified': False, 'original_numerical_protocol': report['protocol']})
    if not args.execute:
        return 0
    from kernel_lab.queue import Queue
    queue = out / 'queue.sqlite'
    instance = Queue(queue)
    instance.close()
    run_id = 'local-packed-decoder-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    policy = write_new(out / 'local-policy.json', {'kind': 'independent-local-app-reproduction',
        'run_id': run_id, 'queue': str(queue), 'gpu_queue_paths': [str(queue)], 'model_requests': 0})
    budget = write_new(out / 'budget-argv.json', [sys.executable, str(Path(__file__).resolve()),
        '--local-status', policy['path'], '--local-status-sha256', policy['sha256']])
    execution_args = SimpleNamespace(gpu=args.gpu, accounting_queue=queue, run_id=run_id,
                                    max_wall_seconds=900, out=out / 'application-result.json')
    checker = benchmark.BudgetChecker(budget['path'], budget['sha256'])
    report['budget_bindings'] = {'registry': policy['path'], 'registry_sha256': policy['sha256'],
                                 'argv': budget['path'], 'argv_sha256': budget['sha256']}
    code = 0
    try:
        benchmark.validate_registry(policy['path'], policy['sha256'], queue)
        report['last_budget_snapshot'] = checker.check()
        benchmark.atomic_json(execution_args.out, report)
        with benchmark.torch._dynamo.config.patch(cache_size_limit=32, accumulated_cache_size_limit=256):
            benchmark.run_gpu(execution_args, bindings, report, checker)
    except BaseException as exc:
        code = 2
        report.update(status='BUDGET_STOPPED' if isinstance(exc, benchmark.BudgetStopped) else 'FAILED',
                      error_type=type(exc).__name__, error=str(exc)[:500], completed_at=benchmark.utcnow())
        benchmark.atomic_json(execution_args.out, report)
    write_new(out / 'execution.json', {'exit_code': code, 'executor': reference(__file__),
        'final_local_accounting': local_status(json.loads(Path(policy['path']).read_bytes())),
        'result': reference(execution_args.out) if execution_args.out.exists() else None})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
