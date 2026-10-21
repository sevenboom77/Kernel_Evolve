"""Independently accounted portable reproduction of the frozen cached decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent / 'application'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reproduce_application import local_status, reference, validate_fresh_output, write_new


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--gpu', type=int, choices=range(8), default=0)
    p.add_argument('--operator-manifest', type=Path)
    p.add_argument('--operator-manifest-sha256')
    p.add_argument('--k12-manifest', type=Path)
    p.add_argument('--k12-manifest-sha256')
    p.add_argument('--local-status', type=Path)
    p.add_argument('--local-status-sha256')
    a = p.parse_args(argv)
    # The live local budget command imports only the standard library; Torch
    # startup must not consume the two-second observation freshness interval.
    if a.local_status:
        try:
            if reference(a.local_status)['sha256'] != a.local_status_sha256:
                raise ValueError('local policy changed')
            print(json.dumps(local_status(json.loads(a.local_status.read_bytes())), allow_nan=False))
            return 0
        except Exception:
            print(json.dumps({'coverage_complete': False, 'should_stop': True, 'error_code': 'LOCAL_ACCOUNTING_REJECTED'}))
            return 2
    if a.out is None:
        p.error('--out required')
    for flag in ('operator', 'k12'):
        if bool(getattr(a, flag + '_manifest')) != bool(getattr(a, flag + '_manifest_sha256')):
            p.error('each explicit manifest requires its SHA256')
    out = validate_fresh_output(a.out)
    out.mkdir(parents=True, mode=0o700)
    import benchmark_cached_decoder_v2 as benchmark
    import cached_decoder
    bindings = cached_decoder.verify_sources(
        a.operator_manifest or cached_decoder.base.DEFAULT_MANIFEST,
        a.operator_manifest_sha256 or cached_decoder.base.DEFAULT_MANIFEST_SHA256,
        a.k12_manifest or cached_decoder.K12_MANIFEST,
        a.k12_manifest_sha256 or cached_decoder.K12_MANIFEST_SHA256)
    report = benchmark.make_plan(bindings, benchmark.make_protocol())
    write_new(out / 'benchmark-plan.json', report)
    write_new(out / 'reproduction-plan.json', {
        'status': 'EXECUTE_REQUESTED' if a.execute else 'CPU_PLAN_ONLY', 'executor': reference(__file__),
        'benchmark': reference(APP / 'benchmark_cached_decoder_v2.py'), 'gpu': a.gpu,
        'operator_manifests': list(bindings.manifests), 'model_requests': 0,
        'production_ledger_modified': False, 'original_numerical_protocol': report['protocol']})
    if not a.execute:
        return 0
    from kernel_lab.queue import Queue
    import hashlib
    from types import SimpleNamespace
    queue = out / 'queue.sqlite'
    q = Queue(queue)
    q.close()
    run_id = 'local-cached-decoder-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    policy = write_new(out / 'local-policy.json', {'kind': 'independent-local-app-reproduction',
        'run_id': run_id, 'queue': str(queue), 'gpu_queue_paths': [str(queue)], 'model_requests': 0})
    budget = write_new(out / 'budget-argv.json', [sys.executable, str(Path(__file__).resolve()),
        '--local-status', policy['path'], '--local-status-sha256', policy['sha256']])
    args = SimpleNamespace(gpu=a.gpu, accounting_queue=queue, run_id=run_id,
                           max_wall_seconds=900, out=out / 'application-result.json')
    checker = benchmark.frozen.BudgetChecker(budget['path'], budget['sha256'])
    report['budget_bindings'] = {'registry': policy['path'], 'registry_sha256': policy['sha256'],
                                 'argv': budget['path'], 'argv_sha256': budget['sha256']}
    code = 0
    try:
        benchmark.frozen.validate_registry(policy['path'], policy['sha256'], queue)
        report['last_budget_snapshot'] = checker.check()
        benchmark.frozen.atomic_json(args.out, report)
        with benchmark.compiler_scope():
            benchmark.frozen.run_gpu(args, bindings, report, checker)
    except BaseException as exc:
        code = 2
        report.update(status='BUDGET_STOPPED' if isinstance(exc, benchmark.frozen.BudgetStopped) else 'FAILED',
                      error_type=type(exc).__name__, error=str(exc)[:500], completed_at=benchmark.frozen.utcnow())
        benchmark.frozen.atomic_json(args.out, report)
    write_new(out / 'execution.json', {'exit_code': code, 'executor': reference(__file__),
        'final_local_accounting': local_status(json.loads(Path(policy['path']).read_bytes())),
        'result': reference(args.out) if args.out.exists() else None})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
