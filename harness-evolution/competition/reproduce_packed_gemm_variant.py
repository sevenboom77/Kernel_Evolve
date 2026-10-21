"""Independently account the separate H0 GEMM application on a clean local GPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HELPERS_SHA = '00c1ee72a3b149a3ed3166f0585547ef835b0fb80d56dbe098193f295d3c7d66'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    helpers_path = HERE / 'reproduce_application.py'
    if hashlib.sha256(helpers_path.read_bytes()).hexdigest() != HELPERS_SHA:
        raise ValueError('original independent local accounting helpers changed')
    helper = load_module('private_packed_gemm_local_accounting', helpers_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(8), default=0)
    parser.add_argument('--operator-manifest', type=Path)
    parser.add_argument('--operator-manifest-sha256')
    parser.add_argument('--variant-manifest', type=Path)
    parser.add_argument('--variant-manifest-sha256')
    parser.add_argument('--local-status', type=Path)
    parser.add_argument('--local-status-sha256')
    args = parser.parse_args(argv)
    if args.local_status:
        try:
            if helper.reference(args.local_status)['sha256'] != args.local_status_sha256:
                raise ValueError('local policy changed')
            print(json.dumps(helper.local_status(json.loads(args.local_status.read_bytes())), allow_nan=False))
            return 0
        except Exception:
            print(json.dumps({'coverage_complete': False, 'should_stop': True, 'error_code': 'LOCAL_ACCOUNTING_REJECTED'}))
            return 2
    if args.out is None:
        parser.error('--out required')
    for name in ('operator', 'variant'):
        if bool(getattr(args, name + '_manifest')) != bool(getattr(args, name + '_manifest_sha256')):
            parser.error('each explicit manifest requires its SHA256')
    out = helper.validate_fresh_output(args.out)
    profile = load_module('private_local_packed_gemm_profile', HERE / 'application/benchmark_packed_gemm_variant.py')
    base = profile.load_frozen_base()
    original_verify = base.verify_sources
    if args.operator_manifest:
        original_verify = lambda: base.verify_sources(args.operator_manifest, args.operator_manifest_sha256)
    manifest = args.variant_manifest or profile.VARIANT_MANIFEST
    if args.variant_manifest and helper.reference(manifest)['sha256'] != args.variant_manifest_sha256:
        raise ValueError('explicit variant manifest changed')
    bindings = profile.verify_variant_sources(original_verify, manifest)
    base = profile.adapt_private_base(base)
    report = base.make_plan(bindings, base.make_protocol())
    out.mkdir(parents=True, mode=0o700)
    helper.write_new(out / 'benchmark-plan.json', report)
    helper.write_new(out / 'reproduction-plan.json', {
        'status': 'EXECUTE_REQUESTED' if args.execute else 'CPU_PLAN_ONLY',
        'executor': helper.reference(__file__), 'benchmark': helper.reference(profile.__file__),
        'gpu': args.gpu, 'operator_manifests': list(bindings.manifests),
        'model_requests': 0, 'production_ledger_modified': False,
        'original_numerical_protocol': report['protocol']})
    if not args.execute:
        return 0
    original_paths = sys.path[:]
    sys.path.insert(0, str(ROOT))
    try:
        from kernel_lab.queue import Queue
        queue_file = out / 'queue.sqlite'
        queue = Queue(queue_file)
        queue.close()
        run_id = 'local-packed-gemm-variant-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
        policy = helper.write_new(out / 'local-policy.json', {'kind': 'independent-local-app-reproduction',
            'run_id': run_id, 'queue': str(queue_file), 'gpu_queue_paths': [str(queue_file)], 'model_requests': 0})
        budget = helper.write_new(out / 'budget-argv.json', [sys.executable, str(Path(__file__).resolve()),
            '--local-status', policy['path'], '--local-status-sha256', policy['sha256']])
        execution_args = SimpleNamespace(gpu=args.gpu, accounting_queue=queue_file, run_id=run_id,
                                        max_wall_seconds=900, out=out / 'application-result.json')
        checker = base.BudgetChecker(budget['path'], budget['sha256'])
        report['budget_bindings'] = {'registry': policy['path'], 'registry_sha256': policy['sha256'],
                                     'argv': budget['path'], 'argv_sha256': budget['sha256']}
        code = 0
        try:
            base.validate_registry(policy['path'], policy['sha256'], queue_file)
            report['last_budget_snapshot'] = checker.check()
            base.atomic_json(execution_args.out, report)
            with base.torch._dynamo.config.patch(cache_size_limit=32, accumulated_cache_size_limit=256):
                base.run_gpu(execution_args, bindings, report, checker)
        except BaseException as exc:
            code = 2
            report.update(status='BUDGET_STOPPED' if isinstance(exc, base.BudgetStopped) else 'FAILED',
                          error_type=type(exc).__name__, error=str(exc)[:500], completed_at=base.utcnow())
            base.atomic_json(execution_args.out, report)
        helper.write_new(out / 'execution.json', {'exit_code': code, 'executor': helper.reference(__file__),
            'final_local_accounting': helper.local_status(json.loads(Path(policy['path']).read_bytes())),
            'result': helper.reference(execution_args.out) if execution_args.out.exists() else None})
        return code
    finally:
        sys.path[:] = original_paths


if __name__ == '__main__':
    raise SystemExit(main())
