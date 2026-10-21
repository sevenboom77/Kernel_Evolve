"""Prepare or run the frozen citation graph protocol in a fresh local scope."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent / 'application'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_application import local_status, reference, validate_fresh_output, write_new

BENCHMARK = APP / 'benchmark_citation_graph_split_csr.py'
BENCHMARK_SHA256 = '7fea9e8873d130ff2616774e127e60c343bf999fe8b841a360163423e98bad9c'
CANDIDATE = APP / 'citation_graph_split_csr.py'
CANDIDATE_SHA256 = '045663fab1a66b8b24944234babb65d27dc5827149debb4899d08081097fa937'
ARCHIVE_SHA256 = '49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276'


def load_benchmark():
    if reference(BENCHMARK)['sha256'] != BENCHMARK_SHA256:
        raise ValueError('original citation graph benchmark source changed')
    spec = importlib.util.spec_from_file_location('independent_original_graph_benchmark', BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-sha256', default=ARCHIVE_SHA256)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(8), default=0)
    parser.add_argument('--case', choices=('all', 'arxiv-full', 'arxiv-first-2048', 'boundary-fixture'), default='all')
    parser.add_argument('--local-status', type=Path)
    parser.add_argument('--local-status-sha256')
    args = parser.parse_args(argv)
    if args.local_status:
        try:
            if reference(args.local_status)['sha256'] != args.local_status_sha256:
                raise ValueError('local policy identity changed')
            print(json.dumps(local_status(json.loads(args.local_status.read_text())), allow_nan=False))
            return 0
        except (OSError, ValueError, KeyError, sqlite3.Error):
            print(json.dumps({'coverage_complete': False, 'should_stop': True,
                              'error_code': 'LOCAL_ACCOUNTING_REJECTED'}))
            return 2
    if args.archive is None or args.out is None:
        parser.error('--archive and --out are required; data is never downloaded')
    out = validate_fresh_output(args.out)
    if args.archive_sha256 != ARCHIVE_SHA256:
        raise ValueError('original citation graph archive identity changed')
    if reference(CANDIDATE)['sha256'] != CANDIDATE_SHA256:
        raise ValueError('original split citation graph candidate source changed')
    benchmark = load_benchmark()
    graph = benchmark.strong_controls().original().load_graph_archive(args.archive, args.archive_sha256)
    report = benchmark.make_plan(graph, args.archive_sha256, args.case)
    report.update(reproduction_scope='INDEPENDENT_LOCAL_GRAPH_REPRODUCTION',
                  independent_entrypoint=reference(__file__), author_primary_admission_claimed=False,
                  primary_series_qualification_claimed=False, RSI_proven=False)
    out.mkdir(parents=True, mode=0o700)
    write_new(out / 'benchmark-plan.json', report)
    write_new(out / 'reproduction-plan.json', {
        'status': 'EXECUTE_REQUESTED' if args.execute else 'CPU_PLAN_ONLY',
        'scope': 'INDEPENDENT_LOCAL_GRAPH_REPRODUCTION', 'executor': reference(__file__),
        'benchmark': reference(BENCHMARK), 'candidate': reference(CANDIDATE),
        'archive_path': str(args.archive.resolve()), 'archive_sha256': args.archive_sha256,
        'gpu': args.gpu, 'selected_case': args.case, 'data_downloaded': False,
        'model_requests': 0, 'production_ledger_modified': False,
        'primary_series_qualification_claimed': False, 'RSI_proven': False,
        'original_protocol_sha256': report['protocol_sha256'],
        'original_case_ids': [case['case_id'] for case in report['cases']]})
    if not args.execute:
        return 0
    from kernel_lab.queue import Queue
    queue = out / 'queue.sqlite'
    instance = Queue(queue)
    instance.close()
    run_id = 'local-citation-graph-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    policy = write_new(out / 'local-policy.json', {
        'kind': 'independent-local-app-reproduction', 'run_id': run_id,
        'queue': str(queue), 'gpu_queue_paths': [str(queue)], 'model_requests': 0})
    budget = write_new(out / 'budget-argv.json', [sys.executable, str(Path(__file__).resolve()),
        '--local-status', policy['path'], '--local-status-sha256', policy['sha256']])
    common = benchmark.accounting()
    actual = argparse.Namespace(gpu=args.gpu, case=args.case, accounting_queue=queue,
        run_id=run_id, operator_source=CANDIDATE, out=out / 'application-result.json',
        registry_reference=Path(policy['path']), registry_reference_sha256=policy['sha256'])
    code = 0
    try:
        common.validate_registry(actual.registry_reference, actual.registry_reference_sha256, queue)
        checker = common.BudgetChecker(Path(budget['path']), budget['sha256'])
        report['last_budget_snapshot'] = checker.check()
        common.atomic_json(actual.out, report)
        benchmark.run_gpu(actual, graph, report, checker, common)
    except BaseException as exc:
        code = 2
        report.update(failure_observed_after_status=report['status'],
            status='BUDGET_STOPPED' if isinstance(exc, common.BudgetStopped) else 'FAILED',
            error_type=type(exc).__name__, diagnostic=str(exc)[:4000])
        common.atomic_json(actual.out, report)
    try:
        final = local_status(json.loads(Path(policy['path']).read_text()))
    except (OSError, ValueError, KeyError, sqlite3.Error):
        code = 2
        final = {'coverage_complete': False, 'should_stop': True,
                 'error_code': 'LOCAL_ACCOUNTING_REJECTED'}
    write_new(out / 'execution.json', {
        'exit_code': code, 'executor': reference(__file__),
        'scope': 'INDEPENDENT_LOCAL_GRAPH_REPRODUCTION', 'final_local_accounting': final,
        'result': reference(actual.out) if actual.out.exists() else None,
        'performance_measured': report.get('performance_measured', False),
        'production_ledger_modified': False, 'primary_series_qualification_claimed': False,
        'RSI_proven': False})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
