"""Accounted real-graph GPU qualification and paired whole-computation timing."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STRONG_SOURCE_SHA = '07a50890a51e60efeb65a9ae98f6404b4a67dd3b417f941ea0f82296a01679b9'
ACCOUNTING_SOURCE_SHA = 'd5d38b76c3ab25d4257f1042e19397bda373dbab90d76830e237f9eaf488c685'
TOLERANCE = {'rtol': 0.0003, 'atol': 0.00001}


def load(name, path, expected=None):
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Qualified original source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strong_controls():
    return load('source_bound_graph_strong_GPU_controls', HERE / 'citation_graph_strong_controls.py', STRONG_SOURCE_SHA)


def accounting():
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    return load('original_graph_GPU_accounting', HERE / 'benchmark_decoder_v3.py', ACCOUNTING_SOURCE_SHA)


def tensor_sha(value):
    data = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data).cast('B')).hexdigest()


def qualify(fn, reference, *, inputs, sync, empty_rows=None, budget_stop_type=()):
    before = [tensor_sha(value) for value in inputs]
    receipt = {'status': 'PREPARED_UNQUALIFIED', 'tolerance': TOLERANCE.copy(),
               'fallback_used': False, 'timing_admitted': False, 'input_mutation_detected': False}
    try:
        actual = fn()
        sync()
        if any(tensor_sha(value) != previous for value, previous in zip(inputs, before)):
            receipt['input_mutation_detected'] = True
            raise AssertionError('Original inputs or weights mutated')
        if (actual.shape != reference.shape or actual.dtype != torch.float32
                or not torch.isfinite(actual).all().item()):
            raise AssertionError('Wrong shape, dtype or nonfinite output')
        if any(actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() for value in inputs):
            raise AssertionError('Output storage aliases original inputs or weights')
        delta = (actual.double() - reference.double()).abs()
        receipt['max_absolute_error'] = delta.max().item()
        torch.testing.assert_close(actual.double(), reference.double(), **TOLERANCE, equal_nan=False)
        if empty_rows is not None and torch.count_nonzero(actual[empty_rows]).item() != 0:
            raise AssertionError('Original empty segments must be exactly zero')
        receipt.update(status='AVAILABLE', timing_admitted=True, input_bytes_preserved=True,
                       independent_output_storage=True)
        return fn, receipt
    except Exception as exc:
        if budget_stop_type and isinstance(exc, budget_stop_type):
            raise
        # A failing implementation may mutate inputs before raising itself.
        receipt['input_mutation_detected'] |= any(tensor_sha(value) != previous for value, previous in zip(inputs, before))
        receipt.update(status='INCORRECT' if isinstance(exc, AssertionError) else 'UNAVAILABLE',
                       error_type=type(exc).__name__, diagnostic=str(exc)[:4000])
        return None, receipt


def verify_operator_reproduction(path, sha256, gates):
    value = gates.bound({'path': str(Path(path).resolve()), 'sha256': sha256})
    expected = strong_controls().original().CSR_SHA256
    if (value.get('status') != 'ORIGINAL_K15_FULL_PROTOCOL_TWO_DEVICE_CLOSED'
            or value.get('operator_sha256') != expected or value.get('all_original_21_cases_correct') is not True
            or value.get('independent_verification_rounds') != 5 or len(value.get('replicas', [])) != 2
            or value.get('model_API_calls') != 0 or value.get('foreign_processes_signaled') != 0):
        raise ValueError('Actual unchanged original21-case/five-round two-device CSR reproduction required')
    devices = set()
    for row in value['replicas']:
        result = gates.bound(row['result'])
        if (result.get('owned_resource_closure_complete') is not True
                or result.get('correctness_case_count') != 21 or result.get('independent_verification_rounds') != 5
                or result.get('original_timing_shapes') != 5 or result.get('operator_sha256') != expected):
            raise ValueError('Each complete original CSR replica must qualify')
        devices.add(result['physical_gpu_uuid'])
    if len(devices) != 2:
        raise ValueError('Two distinct physical CSR verification devices required')
    return {'path': str(Path(path).resolve()), 'sha256': sha256}


def make_plan(graph, archive_sha256, case):
    strong = strong_controls()
    value = strong.cpu_plan(graph, archive_sha256=archive_sha256)
    value.update(kind='source-bound-real-citation-graph-GPU-benchmark',
                 protocol='real-citation-graph-strong-controls-GPU-v4',
                 GPU_execution_supported_by_this_entrypoint=True, selected_case=case,
                 performance_measured=False,
                 timing={'paired_rounds': 30, 'warmup': 20, 'order_seed': 20261006,
                         'cuda_events': True, 'host_wall_samples': True, 'cuda_graphs': False},
                 max_wall_seconds=900, original_operator_cross_device_reproduction_required=True)
    sources = [Path(__file__).resolve(), HERE / 'citation_graph_strong_controls.py', HERE / 'citation_graph.py',
               HERE / 'benchmark_decoder_v3.py', HERE / 'benchmark_compiled_pretrained_first_error.py']
    package = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    value['source_package'] = package
    value['source_package_sha256'] = hashlib.sha256(json.dumps(package, sort_keys=True).encode()).hexdigest()
    protocol = {key: value[key] for key in ('protocol', 'cases', 'computation', 'tolerance', 'strong_baselines_required', 'timing', 'max_wall_seconds')}
    value['protocol_sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return value


def run_gpu(args, graph, report, checker, common):
    strong = strong_controls(); base = strong.original()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with common.wall_limit(900), common.NumericGpuLock(ROOT, args.gpu):
        snapshot = common.clean_snapshot(args.gpu)
        checker.check()
        with common.ApplicationJob(args.accounting_queue, args.run_id, report['protocol_sha256'],
                                   report['source_package_sha256'], args.gpu, 900) as job:
            report.update(status='RUNNING', measurements={}, gpu_snapshots=[snapshot],
                          accounting={'queue': str(args.accounting_queue.resolve()), 'run_id': args.run_id,
                                      'job_id': job.job['job_id'], 'physical_gpu': args.gpu})
            job.result['report_path'] = str(args.out.resolve())
            def checkpoint():
                report['last_budget_snapshot'] = checker.check()
                report['gpu_snapshots'].append(common.clean_snapshot(args.gpu))
                if not job.queue.heartbeat(job.job['job_id'], job.job['lease_owner'], 1020):
                    raise common.BenchmarkError('Original application accounting lease lost')
            try:
                with common.cuda_completion(), torch.inference_mode(), torch._dynamo.config.patch(
                        cache_size_limit=32, accumulated_cache_size_limit=256):
                    if not torch.cuda.is_available():
                        raise common.BenchmarkError('CUDA unavailable after official admission')
                    torch.set_num_threads(2)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    torch.set_float32_matmul_precision('highest')
                    torch.cuda.set_device(0)
                    # set_device alone need not create a visible compute context.
                    # Allocate only after the official lock, job and budget admission.
                    device_binding_probe = torch.empty(1, device='cuda')
                    torch.cuda.synchronize()
                    from benchmark_pretrained_gpt2 import verify_device_binding
                    report['actual_physical_gpu_binding'] = verify_device_binding(args.gpu)
                    report['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'physical_gpu': args.gpu, 'device_name': torch.cuda.get_device_properties(0).name, 'tf32': False}
                    for case in report['cases']:
                        label = case['case_id']
                        if args.case != 'all' and args.case != label:
                            continue
                        checkpoint()
                        selected = base.boundary_fixture() if label == 'boundary-fixture' else graph
                        begin, end = 0, min(2048, selected.features.shape[0]) if label == 'arxiv-first-2048' else selected.features.shape[0]
                        weights = base.fixed_weights()
                        oracle = base.message_passing_reference(selected, weights, node_begin=begin, node_end=end)
                        inputs = base.prepare_tensors(selected, device='cuda', node_begin=begin, node_end=end)
                        weights = base.prepare_weights(weights, device='cuda')
                        adjacency = strong.prepare_adjacency(selected, device='cuda', node_begin=begin, node_end=end)
                        reference = torch.tensor(oracle, device='cuda')
                        shared = [inputs.features, inputs.sources, inputs.offsets, inputs.destinations, inputs.degrees,
                                  adjacency.crow_indices(), adjacency.col_indices(), adjacency.values(),
                                  *(offsets for _, offsets in inputs.tiles), *(value for layer in weights for value in layer)]
                        before = [tensor_sha(value) for value in shared]
                        row = {'status': 'RUNNING', 'backend_qualification': {}, 'samples': {}, 'strong_gain': None,
                               'input_sha256': before[0], 'case_id': label}
                        report['measurements'][label] = row
                        fns = {}
                        for name in (*strong.STRONG_BASELINES, 'candidate'):
                            checkpoint()
                            fn, prepared = strong.prepare_backend(name, inputs, weights, adjacency=adjacency,
                                candidate_source=args.operator_source)
                            if fn is None:
                                row['backend_qualification'][name] = prepared
                            else:
                                actual, qualification = qualify(fn, reference, inputs=shared,
                                    sync=torch.cuda.synchronize, budget_stop_type=common.BudgetStopped)
                                qualification.update(prepared=prepared, aggregation_check='pending separately')
                                row['backend_qualification'][name] = qualification
                                if qualification['input_mutation_detected']:
                                    raise common.BenchmarkError('Shared original graph inputs mutated; retain and stop case')
                                if actual is not None:
                                    fns[name] = actual
                            common.atomic_json(args.out, report)
                        # Aggregation qualification is required before every admitted timing.
                        # Full pipeline correctness alone cannot hide an incorrect CSR sum.
                        messages = inputs.features.index_select(0, inputs.sources)
                        from scipy.sparse import csr_matrix
                        offsets = selected.offsets[begin:end + 1] - selected.offsets[begin]
                        sources = selected.sources[selected.offsets[begin]:selected.offsets[end]]
                        csr = csr_matrix((np.ones(sources.size), sources, offsets),
                            shape=(end-begin, selected.features.shape[0]))
                        scale = max(1., float(np.max(np.abs(selected.features))))
                        sum_oracle = torch.tensor(csr @ (selected.features / scale).astype(np.float64), device='cuda')
                        for name in tuple(fns):
                            checkpoint()
                            if name == 'sparse_mm':
                                aggregate = lambda: torch.sparse.mm(adjacency, inputs.features)
                            else:
                                reducer = (base.tiled_aggregation(inputs, base.SourceBoundCSR(args.operator_source))
                                    if name == 'candidate' else base.stock_aggregation(
                                        'torch.segment_reduce' if 'segment_reduce' in name else 'torch.index_add_', inputs))
                                aggregate = lambda reducer=reducer: reducer(messages, inputs.offsets)
                                if name.startswith('compiled_'):
                                    aggregate = torch.compile(aggregate, mode='default', fullgraph=True, dynamic=False)
                            _, result = qualify(aggregate, sum_oracle, inputs=[messages, *shared],
                                empty_rows=inputs.degrees == 0, sync=torch.cuda.synchronize,
                                budget_stop_type=common.BudgetStopped)
                            row['backend_qualification'][name]['aggregation_check'] = result
                            if result['input_mutation_detected']:
                                raise common.BenchmarkError('Original aggregate inputs mutated; retain and stop case')
                            if result['status'] != 'AVAILABLE':
                                row['backend_qualification'][name].update(status=result['status'], timing_admitted=False)
                                del fns[name]
                            common.atomic_json(args.out, report)
                        from kernel_lab import timing
                        timing.configure(True)
                        common.measure_paired(fns, 20, 30, 20261006 + report['cases'].index(case), checkpoint,
                                              partial=row['samples'])
                        checkpoint()
                        if [tensor_sha(value) for value in shared] != before:
                            raise AssertionError('Original graph inputs or weights mutated during timing')
                        row['strong_gain'] = strong.strong_gain({name: samples['gpu_us_samples'] for name, samples in row['samples'].items()})
                        row.update(status='MEASURED_CLEAN', strong_baseline_complete=all(name in fns for name in strong.STRONG_BASELINES))
                        common.atomic_json(args.out, report)
                    checkpoint()
                    report.update(status='MEASURED_CLEAN', performance_measured=True, model_api_calls=0, rsi_promotion=False)
            except BaseException as exc:
                report.update(status='BUDGET_STOPPED' if isinstance(exc, common.BudgetStopped) else 'FAILED',
                              error_type=type(exc).__name__, diagnostic=str(exc)[:4000])
                raise
            finally:
                common.atomic_json(args.out, report)
                job.result.update(report_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(), report_status=report['status'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--operator-source', type=Path, default=HERE.parents[1] / 'codex-rsi/operators/g000-v18-dev2/K15/kernel.py')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=('all', 'arxiv-full', 'arxiv-first-2048', 'boundary-fixture'), default='all')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(8), default=3)
    parser.add_argument('--run-id', default='real-citation-graph-v3')
    for name in ('primary-series-completion', 'operator-reproduction', 'accounting-queue', 'registry-reference', 'budget-argv-json'):
        parser.add_argument('--' + name, type=Path)
        if name != 'accounting-queue':
            parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Fresh output required; preserve every earlier failure')
    strong = strong_controls(); base = strong.original()
    if base.sha256(args.operator_source) != base.CSR_SHA256:
        parser.error('Unchanged verified original K15 source required')
    graph = base.load_graph_archive(args.archive, args.archive_sha256)
    report = make_plan(graph, args.archive_sha256, args.case)
    common = accounting()
    if not args.execute:
        common.atomic_json(args.out, report)
        return 0
    required = ('primary_series_completion', 'primary_series_completion_sha256', 'operator_reproduction',
        'operator_reproduction_sha256', 'accounting_queue', 'registry_reference', 'registry_reference_sha256',
        'budget_argv_json', 'budget_argv_json_sha256')
    if any(getattr(args, name) is None for name in required):
        parser.error('Actual whole primary closure, operator reproduction and fresh carried accounting required')
    gates = load('graph_post_V19_admission_gates', HERE / 'benchmark_compiled_pretrained_first_error.py')
    try:
        closure = gates.verify_closed_v19(args.primary_series_completion, args.primary_series_completion_sha256)
        report['primary_closure'] = closure
        report['operator_cross_device_reproduction'] = verify_operator_reproduction(
            args.operator_reproduction, args.operator_reproduction_sha256, gates)
        report['accounting_admission'] = gates.verify_post_v19_registry(args.registry_reference,
            args.registry_reference_sha256, args.accounting_queue, closure)
        common.validate_registry(args.registry_reference, args.registry_reference_sha256, args.accounting_queue)
        checker = common.BudgetChecker(args.budget_argv_json, args.budget_argv_json_sha256)
        report['last_budget_snapshot'] = checker.check()
        common.atomic_json(args.out, report)
        run_gpu(args, graph, report, checker, common)
        return 0
    except BaseException as exc:
        if report['status'] not in ('FAILED', 'BUDGET_STOPPED'):
            report.update(status='BUDGET_STOPPED' if isinstance(exc, common.BudgetStopped) else 'FAILED',
                          error_type=type(exc).__name__, diagnostic=str(exc)[:4000])
            common.atomic_json(args.out, report)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
