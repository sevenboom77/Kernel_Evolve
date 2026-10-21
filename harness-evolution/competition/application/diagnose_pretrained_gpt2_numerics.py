"""Locate pretrained numerical errors; no timings or performance promotion."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL_BENCHMARK_SHA256 = 'f2152dcaf06bb643df35b50292b3fcd0399457a9771e7420c071860bd90ca1a9'
ORIGINAL_READER_SHA256 = 'a36381226052accd2cac1dbd592df2e25b67fa13717f46d0435767b3d52de481'


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def bound(ref):
    if reference(ref['path']) != ref:
        raise ValueError('source or evidence binding changed')
    return json.loads(Path(ref['path']).read_bytes())


def load_frozen_base():
    path = HERE / 'benchmark_pretrained_gpt2.py'
    if reference(path)['sha256'] != ORIGINAL_BENCHMARK_SHA256:
        raise ValueError('original pretrained benchmark changed')
    original_paths = list(sys.path)
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location('private_pretrained_numerical_base', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_paths


def diagnostic_protocol(base):
    protocol = copy.deepcopy(base.make_protocol())
    protocol.update(name='pretrained-gpt2-original-k02-numerical-diagnosis-v3',
                    requested_backends=['fp16_library', 'reference_glue', 'original_K02'],
                    timing=None, compile=None, strong_baseline_required=[], strong_gain=None,
                    performance_measured=False,
                    purpose='Original weights and inputs; reference wiring, twelve real norm inputs, and layerwise numerical error.',
                    optimization='No new operator or compiler profile; diagnose the original K02.')
    return protocol


def verify_closed_primary(path, sha256):
    completed_ref = {'path': str(Path(path).resolve()), 'sha256': sha256}
    done = bound(completed_ref)
    if done.get('status') != 'EIGHT_ARMS_CLOSED_AND_AUDITED':
        raise ValueError('all eight primary arms must be closed')
    audit, index = bound(done['final_audit']), bound(done['final_index'])
    if (audit.get('closed_arm_count') != 8 or len(audit.get('arms', [])) != 8
            or any(arm.get('execution_status') != 'CLOSED' or len(arm.get('rows', [])) != 20
                   or not all(row.get('terminal') is True for row in arm['rows'])
                   for arm in audit['arms'])):
        raise ValueError('all eight twenty-task primary arms must be closed')
    series = bound(audit['series'])
    reader = HERE.parents[1] / 'codex-rsi/audit_measurement_series_campaign_v16.py'
    if (reference(reader)['sha256'] != ORIGINAL_READER_SHA256
            or index['series'] != audit['series'] or series['reliability_revision'] != 16
            or len(index['arms']) != 8):
        raise ValueError('original V16 primary series or reader differs')
    for ref in audit['evidence_references']:
        if reference(ref['path']) != ref:
            raise ValueError('original complete primary evidence changed')
    last = series['sequence'][-1]['batch_id']
    endpoint_ref = index['arms'][last]['budget_after']
    endpoint = bound(endpoint_ref)
    if (endpoint['coverage_complete'] is not True or endpoint['should_stop'] is not False
            or endpoint['active_http_count'] != 0 or endpoint['gpu_active_span_count'] != 0):
        raise ValueError('actual closed primary resource endpoint required')
    return {'completion': completed_ref, 'final_audit': done['final_audit'],
            'final_index': done['final_index'], 'final_primary_endpoint': endpoint_ref,
            'all_eight_primary_arms_closed': True}


def compare_logits(base, actual, expected):
    try:
        return {'status': 'AVAILABLE', 'correctness': base.correctness(actual, expected)}
    except AssertionError as exc:
        result = {'status': 'INCORRECT', 'error_type': type(exc).__name__,
                  'diagnostic': str(exc)[:4000]}
        if isinstance(exc, base.LogitMismatch):
            result['correctness'] = exc.evidence
        return result


def tensor_error(base, actual, expected):
    torch = base.torch
    if actual.shape != expected.shape:
        return {'status': 'WRONG_SHAPE', 'actual_shape': list(actual.shape),
                'expected_shape': list(expected.shape)}
    if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
        return {'status': 'NONFINITE', 'shape': list(actual.shape)}
    delta = (actual.float() - expected.float()).abs()
    return {'status': 'FINITE', 'shape': list(actual.shape),
            'max_absolute_error': delta.max().item(),
            'mean_squared_error': delta.square().mean().item(),
            'bitwise_equal': torch.equal(actual, expected)}


def observe_case(base, library, glue, candidate, normalizer, ids, checkpoint):
    """Capture real residual/attention pairs without altering any model output."""
    torch = base.torch
    handles, norm_rows, reference_layers = [], [], []
    pairs = {}

    def pre_block(index):
        def hook(module, inputs):
            pairs[index] = {'residual': inputs[0]}
        return hook

    def post_attention(index):
        def hook(module, inputs, output):
            pairs[index]['attention'] = output[0]
        return hook

    def norm_hook(index):
        def hook(module, inputs, output):
            checkpoint()
            x, residual = pairs[index]['attention'], pairs[index]['residual']
            gamma, beta, eps = module.weight, module.bias, module.eps
            base.model_app.validate_norm_inputs(x, residual, gamma, beta)
            expected_r = (x.float() + residual.float()).half()
            row = {'block_index': index, 'shape': list(x.shape), 'epsilon': eps,
                   'input_sha256': [base.tensor_sha(v) for v in (x, residual, gamma, beta)],
                   'kernel_tolerance': copy.deepcopy(base.KERNEL_TOLERANCE),
                   'original_k02_invocation_returned': False,
                   'reference_residual_bitwise_equal': torch.equal(expected_r, inputs[0])}
            if not row['reference_residual_bitwise_equal']:
                raise AssertionError('reference residual reconstruction differs from real LayerNorm input')
            try:
                actual_y, actual_r = normalizer(x, residual, gamma, beta, eps)
                row['original_k02_invocation_returned'] = True
                row['errors'] = [tensor_error(base, actual_y, output),
                                 tensor_error(base, actual_r, inputs[0])]
                torch.testing.assert_close(actual_y, output, **base.KERNEL_TOLERANCE, equal_nan=False)
                torch.testing.assert_close(actual_r, inputs[0], **base.KERNEL_TOLERANCE, equal_nan=False)
                row['status'] = 'AVAILABLE'
            except base.BudgetStopped:
                raise
            except Exception as exc:
                row.update(status='INCORRECT' if isinstance(exc, AssertionError) else 'UNAVAILABLE',
                           error_type=type(exc).__name__, diagnostic=str(exc)[:4000])
            norm_rows.append(row)
        return hook

    def layer_hook(target):
        def hook(module, inputs, output):
            target.append(output[0].detach().clone())
        return hook

    try:
        for index, block in enumerate(library.transformer.h):
            handles.extend([block.register_forward_pre_hook(pre_block(index)),
                            block.attn.register_forward_hook(post_attention(index)),
                            block.ln_2.register_forward_hook(norm_hook(index)),
                            block.register_forward_hook(layer_hook(reference_layers))])
        expected = base.Logits(library)(ids)
    finally:
        for handle in handles:
            handle.remove()
    if len(norm_rows) != 12 or len(reference_layers) != 12:
        raise AssertionError('all twelve original reference blocks must be inspected')
    result = {'original_k02_checks': norm_rows,
              'backend_qualification': {'fp16_library': compare_logits(base, expected, expected)},
              'layer_output_errors': {}, 'timing_measured': False}
    for name, model in [('reference_glue', glue), ('original_K02', candidate)]:
        observed, handles = [], []
        try:
            for block in model.transformer.h:
                handles.append(block.register_forward_hook(layer_hook(observed)))
            checkpoint()
            try:
                actual = base.Logits(model)(ids)
                result['backend_qualification'][name] = compare_logits(base, actual, expected)
                del actual
            except base.BudgetStopped:
                raise
            except Exception as exc:
                result['backend_qualification'][name] = {'status': 'UNAVAILABLE',
                    'error_type': type(exc).__name__, 'diagnostic': str(exc)[:4000]}
            result['layer_output_errors'][name] = [
                {'block_index': index, **tensor_error(base, value, reference_layers[index])}
                for index, value in enumerate(observed)]
            if result['backend_qualification'][name]['status'] == 'AVAILABLE' and len(observed) != 12:
                raise AssertionError('complete backend must capture twelve block outputs')
        finally:
            for handle in handles:
                handle.remove()
    return result


def run_gpu(base, args, binding, report, checker, tokens):
    torch = base.torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with base.wall_limit(900), base.NumericGpuLock(base.ROOT, args.gpu):
        preflight = base.clean_snapshot(args.gpu)
        checker.check()
        with base.ApplicationJob(args.accounting_queue, args.run_id, report['protocol_sha256'],
                                 report['source_package_sha256'], args.gpu, 900) as job:
            report.update(status='RUNNING', measurements={}, gpu_snapshots=[preflight],
                accounting={'queue': str(args.accounting_queue.resolve()), 'run_id': args.run_id,
                            'job_id': job.job['job_id'], 'gpu_physical_index': args.gpu,
                            'gpu_span_semantics': 'official claim-to-finish conservative wall time'})
            job.result['report_path'] = str(args.out.resolve())
            try:
                with base.cuda_completion(), torch.inference_mode():
                    if not torch.cuda.is_available():
                        raise base.BenchmarkError('CUDA unavailable after registered admission')
                    torch.set_num_threads(2)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                    torch.set_float32_matmul_precision('highest')
                    torch.cuda.set_device(0)
                    library = base.model_app.load_model(args.model_dir)
                    if sum(p.numel() for p in library.parameters()) != 124439808:
                        raise AssertionError('full pretrained parameter count differs')
                    parameter_sha = base.parameter_sha(library)
                    glue = base.model_app.make_candidate(library, reference_only=True)
                    candidate = base.model_app.make_candidate(library, binding)
                    if any(base.parameter_sha(m) != parameter_sha for m in (glue, candidate)):
                        raise AssertionError('pretrained model parameter bytes differ')
                    library, glue, candidate = [m.cuda() for m in (library, glue, candidate)]
                    report['actual_physical_gpu_binding'] = base.verify_device_binding(args.gpu)
                    report['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'physical_gpu': args.gpu, 'pid': os.getpid(), 'tf32': False,
                        'fp16_reduced_precision_reduction': False, 'compiler': None}
                    normalizer = base.model_app.KernelResidualLayerNorm(binding, library.transformer.h[0].ln_2.eps)

                    def checkpoint():
                        report['last_budget_snapshot'] = checker.check()
                        report['gpu_snapshots'].append(base.clean_snapshot(args.gpu))
                        if not job.queue.heartbeat(job.job['job_id'], job.job['lease_owner'], 1020):
                            raise base.BenchmarkError('numerical diagnostic accounting lease lost')

                    for label in report['selected_case_labels']:
                        checkpoint()
                        ids = tokens[label].cuda()
                        row = observe_case(base, library, glue, candidate, normalizer, ids, checkpoint)
                        row.update(status='DIAGNOSED', input_token_sha256=base.tensor_sha(ids),
                                   parameter_sha256=parameter_sha)
                        if row['input_token_sha256'] != report['source_package']['input_token_sha256'][label]:
                            raise AssertionError('diagnostic token IDs changed')
                        report['measurements'][label] = row
                        base.atomic_json(args.out, report)
                        del ids
                    checkpoint()
                    if any(base.parameter_sha(m) != parameter_sha for m in (library, glue, candidate)):
                        raise AssertionError('diagnostic pretrained parameters mutated')
                    if base.model_app.model_receipt(args.model_dir) != report['model_receipt']:
                        raise AssertionError('original pretrained model files changed')
                    if base.model_app.verify_kernel(args.operator_manifest, args.operator_manifest_sha256) != binding:
                        raise AssertionError('original K02 binding changed')
                    report.update(status='DIAGNOSED_CLEAN', completed_at=base.utcnow(),
                                  all_pretrained_weights_preserved=True,
                                  actual_K02_Triton_executed=any(
                                      check['original_k02_invocation_returned']
                                      for row in report['measurements'].values()
                                      for check in row['original_k02_checks']),
                                  parameter_sha256=parameter_sha, parameter_count=124439808)
            except BaseException as exc:
                report.update(status='BUDGET_STOPPED' if isinstance(exc, base.BudgetStopped) else 'FAILED',
                              error_type=type(exc).__name__, error=str(exc)[:4000], completed_at=base.utcnow())
                raise
            finally:
                base.atomic_json(args.out, report)
                job.result.update(report_sha256=base.model_app.sha(args.out), report_status=report['status'],
                                  diagnosed_cases=len(report['measurements']))


def adapt_private_base(base, closure=None):
    protocol = diagnostic_protocol(base)
    original_package, original_plan = base.source_package, base.make_plan
    base.make_protocol = lambda: copy.deepcopy(protocol)

    def source_package(*args):
        package, _ = original_package(*args)
        package['application_sources'][str(Path(__file__).resolve().relative_to(base.ROOT))] = reference(__file__)['sha256']
        return package, hashlib.sha256(base.canonical(package)).hexdigest()

    def make_plan(*args):
        report = original_plan(*args)
        report.update(kind='source-bound-pretrained-gpt2-numerical-diagnostic',
                      performance_measured=False, actual_K02_Triton_executed=False,
                      primary_closure=closure, strong_baseline_victory_proven=False, RSI_proven=False)
        return report

    base.source_package, base.make_plan = source_package, make_plan
    base.run_gpu = lambda *args: run_gpu(base, *args)
    return base


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--primary-series-completion', type=Path)
    parser.add_argument('--primary-series-completion-sha256')
    args, remaining = parser.parse_known_args(argv)
    closure = None
    if '--execute' in remaining:
        if args.primary_series_completion is None or args.primary_series_completion_sha256 is None:
            raise ValueError('GPU diagnosis requires original closed primary series evidence')
        closure = verify_closed_primary(args.primary_series_completion, args.primary_series_completion_sha256)
    return adapt_private_base(load_frozen_base(), closure).main(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
