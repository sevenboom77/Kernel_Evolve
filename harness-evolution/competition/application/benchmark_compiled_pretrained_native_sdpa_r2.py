"""Actual numerical diagnosis through the original accounting/GPU benchmark."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
DIAGNOSTIC_SHA = '49a3ff72e1bb5cf151f4a03e139a13b840f483b080d25caa6fccc0c587c9e027'
COMPILER_BENCHMARK_SHA = '877cc87d5d2896787a9ad3739d22a4583ee615a7fd03cbee1663659166a308d7'
REPAIR_SOURCE_SHA = 'e98651345c3e2021388299f893d6a7337dc741a7e2e619a75bf1355c52c88888'
V19_SERIES_SHA = 'a332fd15d8f49ace45e0acc19dec780afeeb02fdfd284fbc26161c327b8a133e'


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def bound(ref):
    if reference(ref['path']) != ref:
        raise ValueError('Original evidence binding changed')
    return json.loads(Path(ref['path']).read_bytes())


def load_bound(name, filename, expected):
    path = HERE / filename
    if reference(path)['sha256'] != expected:
        raise ValueError('Qualified original diagnostic/compiler source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diagnostic():
    return load_bound('source_bound_actual_first_error',
                      'diagnose_compiled_pretrained_first_error.py', DIAGNOSTIC_SHA)


def repair_interface():
    name = 'source_bound_native_sdpa_ablation'
    path = HERE / 'fused_welford_compiled_sdpa.py'
    if reference(path)['sha256'] != REPAIR_SOURCE_SHA:
        raise ValueError('Qualified native SDPA source changed')
    if name not in sys.modules:
        sys.modules[name] = load_bound(name, path.name, REPAIR_SOURCE_SHA)
    module = sys.modules[name]
    if Path(module.__file__).resolve() != path:
        raise ValueError('Native SDPA interface belongs to another source path')
    return module


def compiler_benchmark():
    return load_bound('source_bound_original_compiler_benchmark',
                      'benchmark_pretrained_gpt2_welford_compiled.py', COMPILER_BENCHMARK_SHA)


def verify_closed_v19(path, sha256):
    completion = {'path': str(Path(path).resolve()), 'sha256': sha256}
    done = bound(completion)
    if done.get('status') != 'EIGHT_ARMS_CLOSED_AND_AUDITED':
        raise ValueError('All eight original V19 arms must close before GPU admission')
    audit, index = bound(done['final_audit']), bound(done['final_index'])
    series = bound(audit['series'])
    if (audit['series']['sha256'] != V19_SERIES_SHA
            or index['series'] != audit['series'] or series.get('reliability_revision') != 19
            or audit.get('closed_arm_count') != 8 or len(index.get('arms', {})) != 8
            or len(audit.get('arms', [])) != 8
            or any(arm.get('execution_status') != 'CLOSED' or len(arm.get('rows', [])) != 20
                   or any(row.get('terminal') is not True for row in arm['rows']) for arm in audit['arms'])):
        raise ValueError('Exact qualified-r3 V19 with all160 original terminal positions required')
    for ref in audit['evidence_references']:
        bound(ref) if str(ref['path']).endswith('.json') else reference_matches(ref)
    endpoint_ref = index['arms'][series['sequence'][-1]['batch_id']]['budget_after']
    endpoint = bound(endpoint_ref)
    if (endpoint.get('coverage_complete') is not True or endpoint.get('should_stop') is not False
            or endpoint.get('active_http_count') != 0 or endpoint.get('gpu_active_span_count') != 0):
        raise ValueError('Actually closed original primary resource endpoint required')
    return {'completion': completion, 'final_audit': done['final_audit'], 'final_index': done['final_index'],
            'final_primary_endpoint': endpoint_ref, 'all_eight_primary_arms_closed': True}


def reference_matches(ref):
    if reference(ref['path']) != ref:
        raise ValueError('Original complete primary evidence changed')


def verify_post_v19_registry(path, sha256, queue, closure):
    registry = bound({'path': str(Path(path).resolve()), 'sha256': sha256})
    prior_ref = registry.get('prior_post_v19_closed_phase')
    if not isinstance(prior_ref, dict):
        raise ValueError('Fresh exact post-V19 carried accounting required')
    prior = bound(prior_ref)
    if (prior.get('primary_completion') != closure['completion']
            or prior.get('pending_requests') != 0 or prior.get('refunds_or_ledger_reset') is not False
            or prior['totals'].get('active_http_count') != 0
            or prior['totals'].get('gpu_active_span_count') != 0
            or prior['totals'].get('coverage_complete') is not True):
        raise ValueError('Full closed V19 cost history must be preserved')
    old_policy = bound(prior['source_policy'])
    queue_path = str(Path(queue).resolve())
    if (Path(registry['ledger_path']).resolve() == Path(old_policy['ledger_path']).resolve()
            or queue_path not in registry['gpu_queue_paths']
            or queue_path in prior['registered_queues']
            or any(registry.get(key) is not None for key in
                   ('max_tokens', 'max_usd', 'max_gpu_seconds', 'deadline_at'))):
        raise ValueError('Fresh registered ledger/queue with the authorized resource scope required')
    return {'registry': {'path': str(Path(path).resolve()), 'sha256': sha256},
            'prior_V19_carry': prior_ref, 'fresh_registered_queue': queue_path,
            'history_refunded': False}


def make_diagnostic_qualifier(original_qualifier, compare_logits, interface, *,
                              checkpoint, compile_factory=None, budget_stop_type=()):
    primitives = diagnostic()
    state = {}

    def qualify(block, ids, reference_logits, *, compiled):
        if compiled and state.get('ids') is not ids:
            raise ValueError('Capture the original library reference before compiled diagnosis')
        fn, receipt = original_qualifier(block, ids, reference_logits, compiled=compiled)
        if not compiled:
            if state.get('ids') is not ids:
                with torch.inference_mode():
                    logits, hidden = primitives.HiddenLogits(block.model)(ids)
                fidelity = primitives.tensor_error(logits, reference_logits)
                if fidelity.get('bitwise_equal') is not True:
                    raise ValueError('Native eager original library instrumentation changed output bytes')
                state.update(ids=ids, hidden=hidden, eager_reference_fidelity=fidelity)
            return fn, receipt
        # Keep the original incorrect/unavailable verdict and its timing refusal.
        try:
            prepared = interface.prepare_compiled_model(block)
            receipt['first_error_diagnostic'] = primitives.observe_backend(
                prepared, ids, reference_logits, state['hidden'], qualify_logits=compare_logits,
                checkpoint=checkpoint, compile_factory=compile_factory)
            receipt['first_error_diagnostic']['eager_library_reference_fidelity'] = state['eager_reference_fidelity']
        except Exception as exc:
            if budget_stop_type and isinstance(exc, budget_stop_type):
                raise
            receipt['first_error_diagnostic'] = {
                'status': 'DIAGNOSTIC_UNAVAILABLE', 'error_type': type(exc).__name__,
                'diagnostic': str(exc)[:4000], 'root_cause_proven': False,
                'performance_measured': False, 'original_verdict_preserved': True}
        return fn, receipt
    return qualify


def adapt(base, compiler, closure):
    original_protocol = base.make_protocol()
    protocol = copy.deepcopy(original_protocol)
    protocol.update(name='pretrained-gpt2-native-sdpa-compiler-ablation-v5', native_sdpa_both_compiled_backends=True, new_arithmetic_kernel=False, full_model_eager_fallback=False,
                    purpose='Preserve uninstrumented verdict; qualify native hidden-state fidelity before first-error interpretation',
                    timing=None, strong_gain=None, performance_measured=False)
    base.make_protocol = lambda: copy.deepcopy(protocol)
    source_package, make_plan, run_gpu = base.source_package, base.make_plan, base.run_gpu
    previous_qualifier = base.qualify_backend
    def original_qualifier(block, ids, reference_logits, *, compiled):
        prepared = repair_interface().prepare_compiled_model(block) if compiled else block
        return previous_qualifier(prepared, ids, reference_logits, compiled=compiled)

    def package(*args):
        value, _ = source_package(*args)
        for source in (Path(__file__).resolve(), HERE / 'diagnose_compiled_pretrained_first_error.py', HERE / 'fused_welford_compiled_addmm.py', HERE / 'fused_welford_compiled_sdpa.py'):
            value['application_sources'][str(source.relative_to(base.ROOT))] = reference(source)['sha256']
        return value, hashlib.sha256(base.canonical(value)).hexdigest()

    def plan(*args):
        value = make_plan(*args)
        value.update(kind='source-bound-native-sdpa-compiler-ablation', native_sdpa_both_compiled_backends=True, new_arithmetic_kernel=False, full_model_eager_fallback=False,
                     primary_v19_closure=closure, performance_measured=False,
                     first_error_primitives=diagnostic().cpu_plan(), root_cause_proven=False,
                     GPU_execution_requires_fresh_carried_accounting=True)
        return value

    def gpu(args, binding, report, checker, tokens):
        if closure is None:
            raise ValueError('Actual full V19 primary closure required')
        fresh = verify_closed_v19(closure['completion']['path'], closure['completion']['sha256'])
        if fresh != closure:
            raise ValueError('Original primary closure changed')
        report['post_v19_accounting_admission'] = verify_post_v19_registry(
            args.registry_reference, args.registry_sha256, args.accounting_queue, closure)
        def checkpoint():
            report['last_budget_snapshot'] = checker.check()
            report['gpu_snapshots'].append(base.clean_snapshot(args.gpu))
        base.qualify_backend = make_diagnostic_qualifier(original_qualifier,
            lambda actual, expected: compiler.previous().load_diagnostic().compare_logits(base, actual, expected),
            repair_interface(), checkpoint=checkpoint, budget_stop_type=base.BudgetStopped)
        try:
            return run_gpu(args, binding, report, checker, tokens)
        finally:
            base.qualify_backend = original_qualifier

    base.source_package, base.make_plan, base.run_gpu = package, plan, gpu
    return base


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--primary-series-completion', type=Path)
    parser.add_argument('--primary-series-completion-sha256')
    parser.add_argument('--numerical-qualification', type=Path)
    parser.add_argument('--numerical-qualification-sha256')
    args, remaining = parser.parse_known_args(argv)
    closure = qualification = None
    compiler = compiler_benchmark()
    diagnostic().cpu_plan()
    if '--execute' in remaining:
        if any(value is None for value in vars(args).values()):
            parser.error('Actual closed primary and unchanged original numerical qualification bindings required')
        closure = verify_closed_v19(args.primary_series_completion, args.primary_series_completion_sha256)
        qualification = compiler.previous().verify_numerical_qualification(
            args.numerical_qualification, args.numerical_qualification_sha256)
    base = compiler.previous().load_diagnostic().load_frozen_base()
    base = compiler.adapt(base, qualification, numerical_only=True)
    return adapt(base, compiler, closure).main(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
