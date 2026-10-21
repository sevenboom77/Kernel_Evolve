"""Separate common native-LayerNorm compiler profile, preserving v1 results."""
import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS_SHA = '5d5ad1a5f5a16e4e7ae7731eda278d412e8d023f5285e211a4ed7c0839499556'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def previous():
    path = HERE / 'benchmark_pretrained_gpt2_welford.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PREVIOUS_SHA:
        raise ValueError('original Welford performance source changed')
    return load('frozen_Welford_performance_v1', path)


def interface():
    name = 'kernelevolve_welford_compiled_runtime_v2'
    path = HERE / 'fused_welford_compiled.py'
    if name not in sys.modules:
        sys.modules[name] = load(name, path)
    module = sys.modules[name]
    if Path(module.__file__).resolve() != path:
        raise ValueError('compiled interface belongs to another source path')
    return module


def compiler_protocol(base, *, numerical_only):
    protocol = copy.deepcopy(base.make_protocol())
    protocol.update(name='pretrained-gpt2-Welford-native-LayerNorm-compiler-v2',
                    optimization='Original Welford CUDA through a registered custom operator',
                    compiler_profile_is_distinct_from_v1=True,
                    original_v1_compiler_failures_preserved=True,
                    performance_measured=False)
    protocol['compile'].update(emulate_precision_casts=True,
                              native_layernorm_both_compiled_backends=True,
                              native_layernorm='Unchanged stock PyTorch FP16 LayerNorm behind a CUDA-only custom operator')
    if numerical_only:
        protocol.update(name=protocol['name'] + '-numerical-qualification', timing=None,
                        strong_gain=None, performance_measured=False)
    return protocol


def verify_compiled_qualification(path, sha256):
    old = previous()
    value = old.bound({'path': str(Path(path).resolve()), 'sha256': sha256})
    if (value.get('status') != 'ALL_COMPILED_NUMERICAL_PARTITIONS_CLOSED'
            or value.get('exit_code') != 0 or value.get('all_four_compiler_profile_cases_correct') is not True
            or value.get('owned_RPC_formally_closed') is not True
            or value.get('foreign_processes_signaled') != 0):
        raise ValueError('completed compiled numerical qualification required')
    freeze = old.bound(value['source_freeze'])
    for ref in freeze['sources']:
        if old.reference(ref['path']) != ref:
            raise ValueError('compiled qualification source changed')
    labels = []
    for item in value['reports']:
        report = old.bound(item['report'])
        if report['status'] != 'NUMERICAL_QUALIFIED_CLEAN' or report.get('performance_measured') is not False:
            raise ValueError('compiled numerical qualification required before timing')
        for label, row in report['measurements'].items():
            labels.append(label)
            if (row.get('samples') or len(row['fused_Welford_checks']) != 12
                    or any(q['status'] != 'AVAILABLE' for q in row['backend_qualification'].values())
                    or set(row['backend_qualification']) != {'fp16_library', 'compiled_fp16', 'candidate', 'compiled_candidate'}):
                raise ValueError('all four original backends and twelve checks must qualify')
    if sorted(labels) != ['B1_S128', 'B1_S33', 'B1_S512', 'B2_S1024']:
        raise ValueError('all four compiled numerical qualification cases required')
    return {'execution': {'path': str(Path(path).resolve()), 'sha256': sha256},
            'all_four_compiler_profile_cases_correct': True}


def adapt(base, old_qualification, *, numerical_only, compiled_qualification=None):
    old = previous()
    protocol = compiler_protocol(base, numerical_only=numerical_only)
    base = old.adapt_private_base(base, old_qualification)
    make_candidate = base.model_app.make_candidate
    base.model_app.make_candidate = lambda model, binding=None, reference_only=False: (
        make_candidate(model, binding, reference_only=True) if reference_only
        else interface().make_candidate(model))
    base.model_app.KernelResidualLayerNorm = lambda binding, eps: interface().ResidualLayerNorm(eps)
    base.make_protocol = lambda: copy.deepcopy(protocol)
    source_package, make_plan, qualify_backend, atomic_json = (
        base.source_package, base.make_plan, base.qualify_backend, base.atomic_json)

    def package(*args):
        value, _ = source_package(*args)
        for source in (Path(__file__).resolve(), HERE / 'fused_welford_compiled.py'):
            value['application_sources'][str(source.relative_to(base.ROOT))] = old.reference(source)['sha256']
        import torch._inductor.config as config
        value['compiler_precision_profile'] = {
            'emulate_precision_casts': True, 'native_layernorm_both_compiled_backends': True,
            'installed_config_source_sha256': old.reference(config.__file__)['sha256']}
        return value, hashlib.sha256(base.canonical(value)).hexdigest()

    def plan(*args):
        value = make_plan(*args)
        value.update(kind='source-bound-pretrained-Welford-compiler-v2',
                     numerical_only=numerical_only, compiled_qualification=compiled_qualification,
                     compiler_profile_is_distinct_from_v1=True)
        return value

    def qualify(block, ids, reference, *, compiled):
        prepared = interface().prepare_compiled_model(block) if compiled else block
        if base.parameter_sha(prepared) != base.parameter_sha(block):
            raise ValueError('common compiler profile changed parameter bytes')
        import torch._inductor.config as config
        with config.patch({'emulate_precision_casts': True}):
            fn, receipt = qualify_backend(prepared, ids, reference, compiled=compiled)
        receipt.update(native_layernorm_interface=compiled, parameter_bytes_preserved=True,
                       emulate_precision_casts=compiled)
        return fn, receipt

    def atomic(path, report):
        if numerical_only:
            if report.get('status') == 'MEASURED_CLEAN':
                report['status'] = 'NUMERICAL_QUALIFIED_CLEAN'
            for row in (report.get('measurements') or {}).values():
                if row.get('status') == 'MEASURED_CLEAN':
                    row['status'] = 'NUMERICAL_QUALIFIED_CLEAN'
            report['performance_measured'] = False
        return atomic_json(path, report)

    base.source_package, base.make_plan = package, plan
    base.qualify_backend, base.atomic_json = qualify, atomic
    if numerical_only:
        base.measure_paired = lambda *args, **kwargs: None
        base.paired_gain = lambda *args, **kwargs: None
        base.strong_gain = lambda *args, **kwargs: None
    return base


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--numerical-only', action='store_true')
    parser.add_argument('--compiled-qualification', type=Path)
    parser.add_argument('--compiled-qualification-sha256')
    parser.add_argument('--numerical-qualification', type=Path)
    parser.add_argument('--numerical-qualification-sha256')
    args, remaining = parser.parse_known_args(argv)
    old_qualification = compiled_qualification = None
    if '--execute' in remaining:
        if not args.numerical_only:
            if args.compiled_qualification is None or args.compiled_qualification_sha256 is None:
                raise ValueError('completed compiled numerical qualification required before timing')
            compiled_qualification = verify_compiled_qualification(
                args.compiled_qualification, args.compiled_qualification_sha256)
        if args.numerical_qualification is None or args.numerical_qualification_sha256 is None:
            raise ValueError('original four-case Welford numerical qualification required')
        old_qualification = previous().verify_numerical_qualification(
            args.numerical_qualification, args.numerical_qualification_sha256)
    base = previous().load_diagnostic().load_frozen_base()
    return adapt(base, old_qualification, numerical_only=args.numerical_only,
                 compiled_qualification=compiled_qualification).main(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
