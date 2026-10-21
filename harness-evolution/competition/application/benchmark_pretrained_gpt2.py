"""Paired whole-model GPT-2 inference with the original, verified H0 K02."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

APP = Path(__file__).resolve().parent
ROOT = APP.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

import torch
from kernel_lab import timing
from kernel_lab.util import utcnow
import pretrained_gpt2 as model_app
from benchmark_decoder_v3 import (ApplicationJob, BenchmarkError, BudgetChecker,
    BudgetStopped, NumericGpuLock, atomic_json, canonical, clean_snapshot,
    cuda_completion, measure_paired, paired_gain, validate_registry,
    verify_base_sources, wall_limit)

LIBRARIES = ('fp16_library', 'compiled_fp16')
BACKENDS = (*LIBRARIES, 'candidate', 'compiled_candidate')
LOGIT_TOLERANCE = {'rtol': .005, 'atol': .05}
KERNEL_TOLERANCE = {'rtol': .002, 'atol': .01}
BASE_SHA = 'd5d38b76c3ab25d4257f1042e19397bda373dbab90d76830e237f9eaf488c685'


def make_protocol():
    return {'schema_version': 1, 'name': 'pretrained-gpt2-k02-full-forward-v1',
        'model': 'gpt2', 'revision': model_app.MODEL_REVISION,
        'cases': [{'batch': b, 'sequence': s} for b, s in model_app.SHAPES],
        'case_labels': [f'B{b}_S{s}' for b, s in model_app.SHAPES],
        'requested_backends': list(BACKENDS), 'strong_baseline_required': list(LIBRARIES),
        'use_cache': False, 'output_attentions': False, 'attention_implementation': 'sdpa',
        'workload': 'teacher-forced full pretrained language-model forward, including logits',
        'text_quality_measured': False, 'pretrained_parameters': 124439808,
        'kernel_tolerance': KERNEL_TOLERANCE, 'logit_tolerance': LOGIT_TOLERANCE,
        'compile': {'mode': 'default', 'fullgraph': True, 'dynamic': False,
                    'cache_size_limit': 32, 'accumulated_cache_size_limit': 256},
        'compilation_failure_policy': 'record unavailable/incorrect; no fallback or strong-baseline victory',
        'timing': {'backend': 'cuda_events', 'paired_rounds': 30, 'warmup': 20,
                   'order_seed': 20261006, 'target_batch_ms': 20., 'max_iters': 256,
                   'jit_compile_upload_validation_timed': False, 'cuda_graphs': False},
        'max_wall_seconds': 900, 'parameter_input_policy': 'identical frozen pretrained weights and token IDs',
        'strong_gain': 'per-round minimum of both qualified libraries divided by candidate',
        'optimization': 'attention residual plus ln_2 fused by original H0 K02 in all 12 blocks',
        'unsupported': ['training', 'past/cache', 'cross-attention', 'output attentions',
                         'nonfinite normalization inputs', 'changed model/kernel source']}


def strong_gain(samples, candidate):
    if any(name not in samples for name in (*LIBRARIES, candidate)):
        return None
    vectors = [samples[name]['gpu_us_samples'] for name in (*LIBRARIES, candidate)]
    if any(len(v) != 30 for v in vectors) or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
            for v in vectors for x in v):
        raise ValueError('all 30 paired samples must be finite and positive')
    minima = [min(a, b) for a, b in zip(*vectors[:2])]
    ratios = [a / b for a, b in zip(minima, vectors[-1])]
    rng = random.Random(20261006)
    boot = sorted(statistics.median(rng.choices(ratios, k=30)) for _ in range(2000))
    return {'required_library_backends': list(LIBRARIES), 'paired_rounds': 30,
            'strong_baseline_us_samples': minima, 'paired_ratio_samples': ratios,
            'median': statistics.median(ratios), 'bootstrap_95_percent_interval': [boot[49], boot[1949]],
            'bootstrap_seed': 20261006, 'bootstrap_resamples': 2000, 'victory_proven': boot[49] > 1.}


def tensor_sha(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(canonical({'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}))
        digest.update(bytes.fromhex(tensor_sha(tensor)))
    return digest.hexdigest()


def source_package(binding, model_receipt, tokens, protocol):
    verify_base_sources()
    if model_app.sha(APP / 'benchmark_decoder_v3.py') != BASE_SHA:
        raise BenchmarkError('original accounting/timing implementation changed')
    # Hash installed source without executing its optional-backend detection.
    # Importing modeling_gpt2 here queries CUDA before the accounting admission
    # and before run_gpu fixes device visibility.
    modeling_source = importlib.metadata.distribution('transformers').locate_file(
        'transformers/models/gpt2/modeling_gpt2.py')
    files = [APP / 'pretrained_gpt2.py', Path(__file__), APP / 'benchmark_decoder_v3.py']
    package = {'application_sources': {str(p.relative_to(ROOT)): model_app.sha(p) for p in files},
        'model': {key: model_receipt[key] for key in ('model', 'revision', 'files')},
        'operator_manifest_sha256': binding['manifest']['sha256'],
        'operator_kernel_sha256': binding['source']['sha256'],
        'candidate_id': binding['candidate_id'], 'verify_job_id': binding['verify_job_id'],
        'input_token_sha256': {label: tensor_sha(value) for label, value in tokens.items()},
        'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
        'tokenizers': importlib.metadata.version('tokenizers'),
        'transformers_gpt2_source_sha256': model_app.sha(modeling_source),
        'protocol_sha256': hashlib.sha256(canonical(protocol)).hexdigest()}
    return package, hashlib.sha256(canonical(package)).hexdigest()


def make_plan(binding, receipt, tokens, partition):
    protocol = make_protocol()
    package, package_sha = source_package(binding, receipt, tokens, protocol)
    labels = protocol['case_labels']
    selected = labels if partition == 'all' else labels[int(partition) * 2:int(partition) * 2 + 2]
    return {'schema_version': 1, 'kind': 'source-bound-pretrained-gpt2-application',
        'status': 'GPU_PENDING', 'created_at': utcnow(), 'pretrained_model': True,
        'text_quality_measured': False, 'harness_performance_causality_proven': False,
        'operator_sources': binding, 'model_receipt': receipt, 'source_package': package,
        'source_package_sha256': package_sha, 'protocol': protocol,
        'protocol_sha256': package['protocol_sha256'], 'execution_partition': partition,
        'selected_case_labels': selected, 'measurements': None,
        'limits': ['A CPU plan does not demonstrate actual Triton execution or GPU speed.',
                   'A partition does not cover all four declared shapes.',
                   'Application performance cannot promote the primary AGH comparison.',
                   'No generated-text quality, training, or cached decode claim.']}


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(input_ids=ids, use_cache=False, output_attentions=False, return_dict=False)[0]


class LogitMismatch(AssertionError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def correctness(actual, reference):
    if actual.shape != reference.shape or not torch.isfinite(actual).all().item():
        raise AssertionError('nonfinite or wrong-shaped full pretrained logits')
    delta = (actual.float() - reference.float()).abs()
    evidence = {'shape': list(actual.shape), 'max_absolute_error': delta.max().item(),
        'mean_squared_error': delta.square().mean().item(), 'tolerance': LOGIT_TOLERANCE,
        'all_token_argmax_agreement': (actual.argmax(-1) == reference.argmax(-1)).float().mean().item(),
        'last_token_argmax_agreement': (actual[:, -1].argmax(-1) == reference[:, -1].argmax(-1)).float().mean().item()}
    try:
        torch.testing.assert_close(actual, reference, **LOGIT_TOLERANCE, equal_nan=False)
    except AssertionError as exc:
        raise LogitMismatch(str(exc), evidence) from exc
    return evidence


def qualify_backend(block, ids, reference, *, compiled):
    started = time.monotonic()
    receipt = {'compiled': compiled, 'fullgraph': compiled, 'mode': 'default' if compiled else None}
    try:
        fn = torch.compile(block, mode='default', fullgraph=True, dynamic=False) if compiled else block
        actual = fn(ids)
        torch.cuda.synchronize()
        receipt.update(status='AVAILABLE', correctness=correctness(actual, reference),
                       qualification_seconds=time.monotonic() - started)
        return lambda: fn(ids), receipt
    except BudgetStopped:
        raise
    except Exception as exc:
        receipt.update(status='INCORRECT' if isinstance(exc, AssertionError) else 'UNAVAILABLE',
                       error_type=type(exc).__name__, diagnostic=str(exc)[:4000],
                       qualification_seconds=time.monotonic() - started)
        if isinstance(exc, LogitMismatch):
            receipt['correctness'] = exc.evidence
        return None, receipt


def inspect_kernels(candidate, ids):
    """Check real pretrained inputs at each of the twelve unchanged K02 calls."""
    receipts, handles = [], []
    def hook(module, inputs, outputs):
        x, residual, gamma, beta, eps = inputs
        model_app.validate_norm_inputs(x, residual, gamma, beta)
        expected = model_app.reference_residual_layernorm(*inputs)
        row = {'block_index': len(receipts), 'shape': list(x.shape), 'epsilon': eps,
               'input_sha256': [tensor_sha(value) for value in inputs[:4]],
               'kernel_tolerance': KERNEL_TOLERANCE, 'max_absolute_errors': []}
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, **KERNEL_TOLERANCE, equal_nan=False)
            if not torch.isfinite(actual).all().item():
                raise AssertionError('nonfinite original K02 output')
            row['max_absolute_errors'].append((actual.float() - reference.float()).abs().max().item())
        receipts.append(row)
    try:
        for block in candidate.transformer.h:
            handles.append(block.normalize.register_forward_hook(hook))
        Logits(candidate)(ids)
    finally:
        for handle in handles:
            handle.remove()
    if len(receipts) != 12:
        raise AssertionError('all twelve original K02 calls must be checked')
    return receipts


def verify_device_binding(gpu):
    snapshot = clean_snapshot(gpu)
    if os.getpid() not in snapshot.get('own_pids_present', []):
        raise BenchmarkError('CUDA context absent from the declared physical GPU')
    return {'physical_gpu': gpu, 'uuid': snapshot['uuid'], 'pid': os.getpid()}


def run_gpu(args, binding, report, checker, tokens):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with wall_limit(900), NumericGpuLock(ROOT, args.gpu):
        preflight = clean_snapshot(args.gpu)
        checker.check()
        with ApplicationJob(args.accounting_queue, args.run_id, report['protocol_sha256'],
                            report['source_package_sha256'], args.gpu, 900) as job:
            report.update(status='RUNNING', measurements={}, gpu_snapshots=[preflight],
                accounting={'queue': str(args.accounting_queue.resolve()), 'run_id': args.run_id,
                            'job_id': job.job['job_id'], 'gpu_physical_index': args.gpu,
                            'gpu_span_semantics': 'official claim-to-finish conservative wall time'})
            job.result['report_path'] = str(args.out.resolve())
            try:
                with cuda_completion(), torch.inference_mode(), torch._dynamo.config.patch(
                        cache_size_limit=32, accumulated_cache_size_limit=256):
                    if not torch.cuda.is_available():
                        raise BenchmarkError('CUDA unavailable after official admission')
                    torch.set_num_threads(2)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                    torch.set_float32_matmul_precision('highest')
                    torch.cuda.set_device(0)
                    props = torch.cuda.get_device_properties(0)
                    report['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'python': sys.version.split()[0], 'device_name': props.name,
                        'physical_gpu': args.gpu, 'pid': os.getpid(), 'tf32': False,
                        'fp16_reduced_precision_reduction': False, 'compiler': report['protocol']['compile']}
                    model = model_app.load_model(args.model_dir)
                    report['parameter_count'] = sum(p.numel() for p in model.parameters())
                    if report['parameter_count'] != 124439808:
                        raise AssertionError('full pretrained model parameter count differs')
                    original_weight_sha = parameter_sha(model)
                    candidate = model_app.make_candidate(model, binding)
                    if parameter_sha(candidate) != original_weight_sha:
                        raise AssertionError('candidate pretrained parameter bytes differ')
                    model, candidate = model.cuda(), candidate.cuda()
                    report['actual_physical_gpu_binding'] = verify_device_binding(args.gpu)
                    blocks = {'fp16_library': Logits(model), 'candidate': Logits(candidate)}
                    def checkpoint():
                        report['last_budget_snapshot'] = checker.check()
                        report['gpu_snapshots'].append(clean_snapshot(args.gpu))
                        if not job.queue.heartbeat(job.job['job_id'], job.job['lease_owner'], 1020):
                            raise BenchmarkError('application accounting lease lost')
                    for label in report['selected_case_labels']:
                        index = report['protocol']['case_labels'].index(label)
                        checkpoint()
                        ids = tokens[label].cuda()
                        row = {'config': report['protocol']['cases'][index], 'status': 'RUNNING',
                               'samples': {}, 'backend_qualification': {},
                               'input_token_sha256': tensor_sha(ids), 'parameter_sha256': original_weight_sha}
                        report['measurements'][label] = row
                        atomic_json(args.out, report)
                        reference = blocks['fp16_library'](ids)
                        row['original_k02_checks'] = inspect_kernels(candidate, ids)
                        fns = {}
                        for name in BACKENDS:
                            checkpoint()
                            source = 'fp16_library' if name in LIBRARIES else 'candidate'
                            fn, qualification = qualify_backend(blocks[source], ids, reference,
                                                               compiled=name.startswith('compiled_'))
                            row['backend_qualification'][name] = qualification
                            if fn is not None:
                                fns[name] = fn
                            atomic_json(args.out, report)
                        del reference
                        timing.configure(True)
                        measure_paired(fns, 20, 30, 20261006 + index, checkpoint, partial=row['samples'])
                        checkpoint()
                        if tensor_sha(ids) != report['source_package']['input_token_sha256'][label]:
                            raise AssertionError('timed token IDs mutated')
                        row['candidate_gains'] = {}
                        for name in ('candidate', 'compiled_candidate'):
                            row['candidate_gains'][name] = None if name not in fns else {
                                'against_fp16_library': paired_gain(row['samples'], 'fp16_library', name)
                                    if 'fp16_library' in fns else None,
                                'against_strong_library': strong_gain(row['samples'], name)}
                        row.update(status='MEASURED_CLEAN', strong_baseline_complete=all(k in fns for k in LIBRARIES))
                        atomic_json(args.out, report)
                    checkpoint()
                    if (parameter_sha(model) != original_weight_sha or parameter_sha(candidate) != original_weight_sha
                            or model_app.model_receipt(args.model_dir) != report['model_receipt']):
                        raise AssertionError('timed pretrained model parameters or original model files mutated')
                    report.update(status='MEASURED_CLEAN', completed_at=utcnow(), all_pretrained_weights_preserved=True,
                        actual_K02_Triton_executed=True, parameter_sha256=original_weight_sha,
                        all_strong_baselines_complete=all(r['strong_baseline_complete'] for r in report['measurements'].values()))
            except BaseException as exc:
                report.update(status='BUDGET_STOPPED' if isinstance(exc, BudgetStopped) else 'FAILED',
                              error_type=type(exc).__name__, error=str(exc)[:4000], completed_at=utcnow())
                raise
            finally:
                atomic_json(args.out, report)
                job.result.update(report_sha256=model_app.sha(args.out), report_status=report['status'],
                                  measured_cases=sum(r['status'] == 'MEASURED_CLEAN' for r in report['measurements'].values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partition', choices=('all', '0', '1'), default='all')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(8), default=0)
    parser.add_argument('--run-id', default='pretrained-gpt2-k02-full-forward-v1')
    parser.add_argument('--operator-manifest', type=Path, default=model_app.DEFAULT_MANIFEST)
    parser.add_argument('--operator-manifest-sha256', default=model_app.DEFAULT_MANIFEST_SHA256)
    parser.add_argument('--accounting-queue', type=Path)
    parser.add_argument('--registry-reference', type=Path)
    parser.add_argument('--registry-sha256')
    parser.add_argument('--budget-argv-json', type=Path)
    parser.add_argument('--budget-argv-sha256')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('output must be fresh; preserve previous attempts')
    if args.execute and any(getattr(args, name) is None for name in
            ('accounting_queue', 'registry_reference', 'registry_sha256', 'budget_argv_json', 'budget_argv_sha256')):
        parser.error('GPU execution requires registered accounting and original budget command bindings')
    os.environ['HF_HUB_OFFLINE'] = '1'
    binding = model_app.verify_kernel(args.operator_manifest, args.operator_manifest_sha256)
    receipt = model_app.model_receipt(args.model_dir)
    tokens = model_app.input_ids(args.model_dir)
    report = make_plan(binding, receipt, tokens, args.partition)
    if not args.execute:
        atomic_json(args.out, report)
        return 0
    report['budget_bindings'] = {'registry': str(args.registry_reference.resolve()), 'registry_sha256': args.registry_sha256,
        'argv': str(args.budget_argv_json.resolve()), 'argv_sha256': args.budget_argv_sha256}
    checker = BudgetChecker(args.budget_argv_json, args.budget_argv_sha256)
    try:
        validate_registry(args.registry_reference, args.registry_sha256, args.accounting_queue)
        report['last_budget_snapshot'] = checker.check()
        atomic_json(args.out, report)
        run_gpu(args, binding, report, checker, tokens)
        return 0
    except BaseException as exc:
        if report['status'] not in ('FAILED', 'BUDGET_STOPPED'):
            report.update(status='BUDGET_STOPPED' if isinstance(exc, BudgetStopped) else 'FAILED',
                          error_type=type(exc).__name__, error=str(exc)[:4000], completed_at=utcnow())
            atomic_json(args.out, report)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
