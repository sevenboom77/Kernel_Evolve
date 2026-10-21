"""GPU2 whole-model compile/math qualification, without performance samples."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALIGNED_SHA = 'd2a521ec708e1cec85c660e644400ed6660d388503f672a01c7b27d95b28a63f'


def aligned_module():
    path = HERE / 'benchmark_pretrained_aligned_head.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ALIGNED_SHA:
        raise ValueError('original aligned benchmark changed')
    spec = importlib.util.spec_from_file_location('unchanged_aligned_head', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_original_aligned():
    module = aligned_module()
    return module.adapt(module.load_frozen_base())


def make_protocol(base):
    protocol = copy.deepcopy(base.make_protocol())
    protocol.update(name='pretrained-gpt2-aligned-head-auxiliary-math-r1',
                    timing=None, performance_measured=False,
                    allowed_gpu_indices=[2], clean_timing_claim_permitted=False,
                    resident_process_policy='Known authorized residents retained; no clean timing claim.')
    return protocol


def admit_snapshot(snapshot, gpu, *, before_context):
    if (gpu != 2 or snapshot.get('index') != gpu
            or snapshot.get('available') is not True
            or not isinstance(snapshot.get('uuid'), str)
            or snapshot.get('unknown_processes') != 0):
        raise ValueError('auxiliary device state or known-process binding unavailable')
    if before_context:
        total, used = snapshot.get('mem_total_mib'), snapshot.get('mem_used_mib')
        if (type(total) not in (int, float) or type(used) not in (int, float)
                or not 0 <= used <= total or total - used < 10000
                or snapshot.get('util') != 0):
            raise ValueError('auxiliary preflight needs idle device and 10000MiB capacity')
    return snapshot


def run_gpu(base, args, binding, report, checker, tokens):
    from kernel_lab import gpuinfo
    torch = base.torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with base.wall_limit(900), base.NumericGpuLock(base.ROOT, args.gpu):
        preflight = admit_snapshot(gpuinfo.snapshot(args.gpu), args.gpu, before_context=True)
        checker.check()
        with base.ApplicationJob(args.accounting_queue, args.run_id, report['protocol_sha256'],
                                 report['source_package_sha256'], args.gpu, 900) as job:
            report.update(status='RUNNING', measurements={}, gpu_snapshots=[preflight],
                          performance_measured=False, speedup_claimed=False,
                          primary_AGH_promotion_permitted=False, RSI_proven=False,
                          clean_timing_claim_permitted=False,
                          accounting={'queue': str(args.accounting_queue.resolve()),
                                      'run_id': args.run_id, 'job_id': job.job['job_id'],
                                      'gpu_physical_index': args.gpu,
                                      'gpu_span_semantics': 'official claim-to-finish conservative wall time'})
            job.result['report_path'] = str(args.out.resolve())
            try:
                with base.cuda_completion(), torch._dynamo.config.patch(
                        cache_size_limit=32, accumulated_cache_size_limit=256):
                    if not torch.cuda.is_available():
                        raise base.BenchmarkError('CUDA unavailable after official auxiliary admission')
                    torch.set_num_threads(2)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                    torch.set_float32_matmul_precision('highest')
                    torch.cuda.set_device(0)
                    with torch.inference_mode(False), torch.no_grad():
                        model = base.model_app.load_model(args.model_dir).cuda()
                        if sum(p.numel() for p in model.parameters()) != 124439808:
                            raise AssertionError('full pretrained model parameter count differs')
                        original = base.parameter_sha(model)
                        variants, setup = aligned_module().prepare_models(base, model)
                    physical = admit_snapshot(gpuinfo.snapshot(args.gpu), args.gpu, before_context=False)
                    if os.getpid() not in physical.get('own_pids_present', []):
                        raise base.BenchmarkError('own CUDA context absent from registered GPU2')
                    report.update(parameter_count=124439808, head_preparation=setup,
                                  actual_physical_gpu_binding={'physical_gpu': args.gpu,
                                                              'uuid': physical['uuid'], 'pid': os.getpid()},
                                  actual_K02_Triton_executed=False, original_backbone_preserved=True)
                    blocks = {'fp16_library': base.Logits(model),
                              **{name: base.Logits(m) for name, m in variants.items()}}

                    def checkpoint():
                        report['last_budget_snapshot'] = checker.check()
                        report['gpu_snapshots'].append(admit_snapshot(
                            gpuinfo.snapshot(args.gpu), args.gpu, before_context=False))
                        if not job.queue.heartbeat(job.job['job_id'], job.job['lease_owner'], 1020):
                            raise base.BenchmarkError('auxiliary accounting lease lost')

                    with torch.inference_mode():
                        for label in report['selected_case_labels']:
                            checkpoint()
                            ids = tokens[label].cuda()
                            row = {'status': 'RUNNING', 'backend_qualification': {},
                                   'input_token_sha256': base.tensor_sha(ids),
                                   'parameter_sha256': original}
                            report['measurements'][label] = row
                            reference = blocks['fp16_library'](ids)
                            for name in aligned_module().BACKENDS:
                                checkpoint()
                                source = 'fp16_library' if name in base.LIBRARIES else name.removeprefix('compiled_')
                                _, receipt = base.qualify_backend(blocks[source], ids, reference,
                                                                 compiled=name.startswith('compiled_'))
                                row['backend_qualification'][name] = receipt
                                base.atomic_json(args.out, report)
                            del reference
                            if base.tensor_sha(ids) != report['source_package']['input_token_sha256'][label]:
                                raise AssertionError('qualified token IDs mutated')
                            for candidate in variants.values():
                                candidate.lm_head.validate_cache_binding()
                            row.update(status='QUALIFIED_WITHOUT_TIMING',
                                       all_six_backends_available=all(
                                           r['status'] == 'AVAILABLE' for r in row['backend_qualification'].values()))
                            base.atomic_json(args.out, report)
                        checkpoint()
                        if (base.parameter_sha(model) != original
                                or any(base.parameter_sha(m) != original for m in variants.values())
                                or base.model_app.model_receipt(args.model_dir) != report['model_receipt']):
                            raise AssertionError('original pretrained parameters/files mutated')
                        report.update(status='QUALIFIED_WITHOUT_TIMING', completed_at=base.utcnow(),
                                      all_pretrained_weights_preserved=True, parameter_sha256=original,
                                      all_cache_bindings_preserved=True,
                                      all_six_backends_available=all(
                                          r['all_six_backends_available'] for r in report['measurements'].values()))
            except BaseException as exc:
                report.update(status='BUDGET_STOPPED' if isinstance(exc, base.BudgetStopped) else 'FAILED',
                              error_type=type(exc).__name__, error=str(exc)[:4000], completed_at=base.utcnow())
                raise
            finally:
                base.atomic_json(args.out, report)
                job.result.update(report_sha256=base.model_app.sha(args.out), report_status=report['status'],
                                  performance_measured=False)


def main(argv=None):
    base = load_original_aligned()
    protocol = make_protocol(base)
    original_package = base.source_package
    base.make_protocol = lambda: copy.deepcopy(protocol)

    def source_package(*args):
        package, _ = original_package(*args)
        package['application_sources'][str(Path(__file__).relative_to(base.ROOT))] = base.model_app.sha(__file__)
        return package, hashlib.sha256(base.canonical(package)).hexdigest()

    base.source_package = source_package
    base.run_gpu = lambda *args: run_gpu(base, *args)
    return base.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
