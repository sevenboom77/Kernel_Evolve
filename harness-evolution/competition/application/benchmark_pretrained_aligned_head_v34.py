"""Frozen vanilla GPT-2 full-forward head-layout comparison; CPU plan by default.

Original CUDA accounting, numerical tolerance, paired timing, strong-baseline
mathematics and 900-second limit survive. GPU use requires a fresh registry.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
FROZEN_BENCHMARK_SHA='f2152dcaf06bb643df35b50292b3fcd0399457a9771e7420c071860bd90ca1a9'
BACKENDS=('fp16_library','compiled_fp16','padded_NN','compiled_padded_NN','padded_TN','compiled_padded_TN')


def load_frozen_base():
    path=HERE/'benchmark_pretrained_gpt2.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FROZEN_BENCHMARK_SHA:
        raise ValueError('original pretrained benchmark changed')
    paths=list(sys.path)
    try:
        sys.path.insert(0,str(HERE))
        spec=importlib.util.spec_from_file_location('private_original_aligned_head_basis',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:]=paths


def make_protocol(base):
    protocol=copy.deepcopy(base.make_protocol())
    protocol.update(name='pretrained-gpt2-aligned-head-whole-forward-r2',
        requested_backends=list(BACKENDS),
        optimization='Original trained output head zero-padded to50304 with prepacked NN layout; vanilla pretrained backbone.',
        controls=['unchanged original FP16 library','unchanged fullgraph compiled library','padding-only TN head and compiled TN'],
        output_head={'hidden':768,'original_vocabulary':50257,'padded_vocabulary':50304,
            'extra_zero_columns':47,'candidate_layout':'NN','padding_only_control_layout':'TN',
            'cache_is_nonpersistent':True,'packing_and_hashing_are_setup':True,
            'slice_and_contiguous_copy_inside_every_timed_forward':True,
            'cache_bytes_per_variant':768*50304*2,
            'model_loaded_tracked_outside_inference_mode_after_GPU_admission':True},
        original_K02_or_Welford_replacement_used=False)
    return protocol


FROZEN_OUTPUT_HEAD_SHA='aef9f77646091f902abec3247e66fa515bd83c7654c9474751f26002d65cd34e'


def load_output_head_module():
    path=HERE/'aligned_output_head.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FROZEN_OUTPUT_HEAD_SHA:
        raise ValueError('original aligned output head source changed')
    name='private_actual_output_head'
    existing=sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing,'__file__','')).resolve()!=path.resolve():
            raise ValueError('output head module name belongs to a different source')
        return existing
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    # torch.compile and pickle resolve user classes through their real module.
    sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def prepare_models(base,model):
    module=load_output_head_module()
    original=base.parameter_sha(model);models={};receipts={}
    for layout in ('NN','TN'):
        started=time.monotonic();candidate=copy.deepcopy(model)
        candidate.lm_head=module.AlignedOutputHead(candidate.lm_head,layout=layout)
        candidate.eval()
        if (set(candidate.state_dict())!=set(model.state_dict()) or base.parameter_sha(candidate)!=original):
            raise AssertionError('whole pretrained parameter names or bytes changed')
        # HF GPT-2 ties these exact objects, not merely equal tensors.
        embedding=candidate.get_input_embeddings().weight if hasattr(candidate,'get_input_embeddings') else candidate.wte.weight
        if candidate.lm_head.weight is not embedding:
            raise AssertionError('original tied embedding/output weight alias changed')
        name='padded_'+layout;models[name]=candidate
        receipts[name]={'binding':candidate.lm_head.validate_cache_binding(),
            'packing_copy_hash_validation_setup_seconds':time.monotonic()-started,
            'parameter_sha256':original,'tied_embedding_alias_preserved':True}
    return models,receipts


def run_gpu(base,args,binding,report,checker,tokens):
    torch=base.torch
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    with base.wall_limit(900),base.NumericGpuLock(base.ROOT,args.gpu):
        preflight=base.clean_snapshot(args.gpu);checker.check()
        with base.ApplicationJob(args.accounting_queue,args.run_id,report['protocol_sha256'],
                report['source_package_sha256'],args.gpu,900) as job:
            report.update(status='RUNNING',measurements={},gpu_snapshots=[preflight],
                accounting={'queue':str(args.accounting_queue.resolve()),'run_id':args.run_id,
                    'job_id':job.job['job_id'],'gpu_physical_index':args.gpu,
                    'gpu_span_semantics':'official claim-to-finish conservative wall time'})
            job.result['report_path']=str(args.out.resolve())
            try:
                with base.cuda_completion(),torch._dynamo.config.patch(cache_size_limit=32,accumulated_cache_size_limit=256):
                    if not torch.cuda.is_available():raise base.BenchmarkError('CUDA unavailable after official admission')
                    torch.set_num_threads(2)
                    torch.backends.cuda.matmul.allow_tf32=False
                    torch.backends.cudnn.allow_tf32=False
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
                    torch.set_float32_matmul_precision('highest');torch.cuda.set_device(0)
                    props=torch.cuda.get_device_properties(0)
                    report['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,
                        'python':sys.version.split()[0],'device_name':props.name,'physical_gpu':args.gpu,
                        'pid':os.getpid(),'tf32':False,'fp16_reduced_precision_reduction':False,
                        'compiler':report['protocol']['compile']}
                    # Inference tensors have no version counter. Construct every
                    # original parameter and cache under tracking, after admission.
                    with torch.inference_mode(False),torch.no_grad():
                        started=time.monotonic();model=base.model_app.load_model(args.model_dir).cuda()
                        if sum(p.numel() for p in model.parameters())!=124439808:
                            raise AssertionError('full pretrained model parameter count differs')
                        original=base.parameter_sha(model)
                        variants,setup=prepare_models(base,model)
                        report['model_and_variant_preparation_seconds']=time.monotonic()-started
                    report.update(parameter_count=124439808,head_preparation=setup,
                        actual_physical_gpu_binding=base.verify_device_binding(args.gpu),
                        actual_K02_Triton_executed=False,original_backbone_preserved=True)
                    blocks={'fp16_library':base.Logits(model),**{name:base.Logits(m) for name,m in variants.items()}}
                    def checkpoint():
                        report['last_budget_snapshot']=checker.check()
                        report['gpu_snapshots'].append(base.clean_snapshot(args.gpu))
                        if not job.queue.heartbeat(job.job['job_id'],job.job['lease_owner'],1020):
                            raise base.BenchmarkError('application accounting lease lost')
                    with torch.inference_mode():
                        for label in report['selected_case_labels']:
                            index=report['protocol']['case_labels'].index(label);checkpoint()
                            ids=tokens[label].cuda()
                            row={'config':report['protocol']['cases'][index],'status':'RUNNING',
                                'samples':{},'backend_qualification':{},'input_token_sha256':base.tensor_sha(ids),
                                'parameter_sha256':original}
                            report['measurements'][label]=row;base.atomic_json(args.out,report)
                            reference=blocks['fp16_library'](ids);fns={}
                            for name in BACKENDS:
                                checkpoint()
                                source='fp16_library' if name in base.LIBRARIES else name.removeprefix('compiled_')
                                fn,receipt=base.qualify_backend(blocks[source],ids,reference,compiled=name.startswith('compiled_'))
                                row['backend_qualification'][name]=receipt
                                if fn is not None:fns[name]=fn
                                base.atomic_json(args.out,report)
                            del reference
                            for candidate in variants.values():candidate.lm_head.validate_cache_binding()
                            base.timing.configure(True)
                            base.measure_paired(fns,20,30,20261006+index,checkpoint,partial=row['samples'])
                            checkpoint()
                            if base.tensor_sha(ids)!=report['source_package']['input_token_sha256'][label]:
                                raise AssertionError('timed token IDs mutated')
                            for candidate in variants.values():candidate.lm_head.validate_cache_binding()
                            row['candidate_gains']={name:None if name not in fns else {
                                'against_fp16_library':base.paired_gain(row['samples'],'fp16_library',name) if 'fp16_library' in fns else None,
                                'against_strong_library':base.strong_gain(row['samples'],name)} for name in BACKENDS if name not in base.LIBRARIES}
                            row.update(status='MEASURED_CLEAN',strong_baseline_complete=all(name in fns for name in base.LIBRARIES))
                            base.atomic_json(args.out,report)
                        checkpoint()
                        if (base.parameter_sha(model)!=original or any(base.parameter_sha(m)!=original for m in variants.values())
                                or base.model_app.model_receipt(args.model_dir)!=report['model_receipt']):
                            raise AssertionError('timed pretrained parameters or original model files mutated')
                        report.update(status='MEASURED_CLEAN',completed_at=base.utcnow(),all_pretrained_weights_preserved=True,
                            parameter_sha256=original,all_cache_bindings_preserved=True,
                            all_strong_baselines_complete=all(r['strong_baseline_complete'] for r in report['measurements'].values()))
            except BaseException as exc:
                report.update(status='BUDGET_STOPPED' if isinstance(exc,base.BudgetStopped) else 'FAILED',
                    error_type=type(exc).__name__,error=str(exc)[:4000],completed_at=base.utcnow())
                raise
            finally:
                base.atomic_json(args.out,report)
                job.result.update(report_sha256=base.model_app.sha(args.out),report_status=report['status'],
                    measured_cases=sum(r['status']=='MEASURED_CLEAN' for r in report['measurements'].values()))


def adapt(base):
    protocol=make_protocol(base);original_package=base.source_package
    base.make_protocol=lambda:copy.deepcopy(protocol)
    def source_package(*args):
        package,_=original_package(*args)
        for path in (Path(__file__),HERE/'aligned_output_head.py'):
            package['application_sources'][str(path.relative_to(base.ROOT))]=base.model_app.sha(path)
        package['optimization_scope']='vanilla pretrained output head only; historical K02 operator identity is provenance, not executed'
        return package,hashlib.sha256(base.canonical(package)).hexdigest()
    base.source_package=source_package
    base.run_gpu=lambda *args:run_gpu(base,*args)
    return base


def main(argv=None):
    return adapt(load_frozen_base()).main(argv)


if __name__=='__main__':
    raise SystemExit(main())
