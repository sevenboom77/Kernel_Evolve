#!/usr/bin/env python3
"""Independent small-case checks on the five auxiliary GPUs; never promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent/'application'))
from benchmark_decoder_v2 import BudgetChecker, NumericGpuLock
from decoder_block import DEFAULT_MANIFEST, DEFAULT_MANIFEST_SHA256
from kernel_lab import gpuinfo
from kernel_lab.evaluator import ChildFailure, Evaluator
from kernel_lab.tasks import TaskPackage
from kernel_lab.queue import Queue
from kernel_lab.util import read_json, utcnow

TASKS = ('K04','K05','K06','K07','K09','K10')


def reference(path):
    p=Path(path).resolve()
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read_bound(ref):
    if reference(ref['path'])!=ref:raise ValueError('auxiliary source binding changed')
    return read_json(ref['path'])


def write_new(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
    path.chmod(0o600)
    return reference(path)


def validate_accounting(policy,resource,queue,gpus):
    p=read_bound(policy);r=read_bound(resource)
    if (str(Path(queue).resolve()) not in p['gpu_queue_paths']
            or not gpus or any(type(g) is not int for g in gpus)
            or len(gpus)!=len(set(gpus))
            or not set(gpus).issubset(set(r['gpu_indices'])-set(r['primary_gpu_indices']))):
        raise ValueError('auxiliary queue/device not authorized')
    a=r['authorization_policy']
    if (a['allow_foreign_process_signals'] is not False
            or a['original_contamination_rule_unchanged'] is not True
            or a['allow_resident_foreign_processes_for_auxiliary_execution'] is not True):
        raise ValueError('auxiliary process/pollution authorization changed')


def source_rows():
    d=read_bound({'path':str(DEFAULT_MANIFEST),'sha256':DEFAULT_MANIFEST_SHA256})
    rows=[]
    for row in d['operators']:
        if row['task_id'] in TASKS:
            p=(DEFAULT_MANIFEST.parent/row['source']).resolve()
            if not p.is_relative_to(DEFAULT_MANIFEST.parent):raise ValueError('operator source escapes bundle')
            source=reference(p)
            if source['sha256']!=row['source_file_sha256']:raise ValueError('operator source changed')
            rows.append({'task_id':row['task_id'],'source':source,
                         'expected_sha256':row['source_file_sha256'],
                         'original_verify_job_id':row['verify_job_id']})
    if tuple(r['task_id'] for r in rows)!=TASKS:raise ValueError('required six operators missing')
    return rows


def validate_sources(rows):
    for row in rows:
        if row['source']['sha256']!=row['expected_sha256'] or reference(row['source']['path'])!=row['source']:
            raise ValueError('auxiliary operator source changed')


def task_plan(package):
    return {'task_id':package.task_id,'correctness_cases':package.dev_correctness_cases()[:2],
            'timing_cases':package.dev_timing_cases()[:1],
            'tolerance':list(package.tolerance),'baselines':list(package.baseline_names),
            'used_for_harness_promotion':False}


def install_alias(repo,output,queue,run_id):
    if not run_id or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in run_id):
        raise ValueError('invalid auxiliary run ID')
    alias=Path(repo)/'artifacts'/run_id
    if alias.exists() or alias.is_symlink() or Path(output).exists():
        raise ValueError('auxiliary output/alias must be fresh')
    output=Path(output).resolve();output.mkdir(parents=True,mode=0o700)
    queue=Path(queue).resolve();queue.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    (output/'queue.sqlite').symlink_to(queue)
    alias.parent.mkdir(exist_ok=True)
    alias.symlink_to(output,target_is_directory=True)


def close_result(status,data,before,after):
    contaminated=gpuinfo.contaminated(before,after)
    return {**data,'status':status,'gpu_state_before':before,'gpu_state_after':after,
            'contaminated':contaminated,'timing_status':'INDICATIVE' if contaminated else 'SMALL_SUBSET_ONLY',
            'used_for_harness_promotion':False,'promotion_attempts':0,
            'scope':'two original DEV correctness cases and one original DEV timing case',
            'completed_at':utcnow()}


class AuxiliaryEvaluator(Evaluator):
    def __init__(self,*args,checker,**kwargs):
        super().__init__(*args,**kwargs);self.checker=checker

    def _run_child(self,*args,**kwargs):
        self.checker.check()
        result=super()._run_child(*args,**kwargs)
        self.checker.check()
        return result

    def run_subset(self,row):
        validate_sources([row])
        task=row['task_id'];package=TaskPackage(self.root,task);plan=task_plan(package)
        cases=plan['correctness_cases'];oracle=self.ensure_oracle(task,'dev',cases)
        static={c['case_id']:package.case_static(c) for c in cases}
        out=self.paths.work/'auxiliary_check'/task
        spec={'mode':'check','task_id':task,'candidate_dir':str(Path(row['source']['path']).parent),
              'cases':cases,'inputs_dir':str(oracle/'inputs'),'out_dir':str(out),
              'static_by_case':static,'device':self.device}
        self._run_child(spec,timeout=180,task_id=task,candidate_id='delivered-original')
        runs=read_json(out/'run.json')['results']
        if any(r.get('inputs_mutated') or r.get('aliased') or r['status']!='ok' for r in runs):
            return 'incorrect',{'plan':plan,'candidate_run':runs,'error':'original mutation/alias/execution contract failed'}
        correctness=self._compare_outputs(package,cases,oracle,out)
        if not correctness['ok']:return 'incorrect',{'plan':plan,'correctness':correctness}
        qualified=[];qualifications={}
        for name in plan['baselines']:
            baseline_out=self.paths.work/'auxiliary_baseline_check'/task/name
            check={**spec,'out_dir':str(baseline_out),'baseline':name}
            del check['candidate_dir']
            try:
                self._run_child(check,timeout=180,task_id=task,candidate_id='baseline-'+name)
                verdict=self._compare_outputs(package,cases,oracle,baseline_out)
                baseline_runs=read_json(baseline_out/'run.json')['results']
                ok=verdict['ok'] and all(r['status']=='ok' and not r.get('inputs_mutated') and not r.get('aliased') for r in baseline_runs)
                qualifications[name]={'status':'correct' if ok else 'incorrect','correctness':verdict}
                if ok:qualified.append(name)
            except ChildFailure as exc:
                qualifications[name]={'status':exc.status,'error':str(exc),'detail':exc.detail}
        if not qualified:return 'env_error',{'plan':plan,'correctness':correctness,'baseline_qualification':qualifications}
        timing_out=self.paths.work/'auxiliary_timing'/f'{task}.json'
        timing_spec={'mode':'timing','task_id':task,'candidate_dir':str(Path(row['source']['path']).parent),
             'cases':plan['timing_cases'],'baselines':qualified,'device':self.device,
             'warmup':20,'batches':30,'buffers':2,'run_seed':20261005,'order_seed':20361008,
             'out':str(timing_out)}
        self._run_child(timing_spec,timeout=300,task_id=task,candidate_id='delivered-original')
        data=read_json(timing_out)
        good=all(c.get('status')=='ok' and 'candidate' in c.get('measurements',{}) for c in data['cases'].values()) and len(data['cases'])==1
        return ('complete' if good else 'env_error'),{'plan':plan,'correctness':correctness,
            'baseline_qualification':qualifications,'timing':data,'timing_source':reference(timing_out)}


def prepare(args):
    policy=reference(args.policy);resource=reference(args.resource);queue=Path(args.queue).resolve()
    validate_accounting(policy,resource,queue,[1,2,5,6,7])
    rows=source_rows();validate_sources(rows)
    bindings=[reference(__file__)]
    for name in ['evaluator','children','correctness','tasks','timing','queue','gpuinfo','util','events','selection','paths']:
        bindings.append(reference(ROOT/'kernel_lab'/f'{name}.py'))
    for name in ['benchmark_decoder_v2.py','decoder_block.py']:
        bindings.append(reference(Path(__file__).resolve().parent/'application'/name))
    for task in TASKS:
        for name in ['spec.json','cases.py','reference.py','baselines.py']:
            bindings.append(reference(ROOT/'tasks'/task/name))
    q=Queue(queue)
    try:
        if q.run_row(args.run_id) is not None:raise ValueError('auxiliary run already exists')
    finally:q.close()
    install_alias(ROOT,args.out,queue,args.run_id)
    plan={'schema_version':1,'kind':'independent-small-operator-auxiliary','run_id':args.run_id,
          'gpu_indices':[1,2,5,6,7],'queue':str(queue),'output':str(Path(args.out).resolve()),
          'policy':policy,'resource':resource,'sources':rows,'bindings':bindings,
          'budget_argv':reference(args.budget_argv),'used_for_harness_promotion':False,
          'model_requests':0,'created_at':utcnow()}
    ref=write_new(Path(args.out)/'auxiliary-plan.json',plan)
    q=Queue(queue)
    try:
        q.create_run(args.run_id,'auxiliary-delivered-operators',ref['sha256'])
        for row in rows:q.enqueue(args.run_id,row['task_id'],'auxiliary',source_hash=row['source']['sha256'],max_attempts=1,phase='auxiliary')
    finally:q.close()
    print(json.dumps({'status':'READY_CPU_ONLY','plan':ref}))


def worker(args):
    p=read_bound({'path':args.plan,'sha256':args.plan_sha256})
    for binding in p['bindings']:
        if reference(binding['path'])!=binding:raise ValueError('auxiliary evaluator source changed')
    validate_accounting(p['policy'],p['resource'],p['queue'],[args.gpu]);validate_sources(p['sources'])
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    checker=BudgetChecker(p['budget_argv']['path'],p['budget_argv']['sha256']);checker.check()
    evaluator=AuxiliaryEvaluator(p['run_id'],str(args.gpu),root=ROOT,python_exe=sys.executable,
        protocol={'seeds':{'dev':20261005},'timeouts':{'correctness_s':60}},checker=checker)
    try:
        while True:
            checker.check()
            job=evaluator.q.claim(p['run_id'],evaluator.owner,str(args.gpu),lease_s=1900)
            if job is None:break
            before=gpuinfo.snapshot(args.gpu,own_pid=os.getpid());after=before
            try:
                # Each original child owns its process group and enforces its
                # original timeout. A parent alarm would bypass that cleanup.
                with NumericGpuLock(ROOT,args.gpu):
                    if not before.get('available'):raise ValueError('auxiliary GPU probe unavailable')
                    status,data=evaluator.run_subset(next(r for r in p['sources'] if r['task_id']==job['task_id']))
                    after=gpuinfo.snapshot(args.gpu,own_pid=os.getpid())
            except Exception as exc:
                status=exc.status if isinstance(exc,ChildFailure) else 'env_error'
                if isinstance(exc,ChildFailure) and ('OutOfMemory' in str(exc) or 'out of memory' in str(exc)):
                    status='oom'
                data={'error_type':type(exc).__name__,'error':str(exc),'detail':getattr(exc,'detail',{})}
                after=gpuinfo.snapshot(args.gpu,own_pid=os.getpid())
            result=close_result(status,data,before,after)
            evaluator.q.finish(job,status,result)
            write_new(Path(p['output'])/'reports'/f"{job['task_id']}-{job['job_id']}.json",result)
            print(json.dumps({'task_id':job['task_id'],'GPU':args.gpu,'status':status,'contaminated':result['contaminated']}),flush=True)
    finally:
        evaluator.q.close()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for flag in ['policy','resource','queue','run-id','out','budget-argv']:prep.add_argument('--'+flag,required=True)
    run=sub.add_parser('worker');run.add_argument('--plan',required=True);run.add_argument('--plan-sha256',required=True)
    run.add_argument('--gpu',type=int,required=True,choices=[1,2,5,6,7])
    args=p.parse_args();return prepare(args) if args.command=='prepare' else worker(args)


if __name__=='__main__':main()
