#!/usr/bin/env python3
"""Portable, independently accounted reproduction of the frozen application."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
APP=Path(__file__).resolve().parent/'application'


def reference(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def write_new(path,data):
    with path.open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
    path.chmod(0o600);return reference(path)


def validate_fresh_output(path):
    p=Path(path).resolve()
    if p.is_relative_to(Path('/data/lizhijun/agh-evo')) or p.exists() or Path(path).is_symlink():
        raise ValueError('local reproduction requires a new output outside production experiments')
    return p


def local_status(config):
    """Read every official local claim, including active wall spans, directly."""
    observed=time.time();path=Path(config['queue']).resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        if any(row[0]!=config['run_id'] for row in db.execute('SELECT run_id FROM runs')):
            raise ValueError('foreign local run cannot be dropped from accounting')
        if any(row[0]!='application' or row[1]!='APP_DECODER' for row in db.execute('SELECT kind,task_id FROM jobs')):
            raise ValueError('local reproduction contains a foreign job')
        events=db.execute('SELECT kind,ts,payload_json FROM events ORDER BY id').fetchall()
    active={};seconds=0.;count=0
    for kind,stamp,payload in events:
        job=json.loads(payload).get('job_id')
        if not job:continue
        at=datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp()
        if kind=='job_claimed':
            if job in active:seconds+=max(0,at-active.pop(job));count+=1
            active[job]=at
        elif kind in ('job_finished','job_requeued','job_lost') and job in active:
            seconds+=max(0,at-active.pop(job));count+=1
    seconds+=sum(max(0,observed-start) for start in active.values())
    if time.time()-observed>2:raise ValueError('local actual observation became stale')
    return {'scope':'independent-local-reproduction','coverage_complete':True,'should_stop':False,
            'gpu_seconds':seconds,'gpu_span_count':count+len(active),'gpu_active_span_count':len(active),
            'observed_at':datetime.fromtimestamp(observed,timezone.utc).isoformat(),
            'model_requests':0,'model_tokens':0,'model_usd':0,'production_ledger_modified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--execute',action='store_true',help='run the original GPU benchmark on your own clean device')
    parser.add_argument('--gpu',type=int,choices=range(8),default=0)
    parser.add_argument('--operator-manifest',type=Path)
    parser.add_argument('--operator-manifest-sha256')
    parser.add_argument('--local-status',type=Path)
    parser.add_argument('--local-status-sha256')
    args=parser.parse_args(argv)
    if args.local_status:
        try:
            if reference(args.local_status)['sha256']!=args.local_status_sha256:raise ValueError('local policy changed')
            print(json.dumps(local_status(json.loads(args.local_status.read_text())),allow_nan=False));return 0
        except (OSError,ValueError,sqlite3.Error):
            print(json.dumps({'coverage_complete':False,'should_stop':True,'error_code':'LOCAL_ACCOUNTING_REJECTED'}));return 2
    if args.out is None:parser.error('--out is required')
    if bool(args.operator_manifest)!=bool(args.operator_manifest_sha256):parser.error('operator manifest and SHA256 are required together')
    out=validate_fresh_output(args.out);out.mkdir(parents=True,mode=0o700)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(APP))
    import benchmark_decoder_v3 as benchmark
    manifest=args.operator_manifest or benchmark.DEFAULT_MANIFEST
    sha=args.operator_manifest_sha256 or benchmark.DEFAULT_MANIFEST_SHA256
    bindings=benchmark.verify_sources(manifest,sha)
    protocol=benchmark.make_protocol();planned=benchmark.make_plan(bindings,protocol)
    write_new(out/'benchmark-plan.json',planned)
    write_new(out/'reproduction-plan.json',{'status':'EXECUTE_REQUESTED' if args.execute else 'CPU_PLAN_ONLY',
        'executor':reference(__file__),'benchmark':reference(APP/'benchmark_decoder_v3.py'),
        'operator_manifest':reference(manifest),'gpu':args.gpu,'model_requests':0,
        'production_ledger_modified':False,'original_numerical_protocol':protocol})
    if not args.execute:return 0
    from kernel_lab.queue import Queue
    queue=out/'queue.sqlite';q=Queue(queue);q.close()
    run_id='local-decoder-'+hashlib.sha256(str(out).encode()).hexdigest()[:16]
    policy=write_new(out/'local-policy.json',{'kind':'independent-local-app-reproduction','run_id':run_id,
        'queue':str(queue),'gpu_queue_paths':[str(queue)],'model_requests':0})
    budget=write_new(out/'budget-argv.json',[sys.executable,str(Path(__file__).resolve()),'--local-status',policy['path'],
        '--local-status-sha256',policy['sha256']])
    command=[sys.executable,str(APP/'benchmark_decoder_v3.py'),'--execute','--gpu',str(args.gpu),'--run-id',run_id,
        '--operator-manifest',str(manifest),'--operator-manifest-sha256',sha,'--out',str(out/'application-result.json'),
        '--accounting-queue',str(queue),'--registry-reference',policy['path'],'--registry-sha256',policy['sha256'],
        '--budget-argv-json',budget['path'],'--budget-argv-sha256',budget['sha256']]
    result=subprocess.run(command,cwd=ROOT)
    write_new(out/'execution.json',{'exit_code':result.returncode,'executor':reference(__file__),
        'final_local_accounting':local_status(json.loads(Path(policy['path']).read_text())),
        'result':reference(out/'application-result.json') if (out/'application-result.json').exists() else None})
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
