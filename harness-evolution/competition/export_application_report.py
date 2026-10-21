#!/usr/bin/env python3
"""Validate actual source/sample/terminal-accounting bindings before export."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(values):
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('application latency must be positive and finite')


def metric(samples,baseline,optimized):
    ratios=[a/b for a,b in zip(samples[baseline]['gpu_us_samples'],samples[optimized]['gpu_us_samples'])]
    rng=random.Random(20261005)
    boot=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(2000))
    return {'metric':'median of within-round full-block baseline/candidate latency ratios',
            'paired_ratio_samples':ratios,'median':statistics.median(ratios),
            'bootstrap_95_percent_interval':[boot[49],boot[1949]],
            'bootstrap_seed':20261005,'bootstrap_resamples':2000}


def make_report(raw_file,raw_sha256,queue_file,*,repo,optimized_backend):
    if sha(raw_file)!=raw_sha256:raise ValueError('raw application hash changed')
    raw=json.loads(Path(raw_file).read_text());repo=Path(repo).resolve()
    if raw['status']!='MEASURED_CLEAN':raise ValueError('four-case application must be complete and clean')
    package,protocol=raw['source_package'],raw['protocol']
    if canonical_hash(package)!=raw['source_package_sha256'] or canonical_hash(protocol)!=raw['protocol_sha256'] or package['protocol_sha256']!=raw['protocol_sha256']:
        raise ValueError('source package/protocol binding changed')
    for name,digest in package['application_sources'].items():
        p=(repo/name).resolve()
        if not p.is_relative_to(repo) or sha(p)!=digest:raise ValueError('application source changed')
    manifest=repo/'harness-evolution/codex-rsi/operators/g001/manifest.json'
    if sha(manifest)!=package['manifest_sha256']:raise ValueError('original operator manifest changed')
    for task,digest in package['operator_sources'].items():
        if task not in ['K04','K06','K10'] or sha(manifest.parent/task/'kernel.py')!=digest:
            raise ValueError('original operator source changed')
    if protocol['timing']['paired_rounds']!=30 or len(protocol['cases'])!=4 or len(protocol['case_labels'])!=4:
        raise ValueError('four cases and thirty paired rounds are required')
    if protocol['full_block_output_tolerance']!={'rtol':.005,'atol':.005} or protocol['operator_tolerances']!={'K04':[.0001,.00001],'K06':[.002,.01],'K10':[.002,.01]}:
        raise ValueError('original numerical tolerances changed')
    snapshots=raw['gpu_snapshots']
    if not snapshots or not all(s.get('available') is True and s.get('foreign_processes')==0 for s in snapshots):
        raise ValueError('actual application GPU snapshots are not clean')
    if raw['environment']['tf32'] is not False:raise ValueError('TF32 protocol changed')
    accounting=raw['accounting']
    with sqlite3.connect(Path(queue_file).resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        job=db.execute('SELECT * FROM jobs WHERE run_id=? AND job_id=?',(accounting['run_id'],accounting['job_id'])).fetchone()
        result=db.execute('SELECT * FROM results WHERE job_id=?',(accounting['job_id'],)).fetchone()
        events=db.execute('SELECT kind,ts,payload_json FROM events WHERE run_id=? ORDER BY id',(accounting['run_id'],)).fetchall()
    if (job is None or result is None or job['kind']!='application' or job['task_id']!='APP_DECODER'
            or job['status']!='complete' or result['status']!='complete' or job['source_hash']!=raw['source_package_sha256']
            or json.loads(result['payload_json']).get('report_sha256')!=raw_sha256):
        raise ValueError('actual terminal official job is not bound to raw report')
    claims=[];finishes=[]
    for row in events:
        if json.loads(row['payload_json']).get('job_id')==accounting['job_id']:
            if row['kind']=='job_claimed':claims.append(row['ts'])
            if row['kind']=='job_finished':finishes.append(row['ts'])
    if len(claims)!=1 or len(finishes)!=1:raise ValueError('one complete accounting claim/finish required')
    seconds=(datetime.fromisoformat(finishes[0].replace('Z','+00:00'))-datetime.fromisoformat(claims[0].replace('Z','+00:00'))).total_seconds()
    if seconds<0:raise ValueError('accounting clock moved backward')
    cases={};count=0
    for label,config in zip(protocol['case_labels'],protocol['cases']):
        row=raw['measurements'][label]
        if row['status']!='MEASURED_CLEAN' or row['config']!=config:raise ValueError('case incomplete or relabeled')
        samples=row['samples'];required={'eager','candidate','compiled'}
        if optimized_backend=='compiled_candidate':required.add('compiled_candidate')
        if not required.issubset(samples) or optimized_backend not in samples:raise ValueError('declared backend was not measured')
        for sample in samples.values():
            for field in ['gpu_us_samples','host_us_samples']:
                values=sample[field];positive(values)
                if len(values)!=30 or statistics.median(values)!=sample[field.replace('samples','median')]:
                    raise ValueError('sample count or median mismatch')
            orders=sample['batch_label_order']
            if len(orders)!=30 or any(set(order)!=set(samples) or len(order)!=len(samples) for order in orders):
                raise ValueError('paired label order invalid')
            if orders!=samples[optimized_backend]['batch_label_order']:raise ValueError('timing was not paired')
        group=row['compiled_candidate_speedups'] if optimized_backend=='compiled_candidate' else row['speedups']
        for baseline in ['eager','compiled']:
            expected=metric(samples,baseline,optimized_backend)
            if group['against_'+baseline]!=expected:raise ValueError('stored paired metric/CI differs from raw samples')
        if optimized_backend=='compiled_candidate' and group['against_candidate']!=metric(samples,'candidate',optimized_backend):
            raise ValueError('stored candidate comparison differs')
        safe={key:copy.deepcopy(row[key]) for key in ['status','config','correctness','input_weights_sha256','samples','compiled_baseline','speedups']}
        for key in ['compiled_candidate','compiled_candidate_speedups']:
            if key in row:safe[key]=copy.deepcopy(row[key])
        for key in ['compiled_baseline','compiled_candidate']:
            if key in safe:safe[key].pop('compiler_diagnostic',None)
        cases[label]=safe;count+=30*len(samples)
    selected=[(r['compiled_candidate_speedups'] if optimized_backend=='compiled_candidate' else r['speedups']) for r in cases.values()]
    eager=[r['against_eager']['median'] for r in selected];compiled=[r['against_compiled']['median'] for r in selected]
    version=Path(raw_file).parent.name
    return {'schema_version':1,'kind':'source-bound-decoder-gpu-measurement','status':'MEASURED_CLEAN',
        'benchmark_version':version,'optimized_backend':optimized_backend,
        'source_package':package,'source_package_sha256':raw['source_package_sha256'],
        'protocol':protocol,'protocol_sha256':raw['protocol_sha256'],'raw_result_sha256':raw_sha256,
        'device':{k:v for k,v in raw['environment'].items() if k!='pid'},
        'measurement_cases':cases,'raw_gpu_latency_samples':count,'raw_host_latency_samples':count,
        'clean_measurement':{'snapshot_count':len(snapshots),'all_snapshots_available_and_foreign_processes_zero':True},
        'accounting':{'queue_job_kind':'application','official_queue_job_status':'complete','queue_report_sha256_matches':True,
            'claim_at':claims[0],'finish_at':finishes[0],'conservative_claim_to_finish_seconds':seconds,
            'note':'Includes compilation and host work; not CUDA elapsed time.'},
        'conclusions':{'all_four_cases_correct_and_measured':True,'compiled_baseline_available_in_all_cases':True,
            'beats_compiled_baseline':all(r['against_compiled']['bootstrap_95_percent_interval'][0]>1 for r in selected),
            'candidate_speedup_against_eager_range':[min(eager),max(eager)],
            'candidate_speedup_against_compiled_range':[min(compiled),max(compiled)]},
        'pretrained_model':raw['pretrained_model'],'text_quality_measured':raw['text_quality_measured'],
        'harness_version_performance_causality_proven':False,
        'independent_consistency_scope':'Read-only source/sample/CI/clean-snapshot/official-accounting consistency; not a fresh GPU reproduction.',
        'limits':['Fixed random decoder computation, no pretrained text quality.',
                  'Four predeclared shapes and repeated fixed tensors; source contracts and tolerances preserved.',
                  'The strong torch.compile baseline and all raw paired samples are retained.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--raw-sha256',required=True);p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--optimized-backend',choices=['candidate','compiled_candidate'],required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=make_report(a.raw,a.raw_sha256,a.queue,repo=a.repo,optimized_backend=a.optimized_backend)
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'sample_count':report['raw_gpu_latency_samples'],
        'optimized_backend':report['optimized_backend'],'conclusions':report['conclusions'],'sha256':sha(a.out)}))


if __name__=='__main__':main()
