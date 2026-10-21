"""Closed V47 evidence and G029 source qualification, composed over immutable V46 delivery."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PREFIX='harness-evolution/codex-rsi/reports/'
PARENT='e1dffe9a8eca77cb1fdd81c1a8b5842c8c36370e'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(name):
    path=Path(__file__).parent/name
    spec=importlib.util.spec_from_file_location('closed_v47_'+name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

BASE=module('current_snapshot_v48.py')
NEW_REPORTS={
 'g028-v47-original-two-arm-public-all40-r1.json':'fa7c8d9ef688c0add75324461dd24593f87609fd5c37ab6c025afb9818322d0c',
 'g029-actual-source-full-SDK-qualification-r1.json':'4dd96f367ba81baa0c16a7203f4136e1e094800208ed5b2864c313ce61667179',
 'v48-native20-pre-model-actual-qualification-r1.json':'6c0d0e1ddbf1aae00a9e6b2b04f6abfee5077a73f533d7025f45a9c46573fba0',
}
REPORTS={**BASE.REPORTS,**NEW_REPORTS}
PAIRS={**BASE.PAIRS,47:'G028'}
COMMITS={**BASE.COMMITS,'G029':'34378fbc80c1f0060b4de27a9514bf94895a1fc5'}

def bound_public(repo,name):
    path=Path(repo)/PREFIX/name;digest=REPORTS[name]
    if not path.is_file() or path.is_symlink() or sha(path)!=digest:
        raise ValueError('current snapshot report identity changed: '+name)
    return json.loads(path.read_text()),'records/'+PREFIX+name,digest

def closed_view(repo,revision):
    if revision not in PAIRS:raise ValueError('only original closed pairs can be displayed')
    # A separate module instance extends the qualified reader, without changing
    # the old module's fields, source bytes, historical builder or old outputs.
    reader=module('current_snapshot_v48.py')
    reader.REPORTS=REPORTS.copy();reader.PAIRS=PAIRS.copy();reader.COMMITS=COMMITS.copy()
    return reader.closed_view(repo,revision)

def source_views(repo):
    rows=BASE.source_views(repo)
    old=next(x for x in rows if x['generation']=='G028')
    pair=closed_view(repo,47);candidate=pair['arms'][1]
    old.update(paid_comparison_closed=True,evaluation_complete_count=candidate['evaluation_complete_count'],
        missing_evaluation_task_ids=candidate['missing_evaluation_task_ids'],
        detail='有界数值失败记忆与修复策略的31阶段/1227 SDK正文已验证。V47两组已关闭，各10/20正确；'
        'G028只有19最终评测、K16缺失，原全20分母保留。综合比1.003319、95%CI跨1、GPU成本比1.341300，原门未过、未晋级。')
    p,link,digest=bound_public(repo,'g029-actual-source-full-SDK-qualification-r1.json');sdk=p['SDK'];engineering=p['engineering']
    if (p['source_commit']!=COMMITS['G029'] or p['parent_source_commit']!=PARENT
        or p['engineering_parent_source_commit']!=COMMITS['G028'] or p['engineering_parent_quality_rejected'] is not True
        or p['source_author']!='Codex outer loop' or p['runtime_file_count']!=175 or p['source_file_count']!=3633
        or p['accepted'] is not False or p['RSI_proven'] is not False or p['incumbent_modified'] is not False
        or (sdk['request_cap'],sdk['candidate_cap'],sdk['original_native_tools'],sdk['max_output_tokens'],sdk['context_window'])!=(24,4,23,4096,32768)
        or (sdk['actual_total_HTTP_bodies'],sdk['legacy_actual_HTTP_bodies'],sdk['output_limit_actual_HTTP_bodies'],sdk['output_limit_cases'])!=(1236,1227,9,6)
        or sdk['failed_new_output_fixture_HTTP_bodies_retained']!=7 or sdk['actual_total_HTTP_including_new_output_fixture_failures']!=1243
        or sdk['request_25_and_fifth_candidate_refused'] is not True or sdk['all_owned_lifetimes_closed'] is not True
        or sdk['actual_restarted_output_limit_SDK_turn_tested'] is not False
        or sdk['replay_cancel_scope_live_budget_tested_in_actual_SessionImpl'] is not True
        or sdk['external_model_calls']!=0 or sdk['GPU_calls']!=0 or engineering['full_Core_passed']!=2390):
        raise ValueError('current G029 source/SDK scope differs')
    row={'generation':'G029','commit':p['source_commit'],'measurement_parent_source_commit':PARENT,
        'engineering_parent_source_commit':p['engineering_parent_source_commit'],'engineering_parent_quality_rejected':True,
        'source_author':p['source_author'],'model_authorship_verified':False,'formal_source_performance_promoted':False,
        'paid_comparison_closed':False,'actual_localhost_request_bodies':1236,'legacy_actual_localhost_request_bodies':1227,
        'output_limit_actual_localhost_request_bodies':9,'output_limit_cases':6,'actual_SDK_stages':31,
        'failed_new_output_fixture_HTTP_bodies_retained':7,'actual_total_HTTP_including_new_output_fixture_failures':1243,
        'actual_restarted_output_limit_SDK_turn_tested':False,'replay_cancel_scope_live_budget_tested_in_actual_SessionImpl':True,
        'request_cap':24,'candidate_cap':4,'max_output_tokens':4096,'context_window':32768,
        'source_archive_sha256':p['source_archive_sha256'],'runtime_file_count':175,'source_file_count':3633,
        'full_Core_passed':2390,'title':'输出截断后一次有界续接',
        'detail':'原生kernel回合有剩余额度时，丢弃半截工具调用并允许一次完整续接；第二次截断、取消、耗尽和普通工具保持终止。'
            '2390 Core测试、31原SDK阶段及新增六场景共1236实际本地请求正文通过，另保留探索失败七正文。'
            '重放/取消边界在真实会话实现验证，未实测SDK重启turn。原额度/费用/评分保持，付费对照没有闭账成绩，RSI未证明。',
        'qualification_report':link,'qualification_sha256':digest}
    for key,field in [('source_patch','source_patch'),('incremental_patch','incremental_source_patch')]:
        binding=p[key];patch=Path(repo)/'harness-evolution/codex-rsi'/binding['path']
        if not patch.is_file() or patch.is_symlink() or sha(patch)!=binding['sha256']:
            raise ValueError('current source patch identity changed')
        row[field]='source/'+patch.name;row[field+'_sha256']=binding['sha256']
    return rows+[row]

def readiness_view(repo):
    p,link,digest=bound_public(repo,'v48-native20-pre-model-actual-qualification-r1.json')
    if (p['actual_CPU_checks']!=144 or p['registered_queues']!=250 or p['native_tasks']!=20
        or p['source_parent']!=PARENT or p['source_candidate']!=COMMITS['G029']
        or p['source_benefit_measured'] is not False or p['RSI_proven'] is not False or p['accepted_generations']!=0
        or p['GPU_indices']!=[0,1,2,3,4] or p['evaluators']!=5 or p['daemon_max_workers']!=10
        or p['utilization_required_before_GO_and_job']!=0 or p['idle_residents_allowed'] is not True
        or p['free_memory_floor_MiB']!=3072 or p['prior_formal_requests']!=45537 or p['prior_total_unknown']!=32757):
        raise ValueError('current pre-model scope differs')
    return {'generation':'G029','phase_revision':48,'status':'REGISTERED_PRE_MODEL_QUALIFICATION_SNAPSHOT',
        'CPU_checks':144,'registered_queues':250,'native_parallel_tasks':20,'GPU_indices':p['GPU_indices'],
        'evaluator_concurrency':5,'utilization_required_before_GO_and_job':0,'idle_residents_allowed':True,
        'free_memory_floor_MiB':3072,'performance_result_available':False,'RSI_proven':False,'promotion_permitted':False,
        'report':link,'report_sha256':digest,
        'summary':'V48/G029登记时点：144项CPU接入资格、250队列、20任务/五评测器。'
            '每次GO和GPU作业前实测util0、至少3072MiB剩余显存，允许空闲驻留。该固定记录没有V48最终成绩。'}

def memory_view(repo):return BASE.memory_view(repo)

def extend_data(repo,base):
    d=BASE.extend_data(repo,base)
    d['series_history'].append(d['closed_ablation']);d['closed_ablation']=closed_view(repo,47)
    latest={x['generation']:x for x in source_views(repo)}
    d['source_changes']=[latest.get(x['generation'],x) for x in d['source_changes']]+[latest['G029']]
    d['historical_readiness_v47']=d['pending_source_measurement'];d['pending_source_measurement']=readiness_view(repo)
    stale='G028的1227本地正文和V47的138接入控制属于工程/登记资格，尚无本快照闭合搜索成绩。'
    d['limitations']=[x for x in d['limitations'] if x!=stale]
    d['limitations'].extend(['V47原两组都关闭，G028缺失K16最终评测；各10/20正确、综合比1.003319且CI跨1，GPU成本门失败。',
        'G029的1236本地请求正文是源码/SDK资格，另有七探索失败正文保留；V48登记没有闭账性能结果。',
        '当前accepted0、H0保持、RSI尚未证明；最新完整便携源码、路演PPT/视频及强APP/跨设备仍待完成。'])
    d['campaign']['summary']=d['closed_ablation']['summary'];d['generated_at']=datetime.now(timezone.utc).isoformat()
    return d

def build(repo,base_bundle,out):
    out=Path(out)
    if out.exists() or out.is_symlink():raise ValueError('current snapshot output must be fresh')
    for name in REPORTS:bound_public(repo,name)
    source_views(repo);readiness_view(repo)
    package=module('package_delivery.py');BASE.build(repo,base_bundle,out)
    for old,new in [('current-provenance.json','historical-v46-assembly-receipt.json'),('data.json','historical-v46-data.json'),('evidence-data.js','historical-v46-evidence-data.js')]:
        (out/old).rename(out/new)
    for name in NEW_REPORTS:package.safe_copy(repo,PREFIX+name,out/'records')
    latest=source_views(repo)[-1]
    for field in ['source_patch','incremental_source_patch']:
        package.safe_copy(Path(repo)/'harness-evolution/codex-rsi/source-patches',Path(latest[field]).name,out/'source')
    data=extend_data(repo,json.loads((out/'historical-v37-data.json').read_text()))
    package.write_json(out/'data.json',data)
    (out/'evidence-data.js').write_text('window.EVIDENCE = '+json.dumps(data,ensure_ascii=False,allow_nan=False)+';\n')
    package.safe_copy(Path(__file__).parent/'demo','current_snapshot_v49.js',out)
    html=(out/'index.html').read_text();old='  <script src="current_snapshot_v48.js" defer></script>'
    if html.count(old)!=1:raise ValueError('original dynamic snapshot renderer identity differs')
    (out/'index.html').write_text(html.replace(old,'  <script src="current_snapshot_v49.js" defer></script>'))
    receipt={'status':'CURRENT_V47_G029_ENGINEERING_ASSEMBLED_BROWSER_MEDIA_PORTABILITY_UNVERIFIED',
        'recorded_only':True,'source_and_controller_freeze_edited':False,'new_model_GPU_calls':0,
        'RSI_proven':False,'accepted_generations':0,'closed_revisions':list(PAIRS),
        'current_engineering_source_commit':COMMITS['G029'],'builder_sha256':sha(__file__),
        'bound_reports':REPORTS,'data_sha256':sha(out/'data.json'),'media_updated':False,
        'historical_data_sha256':sha(out/'historical-v37-data.json'),
        'historical_evidence_data_sha256':sha(out/'historical-v37-evidence-data.js'),
        'historical_v46_data_sha256':sha(out/'historical-v46-data.json'),
        'historical_v46_provenance_sha256':sha(out/'historical-v46-assembly-receipt.json'),
        'assets':{name:sha(out/name) for name in ['index.html','app.js','current-snapshot.js','current_snapshot_v49.js','style.css','evidence-data.js']}}
    package.write_json(out/'current-provenance.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--base-bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.repo,a.base_bundle,a.out)))
