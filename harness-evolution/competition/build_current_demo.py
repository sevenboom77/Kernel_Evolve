"""Compose current closed source evidence and the original recorded graph view."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_SHA256 = '1cc1b12677aec1cc779ca8ff42590ae0aecb581a67eeab20d94acd9737efdbe8'


def module(repo, name):
    spec = importlib.util.spec_from_file_location('current_' + name, Path(repo)/'harness-evolution/competition'/name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def bound_public(repo, name):
    package = module(ROOT, 'package_delivery.py')
    rel = 'harness-evolution/codex-rsi/reports/' + name
    expected = package.CURRENT_REPORTS[rel]
    path = Path(repo)/rel
    if package.sha(path) != expected:
        raise ValueError('current closed public identity changed')
    return json.loads(path.read_text()), 'records/' + rel, expected


def closed_series_view(repo, name):
    original, link, digest = bound_public(repo, name)
    revision = original['reliability_revision']
    if revision not in (21, 22, 23, 24, 25, 26, 29) or original['closed_arm_count'] != (8 if revision in (21, 23) else 2):
        raise ValueError('closed current series shape differs')
    promotion = original['original_promotion']
    if promotion['promotion_eligible'] is not False or promotion['rsi_established'] is not False:
        raise ValueError('current closed evidence cannot claim promotion or RSI')
    arms = []
    for arm in original['arms']:
        if arm['execution_status'] != 'CLOSED' or arm['denominator'] != 20 or len(arm['rows']) != 20:
            raise ValueError('all original closed task positions required')
        counts = arm['counts']
        arms.append({
            'batch_id': arm['batch_id'], 'pair': arm['pair'], 'kind': arm['kind'], 'arm': arm['arm'],
            'execution_status': arm['execution_status'], 'source_commit': arm['source_commit'],
            'task_denominator': arm['denominator'], 'terminal_count': counts['terminal'],
            'evaluation_complete_count': counts['evaluation_complete'],
            'environment_error_count': counts['environment_error'], 'correct_count': counts['correct'],
            'verified_gain_count': counts['verified_gain'], 'strict_eligible': arm['all20_eligible'],
            'deployment_geomean': arm['official_deployment_geomean_all_twenty'],
            'task_contributions': arm['rows'], 'missing_evaluation_task_ids': arm['missing_evaluation_task_ids'],
        })
    return {
        'status': 'ALL_ORIGINAL_ARMS_CLOSED_RECORDED_VIEW', 'closed_arm_count': len(arms),
        'arm_count': len(arms), 'task_position_denominator': original['task_position_denominator'],
        'source_patch_author': original['source_patch_author'], 'rsi_established': False,
        'promotion_eligible': False, 'original_promotion': promotion, 'arms': arms,
        'report': link, 'report_sha256': digest, 'original_public_report_sha256': digest,
        'development_geomean_ratio': promotion['deployment_geomean_ratio'],
        'development_paired_ci_low': promotion['paired_ci_low'],
        'development_paired_ci_high': promotion['paired_ci_high'],
        'comparison': {
            'campaign_revision': revision, 'parent_generation': {21: 'H0', 22: 'G009', 23: 'H0', 24: 'G010', 25: 'G010', 26: 'G010', 29: 'G014'}[revision],
            'candidate_generation': {21: 'G009', 22: 'G010', 23: 'G010', 24: 'G011', 25: 'G012', 26: 'G013', 29: 'G016'}[revision],
            'parent_source_commit': original['parent_source_commit'],
            'candidate_source_commit': original['candidate_source_commit'],
            'gpu_indices': [0, 1, 3, 4] if revision == 29 else [1, 3, 4],
            'evaluator_concurrency': 4 if revision == 29 else 3,
        },
        'summary': {
            21: 'V21全八臂160位置保留：159完成评测，0环境错误；原DEV置信区间和成本门未过，H0保持。',
            22: 'V22仅一个DEV消融对：G009正确8/20，G010正确12/20；速度、置信区间与GPU成本门未过，不是H0确认或RSI。',
            23: 'V23独立H0/G010八臂160位置全部评测，0环境错误。DEV综合比1.01650，置信区间含1且两组GPU成本门未过；保留集不补救开发失败，H0保持。',
            24: 'V24仅一个DEV消融对：G010正确11/20，G011正确10/20且缺少K15最终评测。综合比1.00119，置信区间含1；正确性回退，无晋级或RSI。',
            25: 'V25两组40位置全部评测：G010正确13/20，G012正确10/20。综合比1.00918，置信区间含1；正确性和token/USD/GPU成本门未过，无晋级或RSI。',
            26: 'V26两组40位置闭账：两版均正确11/20，G013最终评测19/20、缺失K16保留。综合比1.02375，置信区间含1且GPU成本比1.12764；原门未通过，无晋级或RSI。',
            29: 'V29四卡两组40位置闭账：G014/G016正确10/12，最终评测20/19、缺失K16保留。综合比0.978851，置信区间含1，GPU成本比1.444701；未通过原准入，无晋级或RSI。',
        }[revision],
    }


def current_source_rows(repo):
    package = module(repo, 'package_delivery.py')
    rows = []
    for generation, name, title, detail, count_key in (
        ('g009', 'G009-source-runtime-actual-SDK-CPU-qualification-r1.json', '原生请求与候选预算提示',
         '把实际剩余原生请求与候选额度送到模型正文；不退款、不重置、不扩大24请求/四候选。V21全八臂搜索对照未过原性能门。',
         'physical_localhost_HTTP_requests'),
        ('g010', 'g010-structured-correctness-feedback-actual-CPU-qualification-r1.json', '结构化正确性反馈驱动修复',
         '只在作业/源码身份与失败计数合法时提示先修复数值错误，再优化速度。原容差、权限与结果保持；V22正确数改善尚未形成性能晋级。',
         'actual_local_SDK_requests'),
    ):
        proof, link, digest = bound_public(repo, name)
        binding = package.CURRENT_SOURCES[generation]
        if proof['source_commit'] != binding['commit'] or proof[count_key] != 39:
            raise ValueError('current SDK/source identity differs')
        rows.append({'generation': generation.upper(), 'commit': binding['commit'], 'title': title,
            'detail': detail, 'source_author': 'Codex outer loop', 'model_authorship_verified': False,
            'formal_source_performance_promoted': False, 'paid_comparison_closed': True,
            'actual_localhost_request_bodies': proof[count_key], 'qualification_report': link,
            'qualification_sha256': digest, 'source_patch': 'source/' + Path(binding['patch']).name})
    return rows


def latest_source_rows(repo):
    package = module(repo, 'package_delivery.py')
    rows = current_source_rows(repo)
    for generation, name, title, detail, count_key, closed in (
        ('g011', 'g011-static-compiler-feedback-actual-CPU-qualification-r1.json', '编译失败修复提示',
         '静态编译修复指导已实际构建和验证；V24正确数10对11、缺少一项评测，原晋级门未通过。',
         'actual_SDK_HTTP_bodies', True),
        ('g012', 'g012-unpromoted-feedback-actual-CPU-qualification-r1.json', '正确但未晋级候选的观测反馈',
         '直接从G010分支，保留观测速度与测量下限警示。437项相关测试和39次本地SDK资格；V25正确10/20对13/20，综合比1.00918，正确性、置信区间和成本门未过。',
         'actual_sdk_requests', True),
    ):
        proof, link, digest = bound_public(repo, name)
        binding = package.CURRENT_SOURCES[generation]
        if proof['source_commit'] != binding['commit'] or proof[count_key] != 39:
            raise ValueError('current SDK/source identity differs')
        rows.append({'generation': generation.upper(), 'commit': binding['commit'], 'title': title,
            'detail': detail, 'source_author': 'Codex outer loop', 'model_authorship_verified': False,
            'formal_source_performance_promoted': False, 'paid_comparison_closed': closed,
            'actual_localhost_request_bodies': proof[count_key], 'qualification_report': link,
            'qualification_sha256': digest, 'source_patch': 'source/' + Path(binding['patch']).name})
    proof, link, digest = bound_public(repo, 'g013-bounded-pending-query-actual-CPU-qualification-r1.json')
    binding = package.CURRENT_SOURCES['g013']
    if (proof['source_commit'] != binding['commit'] or proof['engineering_parent_source_commit'] != binding['parent']
        or proof['source_file_count'] != 3629 or proof['runtime_file_count'] != 175
        or proof['original_actual_numerical_cap_body_count'] != 39 or proof['additional_actual_pending_body_count'] != 9
        or proof['actual_terminal_provider_requests'] != {'G010': 3, 'G013': 2}
        or proof['actual_own_source_test_counts']['affected'] != {'tests':525,'failures':0,'errors':0,'skips':0}
        or proof['RSI_proven'] is not False or proof['promoted'] is not False):
        raise ValueError('pending execution SDK/source identity differs')
    rows.append({'generation':'G013', 'commit':binding['commit'],
        'engineering_parent_source_commit':binding['parent'], 'title':'待完成作业的有界续查',
        'detail':'从G010独立分支；保留首次待完成观测，在原剩余超时内最多续查一次。525相关回归、175运行文件和48次实际本地SDK正文验证通过；合成流程请求3→2，CLI均查询两次。V26搜索效果待闭账，尚无性能晋级。',
        'source_author':'Codex outer loop', 'model_authorship_verified':False,
        'formal_source_performance_promoted':False, 'paid_comparison_closed':False,
        'actual_localhost_request_bodies':48,
        'actual_terminal_provider_requests':proof['actual_terminal_provider_requests'],
        'qualification_report':link, 'qualification_sha256':digest,
        'source_patch':'source/'+Path(binding['patch']).name})
    return rows


def pending_measurement_view(repo):
    proof, link, digest = bound_public(repo, 'phase26-pre-model-actual-history-listener-RPC-wire-CPU-qualification-r1.json')
    if (proof['component_checks'] != 92 or proof['skips'] != 0
        or proof['readiness_HTTP200_requests'] != 2 or proof['readiness_GPU_calls'] != 0
        or proof['promotion_permitted'] is not False or proof['RSI_proven'] is not False):
        raise ValueError('pending measurement qualification differs')
    return {'generation': 'G013', 'phase_revision': 26,
        'status': 'CPU_AND_TOOL_CONTINUATION_QUALIFIED_PERFORMANCE_PENDING',
        'CPU_checks': 92, 'readiness_HTTP200_requests': 2, 'readiness_GPU_calls': 0,
        'promotion_permitted': False, 'RSI_proven': False,
        'report': link, 'report_sha256': digest,
        'summary': 'V26已完成92项启动检查和2次真实MiMo工具续写请求；此记录不包含两组闭账搜索结果，G013性能仍待证明。'}


def closed_source_rows(repo):
    """Apply the later original paid result to the historical CPU qualification cards."""
    rows = latest_source_rows(repo)
    ablation = closed_series_view(repo, 'g013-v26-original-two-arm-ablation-r1.json')
    for row in rows:
        if row['generation'] == 'G013':
            row['paid_comparison_closed'] = True
            row['paid_comparison_report'] = ablation['report']
            row['detail'] = ('从G010独立分支；首次待完成观测留档，原剩余超时内最多续查一次。'
                '525相关回归、175运行文件、48次本地SDK资格；V26两版均正确11/20，'
                'G013最终评测19/20、缺失K16保留，综合比1.02375。置信区间和GPU成本门未过，无晋级或RSI。')
    return rows


def closed_history_view(repo, name):
    history, link, _ = bound_public(repo, name)
    result = {'request_rows':history['exact_history_rows'],
        'unknown_all_history':history['historical_unknown_requests_retained'],
        'GPU_seconds':history['GPU_seconds_retained'], 'report':link}
    states = history['descriptive_first_state_counters']['v26-ablation-candidate']
    result.update(running_first_pending=states['running']['first_pending'],
        running_pending_results=states['running']['continued_to_result'],
        queued_first_pending=states['queued']['first_pending'],
        queued_pending_results=states['queued']['continued_to_result'],
        observer_count_is_not_model_requests_saved=True)
    return result


def next_source_rows(repo):
    rows = closed_source_rows(repo)
    package = module(repo, 'package_delivery.py')
    proof, link, digest = bound_public(repo, 'g014-running-only-pending-query-actual-CPU-qualification-r1.json')
    binding = package.CURRENT_SOURCES['g014']
    if (proof['source_commit'] != binding['commit'] or proof['engineering_parent_source_commit'] != binding['parent']
        or proof['source_file_count'] != 3629 or proof['runtime_file_count'] != 175
        or proof['original_actual_numerical_cap_body_count'] != 39 or proof['additional_actual_pending_body_count'] != 13
        or proof['actual_running_provider_requests'] != {'G014':2,'G013':2}
        or proof['actual_queued_provider_requests'] != {'G014':3,'G013':2}
        or proof['actual_own_source_test_counts']['affected'] != {'tests':526,'failures':0,'errors':0,'skips':0}
        or proof['RSI_proven'] is not False or proof['promoted'] is not False):
        raise ValueError('running-only SDK/source identity differs')
    rows.append({'generation':'G014', 'commit':binding['commit'],
        'engineering_parent_source_commit':binding['parent'], 'title':'仅对运行中作业继续有界续查',
        'detail':'直接基于G013；排队首个结果交还模型，运行中最多续查一次。526相关回归、175运行文件、52次实际SDK正文资格；运行路径两版请求均为2次，排队路径G014为3次对G013的2次。取舍保留，付费搜索效果待实测，无晋级或RSI。',
        'source_author':'Codex outer loop', 'model_authorship_verified':False,
        'formal_source_performance_promoted':False, 'paid_comparison_closed':False,
        'actual_localhost_request_bodies':52,
        'actual_running_provider_requests':proof['actual_running_provider_requests'],
        'actual_queued_provider_requests':proof['actual_queued_provider_requests'],
        'qualification_report':link, 'qualification_sha256':digest,
        'source_patch':'source/'+Path(binding['patch']).name})
    return rows


def source_copy_rows(repo):
    """Extend historical cards with the later paid parent result and exact G017 SDK."""
    rows = next_source_rows(repo)
    package = module(repo, 'package_delivery.py')
    ablation = closed_series_view(repo, 'g016-v29-original-two-arm-public-all40-r1.json')
    for row in rows:
        if row['generation'] == 'G014':
            row['paid_comparison_closed'] = True
            row['paid_comparison_report'] = ablation['report']
            row['detail'] = ('基于G013；仅对运行中作业有界续查，排队结果交还模型。'
                '526项相关检查、175运行文件、52次SDK正文资格；运行请求2/2，排队请求3/2取舍保留。'
                '后续V29四卡消融已闭账：G014/G016正确10/12、最终20/19，'
                '综合比0.978851、CI含1、GPU成本比1.444701；G016未接受，H0保持。')
    proof, link, digest = bound_public(repo, 'g017-source-copy-actual-SDK-qualification-r1.json')
    build, build_link, build_digest = bound_public(repo, 'g017-source-independent-build-qualification-r1.json')
    binding = package.CURRENT_SOURCES['g017']
    bodies = proof['actual_local_CPU_HTTP_body_counts']
    if (proof['source_commit'] != binding['commit'] or build['source_commit'] != binding['commit']
        or build['parent_source_commit'] != binding['parent']
        or build['runtime_files'] != 175 or build['archived_tracked_source_files'] != 3630
        or bodies != {'original_numerical_cap': 39, 'pending': 18, 'source_copy': 10, 'total': 67}
        or proof['same_provider_requests_in_copy_fixture'] != {'G017': 5, 'G014': 5}
        or proof['same_actual_file_writes_in_copy_fixture'] != {'G017': 4, 'G014': 4}
        or proof['copy_note_new_path_and_same_path_reaches_next_HTTP_body'] is not True
        or proof['changed_write_latest_result_note_absent'] is not True
        or proof['original_23_tools_and_native24_candidate4_output4096_preserved'] is not True
        or proof['owned_normal_cleanup_passed'] is not True
        or proof['paid_source_gain_proven'] is not False or proof['RSI_proven'] is not False):
        raise ValueError('source-copy SDK/source identity differs')
    rows.append({'generation': 'G017', 'commit': binding['commit'],
        'engineering_parent_source_commit': binding['parent'], 'title': '重复实现的有界反馈',
        'detail': '直接从exact G014独立分支；成功write后，对同lane及可信因果回执的重复实现追加提示。'
            '保留实际写入和hook，不禁止正确候选有意复测。独立546项相关检查、175运行文件、67个实际本地SDK HTTP体通过；'
            '复制流程父子均5请求、4次write。V30四卡20任务付费搜索未实测，无收益晋级或RSI。',
        'source_author': 'Codex outer loop', 'model_authorship_verified': False,
        'formal_source_performance_promoted': False, 'paid_comparison_closed': False,
        'actual_localhost_request_bodies': bodies['total'],
        'actual_copy_provider_requests': proof['same_provider_requests_in_copy_fixture'],
        'actual_copy_file_writes': proof['same_actual_file_writes_in_copy_fixture'],
        'qualification_report': link, 'qualification_sha256': digest,
        'build_report': build_link, 'build_report_sha256': build_digest,
        'source_patch': 'source/' + Path(binding['patch']).name})
    return rows


def timing_history_view(repo):
    history, link, digest = bound_public(repo, 'g016-v29-closed-DEV-diagnosis-exact-history-r1.json')
    if (history['request_count'] != 33912 or history['registered_queue_count'] != 169
        or history['refunds_or_ledger_reset'] is not False
        or history['previous33172_all_fields_and_opaque_receipts_exact'] is not True
        or history['timing_exposure_is_not_causal_gain'] is not True):
        raise ValueError('V29 exact closed history differs')
    return {'request_rows': history['request_count'],
        'unknown_all_history': history['historical_unknown_cost_retained'],
        'GPU_seconds': history['cumulative_GPU_seconds'], 'report': link, 'report_sha256': digest,
        'refunds_or_ledger_reset': history['refunds_or_ledger_reset'],
        'timing_exposure_is_not_causal_gain': history['timing_exposure_is_not_causal_gain'],
        'source_bound_timing_queries': {'G014': 35, 'G016': 42},
        'source_bound_timing_notes': {'G014': 0, 'G016': 42}}


def source_copy_measurement_view(repo):
    proof, link, digest = bound_public(repo, 'phase30-source-copy-actual-RPC-model-ready-GPU-pending-r1.json')
    if (proof['phase_revision'] != 30 or proof['original_CPU_component_checks'] != 95
        or proof['registered_queues'] != 172 or proof['actual_parallel_RPC_status_workers'] != 20
        or not 0 < proof['actual_RPC_cold_start_ms'] < proof['original_RPC_startup_limit_ms'] == 10000
        or proof['original_stale_snapshot_refusal_ms'] != 2000
        or proof['actual_stale_snapshot_refused'] is not True
        or proof['readiness_HTTP200_requests'] != 2 or proof['readiness_GPU_calls'] != 0
        or proof['model'] != 'mimo-v2.6-pro' or proof['model_thinking'] != 'off'
        or proof['primary_paid_arms_started'] != 0 or proof['prepared_native_parent_tasks'] != 20
        or proof['clean_GPU_GO_passed'] is not False or proof['RSI_proven'] is not False):
        raise ValueError('V30 CPU/model readiness scope differs')
    return {'generation': 'G017', 'phase_revision': 30,
        'status': 'CPU_SDK_RPC_MODEL_QUALIFIED_GPU_GO_AND_PAID_PERFORMANCE_PENDING',
        'CPU_checks': 95, 'readiness_HTTP200_requests': 2, 'readiness_GPU_calls': 0,
        'actual_parallel_RPC_status_workers': 20, 'actual_RPC_cold_start_ms': proof['actual_RPC_cold_start_ms'],
        'clean_GPU_GO_passed': False, 'promotion_permitted': False, 'RSI_proven': False,
        'report': link, 'report_sha256': digest,
        'summary': 'V30的95项原组件、20路RPC、8.688秒冷启动、2000ms过期拒绝及2次真实MiMo HTTP200通过。'
            '父版20会话准备完成；原四卡GPU准入和付费效果尚待验证。两次续写检查独立登记，当前请求33914；旧完整闭账携带仍为33912。'}


def build(repo, base_bundle, out):
    repo, base_bundle, out = Path(repo).resolve(), Path(base_bundle).resolve(), Path(out)
    package = module(repo, 'package_delivery.py')
    if out.exists() or out.is_symlink():
        raise ValueError('current recorded demo output must be fresh')
    legacy_inventory = package.verify_inventory(base_bundle)
    base = base_bundle/'demo'
    if package.sha(base/'data.json') != BASE_DATA_SHA256:
        raise ValueError('original recorded base identity changed')
    data = json.loads((base/'data.json').read_text())
    series = closed_series_view(repo, 'g010-v23-original-full-eight-r1.json')
    ablation = closed_series_view(repo, 'g016-v29-original-two-arm-public-all40-r1.json')
    sources = source_copy_rows(repo)
    history24, history24_link, _ = bound_public(repo, 'g011-v24-closed-DEV-diagnosis-exact-history-r1.json')
    history25, history25_link, _ = bound_public(repo, 'g012-v25-closed-DEV-diagnosis-exact-history-r1.json')
    history, history_link, _ = bound_public(repo, 'g010-phase22-actual-closed-DEV-diagnosis-and-exact-history-r1.json')
    csr, csr_link, _ = bound_public(repo, 'CSR-original-protocol-two-device-post-V19-r1.json')
    graph_builder = module(repo, 'build_citation_graph_demo.py')
    samples = repo/'harness-evolution/competition/application/citation-graph-split-complete-public-samples-r1.json'
    if package.sha(samples) != graph_builder.SAMPLE_SHA256:
        raise ValueError('original graph public samples identity changed')
    graph = json.loads(samples.read_text())
    if data['mode'] != 'recorded' or data['rsi_performance_established'] is not False:
        raise ValueError('recorded original non-RSI base required')
    out.mkdir(parents=True, mode=0o700)
    for path in sorted(base.rglob('*')):
        if path.is_file():
            package.safe_copy(base, path.relative_to(base).as_posix(), out)
    package.package_current_reports(out/'records', repo=repo)
    for name in ('index.html', 'app.js', 'style.css'):
        shutil.copyfile(repo/'harness-evolution/competition/demo'/name, out/name)
    for generation in ('g009', 'g010', 'g011', 'g012', 'g013', 'g014', 'g017'):
        rel = Path(package.CURRENT_SOURCES[generation]['patch'])
        package.safe_copy(repo/rel.parent, rel.name, out/'source')
    temporary = out/'graph-assembly'
    graph_builder.build(repo, samples, graph_builder.SAMPLE_SHA256, temporary)
    for path in sorted(temporary.rglob('*')):
        if path.is_file():
            rel = path.relative_to(temporary)
            if (out/rel).exists():
                if package.sha(out/rel) != package.sha(path):
                    raise ValueError('graph/shared recorded asset bytes differ')
            else:
                package.safe_copy(temporary, rel.as_posix(), out)
    shutil.rmtree(temporary)
    data['series_history'].append(data['measurement_series'])
    data['series_history'].extend([
        closed_series_view(repo, 'g009-v21-original-series-full-eight-r1.json'),
        closed_series_view(repo, 'g010-phase22-original-two-arm-ablation-r1.json'),
        closed_series_view(repo, 'g011-v24-original-two-arm-ablation-r1.json'),
        closed_series_view(repo, 'g012-v25-original-two-arm-ablation-r1.json'),
        closed_series_view(repo, 'g013-v26-original-two-arm-ablation-r1.json')])
    data['measurement_series'] = series
    data['closed_ablation'] = ablation
    data['pending_source_measurement'] = source_copy_measurement_view(repo)
    data['source_changes'].extend(sources)
    for row in data['source_changes']:
        if row['generation'] in ('G004', 'G005', 'G006', 'G007', 'G008'):
            row['detail'] = row['detail'].split('搜索收益')[0].split('搜索性能等待')[0].split('付费收益仍待')[0].rstrip('；。') + '；该代工程资格保留，已闭账搜索对照未通过原晋级门。'
    for row in data['operators']:
        if row['source_sha256'] == csr['original_operator_sha256']:
            row['new_cross_device_reproduction_performed'] = csr['cross_device_correctness_and_protocol_reproduced']
            row['baseline_scope'] = '原大比值针对Python分段基线；新增复测一张卡约3.358×对合格编译库，另一张卡约730.74×仅对Python基线，资格集不同。'
    data['closed_history_v22'] = {**history['full_history'], 'report': history_link}
    data['closed_history_v24'] = {'request_rows': history24['exact_history_rows'],
        'unknown_all_history': history24['historical_unknown_requests_retained'],
        'GPU_seconds': history24['GPU_seconds_retained'], 'report': history24_link}
    data['closed_history_v25'] = {'request_rows':history25['exact_history_rows'],
        'unknown_all_history':history25['historical_unknown_requests_retained'],
        'GPU_seconds':history25['GPU_seconds_retained'], 'report':history25_link}
    data['closed_history_v26'] = closed_history_view(repo, 'g013-v26-closed-DEV-diagnosis-exact-history-r1.json')
    data['closed_history_v29'] = timing_history_view(repo)
    data['accounting_snapshot_label'] = '下列token/美元/GPU累计数字为V18历史checkpoint；V29已闭账33912条全历史请求，24482未知预留和原失败另列，均不退款。V30独立续写检查另登记2请求，不能并入旧闭账快照。'
    data['campaign']['summary'] = 'H0仍为incumbent；V29四卡G014/G016正确10/12，最终20/19，K16缺失；综合比0.978851、CI含1、GPU成本比1.444701，G016未接受。G017已通过源码及真实SDK资格，V30付费对照待GPU准入；尚无连续两代已接受改进或RSI证据。'
    data['graph_application'] = {'recorded_only': True, 'source_author': 'Codex',
        'report': 'graph.html', 'sample_report': 'graph-data.json', 'csr_report': csr_link,
        'GPU_sample_count': graph['GPU_sample_count'], 'host_sample_count': graph['host_sample_count'],
        'replicas': [{'gpu': r['physical_gpu'], 'strong_gain': next(c['strong_gain'] for c in r['cases'] if c['case_id'] == 'arxiv-full')} for r in graph['replicas']],
        'summary': '真实ogbn-arxiv两层图计算：169343节点、1166243边、固定未训练投影。双设备全量相对每轮五种合格库最快值约1.59×；窗口与边界也通过原数值门。应用由Codex实现，收益不证明AGH进化。'}
    data['generated_at'] = datetime.now(timezone.utc).isoformat()
    data['limitations'].append('本演示保留V23至V29原闭账记录及G016拒绝；G017仍仅源码与真实SDK资格通过。新便携包独立图GPU复现、V30付费对照和连续两代性能晋级尚未验收。')
    (out/'data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (out/'evidence-data.js').write_text('window.EVIDENCE = '+json.dumps(data, ensure_ascii=False, allow_nan=False)+';\n')
    metadata = {'status': 'CURRENT_CLOSED_DEMO_ASSEMBLED_BROWSER_UNVERIFIED', 'recorded_only': True,
        'new_GPU_model_network_calls': 0, 'browser_verified': False, 'RSI_proven': False,
        'base_inventory_sha256': legacy_inventory['manifest_sha256'], 'base_data_sha256': BASE_DATA_SHA256,
        'closed_report_sha256': [series['original_public_report_sha256'], ablation['original_public_report_sha256']],
        'graph_sample_sha256': graph_builder.SAMPLE_SHA256, 'builder_sha256': package.sha(__file__),
        'package_builder_sha256': package.sha(repo/'harness-evolution/competition/package_delivery.py'),
        'current_assets': {name: package.sha(out/name) for name in ('index.html', 'app.js', 'style.css')},
        'data_sha256': package.sha(out/'data.json')}
    package.write_json(out/'current-provenance.json', metadata)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--base-bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.base_bundle, args.out)))
