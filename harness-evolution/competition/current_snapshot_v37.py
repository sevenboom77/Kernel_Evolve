"""Build a fresh recorded work snapshot without editing frozen experiment sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'harness-evolution/codex-rsi/reports/'
REPORTS = {
    'g017-v30-original-two-arm-public-all40-r1.json': '11b35de136da0761661effced73997c7cf181cd9c1703fbc6fb6223a5de382ba',
    'g018-v32-original-two-arm-public-all40-r1.json': '80856cabfbc766a6663f14c84af4e35fa50f19f2bd7c8a4d1966d4acea0196c5',
    'g019-v33-original-two-arm-public-all40-r1.json': '91035f27977f85fcf652574645df11c39e8e660bf5604739c77c5e051f32ef69',
    'g020-v36-original-two-arm-public-all40-r1.json': '7e2626c2226cfd0b55ffee088ce031871c598426aa1bce99b7749d96ddaf8374',
    'g018-source-built-actual-135-local-SDK-r1.json': 'f99b3cca2cb7b73b8b894e850a8938d2396d642c2e75876de330528032d6b5a1',
    'g019-source-built-actual-135-local-SDK-r1.json': 'e052b9bca2e7757572048969e58a5437c9a854d8d7ad32aa151d41267f5d32ba',
    'g020-r2-actual-source-full-SDK-qualification-r1.json': 'debe2bec4bb26284ee1889bdb04f4e167f282e88836820a43f3096876b7d639a',
    'g020-r2-actual-source-full-SDK-qualification-r2.json': '77718d386ec6afa53dd2c00dac6de60bf20feee914bd9d84be8a6a23588ae771',
    'g021-r2-actual-source-full-SDK-qualification-r1.json': '8f41f8c374e2a296150b66581ce2be5f4c57c2581a2ef402eae6372ba58aabe5',
    'phase37-pre-model-actual-history-listener-RPC-wire-CPU-registration-r1.json': 'b31d1a0ebdf81bc4297ee0cf3bef2e88ee53782dec228900c7ddb1c61bbaa7e1',
    'phase37-r2-actual-pre-model-history-root-GO-and-readiness-r1.json': '8893ad0eb79613934ba1ef246b8df7f3aaf7cbef2391252a75b9e0dcee9d92c1',
    'app35-five-gpu-whole-forward-closed-original-all-cases-r1.json': '45de3ea5f653db7f4041d421d0f803b4647100500fcbb7bd4ccf7d1090c14276',
    'g018-v32-closed-DEV-diagnosis-exact-history-r1.json': '5b5c09c0070c792aebef2c46a514b275aeafb5c9d9f72b50b575f6b8c287a6cd',
    'g019-v33-closed-DEV-diagnosis-exact-history-r1.json': 'd0a89c430128b437faefe9a4e2681f0d62b5f50e0ad38f55dba803949a9da4ca',
    'g020-v36-closed-DEV-diagnosis-exact-history-r1.json': '70c9a92c511cd9d870901f36b565e13ddbc16cd682dc7bc272a9d3ad55039633',
}
PAIRS = {
    30: ('G017', 'g017-v30-original-two-arm-public-all40-r1.json'),
    32: ('G018', 'g018-v32-original-two-arm-public-all40-r1.json'),
    33: ('G019', 'g019-v33-original-two-arm-public-all40-r1.json'),
    36: ('G020-r2', 'g020-v36-original-two-arm-public-all40-r1.json'),
}
PARENT = 'e1dffe9a8eca77cb1fdd81c1a8b5842c8c36370e'
SOURCES = {
    'g018': {
        'commit': 'e5316304513ffc876429b735b8ce091c87b24bbe',
        'archive_sha256': 'f56f02f5d2d009193e1d497375fb06af7f217421f0b905c4c4665df91e78571f',
        'patch': 'G018-compiler-cause-feedback.patch',
        'patch_sha256': 'd797dd25657739c41162da7d2d40d0f25566abeaba42fce0b1b4a003813272e1',
        'report': 'g018-source-built-actual-135-local-SDK-r1.json',
        'title': '缺失编译原因的完整反馈引导',
    },
    'g019': {
        'commit': '66ad7aba9b0d4fce28b494c490ce61d78aee2ae4',
        'archive_sha256': '4c26d19bae764cee694746e70427bc6806a80a141257b25584a204c6c901c353',
        'patch': 'G019-verified-compiler-traceback-feedback.patch',
        'patch_sha256': 'e62030a97ea8af09b119a8b53883b5e68f54b97e1ddd0b98b8a887c3544310c2',
        'report': 'g019-source-built-actual-135-local-SDK-r1.json',
        'title': '保留实际编译 traceback 的诊断反馈',
    },
    'g020-r2': {
        'commit': '5b9ba2ff0c23fd3fca377b8bf2787d25bd6905ab',
        'archive_sha256': '6d90b28c441bc581a6d827a9387d3980c7866c8aa14338f81fbc21d96088c45a',
        'patch': 'G020-live-task-contract-correctness-first-r2.patch',
        'patch_sha256': '623eb37e26aba12dc8e7f2acaff2f5efd2b225da2806d3edf7ffaca21de24914',
        'report': 'g020-r2-actual-source-full-SDK-qualification-r2.json',
        'title': '实际任务与剩余预算中的正确性优先指导',
    },
    'g021-r2': {
        'commit': 'bfdb470b632c74b85abe29e70a5ff9af37970e65',
        'archive_sha256': '20002bf3a8f3e010ba066e8352c110e7489421b892ae1b92da34cb0020d36278',
        'patch': 'G021-complete-structural-correctness-feedback-r2.patch',
        'patch_sha256': 'f86d9051ccce03048bca81c1971afa7bfdc96d5f7a86d7aeb37f75a515c26556',
        'report': 'g021-r2-actual-source-full-SDK-qualification-r1.json',
        'title': '完整反馈中的缺失输出与输入别名诊断',
    },
}


def module(name):
    path = Path(__file__).parent / name
    spec = importlib.util.spec_from_file_location('snapshot_' + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def bound_public(repo, name):
    package = module('package_delivery.py')
    path = Path(repo) / PREFIX / name
    digest = REPORTS[name]
    if not path.is_file() or path.is_symlink() or package.sha(path) != digest:
        raise ValueError('current snapshot report identity changed: ' + name)
    return json.loads(path.read_text()), 'records/' + PREFIX + name, digest


def closed_view(repo, revision):
    candidate, name = PAIRS[revision]
    original, link, digest = bound_public(repo, name)
    promotion = original['original_promotion']
    if (original['reliability_revision'] != revision or original['closed_arm_count'] != 2
        or original['task_position_denominator'] != 40 or original['parent_source_commit'] != PARENT
        or promotion['promotion_eligible'] is not False or promotion['rsi_established'] is not False):
        raise ValueError('closed pair identity or scope differs')
    arms = []
    for arm in original['arms']:
        if (arm['execution_status'] != 'CLOSED' or arm['denominator'] != 20
            or len(arm['rows']) != 20
            or {r['task_id'] for r in arm['rows']} != {f'K{i:02}' for i in range(1, 21)}):
            raise ValueError('all original closed task positions required')
        c = arm['counts']
        arms.append({
            'batch_id': arm['batch_id'], 'pair': arm['pair'], 'kind': arm['kind'], 'arm': arm['arm'],
            'execution_status': arm['execution_status'], 'source_commit': arm['source_commit'],
            'task_denominator': 20, 'terminal_count': c['terminal'],
            'evaluation_complete_count': c['evaluation_complete'], 'environment_error_count': c['environment_error'],
            'correct_count': c['correct'], 'verified_gain_count': c['verified_gain'],
            'strict_eligible': arm['all20_eligible'],
            'deployment_geomean': arm['official_deployment_geomean_all_twenty'],
            'task_contributions': arm['rows'], 'missing_evaluation_task_ids': arm['missing_evaluation_task_ids'],
        })
    ratio, low, high = (promotion[k] for k in ('deployment_geomean_ratio', 'paired_ci_low', 'paired_ci_high'))
    return {
        'status': 'ALL_ORIGINAL_ARMS_CLOSED_RECORDED_VIEW', 'closed_arm_count': 2, 'arm_count': 2,
        'task_position_denominator': 40, 'source_patch_author': original['source_patch_author'],
        'rsi_established': False, 'promotion_eligible': False, 'original_promotion': promotion, 'arms': arms,
        'report': link, 'report_sha256': digest, 'original_public_report_sha256': digest,
        'development_geomean_ratio': ratio, 'development_paired_ci_low': low, 'development_paired_ci_high': high,
        'comparison': {'campaign_revision': revision, 'parent_generation': 'G014', 'candidate_generation': candidate,
            'parent_source_commit': original['parent_source_commit'], 'candidate_source_commit': original['candidate_source_commit'],
            'gpu_indices': [0, 1, 3, 4], 'evaluator_concurrency': 4},
        'summary': f"V{revision}两组40位置闭账：G014/{candidate}正确{arms[0]['correct_count']}/{arms[1]['correct_count']}，"
            f"最终评测{arms[0]['evaluation_complete_count']}/{arms[1]['evaluation_complete_count']}；"
            f"综合比{ratio:.6f}、95%CI[{low:.6f},{high:.6f}]。保留原缺失和成本，未晋级，H0保持，RSI未证明。",
    }


def source_views(repo):
    package = module('package_delivery.py')
    rows = []
    for generation, binding in SOURCES.items():
        p, link, digest = bound_public(repo, binding['report'])
        patch = Path(repo) / 'harness-evolution/codex-rsi/source-patches' / binding['patch']
        if package.sha(patch) != binding['patch_sha256']:
            raise ValueError('current source patch identity changed')
        if generation == 'g018':
            commit, parent = p['candidate_source_commit'], p['parent_source_commit']
            bodies = p['actual_local_HTTP_bodies']['total']
            cap, candidates, tools = p['native_provider_cap_per_task'], p['source_candidate_cap_per_task'], p['catalog_tools']
        elif generation == 'g019':
            commit, parent = p['source_commit'], p['parent_source_commit']
            bodies = p['total_HTTP_bodies']
            cap, candidates, tools = p['native_request_cap'], p['candidate_cap'], p['actual_catalog_tools']
        else:
            commit, parent = p['source_commit'], p['exact_parent_commit']
            s = p['actual_SDK']
            bodies, cap, candidates, tools = s['total_HTTP_bodies'], s['request_cap'], s['candidate_cap'], s['tools']
        if (commit != binding['commit'] or parent != PARENT or p['source_author'] != 'Codex outer loop'
            or (cap, candidates, tools) != (24, 4, 23) or bodies != (291 if generation == 'g021-r2' else 135)
            or p['RSI_proven'] is not False):
            raise ValueError('current source/SDK scope differs')
        detail = {
            'g018': '缺失编译原因时引导读取原完整反馈；532项相关检查、135次实际本地SDK正文。V32正确13→9，综合比0.911445；原门未过。',
            'g019': '读取实际traceback，避免对已有完整原因的误提示；544项相关检查、135次SDK正文。V33正确11→11，子版18/20最终评测、2环境错误；未接受。',
            'g020-r2': '只在完整kernel_get_task成功后，使用实际budget.remaining提示先完成正确候选；541项相关检查、135次SDK正文。V36正确11→10、K17缺评测，综合比1.026182、CI含1；未接受。',
            'g021-r2': '摘要没有具体数值失败时请求既有detail=full；只有完整且绑定的反馈才诊断缺失输出或输入突变/别名。541项相关检查、291次实际本地SDK正文、175运行文件。G021-r1失败保留；V37最终付费成绩待闭账，RSI未证明。',
        }[generation]
        row = {'generation': generation.upper().replace('-R', '-r'), 'commit': commit,
            'engineering_parent_source_commit': parent, 'source_author': 'Codex outer loop',
            'model_authorship_verified': False, 'formal_source_performance_promoted': False,
            'paid_comparison_closed': generation != 'g021-r2', 'actual_localhost_request_bodies': bodies,
            'source_archive_sha256': binding['archive_sha256'], 'runtime_file_count': 175,
            'source_file_count': 3629, 'title': binding['title'], 'detail': detail,
            'qualification_report': link, 'qualification_sha256': digest,
            'source_patch': 'source/' + binding['patch']}
        if generation == 'g020-r2':
            old, old_link, old_digest = bound_public(repo, 'g020-r2-actual-source-full-SDK-qualification-r1.json')
            correction = p['report_correction']
            if correction['original_report_sha256'] != old_digest or old['source_file_count'] != 2:
                raise ValueError('original source count correction identity differs')
            row.update(original_wrong_count_report_preserved=True, original_wrong_count_report=old_link)
        if generation == 'g021-r2':
            if (s['summary_requests_full_feedback_in_body5'] is not True
                or s['structural_guidance_after_complete_bound_feedback_in_body7'] is not True
                or s['malformed_metadata_not_interpreted'] is not True
                or p['preserved_failures']['G021_r1_initial_summary_positive_actual_SDK_failed'] is not True):
                raise ValueError('actual summary/full structural source qualification differs')
            row.update(summary_requests_full_feedback=True, structural_guidance_requires_complete_bound_feedback=True)
        rows.append(row)
    return rows


def check_source(archive, generation, repo=ROOT):
    package = module('package_delivery.py')
    binding = SOURCES[generation]
    if package.sha(archive) != binding['archive_sha256']:
        raise ValueError('current source archive identity changed')
    source_views(repo)
    files = []
    with tarfile.open(archive, 'r:') as bundle:
        for item in bundle.getmembers():
            package.safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('current source archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                package.inspect_bytes(bundle.extractfile(item).read(), item.name)
    required = {'LICENSE', 'NOTICE', 'packages/core/src/step/job-feedback.ts', 'packages/core/src/step/pending-job-query.ts'}
    if len(files) != 3629 or len(set(files)) != 3629 or not required.issubset(files):
        raise ValueError('current source/legal inventory changed')
    return {'source_commit': binding['commit'], 'parent_source_commit': PARENT,
        'source_author': 'Codex outer loop', 'archive_sha256': binding['archive_sha256'],
        'patch_sha256': binding['patch_sha256'], 'tracked_source_files': 3629, 'runtime_file_count': 175,
        'model_authorship_verified': False, 'accepted': False, 'RSI_proven': False}


def readiness_view(repo):
    p, link, digest = bound_public(repo, 'phase37-r2-actual-pre-model-history-root-GO-and-readiness-r1.json')
    if (p['CPU_component_checks'] != 84 or p['registered_queues'] != 197
        or p['actual_original_initializer_10s_limit_pass'] is not True
        or p['actual20_RPC_clients_original5s_limit_pass'] is not True
        or p['actual_original2s_snapshot_age_pass'] is not True
        or p['actual_readiness_requests'] != 2 or p['RSI_proven'] is not False):
        raise ValueError('current pre-model scope differs')
    return {'generation': 'G021-r2', 'phase_revision': 37, 'status': 'PRE_MODEL_SNAPSHOT_FINAL_PERFORMANCE_PENDING',
        'CPU_checks': 84, 'registered_queues': 197, 'native_parallel_tasks': 20,
        'clean_GPU_GO_passed': p['fresh_physical_four_card_GO_pass'], 'readiness_HTTP200_requests': 2,
        'readiness_GPU_calls': 0, 'performance_result_available': False, 'RSI_proven': False,
        'promotion_permitted': False, 'failed_initialization_preserved': p['failed_r1_initialization_preserved'],
        'report': link, 'report_sha256': digest,
        'summary': 'V37-r2预模型快照：84项CPU检查、197登记队列、原10秒启动/5秒状态/2秒快照门、两次实际MiMo HTTP200通过。四卡GO通过，GPU2另授权辅助。G021-r2最终对照成绩仍待产生；r1启动失败保留，无RSI或晋级。'}


def application_view(repo):
    p, link, digest = bound_public(repo, 'app35-five-gpu-whole-forward-closed-original-all-cases-r1.json')
    if (p['strong_baseline_complete'] is not False or p['source_effect_or_RSI_proven'] is not False
        or p['all_owned_processes_and_RPC_closed'] is not True
        or p['backend_status_counts'] != {'INCORRECT': 12, 'UNAVAILABLE': 24, 'AVAILABLE': 36}):
        raise ValueError('closed application original failure scope differs')
    return {**p, 'report': link, 'report_sha256': digest,
        'summary': 'APP35已在0/1/2/3/4五卡闭账；GPU2仅辅助数值、不记计时收益。12编译FP16不正确、24编译候选不可用，36 eager库后端正确；强基线未齐，不主张稳定整模型加速或AGH进化。'}


def build(repo, base_bundle, out, archive_root=None):
    repo, out = Path(repo).resolve(), Path(out)
    if out.exists() or out.is_symlink():
        raise ValueError('current snapshot output must be fresh')
    package, legacy = module('package_delivery.py'), module('build_current_demo.py')
    pairs = [closed_view(repo, rev) for rev in PAIRS]
    sources, readiness, application = source_views(repo), readiness_view(repo), application_view(repo)
    history, history_link, history_digest = bound_public(repo, 'g020-v36-closed-DEV-diagnosis-exact-history-r1.json')
    for name in REPORTS:
        bound_public(repo, name)
    archives = {}
    if archive_root is not None:
        for generation in SOURCES:
            version = generation + '-r1' if generation in ('g018', 'g019') else generation
            path = Path(archive_root) / version / 'source.tar'
            archives[generation] = (path, check_source(path, generation, repo))
    legacy.build(repo, base_bundle, out)
    (out / 'current-provenance.json').rename(out / 'historical-v29-assembly-receipt.json')
    for name in REPORTS:
        package.safe_copy(repo, PREFIX + name, out / 'records')
    for generation, binding in SOURCES.items():
        patch_root = repo / 'harness-evolution/codex-rsi/source-patches'
        package.safe_copy(patch_root, binding['patch'], out / 'source')
        if generation in archives:
            path, receipt = archives[generation]
            destination = out / 'source' / generation
            package.safe_copy(path.parent, path.name, destination)
            package.write_json(destination / 'provenance.json', receipt)
            row = next(r for r in sources if r['commit'] == binding['commit'])
            row['source_archive'] = f'source/{generation}/source.tar'
    data = json.loads((out / 'data.json').read_text())
    data['series_history'].append(data['closed_ablation'])
    data['series_history'].extend(pairs[:-1])
    data['closed_ablation'] = pairs[-1]
    data['historical_readiness_v30'] = data['pending_source_measurement']
    data['pending_source_measurement'] = readiness
    data['source_changes'].extend(sources)
    for row in data['source_changes']:
        if row['generation'] == 'G017':
            row.update(paid_comparison_closed=True, paid_comparison_report=pairs[0]['report'],
                detail='重复write反馈保留实际写入；546项相关检查、67次SDK工程资格。V30正确9→11、40位置均评测，综合比1.024963、CI含1；未接受，H0保持。')
    data['closed_history_v36'] = {'request_rows': history['exact_history_rows'],
        'unknown_all_history': history['historical_unknown_requests_retained'],
        'GPU_seconds': history['GPU_seconds_retained'], 'report': history_link, 'report_sha256': history_digest}
    data['whole_model_application_APP35'] = application
    data['accounting_snapshot_label'] = '旧累计数字是V18历史checkpoint；V36闭账37020全历史请求、26770未知预留、221702.885 GPU秒另列，原失败均保留。V37预模型及进行中的消费不并入这个闭账时点。'
    data['campaign']['summary'] = pairs[-1]['summary'] + ' G021-r2工程资格291实际SDK正文通过；V37最终成绩待闭账。'
    data['historical_limitations_v29'] = data['limitations']
    data['limitations'] = [v for v in data['limitations'] if not v.startswith('本演示保留V23至V29')]
    data['limitations'].extend(['V30/V32/V33/V36均未接受；外层源码作者Codex，H0保持，无连续两代晋级或RSI。',
        'G021-r2工程资格与V37预模型快照不能代替最终付费成绩。',
        'APP35强编译后端未齐，GPU2只记辅助数值；不主张整模型稳定收益。',
        '旧PPT和视频属于历史作品，尚未同步为本快照的路演材料。'])
    data['generated_at'] = datetime.now(timezone.utc).isoformat()
    (out / 'data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (out / 'evidence-data.js').write_text('window.EVIDENCE = ' + json.dumps(data, ensure_ascii=False, allow_nan=False) + ';\n')
    asset_root = Path(__file__).parent / 'demo'
    package.safe_copy(asset_root, 'current_snapshot_v37.js', out)
    (out / 'current_snapshot_v37.js').rename(out / 'current-snapshot.js')
    html = (out / 'index.html').read_text()
    (out / 'index.html').write_text(html.replace('</head>', '  <script src="current-snapshot.js" defer></script>\n</head>'))
    receipt = {'status': 'CURRENT_CLOSED_SNAPSHOT_ASSEMBLED_BROWSER_UNVERIFIED', 'recorded_only': True,
        'source_and_controller_freeze_edited': False, 'new_model_GPU_calls': 0, 'RSI_proven': False,
        'closed_revisions': list(PAIRS), 'current_engineering_source_commit': SOURCES['g021-r2']['commit'],
        'historical_v29_assembly_receipt_sha256': package.sha(out / 'historical-v29-assembly-receipt.json'),
        'builder_sha256': package.sha(__file__), 'bound_reports': REPORTS,
        'source_archives_included': list(archives), 'data_sha256': package.sha(out / 'data.json'),
        'assets': {name: package.sha(out / name) for name in ('index.html', 'app.js', 'current-snapshot.js', 'style.css', 'evidence-data.js')},
        'media_updated': False, 'primary_V37_final_score_available': False}
    package.write_json(out / 'current-provenance.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--base-bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.base_bundle, args.out, args.archive_root)))
