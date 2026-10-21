#!/usr/bin/env python3
"""Extend a hash-bound recorded demonstration with completed public evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

LABELS = ('B1_S33', 'B1_S128', 'B1_S512', 'B2_S1024')
G008 = '9253a7d641fb87a9518866e8ed8493d31292a16d'
G007 = 'c2de21af179b3fb24f1a67147ba871322f94493a'
G000 = '54b05f2a959b5f6328ac2bf572fbd981bf567094'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and digest(path) == expected, 'evidence hash or source path changed')
    return json.loads(path.read_text())


def g008_result(path, expected):
    v = bound(path, expected)
    require(v['status'] == 'ACTUAL_SOURCE_BOUND_G008_ENGINEERING_AND_SDK_PASS' and
            v['source_commit'] == G008 and v['parent_source_commit'] == G007 and
            v['runtime_file_count'] == 175 and v['owned_workspace_aliases'] == 182 and
            v['physical_localhost_requests'] == 78 and v['request_25_refused_without_HTTP'] is True and
            v['fifth_candidate_refused'] is True and v['original_structured_error_and_query_status_preserved'] is True and
            v['owned_daemons_stopped'] is True and v['original_credentials_read'] is False and
            v['external_model_calls'] == v['GPU_calls'] == 0 and
            v['quality_promotion'] is v['RSI_proven'] is False and v['held_trajectory_used_for_source_optimization'] is False,
            'G008 engineering evidence cannot imply paid promotion')
    require([(r['tests'], r['skipped']) for r in v['engineering_checks']] == [(368, 0), (553, 2)],
            'G008 original engineering checks differ')
    return {'generation': 'G008', 'commit': G008, 'title': '优先使用反馈中已有的异常诊断',
            'detail': '保留原异常、源码身份和任务状态；异常已随反馈返回时，提示先据此修复。368项相关测试通过，guard通过551项并保留2项原跳过；78次实际localhost请求完成，付费收益仍待对照。',
            'source_author': v['source_author'], 'paid_improvement_measured': False,
            'model_authorship_verified': False, 'qualification_report': 'source/' + Path(path).name,
            'qualification_sha256': expected}


def csr_result(path, expected):
    v = bound(path, expected)
    require(v['kind'] == 'source-bound-verified-kernel-delivery' and v['agh_source_commit'] == G000 and
            v['batch_id'] == 'v18-d2-parent' and v['denominator'] == 20 and v['all20_eligible'] is True and
            v['primary_environment_error_count'] == 0 and v['original_verified_gain_count'] == 8 and
            v['verified_gain_count'] == len(v['operators']) == 1 and
            v['harness_performance_causality_proven'] is False and v['new_cross_device_reproduction_performed'] is False,
            'CSR source or original whole-arm attribution differs')
    r = v['operators'][0]; directory = Path(path).parent.resolve(); source = (directory / r['source']).resolve()
    require(source.is_relative_to(directory) and not source.is_symlink() and digest(source) == r['source_file_sha256'] and
            r['task_id'] == 'K15' and r['original_correctness_case_count'] == 21 and r['independent_verification_rounds'] == 5 and
            r['contaminated'] is False and r['verify_job_id'] and set(r['shape_speedup_medians']) == {'t0', 't1', 't2', 'r0', 'r1'} and
            1 < r['verified_ci_low'] <= r['verified_speedup'] <= r['verified_ci_high'], 'CSR five-shape verification differs')
    return {'task_id': 'K15', 'name': 'csr_segment_sum', 'verified_speedup': r['verified_speedup'],
            'ci_low': r['verified_ci_low'], 'ci_high': r['verified_ci_high'], 'shape_speedups': r['shape_speedup_medians'],
            'source_sha256': r['source_file_sha256'], 'source': 'operators/K15/kernel.py',
            'strict_environment_error_preserved': False, 'source_harness_generation': 'H0/G000',
            'original_correctness_case_count': 21, 'independent_verification_rounds': 5,
            'new_cross_device_reproduction_performed': False,
            'baseline_scope': '原已资格验证的 Python 分段求和基线；723×不表示胜过所有强库或整模型。'}


def welford_result(path, expected):
    v = bound(path, expected)
    require(v['kind'] == 'original-paired-full-GPT2-fused-Welford-performance' and v['status'] == 'MEASURED_CLEAN' and
            v['numerical_all_four_cases_correct'] is True and v['original_tolerances_and_four_cases_preserved'] is True and
            v['original_parameter_count'] == 124439808 and v['original_paired_rounds'] == 30 and
            v['GPU_samples'] == v['host_samples'] == 240 and v['strong_library_victory_proven'] is False and
            v['RSI_proven'] is False and v['owned_processes_and_RPC_closed'] is True and
            set(v['measurements']) == set(LABELS), 'Welford original performance identity differs')
    cases = []
    for label in LABELS:
        r = v['measurements'][label]; backends = r['backend_qualification']
        require(backends['candidate']['status'] == backends['fp16_library']['status'] == 'AVAILABLE' and
                backends['candidate']['correctness']['max_absolute_error'] == 0, 'Welford eager numerical qualification differs')
        samples = r['samples']; gain = r['candidate_gains']['candidate']['against_fp16_library']
        for backend in ('candidate', 'fp16_library'):
            require(len(samples[backend]['gpu_us_samples']) == len(samples[backend]['host_us_samples']) == 30,
                    'Welford original paired samples missing')
        ratios = [a / b for a, b in zip(samples['fp16_library']['gpu_us_samples'], samples['candidate']['gpu_us_samples'])]
        require(len(gain['paired_ratio_samples']) == 30 and all(math.isfinite(x) and x > 0 for x in ratios) and
                all(math.isclose(x, y, rel_tol=1e-12) for x, y in zip(ratios, gain['paired_ratio_samples'])) and
                math.isclose(statistics.median(ratios), gain['median'], rel_tol=1e-12), 'Welford ratio changed from its paired samples')
        lo, hi = gain['bootstrap_95_percent_interval']
        cases.append({'label': label, 'GPU': r['GPU'], 'eager_max_absolute_error': 0,
                      'against_fp16_library': {'median': gain['median'], 'ci_low': lo, 'ci_high': hi},
                      'compiled_library_status': backends['compiled_fp16']['status'],
                      'compiled_candidate_status': backends['compiled_candidate']['status']})
    return {'cases': cases, 'GPU_samples': 240, 'host_samples': 240, 'report': 'application/' + Path(path).name,
            'report_sha256': expected, 'summary': '原四形状完整输出逐位一致，但融合版相对普通FP16库约0.90–0.95×；未取得端到端加速。完整240个GPU和240个主机样本保留。'}


def compiled_welford_result(path, expected):
    v = bound(path, expected)
    require(v['status'] == 'ACTUAL_FOUR_CASE_COMPILED_NUMERICAL_DIAGNOSIS_CLOSED' and
            v['all_four_compiler_profile_cases_correct'] is False and v['eager_correct_cases'] == 4 and
            v['compiled_library_correct_cases'] == v['compiled_candidate_correct_cases'] == 1 and
            v['compile_interface_graph_break_fixed'] is True and v['performance_measured'] is False and
            v['strong_baseline_victory_proven'] is v['RSI_proven'] is False and v['owned_RPC_formally_closed'] is True and
            v['original_logit_tolerance'] == {'rtol': .005, 'atol': .05} and
            v['original_local_tolerance'] == {'rtol': .002, 'atol': .01} and set(v['measurements']) == set(LABELS),
            'Separate compiler profile cannot inherit a performance claim')
    cases = []
    for label in LABELS:
        r = v['measurements'][label]; backends = r['backends']
        require(r['samples'] == {} and r['performance_measured'] is False and len(r['local_checks']) == 12,
                'Numerical-only compiler diagnostic must contain no timing samples')
        cases.append({'label': label, 'candidate_status': backends['compiled_candidate']['status'],
                      'library_status': backends['compiled_fp16']['status'],
                      'candidate_max_absolute_error': backends['compiled_candidate']['correctness']['max_absolute_error']})
    require(sum(x['candidate_status'] == 'AVAILABLE' for x in cases) == 1 and
            sum(x['candidate_status'] == 'INCORRECT' for x in cases) == 3, 'Compiler failure count differs')
    return {'cases': cases, 'correct_case_count': 1, 'performance_measured': False,
            'report': 'application/' + Path(path).name, 'report_sha256': expected,
            'summary': '新的共同编译配置已消除接口图中断；仅1/4形状通过原数值门，三个长形状仍失败，尚未计时。旧数值失败与计时结果保留。'}


def extend(base, base_sha256, repo, output):
    base, repo, output = Path(base), Path(repo), Path(output)
    data = bound(base / 'data.json', base_sha256)
    require(data['mode'] == 'recorded' and data['rsi_performance_established'] is False and
            data['measurement_series']['closed_arm_count'] == 8 and not output.exists(), 'Fresh full-eight recorded base required')
    require(not any(p.is_symlink() for p in base.rglob('*')), 'Recorded base cannot contain symlinks')
    h = repo / 'harness-evolution/codex-rsi'; app = repo / 'harness-evolution/competition/application'
    g8 = h / 'reports/g008-source-bound-engineering-actual-verification.json'
    csr = h / 'operators/g000-v18-dev2/manifest.json'
    eager = app / 'pretrained-Welford-gpu-verification-v18-r1.json'
    compiled = app / 'pretrained-Welford-compiled-numerical-verification-v18-r1.json'
    row = csr_result(csr, digest(csr)); data['operators'].append(row)
    require(len({r['task_id'] for r in data['operators']}) == len(data['operators']), 'Operator source set duplicated')
    require('G008' not in {r['generation'] for r in data['source_changes']}, 'G008 already present')
    data['source_changes'].append(g008_result(g8, digest(g8)))
    data['welford_application'] = welford_result(eager, digest(eager))
    data['compiled_welford_numerical'] = compiled_welford_result(compiled, digest(compiled))
    shutil.copytree(base, output)
    for source, target in [(g8, output / 'source' / g8.name), (eager, output / 'application' / eager.name),
                           (compiled, output / 'application' / compiled.name),
                           (csr.parent / 'K15/kernel.py', output / 'operators/K15/kernel.py')]:
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    for name in ('app.js', 'index.html', 'style.css'):
        source = repo / 'harness-evolution/competition/demo' / name
        if source.exists(): shutil.copyfile(source, output / name)
    for source in (g8, csr, eager, compiled):
        data['evidence'].append({'file': source.name, 'sha256': digest(source)})
    (output / 'data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (output / 'evidence-data.js').write_text('window.EVIDENCE = ' + json.dumps(data, ensure_ascii=False, allow_nan=False) + ';\n')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--base-data-sha256', required=True)
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); data = extend(a.base, a.base_data_sha256, a.repo, a.out)
    print(json.dumps({'status': 'RECORDED_FULL_EIGHT_EXTENDED', 'operators': len(data['operators']),
                      'generations': len(data['source_changes']), 'rsi_performance_established': False, 'GPU_calls': 0}))


if __name__ == '__main__':
    main()
