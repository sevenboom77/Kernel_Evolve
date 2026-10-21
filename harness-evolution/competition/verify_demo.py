#!/usr/bin/env python3
"""Check offline/HTTP playback, real source links and missing-evidence behavior."""
from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def check_page(browser, url, data, screenshot=None):
    page = browser.new_page(viewport={'width': 1440, 'height': 1080})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url)
    assert page.locator('#operator-count').inner_text() == str(len(data['operators']))
    assert page.locator('#positions').inner_text() == str(data['all_eight_gpu_runs']['task_positions'])
    assert page.locator('#claims').inner_text() == str(data['all_eight_gpu_runs']['job_claim_count'])
    assert page.locator('#rsi-status').inner_text() == ('已验证' if data['rsi_performance_established'] else '待证明')
    if data['application']['status'] == 'MEASURED_CLEAN':
        assert page.locator('#application-status').inner_text() == '实测完成'
        assert page.locator('.application-case').count() == len(data['application']['cases'])
        for case in data['application']['cases']:
            row = page.locator('.application-case').filter(has_text=case['label'])
            assert f"相对 compiled {case['against_compiled']['median']:.2f}×" in row.inner_text()
        report = Path(data['application']['report'])
        assert not report.is_absolute()
        layout = page.locator('.application-cases').first.evaluate(
            '(node) => ({display:getComputedStyle(node).display, columns:getComputedStyle(node).gridTemplateColumns.split(" ").length})')
        assert layout == {'display': 'grid', 'columns': 2}, 'desktop application cases must remain readable in two columns'
    if data.get('cached_application'):
        cached = data['cached_application']
        assert page.locator('.cached-case').count() == len(cached['cases'])
        for case in cached['cases']:
            row = page.locator('.cached-case').filter(has_text=case['label'])
            assert f"相对每轮最快强库 {case['against_strong_library']['median']:.3f}×" in row.inner_text()
            assert row.locator('dt').count() == 6
    if data.get('packed_application'):
        packed = data['packed_application']
        assert page.locator('.packed-case').count() == len(packed['cases'])
        for case in packed['cases']:
            row = page.locator('.packed-case').filter(has_text=case['label'])
            assert f"相对每轮最快六库 {case['against_strong_library']['median']:.3f}×" in row.inner_text()
            assert row.locator('dt').count() == 8
        assert packed['summary'] in page.locator('#application-result').inner_text()
    if data.get('independent_cached'):
        assert data['independent_cached']['summary'] in page.locator('#application-result').inner_text()
    if data.get('pretrained_application'):
        pretrained = data['pretrained_application']
        assert page.locator('.pretrained-case').count() == len(pretrained['cases']) == 4
        for case in pretrained['cases']:
            row = page.locator('.pretrained-case').filter(has_text=case['label'])
            assert row.locator('dt').count() == 4
            assert 'INCORRECT' in row.inner_text()
            if case['against_fp16_library']:
                assert f"相对普通FP16库 {case['against_fp16_library']['median']:.3f}×" in row.inner_text()
        assert pretrained['summary'] in page.locator('#application-result').inner_text()
    if data.get('packed_variant_application'):
        app = data['packed_variant_application']
        assert page.locator('.packed-variant-case').count() == 8
        for index, case in enumerate(c for replica in app['replicas'] for c in replica['cases']):
            row = page.locator('.packed-variant-case').nth(index)
            assert row.locator('dt').count() == 8
            assert f"相对每轮最快六库 {case['against_strong_library']['median']:.3f}×" in row.inner_text()
        assert app['summary'] in page.locator('#application-result').inner_text()
    if data.get('pretrained_precision_application'):
        app = data['pretrained_precision_application']
        assert page.locator('.precision-case').count() == 4
        for index, case in enumerate(app['cases']):
            row = page.locator('.precision-case').nth(index)
            assert row.locator('dt').count() == 4
            if index > 0: assert '候选数值失败，未生成加速比' in row.inner_text()
            if case['against_strong_library']:
                assert f"相对最快库基线 {case['against_strong_library']['median']:.3f}×" in row.inner_text()
        assert app['summary'] in page.locator('#application-result').inner_text()
    if data.get('pretrained_numerical_diagnosis'):
        diagnosis = data['pretrained_numerical_diagnosis']
        assert page.locator('.numerical-diagnosis-case').count() == 4
        for index, case in enumerate(diagnosis['cases']):
            row = page.locator('.numerical-diagnosis-case').nth(index)
            assert case['label'] in row.inner_text()
            assert '局部检查 12/12通过' in row.inner_text()
            assert f"完整输出 {case['candidate_status']}" in row.inner_text()
            assert '计时未测' in row.inner_text()
        assert diagnosis['summary'] in page.locator('#application-result').inner_text()
    if data.get('welford_application'):
        app = data['welford_application']
        assert page.locator('.welford-case').count() == 4
        for index, case in enumerate(app['cases']):
            row = page.locator('.welford-case').nth(index)
            assert case['label'] in row.inner_text()
            assert '原完整输出逐位一致' in row.inner_text()
            assert f"相对普通FP16库 {case['against_fp16_library']['median']:.3f}×" in row.inner_text()
        assert app['summary'] in page.locator('#application-result').inner_text()
    if data.get('compiled_welford_numerical'):
        app = data['compiled_welford_numerical']
        assert page.locator('.compiled-welford-case').count() == 4
        for index, case in enumerate(app['cases']):
            text = page.locator('.compiled-welford-case').nth(index).inner_text()
            assert f"候选 {case['candidate_status']}" in text and '尚未计时' in text
        assert app['summary'] in page.locator('#application-result').inner_text()
    if screenshot:
        page.screenshot(path=str(screenshot), full_page=True)
    for operator in data['operators']:
        row = page.locator('.operator').filter(has_text=operator['task_id'])
        row.locator('summary').click()
        assert row.locator('a').get_attribute('href') == operator['source']
        if operator.get('baseline_scope'):
            assert operator['baseline_scope'] in row.inner_text()
            if operator.get('new_cross_device_reproduction_performed') is False:
                assert '跨设备复现尚未执行' in row.inner_text()
    assert page.locator('.operator-variant').count() == len(data.get('operator_variants', []))
    for variant in data.get('operator_variants', []):
        row = page.locator('.operator-variant').filter(has_text=variant['task_id'])
        assert f"{variant['verified_speedup']:.2f}×" in row.inner_text()
        row.locator('summary').click()
        assert row.locator('a').get_attribute('href') == variant['source']
        assert variant['source_harness_generation'] in row.inner_text()
    page.locator('[data-tab="evolution"]').click()
    assert page.locator('.change').count() == len(data['source_changes'])
    if data.get('measurement_series'):
        series = data['measurement_series']
        assert page.locator('.series-arm').count() == 8
        assert f"{series['closed_arm_count']} / 8臂严格关闭" in page.locator('#campaign-series').inner_text()
        for arm, row in zip(series['arms'], page.locator('.series-arm').all(), strict=True):
            assert f"正确 {arm['correct_count']}/20" in row.inner_text()
            assert f"完成评测 {arm['evaluation_complete_count']}/20" in row.inner_text()
    if data.get('closed_ablation'):
        ablation = data['closed_ablation']
        assert page.locator('#closed-ablation').is_visible()
        assert page.locator('.ablation-arm').count() == 2
        assert ablation['summary'] in page.locator('#closed-ablation').inner_text()
        for arm, row in zip(ablation['arms'], page.locator('.ablation-arm').all(), strict=True):
            assert f"正确 {arm['correct_count']}/20" in row.inner_text()
        assert f"GPU耗时比 {ablation['original_promotion']['pairs'][0]['cost_ratios']['gpu_seconds']:.4f}" in page.locator('#closed-ablation').inner_text()
    if data.get('graph_application'):
        page.locator('[data-tab="results"]').click()
        graph = data['graph_application']
        assert page.locator('#graph-featured').is_visible()
        assert page.locator('#graph-featured a').get_attribute('href') == graph['report']
        for replica in graph['replicas']:
            assert f"全量强库 {replica['strong_gain']['median']:.4f}×" in page.locator('#graph-featured').inner_text()
    page.locator('[data-tab="trace"]').click()
    for _ in data['trace_replay']:
        page.locator('#step').click()
    assert page.locator('#trace-events li').count() == len(data['trace_replay'])
    assert page.locator('#trace-events .error').count() == sum(bool(e.get('is_error') or e.get('job_failed')) for e in data['trace_replay'])
    for event in data['trace_replay']:
        if event.get('job_status'):
            assert f"作业 {event['job_status']}" in page.locator('#trace-events').inner_text()
    page.locator('#reset').click()
    assert page.locator('#trace-events li').count() == 0
    page.locator('[data-tab="integrity"]').click()
    assert page.locator('.gpu-cell').count() == 8
    assert ('资格 ' + ('合格' if data['latest_batch']['eligible'] else '不合格')) in page.locator('#failure-card').inner_text()
    assert not errors, errors
    page.close()
    return {'page_errors': 0, 'real_trace_events': len(data['trace_replay']),
            'real_error_events': sum(bool(e.get('is_error')) for e in data['trace_replay']),
            'real_terminal_job_failures': sum(bool(e.get('job_failed')) for e in data['trace_replay']),
            'source_links': len(data['operators']),
            'variant_source_links': len(data.get('operator_variants', [])), 'tabs_checked': 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    directory = args.demo.resolve()
    data = json.loads((directory / 'data.json').read_text())
    assert data['mode'] == 'recorded'
    for operator in data['operators']:
        assert hashlib.sha256((directory / operator['source']).read_bytes()).hexdigest() == operator['source_sha256']
    for variant in data.get('operator_variants', []):
        assert hashlib.sha256((directory / variant['source']).read_bytes()).hexdigest() == variant['source_sha256']
    if data['application']['status'] == 'MEASURED_CLEAN':
        assert hashlib.sha256((directory / data['application']['report']).read_bytes()).hexdigest() == data['application']['report_sha256']
    for key in ('cached_application', 'measurement_series', 'packed_application', 'independent_cached', 'pretrained_application'):
        if data.get(key):
            assert hashlib.sha256((directory / data[key]['report']).read_bytes()).hexdigest() == data[key]['report_sha256']
    for key in ('packed_variant_application', 'pretrained_precision_application', 'pretrained_numerical_diagnosis',
                'welford_application', 'compiled_welford_numerical'):
        if data.get(key):
            assert hashlib.sha256((directory / data[key]['report']).read_bytes()).hexdigest() == data[key]['report_sha256']
    for replica in data.get('packed_variant_application', {}).get('replicas', []):
        assert hashlib.sha256((directory / replica['report']).read_bytes()).hexdigest() == replica['report_sha256']
    for prior in data.get('series_history', []):
        assert hashlib.sha256((directory / prior['report']).read_bytes()).hexdigest() == prior['report_sha256']
    for change in data['source_changes']:
        if change.get('qualification_report'):
            assert hashlib.sha256((directory / change['qualification_report']).read_bytes()).hexdigest() == change['qualification_sha256']
    args.out.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/index.html'
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=['--disable-gpu'])
            offline = check_page(browser, (directory / 'index.html').as_uri(), data, args.out / 'demo-desktop.png')
            served = check_page(browser, url, data)
            page = browser.new_page(viewport={'width': 390, 'height': 844})
            page.goto(url)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            if data['application']['status'] == 'MEASURED_CLEAN':
                assert page.locator('.application-cases').first.evaluate(
                    '(node) => getComputedStyle(node).gridTemplateColumns.split(" ").length') == 1
            page.screenshot(path=str(args.out / 'demo-mobile.png'), full_page=True)
            for name in ('evolution', 'trace', 'integrity'):
                page.locator(f'[data-tab="{name}"]').click()
                if name == 'trace':
                    for _ in data['trace_replay']:
                        page.locator('#step').click()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'mobile overflow in {name}'
            page.close()
            missing = browser.new_page()
            missing.route('**/evidence-data.js', lambda route: route.abort())
            missing.goto(url)
            assert missing.locator('#load-error').is_visible()
            assert missing.locator('#operator-count').inner_text() == '—'
            missing.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    proof = {'schema_version': 1, 'status': 'PASS', 'mode': 'recorded',
             'data_sha256': hashlib.sha256((directory / 'data.json').read_bytes()).hexdigest(),
             'offline': offline, 'http': served, 'mobile_no_horizontal_overflow': True,
             'missing_evidence_refused': True, 'all_source_link_hashes_verified': True,
             'external_model_calls': 0, 'GPU_calls': 0}
    (args.out / 'browser-verification.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
