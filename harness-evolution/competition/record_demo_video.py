#!/usr/bin/env python3
"""Record a four-minute evidence walkthrough from the actual offline browser UI."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def g013_caption(data):
    ablation = data.get('closed_ablation', {})
    if (ablation.get('comparison', {}).get('campaign_revision') == 26
        or any(s.get('comparison', {}).get('campaign_revision') == 26 for s in data.get('series_history', []))):
        return ('525相关回归、175运行文件、48次本地SDK正文；V26两版均正确11/20，'
                'G013最终评测19/20、缺失K16，综合比1.0238、GPU成本比1.1276。'
                '原门未过；外层Codex源码，RSI未证明。')
    return ('首次待完成观测留档；525相关回归、175运行文件与48次本地SDK正文。'
            '合成流程请求3→2，V26搜索效果待闭账；外层Codex源码，RSI未证明。')


def g014_caption(data):
    if data.get('closed_ablation', {}).get('comparison', {}).get('campaign_revision') == 29:
        return ('526项相关检查和52次SDK正文；运行路径请求2/2，排队路径3/2取舍保留。'
                '后续V29四卡G014/G016正确10/12、最终20/19，G016未接受；外层Codex源码，RSI未证明。')
    return ('526相关回归与52次实际SDK正文；运行路径两版请求均为2次，排队路径G014为3次对G013的2次。'
            '取舍保留，付费待测；外层Codex源码，RSI未证明。')


def g017_caption(data):
    if not any(r['generation'] == 'G017' for r in data['source_changes']):
        raise ValueError('qualified G017 source card required')
    return ('G017从exact G014分支；重复实现提示进入下一模型请求，实际write与hook保留。'
            '546项相关检查、175运行文件、67个实际CPU HTTP体；父子复制均5请求、4次write。'
            'V30待四卡GPU准入，付费效果未实测；外层Codex源码，RSI未证明。')


def planned_scene_seconds(data):
    latest = bool(data.get('packed_variant_application') or data.get('pretrained_precision_application'))
    current = bool(data.get('welford_application') or data.get('compiled_welford_numerical'))
    seconds = (232 + (12 if data.get('packed_application') else 0)
               + (8 if data.get('independent_cached') else 0)
               + (12 if data.get('pretrained_application') else 0)
               + (8 if data.get('packed_variant_application') else 0)
               + (6 if data.get('pretrained_precision_application') else 0)
               + (8 if data.get('pretrained_numerical_diagnosis') else 0)
               - (8 if latest else 0)
               + (8 if data.get('welford_application') else 0)
               + (8 if data.get('compiled_welford_numerical') else 0)
               - (28 if current else 0)
               + (6 if data.get('graph_application') else 0)
               + (10 if data.get('closed_ablation') else 0))
    if not 180 <= seconds <= 300:
        raise ValueError('declared scenes must fit the three-to-five-minute video')
    return seconds


def ablation_caption(data):
    ablation = data['closed_ablation']
    comparison = ablation['comparison']
    parent = next(a for a in ablation['arms'] if a['arm'] == 'parent')
    candidate = next(a for a in ablation['arms'] if a['arm'] == 'candidate')
    gpu_ratio = ablation['original_promotion']['pairs'][0]['cost_ratios']['gpu_seconds']
    cost = '通过原1.1成本门' if gpu_ratio <= 1.1 else '超过原1.1成本门'
    missing = candidate['missing_evaluation_task_ids']
    incomplete = f"候选缺少{ '、'.join(missing) }最终评测，仍保留20分母；" if missing else ''
    return (f"{comparison['parent_generation']}正确{parent['correct_count']}/20，"
            f"{comparison['candidate_generation']}正确{candidate['correct_count']}/20；"
            f"{incomplete}全20比{ablation['development_geomean_ratio']:.4f}，"
            f"95%CI[{ablation['development_paired_ci_low']:.4f},{ablation['development_paired_ci_high']:.4f}]，"
            f"GPU成本比{gpu_ratio:.4f}{cost}。其他晋级门未通过；H0仍为incumbent，RSI未证明。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    directory = args.demo.resolve()
    data = json.loads((directory / 'data.json').read_text())
    if data['mode'] != 'recorded' or data['application']['status'] != 'MEASURED_CLEAN':
        raise ValueError('a recorded, measured application evidence bundle is required')
    if args.out.exists():
        raise ValueError('video output must be fresh; preserve previous recordings')
    args.out.mkdir(parents=True)
    declared_seconds = planned_scene_seconds(data)
    latest = bool(data.get('packed_variant_application') or data.get('pretrained_precision_application'))
    current = bool(data.get('welford_application') or data.get('compiled_welford_numerical'))
    (args.out / 'recording-plan.json').write_text(json.dumps({
        'data_sha256': sha(directory / 'data.json'), 'source_script_sha256': sha(__file__),
        'planned_scene_seconds': declared_seconds, 'packed_scene': bool(data.get('packed_application')),
        'independent_scene': bool(data.get('independent_cached')),
        'pretrained_scene': bool(data.get('pretrained_application')),
        'two_replica_variant_scene': bool(data.get('packed_variant_application')),
        'precision_diagnostic_scene': bool(data.get('pretrained_precision_application')),
        'GPU_numerical_diagnosis_scene': bool(data.get('pretrained_numerical_diagnosis')),
        'Welford_eager_slowdown_scene': bool(data.get('welford_application')),
        'Welford_compiled_numerical_failure_scene': bool(data.get('compiled_welford_numerical')),
        'recorded_graph_normal_and_boundary_scenes': bool(data.get('graph_application')),
        'closed_G009_G010_ablation_scene': bool(data.get('closed_ablation')),
        'closed_ablation_revision': data.get('closed_ablation', {}).get('comparison', {}).get('campaign_revision')}, indent=2) + '\n')
    (args.out / 'recorder-source.py').write_bytes(Path(__file__).read_bytes())
    errors = []
    external_requests = []
    scenes = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=['--disable-gpu'])
        context = browser.new_context(viewport={'width': 1280, 'height': 720},
            record_video_dir=str(args.out / 'browser-original'),
            record_video_size={'width': 1280, 'height': 720})
        context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(directory.as_uri() + '/')
                      else (external_requests.append(route.request.url), route.abort())[-1])
        page = context.new_page()
        video = page.video
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto((directory / 'index.html').as_uri())
        def install_caption():
            page.add_style_tag(content='''
            #walkthrough-caption{position:fixed;bottom:0;left:0;right:0;z-index:9999;
              background:#0c1017f5;border-top:1px solid #a9f06e;padding:14px 28px;
              color:#edf2f7;font:16px/1.5 "Noto Sans CJK SC",sans-serif;min-height:84px}
            #walkthrough-caption b{color:#a9f06e;display:block;font-size:12px;margin-bottom:3px}
            body{padding-bottom:110px}html{scroll-behavior:smooth}
            ''')
            page.evaluate('''() => {const p=document.createElement('aside');p.id='walkthrough-caption';document.body.append(p)}''')

        install_caption()

        def caption(title, text):
            page.evaluate('''([title,text]) => {const p=document.getElementById('walkthrough-caption');
              p.replaceChildren(); const b=document.createElement('b');b.textContent=title;
              const s=document.createElement('span');s.textContent=text;p.append(b,s)}''', [title, text])

        def hold(label, seconds):
            scenes.append({'name': label, 'seconds': seconds})
            print(json.dumps({'scene': label, 'seconds': seconds}), flush=True)
            page.wait_for_timeout(seconds * 1000)

        def tab(name):
            page.locator(f'[data-tab="{name}"]').click()
            page.locator('.tabs').scroll_into_view_if_needed()

        caption('01 / 硕博组 · 已记录证据 · 非实时运行',
                'KernelEvolve：20路AGH搜索GPU算子，整批轨迹驱动外层修改AGH；收益以独立对照和应用实测验收。')
        hold('introduction', (18 if latest else 20) - (4 if current else 0))
        tab('results')
        gains = [row['verified_speedup'] for row in data['operators']]
        caption(f"02 / {len(gains)}个原文件算子 · 独立GPU复验",
                f'相对项目冻结基线约{min(gains):.2f}–{max(gains):.2f}倍。显示五种形状和原文件SHA256，单算子加速比不能相乘推断应用收益。')
        hold('operators-overview', (16 if latest else 18) - (4 if current else 0) - (4 if data.get('graph_application') else 0))
        k06 = page.locator('.operator').filter(has_text='K06')
        k06.locator('summary').click()
        k06.scroll_into_view_if_needed()
        caption('02 / 边界与弱收益也保留',
                'SiLU乘法的大形状收益仅约1.002倍。原搜索环境异常标记与独立复验结果同时保留。')
        has_csr = any(row.get('baseline_scope') for row in data['operators'])
        hold('operator-shapes', 12 if has_csr else 18)
        if has_csr:
            csr = page.locator('.operator').filter(has_text='csr_segment_sum')
            csr.locator('summary').click(); csr.scroll_into_view_if_needed()
            caption('02 / CSR基线范围与复验边界',
                    '原大比值对应Python分段基线；新增复测一张卡约3.358倍对编译库，另一张卡仅对Python基线。资格集与失败保留。' if data.get('graph_application') else '约724倍对应原Python分段基线，21正确性与五轮原独立验证保留；未新增跨设备复验，不表示全强库或整模型胜出。')
            hold('CSR-original-baseline-scope', 6)
        tab('evolution')
        if any(row['generation'] == 'G007' for row in data['source_changes']):
            page.locator('.change').last.scroll_into_view_if_needed()
        caption('03 / 源码修改有具体提交和工程回归',
                '编辑失败、审批拒绝、上下文压缩及工作区搜索恢复提示均有源码证据。CPU机制与运行包资格仍不能代替付费搜索性能对照。')
        if any(row['generation'] == 'G008' for row in data['source_changes']):
            caption('03 / G008优先使用已有异常诊断',
                    '保留原异常与源码身份，先据已有反馈修复。368项相关回归、551项guard通过，保留两项原跳过；78次本地SDK请求不等于付费搜索收益。')
        if any(row['generation'] == 'G010' for row in data['source_changes']):
            caption('03 / G009预算提示 · G010结构化数值失败修复',
                    '真实SDK正文资格通过；保留原24请求/四候选与数值容差。源码由外层Codex修改，工程资格仍须接受搜索对照。')
        if any(row['generation'] == 'G012' for row in data['source_changes']):
            caption('03 / G012正确但未晋级候选的观测反馈',
                    '直接从G010分支；437相关回归、39次本地SDK资格。V25正确10/20对13/20，原正确性、置信区间与成本门未过；源码由外层Codex修改，RSI未证明。')
        if any(row['generation'] == 'G013' for row in data['source_changes']):
            caption('03 / G013在原剩余超时内最多续查一次',
                    g013_caption(data))
        if any(row['generation'] == 'G014' for row in data['source_changes']):
            caption('03 / G014仅对运行中的作业续查',
                    g014_caption(data))
        if any(row['generation'] == 'G017' for row in data['source_changes']):
            caption('03 / G017重复实现的可信回执反馈', g017_caption(data))
        hold('source-evolution', 18 if current else 22)
        page.locator('#campaign').scroll_into_view_if_needed()
        caption('03 / 原始AGH与改进AGH同条件比较',
                data.get('campaign', {}).get('summary', '三组开发配对加独立留出；每轮完整20任务，不继承冠军。'))
        hold('matched-comparison-status', 18)
        if data.get('closed_ablation'):
            page.locator('#closed-ablation').scroll_into_view_if_needed()
            caption('03 / 单DEV消融保留全部原位置与失败', ablation_caption(data))
            comparison = data['closed_ablation']['comparison']
            hold(f"closed-{comparison['parent_generation']}-{comparison['candidate_generation']}-ablation-failed-gates", 10)
        tab('trace')
        caption('04 / AGH原生工具事件回放',
                '真实trace的调用、返回和失败顺序；仅公开安全结构字段。红色失败后续的读取、编辑与反馈查询可直接观察。')
        page.locator('#play').click()
        hold('native-tool-replay', 20 if current else 26)
        page.locator('#reset').click()
        page.locator('#step').click()
        page.locator('#step').click()
        caption('04 / 可暂停、逐步查看和重新回放',
                '回放不会调用模型或GPU；模型文本、工具参数、HTTP头及凭据不在公开演示中。')
        hold('native-step-controls', 14 if current else 18)
        tab('results')
        page.locator('.application').scroll_into_view_if_needed()
        caption('05 / 完整decoder计算块的真实GPU测量', data['application']['summary'])
        hold('application-overview', (20 if latest else 24) - (2 if current else 0) - (6 if data.get('graph_application') else 0))
        page.locator('.application-case').last.scroll_into_view_if_needed()
        caption('05 / 正常、边界与强基线',
                '同一固定权重和输入，30轮配对。完整输出通过；报告公开全部原始GPU/host样本、置信区间与源文件绑定。')
        hold('application-boundary-and-baseline', 20)
        page.locator('.cached-case').first.scroll_into_view_if_needed()
        caption('05 / 单token与KV缓存 · 六后端独立配对',
                data['cached_application']['summary'])
        hold('cached-application-and-strong-baseline', 12)
        if data.get('packed_application'):
            page.locator('.packed-case').last.scroll_into_view_if_needed()
            caption('05 / 合并投影 · 八后端与六个强库基线', data['packed_application']['summary'])
            hold('packed-application-boundary', 12)
        if data.get('independent_cached'):
            caption('05 / 不同物理GPU独立重复', data['independent_cached']['summary'])
            hold('independent-cached-reproduction', 8)
        if data.get('pretrained_application'):
            page.locator('.pretrained-case').first.scroll_into_view_if_needed()
            caption('05 / 真实预训练GPT-2 · 124M参数与12层',
                    '短序列编译候选相对普通FP16库约2.02倍。只有1/4形状通过，不能宣称完整应用收益。')
            hold('pretrained-short-sequence', 6)
            page.locator('.pretrained-case').last.scroll_into_view_if_needed()
            caption('05 / 长序列失败完整保留',
                    '长序列候选与编译库基线未通过原数值门；失败后端不计入加速比，没有强基线胜出或文本质量结论。')
            hold('pretrained-long-sequence-failure', 6)
        if data.get('packed_variant_application'):
            page.locator('.packed-variant-case').first.scroll_into_view_if_needed()
            caption('05 / 新H0 GEMM · 两张GPU完整独立复测', data['packed_variant_application']['summary'])
            hold('new-GEMM-two-replicas', 4)
            page.locator('.packed-variant-case').last.scroll_into_view_if_needed()
            caption('05 / 新GEMM边界损失保留',
                    '两卡四形状、原30轮配对和六强库；边界相对强库约0.845/0.876倍。原900秒失败与成本不删除。')
            hold('new-GEMM-strong-boundary-loss', 4)
        if data.get('pretrained_precision_application'):
            page.locator('.precision-case').first.scroll_into_view_if_needed()
            caption('05 / GPT-2精度诊断 · 原权重和容差',
                    '短形状强库比约1.000，置信区间跨1；编译参数调整未证明显著强库收益。')
            hold('precision-short-strong-baseline', 3)
            page.locator('.precision-case').last.scroll_into_view_if_needed()
            caption('05 / 精度诊断仍未修复长形状',
                    '后三形状候选及编译库数值失败仍保留；旧失败报告不覆盖，失败后端没有加速比。')
            hold('precision-long-failures', 3)
        if data.get('pretrained_numerical_diagnosis'):
            page.locator('.numerical-diagnosis-case').first.scroll_into_view_if_needed()
            caption('05 / 真实GPU数值定位 · 局部通过不等于完整正确',
                    '原权重、输入与容差：48次局部检查通过，参考接线四形状逐位相等；原K02完整输出只有一形状通过。')
            hold('GPU-numerical-local-versus-full', 4)
            page.locator('.numerical-diagnosis-case').last.scroll_into_view_if_needed()
            caption('05 / 逐层误差累积仍待修复',
                    '后三形状完整输出不合格；局部归一化小误差会逐层累积，本定位没有计时或新增加速比。')
            hold('GPU-numerical-long-shape-failures', 4)
        if data.get('welford_application'):
            page.locator('.welford-case').last.scroll_into_view_if_needed()
            caption('05 / Welford四形状正确，但计时未加速', data['welford_application']['summary'])
            hold('Welford-eager-original-paired-slowdown', 8)
        if data.get('compiled_welford_numerical'):
            page.locator('.compiled-welford-case').last.scroll_into_view_if_needed()
            caption('05 / 新编译接口仅通过1/4原数值门', data['compiled_welford_numerical']['summary'])
            hold('Welford-compiled-one-of-four-no-timings', 8)
        if data.get('graph_application'):
            page.goto((directory / 'graph.html').as_uri())
            install_caption()
            page.select_option('#graph-device', str(data['graph_application']['replicas'][0]['gpu']))
            page.select_option('#graph-case', 'arxiv-full')
            page.locator('#graph-ratio').scroll_into_view_if_needed()
            caption('05 / 真实ogbn-arxiv图计算 · 双设备强库约1.59倍',
                    '169343节点、1166243边；每轮五种合格库最快值，30配对轮与原容差。固定投影未训练，源码作者Codex，应用收益不证明RSI。')
            hold('recorded-graph-full-strong-library-gain', 8)
            page.select_option('#graph-case', 'boundary-fixture')
            caption('05 / 图计算边界也通过，逐轮原样本可核查',
                    '空行、尾部与高阶节点保留；两设备、三案例、六后端全部样本已公开。本视频是已记录证据，未重新运行GPU。')
            hold('recorded-graph-boundary-and-original-samples', 8)
            page.goto((directory / 'index.html').as_uri())
            install_caption()
        tab('integrity')
        caption('06 / 失败和未知不删除',
                '历史八卡实际206个job claims；执行记录不代替有效对照。未知请求保守预留不是确认账单，新的早停诊断与历史未解析原因分别保留。')
        hold('failure-and-accounting', 20 if current else 22)
        page.locator('#evidence-files').scroll_into_view_if_needed()
        caption('06 / 可以独立核查的交付',
                '源码、运行包、队列和实际计时按SHA256绑定。完整对照通过才宣称性能进化；当前结论保持在实测证据范围内。')
        hold('evidence-conclusion', 12 if current else 14)
        page.screenshot(path=str(args.out / 'final-recorded-frame.png'))
        page.close()
        context.close()
        original = Path(video.path())
        browser.close()
    if errors or external_requests:
        raise ValueError(f'browser playback failed: {errors}')
    target = args.out / 'KernelEvolve-硕博组-证据演示.mp4'
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-nostdin', '-y', '-i', str(original), '-an', '-vf', 'fps=24',
        '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast', '-crf', '28', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(target)], check=True, stdout=subprocess.DEVNULL,
        stderr=(args.out / 'encoding.log').open('w'))
    probe = subprocess.run([ffmpeg, '-i', str(target)], capture_output=True, text=True)
    (args.out / 'media-inspection.txt').write_text(probe.stderr)
    assert sum(scene['seconds'] for scene in scenes) == declared_seconds
    manifest = {'schema_version': 1, 'kind': 'actual-browser-recorded-evidence-walkthrough',
        'recorded': True, 'live_experiment_claimed': False, 'planned_scene_seconds': declared_seconds,
        'scenes': scenes, 'data_sha256': sha(directory / 'data.json'),
        'source_script_sha256': sha(__file__), 'original_browser_webm_sha256': sha(original),
        'video_file': target.name, 'video_sha256': sha(target), 'video_bytes': target.stat().st_size,
        'resolution': [1280, 720], 'audio': 'none; Chinese on-screen captions',
        'browser_page_errors': [], 'external_page_requests': external_requests,
        'external_model_calls': 0, 'GPU_calls': 0}
    (args.out / 'video-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
