"""Record the current offline work UI with measured-failure captions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
from playwright.sync_api import sync_playwright


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_plan(data):
    path = Path(__file__).parent / 'generate_current_materials_v38.py'
    s = importlib.util.spec_from_file_location('current_video_content_scope', path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m.content(data)
    h = data['closed_history_v37']
    pair = data['closed_ablation']
    p, c = pair['arms']
    rows = [
        ('intro', 18, 'KernelEvolve：20路原生AGH搜索、GPU验证、闭账轨迹诊断、真实源码修改与同条件复测。记录证据演示；外层Codex驱动，RSI未成立。'),
        ('operators', 31, '独立验证的算子：形状、原数值门、源码SHA和verify身份均保留。弱收益和失败也公开，单算子倍率不能相乘推断整模型。'),
        ('gemm', 24, '历史small-M GEMM约4.273×，仅对应原指定五形状和独立验证。它来自原H0搜索，不能归因于G022或当作整模型收益。'),
        ('graph', 22, '真实ogbn-arxiv全图：169343节点、1166243边，两设备相对每轮合格强库最快值约1.59×。固定随机未训练投影，不证明分类精度或AGH源码进化。'),
        ('APP35', 31, 'APP35真实预训练GPT-2五卡闭账：12编译后端不正确、24不可用、36普通后端正确。GPU2仅辅助数值，无计时收益。强编译基线未齐，不报告稳定整模型加速。'),
        ('G022', 44, 'G022：完整绑定的Python缺失、重复或未知参数异常反馈JIT签名、Config与launch约定，澄清普通function调用边界。543源码检查、408实际本地SDK正文；四正控、两负控和父版均核最新下一工具结果。Codex源码，V38最终成绩待闭账。'),
        ('V37', 34, f"V37闭账：G014正确{p['correct_count']}/20，G021-r2正确{c['correct_count']}/20，均完成20项评测；旧V36的K17缺失仍保留。综合比{pair['development_geomean_ratio']:.4f}、95%CI[{pair['development_paired_ci_low']:.4f},{pair['development_paired_ci_high']:.4f}]，未接受，H0保持。"),
        ('trace', 26, '逐步回放真实工具事件：成功、失败、后续读取与编辑都保留。这是既有记录回放，不是实时模型调用，也不单凭修复提示声称性能进化。'),
        ('history', 18, f"V37闭账时点：{h['request_rows']}全历史请求、{h['unknown_all_history']}未知预留，原失败不退款。V38新增消费另计；原10秒启动、5秒状态、2秒快照门保留。"),
        ('next', 18, '下一步：完成当前两组对照并闭账，再用DEV轨迹修改AGH。至少14/20正确、稳定提升10%、重复与独立任务验证及连续两代接受，才能证明持续进化。当前RSI未成立。'),
    ]
    if not 180 <= sum(seconds for _, seconds, _ in rows) <= 300:
        raise ValueError('original recording duration range exceeded')
    return [{'id': name, 'seconds': seconds, 'caption': caption} for name, seconds, caption in rows]


def record(demo, out):
    demo, out = Path(demo).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise ValueError('current recorded video output must be fresh')
    data_path = demo / 'data.json'
    provenance = json.loads((demo / 'current-provenance.json').read_text())
    if sha(data_path) != provenance['data_sha256']:
        raise ValueError('current recorded data identity changed')
    data = json.loads(data_path.read_text())
    scenes = scene_plan(data)
    out.mkdir(parents=True, mode=0o700)
    (out / 'scene-plan.json').write_text(json.dumps(scenes, ensure_ascii=False, indent=2) + '\n')
    errors, external, frames = [], [], []
    started = time.monotonic()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--disable-gpu'])
        context = browser.new_context(viewport={'width': 1440, 'height': 810},
            record_video_dir=str(out / 'raw'), record_video_size={'width': 1280, 'height': 720})
        page = context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        def route(r):
            if r.request.url.startswith(('file:', 'data:')):
                r.continue_()
            else:
                external.append(r.request.url)
                r.abort()
        page.route('**/*', route)
        page.goto((demo / 'index.html').as_uri())
        assert page.locator('#load-error').is_hidden()
        for scene in scenes:
            name = scene['id']
            if name == 'graph':
                page.goto((demo / 'graph.html').as_uri())
            else:
                if name == 'APP35':
                    page.goto((demo / 'index.html').as_uri())
                tab = 'evolution' if name in ('G022', 'V37') else 'trace' if name == 'trace' else 'integrity' if name in ('history', 'next') else 'results'
                page.locator(f'[data-tab="{tab}"]').click()
                target = {'operators': '#operators', 'gemm': '#operator-variants',
                    'APP35': '#whole-model-app35', 'V37': '#closed-ablation',
                    'trace': '#trace-events', 'history': '#closed-history-v37', 'next': '#limitations'}.get(name)
                if name == 'G022':
                    page.locator('.change').filter(has=page.locator('.generation', has_text='G022')).scroll_into_view_if_needed()
                elif target:
                    page.locator(target).scroll_into_view_if_needed()
            page.evaluate('''caption => {
              let n = document.getElementById('current-video-caption');
              if (!n) { n = document.createElement('div'); n.id = 'current-video-caption'; document.body.append(n); }
              n.textContent = caption;
              n.style.cssText = 'position:fixed;bottom:14px;left:30px;right:30px;z-index:9999;background:rgba(8,12,18,.96);color:#edf2f7;border:1px solid #4f6b83;border-radius:12px;padding:18px 22px;font:23px/1.5 "Noto Sans CJK SC",sans-serif;white-space:normal';
            }''', scene['caption'])
            page.screenshot(path=str(out / f'scene-{name}.png'))
            frames.append({'id': name, 'screenshot': f'scene-{name}.png', 'sha256': sha(out / f'scene-{name}.png')})
            print(json.dumps({'scene': name, 'seconds': scene['seconds'], 'status': 'RECORDING'}), flush=True)
            if name == 'trace':
                for _ in data['trace_replay']:
                    page.locator('#step').click()
                    page.wait_for_timeout(120)
                page.wait_for_timeout(max(1, scene['seconds'] * 1000 - 120 * len(data['trace_replay'])))
            else:
                page.wait_for_timeout(scene['seconds'] * 1000)
        video = page.video
        page.close()
        context.close()
        raw = Path(video.path())
        browser.close()
    assert not errors and not external
    target = out / 'KernelEvolve-current-recorded-V37-G022.mp4'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(raw), '-an', '-vf', 'fps=20',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
    with (out / 'encoding.log').open('xb') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    proof = {'status': 'ACTUAL_CURRENT_BROWSER_RECORDED_DURATION_VERIFICATION_PENDING',
        'data_sha256': sha(data_path), 'recorder_sha256': sha(__file__), 'source_author': 'Codex outer loop',
        'scenes': scenes, 'scene_screenshots': frames, 'planned_scene_seconds': 266,
        'wall_seconds': time.monotonic() - started, 'video_sha256': sha(target), 'video_file': target.name,
        'page_errors': errors, 'external_requests': external, 'owned_browser_closed': True,
        'CPU_encoding_threads': 2, 'new_model_GPU_calls': 0, 'RSI_proven': False,
        'closed_performance_revision': 37, 'current_engineering_generation': 'G022'}
    (out / 'recording.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': proof['status'], 'video': str(target), 'video_sha256': proof['video_sha256']}), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    record(args.demo, args.out)
