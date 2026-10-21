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
    path=Path(__file__).parent/'generate_current_materials_v49.py'
    s=importlib.util.spec_from_file_location('current_video_closed_v47_scope',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.content(data)
    pair=data['closed_ablation'];p,c=pair['arms']
    rows=[
        ('intro',18,'KernelEvolve：20路原生AGH、GPU验证、两组全闭后诊断、批量修改真实源码再同条件实测。固定V47/G029记录作品；外层Codex，accepted0、H0保持、RSI未成立。'),
        ('operators',31,'原算子形状、数值门、源码和独立验证身份可查。弱收益和失败保留；当前实验GPU0–4五评测器，每次GO和GPU作业新探util0/free3072MiB，允许空闲驻留，不结束外部进程。'),
        ('gemm',24,'历史small-M GEMM约4.273×，只对应原指定五形状。它来自H0搜索，不归因于G029，也不能乘其他算子倍率推导整模型收益。'),
        ('graph',22,'ogbn-arxiv全图169343节点、1166243边，两设备相对合格强库最快值约1.59×。投影固定随机、未训练，不证明分类精度或AGH源码进化。'),
        ('APP35',31,'APP35真实GPT-2五卡：12编译后端不正确、24不可用、36普通后端正确。此历史协议GPU2仅辅助正确性，不计加速。强整模型基线未齐，尚无稳定模型收益。'),
        ('G029',44,'G029一次有界输出截断续接：丢弃全部半截tool，保留费用与24请求/四候选。2390 Core与1236实际SDK正文通过，另七失败保留；3633源码/175运行文件。SDK重启turn未测，持久重放/取消在真实会话实现验证。源码作者外层Codex；固定V48登记，没有未闭账性能成绩。'),
        ('V47',34,f"V47 G014正确{p['correct_count']}/20、G028正确{c['correct_count']}/20；候选只有19最终评测、K16缺失，原20分母保留。比{pair['development_geomean_ratio']:.6f}、95%CI[{pair['development_paired_ci_low']:.6f},{pair['development_paired_ci_high']:.6f}]，GPU成本比1.341300；未过原门，未晋级。"),
        ('trace',26,'回放历史真实工具事件与成功、失败、读取和修改。它是固定记录，不是现场模型调用；当前V48中间DEV成绩不读取。生产提示的效果须完整对照，不能从提示文本推断。'),
        ('history',18,'闭账V44/G026单组正向但GPU门失败；V45重复、V46/G027、V47/G028均未接受。V47 carry 45537请求、32757未知，全部原失败留档。V48固定144 CPU/250队列登记没有最终成绩；页面旧V37历史卡仍按原时点保留。'),
        ('next',18,'继续两臂全闭后分析DEV、修改AGH并工程验证和实测。平均14/20、稳定10%、三组开发及独立held/new、连续两代和效率门全过才证明持续收益。当前RSI未成立，强APP/跨设备与最新完整作品还须实际验收。'),
    ]
    if not 180<=sum(seconds for _,seconds,_ in rows)<=300:
        raise ValueError('original recording duration range exceeded')
    return [{'id':name,'seconds':seconds,'caption':caption} for name,seconds,caption in rows]


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
                tab = 'evolution' if name in ('G029', 'V47') else 'trace' if name == 'trace' else 'integrity' if name in ('history', 'next') else 'results'
                page.locator(f'[data-tab="{tab}"]').click()
                target = {'operators': '#operators', 'gemm': '#operator-variants',
                    'APP35': '#whole-model-app35', 'V47': '#closed-v47-original-gates',
                    'trace': '#trace-events', 'history': '#closed-history-v37', 'next': '#limitations'}.get(name)
                if name == 'G029':
                    page.locator('.change').filter(has=page.locator('.generation', has_text='G029')).scroll_into_view_if_needed()
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
    target = out / 'KernelEvolve-current-recorded-V47-G029.mp4'
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
        'closed_performance_revision': 47, 'current_engineering_generation': 'G029'}
    (out / 'recording.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': proof['status'], 'video': str(target), 'video_sha256': proof['video_sha256']}), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    record(args.demo, args.out)
