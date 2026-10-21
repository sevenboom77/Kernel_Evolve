"""Record a four-minute browser walkthrough of the frozen, source-bound review."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

from generate_final_review_materials import content
from verify_final_review_browser import verify_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_video(target, out):
    """Use the available encoder to fully decode, avoiding an optional ffprobe dependency."""
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner',
        '-threads', '2', '-i', str(target), '-map', '0:v:0', '-progress', 'pipe:1',
        '-f', 'null', '-'], capture_output=True, text=True, check=True, timeout=90)
    (out / 'full-decode.log').write_text(result.stderr + '\n' + result.stdout)
    duration_match = re.search(r'Duration: (\d+):(\d+):([\d.]+)', result.stderr)
    if not duration_match:
        raise ValueError('actual encoded media duration unavailable')
    hours, minutes, seconds = map(float, duration_match.groups())
    duration = hours * 3600 + minutes * 60 + seconds
    video_line = result.stderr.split('Video:', 1)[1].splitlines()[0]
    geometry = re.search(r'(\d{2,5})x(\d{2,5})', video_line)
    if not geometry or 'h264' not in video_line:
        raise ValueError('actual encoded video format unavailable')
    progress = re.findall(r'out_time_us=(\d+)', result.stdout)
    if not progress or 'progress=end' not in result.stdout:
        raise ValueError('actual full video decoding did not close')
    assert abs(float(progress[-1]) / 1000000 - duration) <= 1
    assert 180 <= duration <= 300
    return {'method': 'bundled_ffmpeg_full_CPU_decode', 'duration_seconds': duration,
        'width': int(geometry[1]), 'height': int(geometry[2]), 'codec': 'h264',
        'full_decode_exit_code': result.returncode, 'decoded_last_out_time_us': int(progress[-1]),
        'log_sha256': sha(out / 'full-decode.log')}


def finalize_existing(demo, out):
    """Close the already encoded recording after the optional probe command failed."""
    demo, out = Path(demo).resolve(), Path(out).resolve()
    verify_manifest(demo)
    failure_path = out / 'missing-ffprobe-stage-failure.json'
    failure = json.loads(failure_path.read_text())
    original = Path(failure['original_actor_preserved'])
    assert sha(original) == failure['original_actor_sha256']
    assert not (out / 'recording.json').exists()
    scenes = json.loads((out / 'scene-plan.json').read_text())
    assert len(scenes) == 9 and sum(s['seconds'] for s in scenes) == 240
    frames = []
    for scene in scenes:
        path = out / ('scene-' + scene['id'] + '.png')
        frames.append({'scene': scene['id'], 'path': path.name, 'sha256': sha(path)})
    target = out / 'KernelEvolve-闭合证据-20261009.mp4'
    measured = probe_video(target, out)
    proof = {'status': 'ACTUAL_FOUR_MINUTE_BROWSER_RECORDING_EXISTING_ENCODING_FULL_DECODE_PASS',
        'data_sha256': sha(demo / 'data.json'), 'review_manifest_sha256': sha(demo / 'review-manifest.json'),
        'original_recording_actor_sha256': sha(original), 'finalizer_sha256': sha(__file__),
        'retained_probe_stage_failure': {'path': str(failure_path), 'sha256': sha(failure_path)},
        'scenes': scenes, 'scene_screenshots': frames, 'planned_seconds': 240,
        'measured_duration_seconds': measured['duration_seconds'], 'media_probe': measured,
        'video_file': target.name, 'video_sha256': sha(target), 'owned_browser_closed_before_probe_failure': True,
        'browser_not_re_recorded': True, 'audio_track': False, 'captioned_recorded_evidence_only': True,
        'new_model_GPU_calls': 0, 'RSI_proven': False}
    (out / 'recording.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    return {'video': str(target), 'duration': measured['duration_seconds'], 'video_sha256': proof['video_sha256']}


def record(demo, out):
    demo, out = Path(demo).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('fresh video output required')
    verify_manifest(demo)
    data = json.loads((demo / 'data.json').read_text())
    slides = content(data)
    sections = ['intro', 'native-chain', 'native', 'graph', 'engineering', 'measurement', 'failures', 'reproduction', 'downloads']
    seconds = [20, 25, 30, 30, 35, 30, 25, 25, 20]
    assert len(slides) == len(sections) == 9 and sum(seconds) == 240
    scenes = [{'id': section, 'seconds': duration, 'caption': note.split(' ', 1)[1]}
              for section, duration, (_, _, note) in zip(sections, seconds, slides)]
    out.mkdir(parents=True, mode=0o700)
    (out / 'scene-plan.json').write_text(json.dumps(scenes, ensure_ascii=False, indent=2) + '\n')
    errors, external, frames, clicks = [], [], [], []
    start = time.monotonic()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--disable-gpu'])
        context = browser.new_context(viewport={'width': 1440, 'height': 810},
            record_video_dir=str(out / 'raw'), record_video_size={'width': 1280, 'height': 720})
        try:
            page = context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))

            def route(request):
                if request.request.url.startswith(('file:', 'data:')):
                    request.continue_()
                else:
                    external.append(request.request.url)
                    request.abort()

            page.route('**/*', route)
            page.goto((demo / 'index.html').as_uri())
            for scene in scenes:
                scene_start = time.monotonic()
                page.locator('#' + scene['id']).scroll_into_view_if_needed()
                page.evaluate('''caption => {
                    let node = document.getElementById('video-caption');
                    if (!node) { node = document.createElement('div'); node.id = 'video-caption'; document.body.append(node); }
                    node.textContent = caption;
                    node.style.cssText = 'position:fixed;bottom:10px;left:24px;right:24px;z-index:9999;background:rgba(8,12,18,.97);color:#edf2f7;border:1px solid #4f6b83;border-radius:10px;padding:14px 20px;font:21px/1.5 "Noto Sans CJK SC",sans-serif';
                }''', scene['caption'])
                if scene['id'] == 'failures':
                    for detail in page.locator('details').all():
                        detail.locator('summary').click()
                        clicks.append({'scene': scene['id'], 'action': 'open failure details'})
                    page.locator('#failures').scroll_into_view_if_needed()
                screenshot = out / ('scene-' + scene['id'] + '.png')
                page.screenshot(path=str(screenshot))
                frames.append({'scene': scene['id'], 'path': screenshot.name, 'sha256': sha(screenshot)})
                print(json.dumps({'scene': scene['id'], 'seconds': scene['seconds'], 'status': 'ACTUAL_BROWSER_RECORDING'}), flush=True)
                if scene['id'] == 'graph':
                    for case_id in ('arxiv-full', 'arxiv-first-2048', 'boundary-fixture'):
                        page.locator(f'[data-select="{case_id}"]').click()
                        assert page.locator(f'[data-case="{case_id}"]').is_visible()
                        clicks.append({'scene': 'graph', 'case': case_id})
                        page.wait_for_timeout(8500)
                remaining = scene['seconds'] - (time.monotonic() - scene_start)
                if remaining > 0:
                    page.wait_for_timeout(remaining * 1000)
            video = page.video
            page.close()
            context.close()
            raw = Path(video.path())
        finally:
            context.close()
            browser.close()
    assert not errors and not external
    target = out / 'KernelEvolve-闭合证据-20261009.mp4'
    with (out / 'encoding.log').open('xb') as stream:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(raw), '-an', '-vf', 'fps=20',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-threads', '2', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(target)], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
    measured = probe_video(target, out)
    duration = measured['duration_seconds']
    proof = {'status': 'ACTUAL_FOUR_MINUTE_FINAL_REVIEW_BROWSER_VIDEO_CLOSED_DURATION_PASS',
        'data_sha256': sha(demo / 'data.json'), 'review_manifest_sha256': sha(demo / 'review-manifest.json'),
        'recorder_sha256': sha(__file__), 'material_content_sha256': sha(Path(__file__).parent / 'generate_final_review_materials.py'),
        'scenes': scenes, 'scene_screenshots': frames, 'actual_graph_and_failure_clicks': clicks,
        'planned_seconds': 240, 'measured_duration_seconds': duration, 'media_probe': measured,
        'elapsed_wall_seconds': time.monotonic() - start, 'video_file': target.name, 'video_sha256': sha(target),
        'page_errors': errors, 'external_requests': external, 'owned_browser_closed': True,
        'audio_track': False, 'captioned_recorded_evidence_only': True,
        'new_model_GPU_calls': 0, 'RSI_proven': False}
    (out / 'recording.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    return {'video': str(target), 'duration': duration, 'video_sha256': proof['video_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--finalize-existing', action='store_true')
    args = parser.parse_args()
    print(json.dumps((finalize_existing if args.finalize_existing else record)(args.demo, args.out)))
