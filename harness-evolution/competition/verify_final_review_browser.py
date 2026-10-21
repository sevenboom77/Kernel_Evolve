"""Actually open the frozen review offline and over local HTTP, and verify downloads."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(demo):
    manifest = json.loads((demo / 'review-manifest.json').read_text())
    for row in manifest['files']:
        path = demo / row['path']
        if not path.resolve().is_relative_to(demo.resolve()) or path.is_symlink() or not path.is_file() or sha(path) != row['sha256']:
            raise ValueError('review evidence missing or changed: ' + row['path'])
    return manifest


def verify(demo, out):
    demo, out = Path(demo).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('fresh browser verification output required')
    manifest = verify_manifest(demo)
    data = json.loads((demo / 'data.json').read_text())
    if data['release_id'] != 'review-20261009-closed-r1' or data['rsi_performance_established'] is not False:
        raise ValueError('closed recorded review required')
    out.mkdir(parents=True, mode=0o700)
    errors, external, views, downloads = [], [], [], []

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(demo)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/'
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--disable-gpu'])
            try:
                for mode, target in [('offline', (demo / 'index.html').as_uri()), ('HTTP', url)]:
                    for width, height in [(1440, 900), (390, 844)]:
                        context = browser.new_context(viewport={'width': width, 'height': height}, accept_downloads=True)
                        page = context.new_page()
                        page.on('pageerror', lambda error: errors.append(str(error)))

                        def route(request):
                            if request.request.url.startswith(('file:', 'data:', url)):
                                request.continue_()
                            else:
                                external.append(request.request.url)
                                request.abort()

                        page.route('**/*', route)
                        page.goto(target, wait_until='networkidle')
                        assert page.locator('#intro').is_visible()
                        assert '1.59' in page.locator('#intro').inner_text()
                        assert '未证明' in page.locator('#intro').inner_text()
                        for case_id in ('arxiv-first-2048', 'boundary-fixture', 'arxiv-full'):
                            page.locator(f'[data-select="{case_id}"]').click()
                            assert page.locator(f'[data-case="{case_id}"]').is_visible()
                            assert page.locator('[data-case]:visible').count() == 1
                        page.locator('#failures').scroll_into_view_if_needed()
                        for detail in page.locator('details').all():
                            detail.locator('summary').click()
                            assert detail.get_attribute('open') is not None
                        overflow = page.evaluate('document.documentElement.scrollWidth > innerWidth')
                        # Record actual dimensions; a mobile-wide download name must not break layout.
                        assert overflow is False, f'page overflows {mode}/{width}'
                        page.locator('#intro').scroll_into_view_if_needed()
                        screenshot = out / f'{mode}-{width}.png'
                        page.screenshot(path=str(screenshot), full_page=True)
                        views.append({'mode': mode, 'viewport': [width, height], 'graph_cases_clicked': 3,
                            'failure_details_opened': 3, 'horizontal_overflow': overflow,
                            'screenshot': screenshot.name, 'sha256': sha(screenshot)})
                        if mode == 'HTTP' and width == 1440:
                            seen = set()
                            for link in page.locator('a[download]').all():
                                href = link.get_attribute('href')
                                if href in seen:
                                    continue
                                seen.add(href)
                                with page.expect_download() as pending:
                                    link.click()
                                download = pending.value
                                destination = out / 'downloads' / href
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                download.save_as(destination)
                                assert download.failure() is None
                                assert sha(destination) == sha(demo / href)
                                downloads.append({'path': href, 'sha256': sha(destination), 'bytes': destination.stat().st_size})
                                # Chromium throttles a burst of many local downloads;
                                # keep the real user clicks separate in time.
                                page.wait_for_timeout(350)
                        context.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not errors and not external
    proof = {'status': 'ACTUAL_FINAL_REVIEW_OFFLINE_HTTP_DESKTOP_MOBILE_DOWNLOADS_PASS',
        'review_manifest_sha256': sha(demo / 'review-manifest.json'), 'data_sha256': manifest['data_sha256'],
        'verifier_sha256': sha(__file__), 'views': views, 'downloads': downloads,
        'page_errors': errors, 'external_requests': external, 'owned_browser_and_HTTP_server_closed': True,
        'new_model_GPU_calls': 0, 'RSI_proven': False}
    with (out / 'verification.json').open('x') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return {'proof': str(out / 'verification.json'), 'views': len(views), 'downloads': len(downloads)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.demo, args.out)))
