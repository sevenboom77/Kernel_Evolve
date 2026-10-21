"""Assemble a recorded graph evidence view without running a browser or GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

SAMPLE_SHA256 = 'ca57bbc438bd0f5002be75b29bb8993b63bb2e1e4ae024e777c87faf34e4b036'
ASSETS = ('graph.html', 'graph-app.js', 'graph-model.js', 'graph-style.css', 'style.css')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(repo, samples, expected_sha256, out):
    repo, samples, out = Path(repo).resolve(), Path(samples), Path(out)
    if out.exists() or out.is_symlink():
        raise ValueError('recorded presentation output must be fresh')
    if expected_sha256 != SAMPLE_SHA256 or sha(samples) != expected_sha256:
        raise ValueError('original public graph sample identity changed')
    data = json.loads(samples.read_text())
    demo = repo / 'harness-evolution/competition/demo'
    for asset in ASSETS:
        if not (demo/asset).is_file() or (demo/asset).is_symlink():
            raise ValueError('required graph presentation asset missing')
    for name, digest in data['source_package'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha(repo/path) != digest:
            raise ValueError('original graph source binding changed')
    out.mkdir(parents=True, mode=0o700)
    for asset in ASSETS:
        shutil.copyfile(demo/asset, out/asset)
    shutil.copyfile(samples, out/'graph-data.json')
    (out/'graph-data.js').write_text('window.GRAPH_EVIDENCE = ' + json.dumps(data, ensure_ascii=True, allow_nan=False) + ';\n')
    for name in data['source_package']:
        target = out/'graph-sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, target)
    metadata = {'status': 'RECORDED_GRAPH_PRESENTATION_ASSEMBLED_BROWSER_UNVERIFIED',
        'recorded_only': True, 'GPU_model_calls': 0, 'browser_verified': False,
        'RSI_proven': False, 'source_author': 'Codex',
        'original_public_sample_sha256': expected_sha256,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'presentation_assets': {asset: sha(out/asset) for asset in ASSETS},
        'source_package': data['source_package'], 'builder_sha256': sha(__file__)}
    (out/'graph-provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--samples-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.samples, args.samples_sha256, args.out)))
