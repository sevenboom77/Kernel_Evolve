"""An isolated packed decoder measurement with the original H0 V15 GEMM bytes."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN_BENCHMARK_SHA = '75d1675e3e29931464bccb9e579ecda687b952042612244fc38383c3f5c558cd'
VARIANT_MANIFEST = ROOT / 'harness-evolution/codex-rsi/operators/g000-v15-dev2/manifest.json'
VARIANT_MANIFEST_SHA = '7c306de80f3c75255d8704a710828f583bbbccbcca21aaecca07147e8a01a06d'
K12_SHA = '22af8289078d96579622e2fcf47ef11e1a50cf3b0f6fa0db6b9da354e52ed276'
PROTOCOL_NAME = 'source-bound-packed-cached-decoder-h0-v15-gemm-v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_profile():
    return {'K12_source_sha256': K12_SHA, 'source_harness_generation': 'H0/G000',
            'original_manifest_sha256': VARIANT_MANIFEST_SHA,
            'original_verify_job_id': '90ba0e93e0304897'}


def load_frozen_base():
    path = HERE / 'benchmark_packed_cached_decoder.py'
    if sha(path) != FROZEN_BENCHMARK_SHA:
        raise ValueError('original packed benchmark changed')
    original_paths = sys.path[:]
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location('private_packed_gemm_variant_base', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_paths


def verify_variant_sources(original_verify, variant_manifest=VARIANT_MANIFEST):
    original = original_verify()
    manifest = Path(variant_manifest).resolve()
    if sha(manifest) != VARIANT_MANIFEST_SHA:
        raise ValueError('original H0 variant manifest changed')
    data = json.loads(manifest.read_bytes())
    for row in data['operators']:
        source = (manifest.parent / row['source']).resolve()
        if not source.is_relative_to(manifest.parent) or sha(source) != row['source_file_sha256']:
            raise ValueError('original H0 variant source changed')
    rows = [row for row in data['operators'] if row['task_id'] == 'K12']
    if len(rows) != 1 or rows[0]['source_file_sha256'] != K12_SHA:
        raise ValueError('original H0 K12 identity changed')
    row = rows[0]
    operators = copy.deepcopy(original.operators)
    operators['K12'] = {'path': str((manifest.parent / row['source']).resolve()),
                        'sha256': K12_SHA, 'candidate_id': row['candidate_id'],
                        'verify_job_id': row['verify_job_id']}
    return type(original)((original.manifests[0], {'path': str(manifest), 'sha256': VARIANT_MANIFEST_SHA}),
                          operators)


def adapt_private_base(base):
    original_verify = base.verify_sources
    original_protocol = base.make_protocol
    original_package = base.source_package

    def make_protocol(*args, **kwargs):
        protocol = copy.deepcopy(original_protocol(*args, **kwargs))
        protocol['name'] = PROTOCOL_NAME
        protocol['operator_variant'] = variant_profile()
        return protocol

    def source_package(*args):
        package, _ = original_package(*args)
        source = Path(__file__).resolve()
        package['application_sources'][str(source.relative_to(base.ROOT))] = sha(source)
        package['operator_variant'] = variant_profile()
        return package, hashlib.sha256(base.canonical(package)).hexdigest()

    base.verify_sources = lambda: verify_variant_sources(original_verify)
    base.make_protocol = make_protocol
    base.source_package = source_package
    return base


def main(argv=None):
    return adapt_private_base(load_frozen_base()).main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
