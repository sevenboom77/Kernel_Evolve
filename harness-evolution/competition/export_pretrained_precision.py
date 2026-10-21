"""Read-only validation of a separately declared compiler precision diagnostic."""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL_EXPORT_SHA = '425597ecdecf73bb21615f71e0b2a1492ad58c9e99e9d9b9dfb3e03c75988b3d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile():
    spec = importlib.util.spec_from_file_location('private_precision_profile',
        HERE / 'application/benchmark_pretrained_gpt2_precise.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_profile_sources():
    profile = load_profile()
    profile.load_frozen_base()  # Refuse drift in the previously measured benchmark.
    paths = [HERE / 'application/benchmark_pretrained_gpt2_precise.py',
             HERE / 'application/benchmark_pretrained_gpt2.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def expected_compiler_profile():
    import torch._inductor.config as config
    return {'emulate_precision_casts': True, 'installed_config_source_sha256': sha(config.__file__)}


def load_validator():
    """Change only this private module's import dependency, never sys.modules."""
    path = HERE / 'export_pretrained_application.py'
    if sha(path) != ORIGINAL_EXPORT_SHA:
        raise ValueError('original read-only exporter changed')
    profile = load_profile()
    benchmark = profile.adapt_private_base(profile.load_frozen_base())
    original_import = builtins.__import__

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'benchmark_pretrained_gpt2' and level == 0:
            return benchmark
        return original_import(name, globals, locals, fromlist, level)

    spec = importlib.util.spec_from_file_location('private_precision_validator', path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__['__builtins__'] = {**vars(builtins), '__import__': local_import}
    original_paths = list(sys.path)
    try:
        sys.path.insert(0, str(HERE))
        sys.path.insert(0, str(HERE / 'application'))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_paths
    return module


def validate_profile(raw):
    profile = load_profile()
    if raw.get('protocol') != profile.make_protocol(profile.load_frozen_base()):
        raise ValueError('explicit precision protocol differs')
    package = raw.get('source_package', {})
    if package.get('compiler_precision_profile') != expected_compiler_profile():
        raise ValueError('actual compiler precision source or flag differs')
    if any(package.get('application_sources', {}).get(path) != digest
           for path, digest in required_profile_sources().items()):
        raise ValueError('precision profile or original benchmark source differs')


def validate_partition(raw):
    validate_profile(raw)
    return load_validator().validate_partition(raw)


def make_report(entries, repo):
    for path, digest, _ in entries:
        if sha(path) != digest:
            raise ValueError('raw precision result changed')
        validate_profile(json.loads(Path(path).read_bytes()))
    report = load_validator().make_report(entries, repo)
    report['precision_diagnostic'] = {
        'emulate_precision_casts': True,
        'original_validator_sha256': ORIGINAL_EXPORT_SHA,
        'exporter_source_sha256': sha(__file__),
        'original_tolerances_preserved': True,
        'prior_failed_results_superseded': False,
        'fresh_GPU_reproduction': False}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition-result', nargs=3, action='append', required=True,
                        metavar=('RAW', 'SHA256', 'QUEUE'))
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = make_report(args.partition_result, args.repo)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'sha256': sha(args.out), 'GPU_samples': report['raw_gpu_latency_samples'],
                      'conclusions': report['conclusions']}))


if __name__ == '__main__':
    main()
