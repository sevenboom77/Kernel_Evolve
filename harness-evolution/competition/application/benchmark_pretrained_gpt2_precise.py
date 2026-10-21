"""A separate, source-bound test of eager FP16 intermediate-cast semantics."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FROZEN_BENCHMARK_SHA = 'f2152dcaf06bb643df35b50292b3fcd0399457a9771e7420c071860bd90ca1a9'


def load_frozen_base():
    path = HERE / 'benchmark_pretrained_gpt2.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_BENCHMARK_SHA:
        raise ValueError('original pretrained benchmark changed')
    original_paths = list(sys.path)
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location('private_pretrained_precision_base', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_paths


def make_protocol(base):
    protocol = copy.deepcopy(base.make_protocol())
    protocol['name'] = 'pretrained-gpt2-k02-precision-casts-v2'
    protocol['compile']['emulate_precision_casts'] = True
    return protocol


def adapt_private_base(base):
    protocol = make_protocol(base)
    original_package = base.source_package
    base.make_protocol = lambda: copy.deepcopy(protocol)

    def source_package(*args):
        package, _ = original_package(*args)
        source = Path(__file__).resolve()
        package['application_sources'][str(source.relative_to(base.ROOT))] = base.model_app.sha(source)
        import torch._inductor.config as config
        package['compiler_precision_profile'] = {
            'emulate_precision_casts': True,
            'installed_config_source_sha256': base.model_app.sha(config.__file__)}
        return package, hashlib.sha256(base.canonical(package)).hexdigest()

    base.source_package = source_package
    return base


def main(argv=None):
    base = adapt_private_base(load_frozen_base())
    import torch._inductor.config as config
    with config.patch({'emulate_precision_casts': True}):
        return base.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
