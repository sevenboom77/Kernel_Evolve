"""Separate experimental GPT-2 wiring for the source-frozen Welford variant.

CPU wiring verification uses a reference double, not the CUDA kernel. Full
pretrained GPU correctness and performance must be measured independently.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ORIGINAL_APPLICATION_SHA256 = 'eeab9a6743cf98390ebe9c23955f38cee8b4ac2da2d5309e32576d4a0089f0cc'
VARIANT_SOURCE_SHA256 = {
    'fused_welford_layernorm.py': 'a2d94217808f467a9aa59e741a7beb326e2705122617c20bf39aeadf161f9bbd',
    'fused_welford_binding.cpp': '84ee88e91fabf2a65e574dba8cbbdbe894a9e0919fe64acefcc647082912e78d',
    'fused_welford_kernel.cu': 'e077bdecd118b15fbd25b7f2030d9c68140199c5bd8a3c2116b22582bf08df96',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original_application():
    path = HERE / 'pretrained_gpt2.py'
    if sha(path) != ORIGINAL_APPLICATION_SHA256:
        raise ValueError('original pretrained application source changed')
    name = 'kernelevolve_welford_original_pretrained'
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    elif Path(module.__file__).resolve() != path:
        raise ValueError('original pretrained module belongs to another path')
    return module


def source_binding():
    required = {'fused_welford_layernorm.py', 'fused_welford_binding.cpp', 'fused_welford_kernel.cu'}
    if set(VARIANT_SOURCE_SHA256) != required:
        raise ValueError('Welford source set differs from CPU-built variant')
    sources = {}
    for name, expected in VARIANT_SOURCE_SHA256.items():
        path = HERE / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Welford source differs from CPU-built variant')
        sources[name] = {'path': str(path), 'sha256': expected}
    return {'kind': 'experimental-source-bound-fused-welford-variant', 'sources': sources,
            'author': 'Codex outer loop', 'GPU_correctness_verified': False,
            'GPU_performance_measured': False, 'RSI_causality_proven': False}


def _kernel_class():
    binding = source_binding()
    path = Path(binding['sources']['fused_welford_layernorm.py']['path'])
    name = 'kernelevolve_welford_variant_' + sha(path)[:16]
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    elif Path(module.__file__).resolve() != path:
        raise ValueError('Welford module belongs to another path')
    return module.ModelNew


class ResidualLayerNorm(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.kernel = _kernel_class()(eps=eps)

    def forward(self, x, residual, gamma, beta, eps):
        if eps != self.eps:
            raise ValueError('normalization epsilon differs from the original block')
        if x.ndim != 3 or x.shape[-1] != 768 or residual.shape != x.shape:
            raise ValueError('declared pretrained residual input shape differs')
        rows = x.shape[0] * x.shape[1]
        y, r = self.kernel(x.reshape(rows, 768), residual.reshape(rows, 768), gamma, beta)
        return y.reshape(x.shape), r.reshape(x.shape)


def make_candidate(model):
    source_binding()
    original = load_original_application()
    candidate = copy.deepcopy(model)
    for index, block in enumerate(candidate.transformer.h):
        normalize = ResidualLayerNorm(block.ln_2.eps)
        candidate.transformer.h[index] = original.FusedResidualLayerNormBlock(block, normalize)
    candidate.eval()
    if set(candidate.state_dict()) != set(model.state_dict()):
        raise ValueError('candidate pretrained parameter names changed')
    return candidate
