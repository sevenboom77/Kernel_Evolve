"""Experimental fused residual norm; no GPU correctness or gain claimed yet.

The declared arithmetic follows the vectorized CUDA reduction geometry of
PyTorch 2.5.1. The original K02 bytes and old application remain unchanged.
Importing or constructing this module does not compile or initialize CUDA.
"""
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _load_extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    files = [root / 'fused_welford_binding.cpp', root / 'fused_welford_kernel.cu']
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()[:16]
    return load(name='kernelevolve_welford_' + digest, sources=[str(p) for p in files],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3'], verbose=False)


class ModelNew(torch.nn.Module):
    def __init__(self, *, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('epsilon must be finite and positive')

    def forward(self, x, residual, gamma, beta):
        values = (x, residual, gamma, beta)
        if any(not isinstance(v, torch.Tensor) or v.device.type != 'cuda' for v in values):
            raise ValueError('declared fused Welford variant requires CUDA tensors')
        if (x.ndim != 2 or x.shape != residual.shape or x.shape[1] != 768
                or not 1 <= x.shape[0] <= 2048 or gamma.shape != (768,) or beta.shape != (768,)
                or any(v.dtype != torch.float16 or v.device != x.device or not v.is_contiguous() for v in values)):
            raise ValueError('declared FP16 contiguous R-by-768 normalization contract violated')
        return tuple(_load_extension().forward(x, residual, gamma, beta, self.eps))
