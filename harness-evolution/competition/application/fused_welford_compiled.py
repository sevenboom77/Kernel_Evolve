"""CUDA-only compiler interfaces; the original numerical kernels stay frozen."""
import copy
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys
from typing import Tuple

import torch

HERE = Path(__file__).resolve().parent


def variant():
    name = 'kernelevolve_compiled_original_welford_wiring'
    path = HERE / 'fused_welford_pretrained.py'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    module = sys.modules[name]
    if Path(module.__file__).resolve() != path:
        raise ValueError('original Welford wiring path differs')
    return module


def validate_fused(x, residual, gamma, beta, eps):
    values = (x, residual, gamma, beta)
    if (x.ndim != 2 or x.shape != residual.shape or x.shape[1] != 768
            or not 1 <= x.shape[0] <= 2048 or gamma.shape != (768,) or beta.shape != (768,)
            or any(v.dtype != torch.float16 or v.device != x.device or not v.is_contiguous()
                   for v in values)):
        raise ValueError('declared FP16 contiguous R-by-768 normalization contract violated')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('epsilon must be finite and positive')


@lru_cache(maxsize=8)
def cuda_module(eps):
    return variant()._kernel_class()(eps=eps)


def cuda_welford(x, residual, gamma, beta, eps):
    return cuda_module(eps)(x, residual, gamma, beta)


@torch.library.custom_op('kernelevolve_welford_v2::fused', mutates_args=(), device_types='cuda')
def fused(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor,
          beta: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_fused(x, residual, gamma, beta, eps)
    return cuda_welford(x, residual, gamma, beta, eps)


@fused.register_fake
def fake_fused(x, residual, gamma, beta, eps):
    validate_fused(x, residual, gamma, beta, eps)
    return torch.empty_like(x), torch.empty_like(x)


class ModelNew(torch.nn.Module):
    def __init__(self, *, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('epsilon must be finite and positive')

    def forward(self, x, residual, gamma, beta):
        validate_fused(x, residual, gamma, beta, self.eps)
        return fused(x, residual, gamma, beta, self.eps)


class ResidualLayerNorm(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.kernel = ModelNew(eps=eps)

    def forward(self, x, residual, gamma, beta, eps):
        if eps != self.eps or x.ndim != 3 or x.shape[-1] != 768 or x.shape != residual.shape:
            raise ValueError('declared pretrained residual input contract violated')
        rows = x.shape[0] * x.shape[1]
        y, r = self.kernel(x.reshape(rows, 768), residual.reshape(rows, 768), gamma, beta)
        return y.reshape(x.shape), r.reshape(x.shape)


def make_candidate(model):
    source_binding = variant().source_binding()
    original = variant().load_original_application()
    candidate = copy.deepcopy(model)
    for index, block in enumerate(candidate.transformer.h):
        candidate.transformer.h[index] = original.FusedResidualLayerNormBlock(
            block, ResidualLayerNorm(block.ln_2.eps))
    candidate.eval()
    if set(candidate.state_dict()) != set(model.state_dict()):
        raise ValueError('candidate pretrained parameter names changed')
    return candidate


def validate_native(x, gamma, beta, eps):
    if (x.ndim < 1 or x.shape[-1] != 768 or gamma.shape != (768,) or beta.shape != (768,)
            or any(v.dtype != torch.float16 or v.device != x.device for v in (x, gamma, beta))):
        raise ValueError('native FP16 LayerNorm contract violated')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('epsilon must be finite and positive')


@torch.library.custom_op('kernelevolve_welford_v2::native_layernorm',
                         mutates_args=(), device_types='cuda')
def native_layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     eps: float) -> torch.Tensor:
    validate_native(x, gamma, beta, eps)
    return torch.nn.functional.layer_norm(x, (768,), gamma, beta, eps)


@native_layernorm.register_fake
def fake_native_layernorm(x, gamma, beta, eps):
    validate_native(x, gamma, beta, eps)
    return torch.empty_like(x)


class NativeLayerNorm(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.normalized_shape != (768,) or original.weight is None or original.bias is None:
            raise ValueError('native LayerNorm profile requires original affine width 768')
        self.weight, self.bias, self.eps = original.weight, original.bias, original.eps

    def forward(self, x):
        return native_layernorm(x, self.weight, self.bias, self.eps)


def prepare_compiled_model(model):
    profiled = copy.deepcopy(model)
    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.LayerNorm):
                setattr(parent, name, NativeLayerNorm(child))
            else:
                replace(child)
    replace(profiled)
    if set(profiled.state_dict()) != set(model.state_dict()):
        raise ValueError('compiler profile changed parameter names')
    return profiled
