"""Disclosed native addmm boundary for both original compiled GPT-2 paths."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ORIGINAL_INTERFACE_SHA = 'dab446dfa18dd0bdb7be08bc02f292b90c245a7c35fd57b5f72142119436db77'


def original():
    path = HERE / 'fused_welford_compiled.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_INTERFACE_SHA:
        raise ValueError('Original compiler interface changed')
    name = 'kernelevolve_welford_compiled_runtime_v2'
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
        raise ValueError('Original compiler interface belongs to another path')
    return module


def validate_addmm(x, weight, bias):
    if (x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1 or x.shape[0] < 1
            or x.shape[1] != weight.shape[0] or bias.shape[0] != weight.shape[1]
            or any(value.dtype != torch.float16 or value.device != x.device
                   for value in (x, weight, bias))):
        raise ValueError('Original FP16 projection shape, dtype and device required')


@torch.library.custom_op('kernelevolve_welford_v3::native_addmm',
                         mutates_args=(), device_types='cuda')
def native_addmm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    validate_addmm(x, weight, bias)
    return torch.addmm(bias, x, weight)


@native_addmm.register_fake
def fake_native_addmm(x, weight, bias):
    validate_addmm(x, weight, bias)
    return x.new_empty((x.shape[0], weight.shape[1]))


class NativeConv1D(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight, self.bias, self.nf = original.weight, original.bias, original.nf

    def forward(self, x):
        size = x.size()[:-1] + (self.nf,)
        return native_addmm(x.reshape(-1, x.size(-1)), self.weight, self.bias).reshape(size)


def prepare_compiled_model(model):
    from transformers.pytorch_utils import Conv1D
    profiled = original().prepare_compiled_model(model)

    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, Conv1D):
                setattr(parent, name, NativeConv1D(child))
            else:
                replace(child)

    replace(profiled)
    if set(profiled.state_dict()) != set(model.state_dict()):
        raise ValueError('Native projection profile changed parameter names')
    return profiled
