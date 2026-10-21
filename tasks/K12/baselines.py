"""Baseline implementations for K12 (small_m_gemm).

Baselines are measured with the same GPU slots, the same input contract and the
same tolerance as a candidate. They are part of the frozen protocol and cannot
be changed by an optimisation worker.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NAMES = ("torch_eager", "torch_compile", "specialized")


def build(name: str, static_params: dict) -> nn.Module:
    """Construct one baseline. Raises NotImplementedError for an absent slot."""
    sp = dict(static_params)
    if name == "torch_eager":
        return _Eager(**sp)
    if name == "torch_compile":
        return torch.compile(_Eager(**sp), fullgraph=True, dynamic=False)
    if name == "specialized":
        return _Specialized(**sp)
    raise KeyError(name)

class _Eager(nn.Module):
    def __init__(self, **sp):
        super().__init__()

    def forward(self, a, b):
        return (torch.mm(a.to(torch.float32), b.to(torch.float32)).to(torch.float16),)


class _Specialized(nn.Module):
    """mv when M == 1, otherwise the plain GEMM."""

    def __init__(self, **sp):
        super().__init__()

    def forward(self, a, b):
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        if a.shape[0] == 1:
            return (torch.mv(b.t(), a[0]).unsqueeze(0).to(torch.float16),)
        return (torch.mm(a, b).to(torch.float16),)
