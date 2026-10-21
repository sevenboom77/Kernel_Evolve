"""Baseline implementations for K19 (resize_normalize).

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
        self._sp = dict(sp)

    def forward(self, x, mean, std):
        ho = int(self._sp["Ho"])
        wo = int(self._sp["Wo"])
        c = x.shape[1]
        y = F.interpolate(x, size=(ho, wo), mode="bilinear", align_corners=False)
        return (y.sub(mean.view(1, c, 1, 1)).div(std.view(1, c, 1, 1)),)


class _Specialized(nn.Module):
    def __init__(self, **sp):
        super().__init__()
        raise NotImplementedError("no fused resize+normalise primitive in torch")
