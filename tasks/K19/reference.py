"""Reference (ground-truth) implementation for K19: resize_normalize.

This file DEFINES the task. It is never timed, never optimised and never
replaced. It runs in an isolated oracle process, so the precision switches
below can never leak into a candidate's measurements.

A candidate kernel must not import this module and must not read its outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class Model(nn.Module):
    """Bilinear resize (align_corners=False) followed by per-channel normalisation."""

    def __init__(self, **static_params):
        super().__init__()
        self.static_params = dict(static_params)

    def forward(self, x, mean, std):
        ho = int(self.static_params["Ho"])
        wo = int(self.static_params["Wo"])
        y = F.interpolate(x, size=(ho, wo), mode="bilinear", align_corners=False)
        c = x.shape[1]
        y = (y - mean.view(1, c, 1, 1)) / std.view(1, c, 1, 1)
        return (y,)
