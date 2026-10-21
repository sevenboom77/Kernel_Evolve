"""Reference (ground-truth) implementation for K12: small_m_gemm.

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
    """Small-M GEMM with fp32 accumulation."""

    def __init__(self, **static_params):
        super().__init__()
        self.static_params = dict(static_params)

    def forward(self, a, b):
        acc = a.to(torch.float32) @ b.to(torch.float32)
        return (acc.to(torch.float16),)
