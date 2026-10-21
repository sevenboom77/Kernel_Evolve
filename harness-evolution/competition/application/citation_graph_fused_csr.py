"""Codex-authored, graph-domain gather plus CSR sum; distinct from original K15."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def validate(features, sources, offsets):
    if (features.ndim != 2 or features.shape[1] != 128
            or not 1 <= features.shape[0] <= 169343 or features.dtype != torch.float32
            or sources.ndim != 1 or not 0 <= sources.numel() <= 1166243
            or offsets.ndim != 1 or not 2 <= offsets.numel() <= 169344
            or sources.dtype != torch.int64 or offsets.dtype != torch.int64
            or any(value.device != features.device or not value.is_contiguous()
                   for value in (features, sources, offsets))):
        raise ValueError('Declared original real-graph FP32 CSR metadata required')


@triton.jit
def _gather_csr_sum(FEATURES, SOURCES, OFFSETS, OUTPUT,
                    BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    segment = tl.program_id(0)
    begin = tl.load(OFFSETS + segment)
    end = tl.load(OFFSETS + segment + 1)
    columns = tl.arange(0, BLOCK_D)
    rows = tl.arange(0, BLOCK_R)
    total = tl.zeros((BLOCK_D,), tl.float32)
    while begin < end:
        edges = begin + rows
        source = tl.load(SOURCES + edges, mask=edges < end, other=0)
        values = tl.load(FEATURES + source[:, None] * 128 + columns[None, :],
                         mask=(edges < end)[:, None], other=0.)
        total += tl.sum(values, axis=0)
        begin += BLOCK_R
    tl.store(OUTPUT + segment * 128 + columns, total)


def aggregate(features, sources, offsets):
    validate(features, sources, offsets)
    if features.device.type != 'cuda':
        raise ValueError('Fused application kernel requires declared CUDA inputs')
    segments = offsets.numel() - 1
    output = torch.empty((segments, 128), dtype=torch.float32, device=features.device)
    _gather_csr_sum[(segments,)](features, sources, offsets, output,
                               BLOCK_R=32, BLOCK_D=128, num_warps=4)
    return output


def message_passing(inputs, weights):
    hidden = inputs.features
    begin, end = inputs.node_begin, inputs.node_end
    for self_weight, neighbor_weight, bias in weights:
        total = aggregate(hidden, inputs.sources, inputs.offsets)
        mean = total / inputs.degrees.clamp(min=1).unsqueeze(1)
        updated = torch.relu(hidden[begin:end] @ self_weight + mean @ neighbor_weight + bias) + hidden[begin:end]
        hidden = updated if begin == 0 and end == hidden.shape[0] else torch.cat((hidden[:begin], updated, hidden[end:]))
    return hidden[begin:end].clone()


class GraphModel(torch.nn.Module):
    def __init__(self, inputs, weights):
        super().__init__()
        self.inputs, self.weights = inputs, weights

    def forward(self):
        return message_passing(self.inputs, self.weights)
