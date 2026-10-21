"""Graph-only CSR variant with parallel high-degree partial reductions."""
from __future__ import annotations

import numpy as np
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


def build_schedule(offsets):
    if offsets.ndim != 1 or offsets.dtype != torch.int64 or not 2 <= offsets.numel() <= 169344:
        raise ValueError('Declared real-graph int64 offsets required')
    cpu = offsets.detach().cpu().numpy()
    degrees = np.diff(cpu)
    if cpu[0] != 0 or cpu[-1] > 1166243 or np.any(degrees < 0) or np.any(degrees > 13155):
        raise ValueError('Original monotone zero-origin offsets and degree<=13155 required')
    small = np.flatnonzero(degrees <= 8)
    medium = np.flatnonzero((degrees > 8) & (degrees <= 32))
    large = np.flatnonzero(degrees > 32)
    counts = (degrees[large] + 127) // 128
    partial_offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
    chunk_begin = np.repeat(cpu[large], counts) + 128 * (
        np.arange(partial_offsets[-1]) - np.repeat(partial_offsets[:-1], counts))
    chunk_end = np.minimum(chunk_begin + 128, np.repeat(cpu[large + 1], counts))
    return {name: torch.tensor(array, dtype=torch.int64, device=offsets.device)
            for name, array in [('small', small), ('medium', medium), ('large', large),
                ('partial_offsets', partial_offsets), ('chunk_begin', chunk_begin), ('chunk_end', chunk_end)]}


@triton.jit
def _short(FEATURES, SOURCES, OFFSETS, TARGETS, OUTPUT, COUNT: tl.constexpr,
           ROWS: tl.constexpr, GROUP: tl.constexpr):
    positions = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = positions < COUNT
    targets = tl.load(TARGETS + positions, mask=valid, other=0)
    begin = tl.load(OFFSETS + targets)
    end = tl.load(OFFSETS + targets + 1)
    edges = begin[:, None] + tl.arange(0, ROWS)[None, :]
    mask = valid[:, None] & (edges < end[:, None])
    source = tl.load(SOURCES + edges, mask=mask, other=0)
    columns = tl.arange(0, 128)
    values = tl.load(FEATURES + source[:, :, None] * 128 + columns[None, None, :],
                     mask=mask[:, :, None], other=0.)
    total = tl.sum(values, axis=1)
    tl.store(OUTPUT + targets[:, None] * 128 + columns[None, :], total, mask=valid[:, None])


@triton.jit
def _partial(FEATURES, SOURCES, BEGIN, END, PARTIAL):
    chunk = tl.program_id(0)
    edges = tl.load(BEGIN + chunk) + tl.arange(0, 128)
    valid = edges < tl.load(END + chunk)
    source = tl.load(SOURCES + edges, mask=valid, other=0)
    columns = tl.arange(0, 128)
    values = tl.load(FEATURES + source[:, None] * 128 + columns[None, :],
                     mask=valid[:, None], other=0.)
    tl.store(PARTIAL + chunk * 128 + columns, tl.sum(values, axis=0))


@triton.jit
def _merge(PARTIAL, PARTIAL_OFFSETS, TARGETS, OUTPUT):
    position = tl.program_id(0)
    target = tl.load(TARGETS + position)
    begin = tl.load(PARTIAL_OFFSETS + position)
    end = tl.load(PARTIAL_OFFSETS + position + 1)
    chunks = begin + tl.arange(0, 128)
    columns = tl.arange(0, 128)
    values = tl.load(PARTIAL + chunks[:, None] * 128 + columns[None, :],
                     mask=(chunks < end)[:, None], other=0.)
    tl.store(OUTPUT + target * 128 + columns, tl.sum(values, axis=0))


def aggregate(features, sources, offsets, schedule=None):
    validate(features, sources, offsets)
    if features.device.type != 'cuda':
        raise ValueError('Split application kernel requires declared CUDA inputs')
    if schedule is None:
        schedule = build_schedule(offsets)
    output = torch.empty((offsets.numel()-1, 128), dtype=torch.float32, device=features.device)
    for name, rows, warps in [('small', 8, 4), ('medium', 32, 8)]:
        count = schedule[name].numel()
        if count:
            _short[(triton.cdiv(count, 4),)](features, sources, offsets, schedule[name], output,
                COUNT=count, ROWS=rows, GROUP=4, num_warps=warps)
    if schedule['large'].numel():
        count = schedule['chunk_begin'].numel()
        partial = torch.empty((count, 128), dtype=torch.float32, device=features.device)
        _partial[(count,)](features, sources, schedule['chunk_begin'], schedule['chunk_end'], partial, num_warps=8)
        _merge[(schedule['large'].numel(),)](partial, schedule['partial_offsets'], schedule['large'], output, num_warps=4)
    return output


def message_passing(inputs, weights, schedule=None):
    if schedule is None:
        schedule = build_schedule(inputs.offsets)
    hidden = inputs.features
    begin, end = inputs.node_begin, inputs.node_end
    for self_weight, neighbor_weight, bias in weights:
        total = aggregate(hidden, inputs.sources, inputs.offsets, schedule)
        mean = total / inputs.degrees.clamp(min=1).unsqueeze(1)
        updated = torch.relu(hidden[begin:end] @ self_weight + mean @ neighbor_weight + bias) + hidden[begin:end]
        hidden = updated if begin == 0 and end == hidden.shape[0] else torch.cat((hidden[:begin], updated, hidden[end:]))
    return hidden[begin:end].clone()


class GraphModel(torch.nn.Module):
    def __init__(self, inputs, weights):
        super().__init__()
        self.inputs, self.weights = inputs, weights
        self.schedule = build_schedule(inputs.offsets)

    def forward(self):
        return message_passing(self.inputs, self.weights, self.schedule)
