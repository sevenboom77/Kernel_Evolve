"""Four physical projections with the original seven FP16 rounding points."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

import cached_decoder as cached

FROZEN_CACHED_SHA = '307044f4c238b095a83eae774e072cca6962c332b54772464497abb8c9b58bbe'
if hashlib.sha256((Path(__file__).parent / 'cached_decoder.py').read_bytes()).hexdigest() != FROZEN_CACHED_SHA:
    raise ValueError('original cached decoder source changed')


class PackedCachedDecoder(cached.CachedDecoder):
    def __init__(self, case, operators=None, projection='fp32_reference'):
        super().__init__(case, operators, projection)
        for name, keys in [('qkv', ('q', 'k', 'v')), ('gate_up', ('gate', 'up'))]:
            joined = torch.cat([case.weights[key] for key in keys], dim=1).contiguous()
            self.register_buffer('w_' + name, joined)
            self.register_buffer('wf_' + name, joined.float())

    def _project_group(self, x, group, keys, trace):
        weight = getattr(self, 'w_' + group)
        if self.projection == 'candidate':
            value = self.operators.gemm(x, weight)
        elif self.projection == 'fp16_library':
            value = x @ weight
        else:
            value = (x.float() @ getattr(self, 'wf_' + group)).half()
        width = weight.shape[1] // len(keys)
        # For B=2 these output-column views have a larger row stride. Original
        # K10 and K06 require contiguous tensors, so materialize explicitly.
        outputs = tuple(value[:, i * width:(i + 1) * width].contiguous() for i in range(len(keys)))
        if trace is not None:
            for key, result in zip(keys, outputs):
                trace[key] = {'inputs': (x, getattr(self, 'w_' + key)), 'output': result}
        return outputs

    def _forward(self, x, trace):
        c = self.config
        shape = (c.batch, 1, c.heads, c.head_dim)
        q, k, v = [value.reshape(shape) for value in self._project_group(x, 'qkv', ('q', 'k', 'v'), trace)]
        rq = self.operators.rope(q, self.cos, self.sin)
        rk = self.operators.rope(k, self.cos, self.sin)
        new_k, new_v = torch.cat((self.past_k, rk), 1), torch.cat((self.past_v, v), 1)
        scores = ((rq.permute(0, 2, 1, 3).float() @ new_k.permute(0, 2, 3, 1).float())
                  / math.sqrt(c.head_dim)).reshape(-1, c.context).contiguous()
        probability = self.operators.softmax(scores, self.mask)
        attended = (probability.reshape(c.batch, c.heads, 1, c.context)
                    @ new_v.permute(0, 2, 1, 3).float()).permute(0, 2, 1, 3).reshape(c.batch, c.embedding).half()
        residual = (x.float() + self._project(attended, 'o', trace).float()).half()
        gate, up = self._project_group(residual, 'gate_up', ('gate', 'up'), trace)
        hidden = self.operators.silu_mul(gate, up)
        result = (residual.float() + self._project(hidden, 'down', trace).float()).half()
        if trace is not None:
            trace.update({'rope_q': {'inputs': (q, self.cos, self.sin), 'output': rq},
                          'rope_k': {'inputs': (k, self.cos, self.sin), 'output': rk},
                          'softmax': {'inputs': (scores, self.mask), 'output': probability},
                          'silu_mul': {'inputs': (gate, up), 'output': hidden}})
        return result, new_k, new_v


def validate_packed_trace(block, trace):
    cached.validate_trace(trace)
    for group, key, count in [('qkv', 'q', 3), ('gate_up', 'gate', 2)]:
        a = trace[key]['inputs'][0]
        weight = getattr(block, 'w_' + group)
        m, k = a.shape
        k2, n = weight.shape
        if k != k2 or m not in (1, 2, 4, 8) or not (128 <= k <= 11008 and 128 <= n <= 11008):
            raise cached.ContractError('merged projection outside original K12 domain')
        cached.base._tensor(weight, (k, n), torch.float16, 8, 'merged K12 weights')
        keys = ('q', 'k', 'v') if count == 3 else ('gate', 'up')
        if not torch.equal(weight, torch.cat([getattr(block, 'w_' + name) for name in keys], 1)):
            raise cached.ContractError('merged weights differ from original FP16 bytes')


def compare_outputs(block, reference, actual, trace):
    validate_packed_trace(block, trace)
    return cached.compare_outputs(reference, actual, trace)
