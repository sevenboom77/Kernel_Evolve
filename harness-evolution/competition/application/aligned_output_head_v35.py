"""Frozen FP16 GPT-2 output-head layout ablation; no performance is asserted.

Construct after the model reaches its declared device, using tracked parameters.
Packing is setup work; output slicing and contiguous copies stay in forward.
The benchmark must verify original parameter/input hashes before and after work.
"""
from __future__ import annotations

import hashlib

import torch

HIDDEN=768
VOCAB=50257
PADDED_VOCAB=50304
MAX_ROWS=2048


def version(tensor):
    try:
        return tensor._version
    except RuntimeError as exc:
        raise ValueError('tracked immutable source/cache tensors are required') from exc


def tensor_sha(tensor):
    values=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(values).cast('B')).hexdigest()


class AlignedOutputHead(torch.nn.Module):
    def __init__(self, source, *, layout='NN'):
        super().__init__()
        if (not isinstance(source,torch.nn.Linear) or source.bias is not None
                or source.in_features!=HIDDEN or source.out_features!=VOCAB
                or source.weight.shape!=(VOCAB,HIDDEN) or source.weight.dtype!=torch.float16
                or not source.weight.is_contiguous() or layout not in {'NN','TN'}):
            raise ValueError('original FP16 GPT-2 weight and declared NN/TN layout required')
        if source.training:
            raise ValueError('frozen inference source is required')
        source_version=version(source.weight)
        if not torch.isfinite(source.weight).all().item():
            raise ValueError('finite original pretrained weight required')
        self.weight=source.weight
        self.layout=layout
        self._source_version=source_version
        self._cache_device=source.weight.device
        # Normal tracked buffers let a later mutation fail explicitly, including
        # when construction takes place inside the outer inference-mode scope.
        with torch.inference_mode(False),torch.no_grad():
            if layout=='NN':
                packed=source.weight.new_zeros((HIDDEN,PADDED_VOCAB))
                packed[:,:VOCAB].copy_(source.weight.detach().t())
            else:
                packed=source.weight.new_zeros((PADDED_VOCAB,HIDDEN))
                packed[:VOCAB].copy_(source.weight.detach())
        self.register_buffer('_packed_weight',packed,persistent=False)
        self._packed_version=version(packed)
        self._source_weight_sha=tensor_sha(source.weight)
        self._packed_weight_sha=tensor_sha(packed)
        self.eval()

    def validate_cache_binding(self):
        """Full byte binding is qualification work outside the timed forward."""
        if (version(self.weight)!=self._source_version
                or version(self._packed_weight)!=self._packed_version
                or tensor_sha(self.weight)!=self._source_weight_sha
                or tensor_sha(self._packed_weight)!=self._packed_weight_sha):
            raise ValueError('cache binding changed; explicitly prepare a fresh output head')
        return {'layout':self.layout,'source_weight_sha256':self._source_weight_sha,
                'packed_weight_sha256':self._packed_weight_sha,'padded_vocabulary':PADDED_VOCAB,
                'original_vocabulary':VOCAB,'zero_columns':PADDED_VOCAB-VOCAB,
                'cache_bytes':self._packed_weight.numel()*self._packed_weight.element_size(),
                'persistent_parameter_names':list(self.state_dict())}

    def forward(self, hidden):
        if self.training or torch.is_grad_enabled():
            raise ValueError('frozen inference without gradients is required')
        source_same = version(self.weight) == self._source_version
        cache_same = version(self._packed_weight) == self._packed_version
        if torch.compiler.is_compiling():
            # Runtime SymBool assertions stay in the complete graph. A Python
            # bool branch would ask Dynamo to specialize a data-dependent
            # tensor version; no optional lambda message is traceable here.
            torch._check(source_same)
            torch._check(cache_same)
        elif not (source_same and cache_same):
            raise ValueError('cache binding changed; explicitly prepare a fresh output head')
        if (self.weight.device!=self._cache_device or self._packed_weight.device!=self._cache_device
                or self.weight.dtype!=torch.float16 or self._packed_weight.dtype!=torch.float16):
            raise ValueError('cache binding changed; explicitly prepare a fresh output head')
        if (hidden.ndim not in (2,3) or hidden.shape[-1]!=HIDDEN
                or hidden.numel()//HIDDEN<1 or hidden.numel()//HIDDEN>MAX_ROWS
                or hidden.dtype!=torch.float16 or hidden.device!=self._cache_device):
            raise ValueError('declared FP16 hidden-state shape and device required')
        matrix=self._packed_weight if self.layout=='NN' else self._packed_weight.t()
        output=torch.matmul(hidden.reshape(-1,HIDDEN),matrix)
        return output[:,:VOCAB].contiguous().reshape(*hidden.shape[:-1],VOCAB)
