"""Disclosed native SDPA boundary, retaining the frozen addmm profile."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention

HERE = Path(__file__).resolve().parent
ADDMM_SHA = '0745a40a5382951b1302cca76ac044fde9dfec1864c1652b2995d0383e31b856'


def addmm_interface():
    path = HERE / 'fused_welford_compiled_addmm.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ADDMM_SHA:
        raise ValueError('Frozen native addmm interface changed')
    name = 'source_bound_native_addmm_ablation'
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
        raise ValueError('Native addmm interface belongs to another source path')
    return module


def validate_sdpa(query, key, value):
    if (query.ndim != 4 or query.shape[0] not in (1, 2)
            or query.shape[1] != 12 or not 1 <= query.shape[2] <= 1024
            or query.shape[3] != 64
            or any(item.shape != query.shape or item.dtype != torch.float16
                   or item.device != query.device for item in (query, key, value))):
        raise ValueError('Original FP16 GPT-2 self-attention domain required')


@torch.library.custom_op('kernelevolve_welford_v4::native_sdpa',
                         mutates_args=(), device_types='cuda')
def native_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                causal: bool) -> torch.Tensor:
    validate_sdpa(query, key, value)
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=causal).contiguous()


@native_sdpa.register_fake
def fake_native_sdpa(query, key, value, causal):
    validate_sdpa(query, key, value)
    return query.new_empty(query.shape)


class NativeSDPAAttention(GPT2SdpaAttention):
    def forward(self, hidden_states, layer_past=None, attention_mask=None,
                head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, use_cache=False,
                output_attentions=False):
        if (self.training or layer_past is not None or attention_mask is not None
                or head_mask is not None or encoder_hidden_states is not None
                or encoder_attention_mask is not None or use_cache
                or output_attentions or self.is_cross_attention):
            raise ValueError('Frozen evaluation self-attention without cache or masks required')
        batch, length, _ = hidden_states.size()
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        output = native_sdpa(query, key, value, length > 1)
        output = output.transpose(1, 2).contiguous().view(batch, length, self.embed_dim)
        return self.resid_dropout(self.c_proj(output)), None, None


def prepare_compiled_model(model):
    prepared = addmm_interface().prepare_compiled_model(model)
    count = 0
    for module in prepared.modules():
        if isinstance(module, GPT2SdpaAttention):
            module.__class__ = NativeSDPAAttention
            count += 1
    if count != 12 or set(prepared.state_dict()) != set(model.state_dict()):
        raise ValueError('All twelve original GPT-2 attention blocks and parameter names required')
    return prepared
