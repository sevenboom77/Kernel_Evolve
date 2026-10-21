"""Use the original K02 bytes for a pretrained GPT-2 residual/LayerNorm pair."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

# This application uses local PyTorch weights only. Optional TensorFlow/Flax
# backend imports must not initialize an unrelated accelerator runtime.
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

ROOT = Path(__file__).resolve().parents[3]
MODEL_REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
MODEL_WEIGHT_SHA256 = '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
K02_SHA256 = '4bc789889609bad37e3f8bc2f46e1965e27f42fe2bce617c375c9a9f8979e13d'
DEFAULT_MANIFEST = ROOT / 'harness-evolution/codex-rsi/operators/g000-v14-dev1/manifest.json'
DEFAULT_MANIFEST_SHA256 = 'b7beb14a51cad3d31e6e596d6799f8709efda279de0d79eca33f9c2ef8612fa7'
MODEL_FILES = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')
MODEL_FILE_SHA256S = {
    'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
    'model.safetensors': MODEL_WEIGHT_SHA256,
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
}
SHAPES = ((1, 33), (1, 128), (1, 512), (2, 1024))
TEXT = ('The experiment records every candidate, including failures. '
        'A faster operation must preserve its numerical contract. '
        'We compare identical model weights and input tokens on a clean device. '
        'The evaluator retains source hashes and paired timing samples. '
        'Independent measurements test whether an optimization repeats. ')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_receipt(directory):
    directory = Path(directory).resolve()
    files = {name: {'sha256': sha(directory / name), 'bytes': (directory / name).stat().st_size}
             for name in MODEL_FILES}
    config = json.loads((directory / 'config.json').read_text())
    if (any(files[name]['sha256'] != MODEL_FILE_SHA256S[name] for name in MODEL_FILES)
            or config.get('model_type') != 'gpt2'
            or any(config.get(k) != v for k, v in {'n_embd': 768, 'n_head': 12, 'n_layer': 12,
                                                   'n_positions': 1024, 'vocab_size': 50257}.items())):
        raise ValueError('the declared pretrained GPT-2 weight or configuration differs')
    return {'model': 'gpt2', 'revision': MODEL_REVISION, 'files': files,
            'weights_included_in_delivery': False, 'model_downloaded': False,
            'model_API_requests': 0, 'model_directory': str(directory)}


def verify_kernel(manifest=DEFAULT_MANIFEST, manifest_sha256=DEFAULT_MANIFEST_SHA256):
    path = Path(manifest).resolve()
    if sha(path) != manifest_sha256:
        raise ValueError('normalization manifest identity changed')
    document = json.loads(path.read_bytes())
    rows = [r for r in document['operators'] if r['task_id'] == 'K02']
    if document.get('kind') != 'source-bound-verified-kernel-delivery' or len(rows) != 1:
        raise ValueError('one source-bound K02 identity required')
    row = rows[0]
    if row['candidate_id'] != 'c003-1b91fa14' or row['verify_job_id'] != 'f27c892668044dc9':
        raise ValueError('original K02 candidate or verification identity changed')
    source = (path.parent / row['source']).resolve()
    if (not source.is_relative_to(path.parent) or row['source_file_sha256'] != K02_SHA256
            or sha(source) != K02_SHA256 or not row['verify_job_id']):
        raise ValueError('original verified K02 kernel bytes changed')
    return {'manifest': {'path': str(path), 'sha256': manifest_sha256},
            'source': {'path': str(source), 'sha256': K02_SHA256},
            'candidate_id': row['candidate_id'], 'verify_job_id': row['verify_job_id'],
            'source_harness_generation': 'H0/G000', 'harness_performance_causality_proven': False}


def validate_norm_inputs(x, residual, gamma, beta):
    if (x.ndim != 3 or x.shape != residual.shape or x.dtype != torch.float16
            or residual.dtype != torch.float16 or not 1 <= x.shape[0] * x.shape[1] <= 2048
            or not 33 <= x.shape[2] <= 11008 or gamma.shape != (x.shape[2],)
            or beta.shape != gamma.shape or gamma.dtype != torch.float16 or beta.dtype != torch.float16
            or not x.is_contiguous() or not residual.is_contiguous()
            or any(t.device != x.device for t in (residual, gamma, beta))):
        raise ValueError('original residual LayerNorm input contract violated')
    if any(not torch.isfinite(t).all().item() for t in (x, residual, gamma, beta)):
        raise ValueError('nonfinite residual LayerNorm input')


def reference_residual_layernorm(x, residual, gamma, beta, eps):
    r = (x.float() + residual.float()).to(torch.float16)
    y = torch.nn.functional.layer_norm(r, (r.shape[-1],), gamma, beta, eps)
    return y, r


class KernelResidualLayerNorm(torch.nn.Module):
    def __init__(self, binding, eps):
        super().__init__()
        source = binding['source']
        if sha(source['path']) != K02_SHA256:
            raise ValueError('K02 executable source changed')
        name = 'pretrained_original_K02'
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, source['path'])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                if sys.modules.get(name) is module:
                    del sys.modules[name]
                raise
        elif Path(module.__file__).resolve() != Path(source['path']).resolve():
            raise ValueError('loaded K02 module belongs to another source')
        self.kernel = module.ModelNew(eps=eps)
        self.eps = eps

    def forward(self, x, residual, gamma, beta, eps):
        if eps != self.eps:
            raise ValueError('normalization epsilon differs from the trained block')
        rows, hidden = x.shape[0] * x.shape[1], x.shape[2]
        y, r = self.kernel(x.reshape(rows, hidden), residual.reshape(rows, hidden), gamma, beta)
        return y.reshape(x.shape), r.reshape(x.shape)


class FusedResidualLayerNormBlock(torch.nn.Module):
    def __init__(self, original, normalize):
        super().__init__()
        if hasattr(original, 'crossattention'):
            raise ValueError('cross-attention is outside this frozen application')
        self.ln_1, self.ln_2 = original.ln_1, original.ln_2
        self.attn, self.mlp = original.attn, original.mlp
        self.normalize = normalize

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False):
        if (self.training or layer_past is not None or use_cache or output_attentions
                or encoder_hidden_states is not None or encoder_attention_mask is not None):
            raise ValueError('only declared inference without cache/cross-attention/output-attentions is supported')
        residual = hidden_states
        attention = self.attn(self.ln_1(hidden_states), layer_past=None, attention_mask=attention_mask,
                              head_mask=head_mask, use_cache=False, output_attentions=False)
        normalized, residual = self.normalize(attention[0], residual, self.ln_2.weight,
                                              self.ln_2.bias, self.ln_2.eps)
        hidden_states = residual + self.mlp(normalized)
        return (hidden_states,) + attention[2:]


def make_candidate(model, binding=None, *, reference_only=False):
    candidate = copy.deepcopy(model)
    for number, block in enumerate(candidate.transformer.h):
        normalize = reference_residual_layernorm if reference_only else KernelResidualLayerNorm(binding, block.ln_2.eps)
        candidate.transformer.h[number] = FusedResidualLayerNormBlock(block, normalize)
    candidate.eval()
    if set(candidate.state_dict()) != set(model.state_dict()):
        raise ValueError('candidate pretrained parameter names changed')
    return candidate


def input_ids(directory):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, use_fast=True,
                                              trust_remote_code=False)
    tokens = tokenizer.encode(TEXT, add_special_tokens=False)
    return {f'B{batch}_S{sequence}': torch.tensor(
        [tokens[i % len(tokens)] for i in range(batch * sequence)], dtype=torch.long).reshape(batch, sequence)
            for batch, sequence in SHAPES}


def load_model(directory):
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained(str(directory), local_files_only=True, use_safetensors=True,
                                           torch_dtype=torch.float16, attn_implementation='sdpa').eval()
