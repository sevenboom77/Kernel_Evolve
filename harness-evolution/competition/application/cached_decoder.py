"""Source-bound single-token decoder with a nonmutating preexisting KV cache.

Seven FP16 projections use FP32 accumulation and explicit FP16 output rounding.
The attention matmuls remain FP32; all implementations share every rounding
point. This fixed-random architecture does not measure pretrained text quality.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import decoder_block as base
from decoder_block import (
    APPLICATION_TOLERANCE, CandidateOperators, ContractError, ReferenceOperators,
    SourceMismatch, sha256_file,
)

K12_MANIFEST = base.ROOT / "harness-evolution/codex-rsi/operators/g003-v13-dev1/manifest.json"
K12_MANIFEST_SHA256 = "51e7aa5f29812f17f165cc5186c4e18bd6cdcc8ea64c355cb6c43d4a2e84bad2"
K12_SHA256 = "65fd89b6c95eb30dd9eb5a9164875a1a326306e3283f23063cfd180013cef780"
OPERATOR_HASHES = {**base.OPERATOR_HASHES, "K12": K12_SHA256}
OPERATOR_TOLERANCES = {**base.OPERATOR_TOLERANCES, "K12": (0.02, 0.02)}
PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")


@dataclass(frozen=True)
class CachedSources:
    manifests: tuple
    operators: dict

    def receipt(self):
        return {"manifests": list(self.manifests), "operators": self.operators}


def verify_sources(manifest=base.DEFAULT_MANIFEST,
                   manifest_sha256=base.DEFAULT_MANIFEST_SHA256,
                   k12_manifest=K12_MANIFEST,
                   k12_manifest_sha256=K12_MANIFEST_SHA256):
    earlier = base.verify_sources(manifest, manifest_sha256)
    path = Path(k12_manifest).resolve()
    try:
        if sha256_file(path) != k12_manifest_sha256:
            raise SourceMismatch("K12 manifest SHA256 mismatch")
        data = json.loads(path.read_bytes())
        if data.get("kind") != "source-bound-verified-kernel-delivery":
            raise SourceMismatch("K12 manifest kind mismatch")
        rows = [r for r in data["operators"] if r["task_id"] == "K12"]
        if len(rows) != 1:
            raise SourceMismatch("K12 must have one original verification identity")
        row = rows[0]
        source = (path.parent / row["source"]).resolve()
        if not source.is_relative_to(path.parent):
            raise SourceMismatch("K12 source escapes bundle")
        if row.get("source_file_sha256") != K12_SHA256 or sha256_file(source) != K12_SHA256:
            raise SourceMismatch("K12 source SHA256 mismatch")
        if not row.get("verify_job_id"):
            raise SourceMismatch("K12 original verification missing")
        entry = {"path": str(source), "sha256": K12_SHA256,
                 "candidate_id": row["candidate_id"], "verify_job_id": row["verify_job_id"]}
        return CachedSources(
            ({"path": str(earlier.manifest), "sha256": earlier.manifest_sha256},
             {"path": str(path), "sha256": k12_manifest_sha256}),
            {**earlier.operators, "K12": entry})
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise SourceMismatch("K12 source bundle unreadable") from exc


@dataclass(frozen=True)
class CachedConfig:
    batch: int = 1
    context: int = 64
    heads: int = 8
    head_dim: int = 64
    mlp_dim: int = 1024

    @property
    def embedding(self):
        return self.heads * self.head_dim

    def validate(self):
        if any(type(v) is not int for v in asdict(self).values()):
            raise ContractError("dimensions must be exact integers")
        if self.batch not in (1, 2) or self.heads != 8 or self.head_dim not in (64, 128):
            raise ContractError("cached decoder requires B1/2 H8 D64/128")
        if not 33 <= self.context <= 128 or self.mlp_dim not in (512, 1024):
            raise ContractError("context33..128 and MLP512/1024 are predeclared")


FIXED_CONFIGS = (
    CachedConfig(context=64), CachedConfig(context=128),
    CachedConfig(context=33, mlp_dim=512),
    CachedConfig(batch=2, context=128, head_dim=128),
)
CASE_LABELS = ("decode-context64", "decode-context128", "min-context33-mlp512", "boundary-b2-d128")


@dataclass
class CachedCase:
    config: CachedConfig
    seed: int
    x: torch.Tensor
    weights: dict
    cos: torch.Tensor
    sin: torch.Tensor
    mask: torch.Tensor
    past_k: torch.Tensor
    past_v: torch.Tensor

    def tensors(self):
        return {name: getattr(self, name) for name in ("x", "cos", "sin", "mask", "past_k", "past_v")} | {
            f"weight.{key}": value for key, value in self.weights.items()}

    def fingerprint(self):
        h = hashlib.sha256(json.dumps({"config": asdict(self.config), "seed": self.seed}, sort_keys=True).encode())
        for key, value in sorted(self.tensors().items()):
            h.update(key.encode())
            h.update(str(value.dtype).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def make_case(config, seed=20261005):
    config.validate()
    if type(seed) is not int or seed < 0:
        raise ContractError("seed must be a nonnegative exact integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    e, f = config.embedding, config.mlp_dim
    x = (torch.randn(config.batch, e, generator=generator) * 0.25).half()
    shapes = {"q": (e, e), "k": (e, e), "v": (e, e), "o": (e, e),
              "gate": (e, f), "up": (e, f), "down": (f, e)}
    weights = {key: (torch.randn(*shape, generator=generator) / math.sqrt(shape[0])).half()
               for key, shape in shapes.items()}
    inv = 10000.0 ** (-torch.arange(0, config.head_dim, 2).float() / config.head_dim)
    angles = torch.arange(config.context).float()[:, None] * inv[None, :]
    raw_k = (torch.randn(config.batch, config.context - 1, config.heads, config.head_dim,
                         generator=generator) * 0.25).half()
    past_k = ReferenceOperators.rope(raw_k, angles[:-1].cos(), angles[:-1].sin())
    past_v = (torch.randn(*raw_k.shape, generator=generator) * 0.25).half()
    # At the final query position all prefix/current tokens are causally valid.
    mask = torch.ones(config.batch * config.heads, config.context, dtype=torch.bool)
    return CachedCase(config, seed, x, weights, angles[-1:].cos(), angles[-1:].sin(),
                      mask, past_k.contiguous(), past_v)


class CachedCandidateOperators(CandidateOperators):
    def gemm(self, a, b):
        return self.models["K12"](a, b)[0]


class CachedDecoder(torch.nn.Module):
    def __init__(self, case, operators=None, projection="fp32_reference"):
        super().__init__()
        case.config.validate()
        if projection not in ("fp32_reference", "fp16_library", "candidate"):
            raise ContractError("unknown projection backend")
        if projection == "candidate" and operators is None:
            raise ContractError("candidate requires a verified operator loader")
        self.config, self.projection = case.config, projection
        self.operators = operators or ReferenceOperators()
        for key, value in case.weights.items():
            # Both buffers derive from the identical FP16 weight bytes. Preparing
            # FP32 reference weights is outside timing but inside the GPU span.
            self.register_buffer(f"w_{key}", value)
            self.register_buffer(f"wf_{key}", value.float())
        for key in ("cos", "sin", "mask", "past_k", "past_v"):
            self.register_buffer(key, getattr(case, key))

    def validate_structure(self, x):
        c = self.config
        if x.shape != (c.batch, c.embedding) or x.dtype != torch.float16 or not x.is_contiguous():
            raise ContractError("input requires contiguous FP16 [B,H*D]")
        if x.device != self.w_q.device:
            raise ContractError("input and weights must share a device")

    def validate_input(self, x):
        self.validate_structure(x)
        if not torch.isfinite(x).all().item():
            raise ContractError("input must be finite")

    def forward(self, x):
        self.validate_structure(x)
        return self._forward(x, None)

    def forward_with_trace(self, x):
        self.validate_input(x)
        trace = {}
        return self._forward(x, trace), trace

    def _project(self, x, key, trace):
        weight = getattr(self, f"w_{key}")
        if self.projection == "candidate":
            value = self.operators.gemm(x, weight)
        elif self.projection == "fp16_library":
            value = x @ weight
        else:
            value = (x.float() @ getattr(self, f"wf_{key}")).half()
        if trace is not None:
            trace[key] = {"inputs": (x, weight), "output": value}
        return value

    def _forward(self, x, trace):
        c = self.config
        shape = (c.batch, 1, c.heads, c.head_dim)
        q = self._project(x, "q", trace).reshape(shape)
        k = self._project(x, "k", trace).reshape(shape)
        v = self._project(x, "v", trace).reshape(shape)
        rq, rk = self.operators.rope(q, self.cos, self.sin), self.operators.rope(k, self.cos, self.sin)
        new_k, new_v = torch.cat((self.past_k, rk), 1), torch.cat((self.past_v, v), 1)
        scores = ((rq.permute(0, 2, 1, 3).float() @ new_k.permute(0, 2, 3, 1).float())
                  / math.sqrt(c.head_dim)).reshape(-1, c.context).contiguous()
        probability = self.operators.softmax(scores, self.mask)
        attended = (probability.reshape(c.batch, c.heads, 1, c.context)
                    @ new_v.permute(0, 2, 1, 3).float()).permute(0, 2, 1, 3).reshape(c.batch, c.embedding).half()
        residual = (x.float() + self._project(attended, "o", trace).float()).half()
        gate, up = self._project(residual, "gate", trace), self._project(residual, "up", trace)
        hidden = self.operators.silu_mul(gate, up)
        out = (residual.float() + self._project(hidden, "down", trace).float()).half()
        if trace is not None:
            trace.update({"rope_q": {"inputs": (q, self.cos, self.sin), "output": rq},
                          "rope_k": {"inputs": (k, self.cos, self.sin), "output": rk},
                          "softmax": {"inputs": (scores, self.mask), "output": probability},
                          "silu_mul": {"inputs": (gate, up), "output": hidden}})
        return out, new_k, new_v


def validate_trace(trace):
    base.validate_operator_trace(trace)
    for key in PROJECTIONS:
        a, b = trace[key]["inputs"]
        m, k = a.shape
        k2, n = b.shape
        if k2 != k or m not in (1, 2, 4, 8) or not (128 <= n <= 11008 and 128 <= k <= 11008):
            raise ContractError("K12 shape outside original domain")
        base._tensor(a, (m, k), torch.float16, 8, "K12 A")
        base._tensor(b, (k, n), torch.float16, 8, "K12 Bmat")
        output = trace[key]["output"]
        base._tensor(output, (m, n), torch.float16, None, "K12 output")
        if output.data_ptr() in (a.data_ptr(), b.data_ptr()):
            raise ContractError("K12 output aliases an input")


def compare_outputs(reference, actual, trace):
    validate_trace(trace)
    diagnostics = {}
    for key, row in trace.items():
        if key in PROJECTIONS:
            task = "K12"
            a, b = row["inputs"]
            expected = (a.float() @ b.float()).half()
        elif key.startswith("rope"):
            task, expected = "K10", ReferenceOperators.rope(*row["inputs"])
        elif key == "softmax":
            task, expected = "K04", ReferenceOperators.softmax(*row["inputs"])
        else:
            task, expected = "K06", ReferenceOperators.silu_mul(*row["inputs"])
        rtol, atol = OPERATOR_TOLERANCES[task]
        torch.testing.assert_close(row["output"], expected, rtol=rtol, atol=atol, equal_nan=False)
        diagnostics[key] = {"task_id": task, "max_abs_error": (row["output"].float() - expected.float()).abs().max().item(),
                            "rtol": rtol, "atol": atol}
    errors = {}
    for label, left, right in zip(("output", "cache_k", "cache_v"), reference, actual):
        torch.testing.assert_close(right, left, **APPLICATION_TOLERANCE, equal_nan=False)
        if not torch.isfinite(right).all().item():
            raise AssertionError("nonfinite decoded output or cache")
        errors[label] = (left.float() - right.float()).abs().max().item()
    return {"operators": diagnostics, "full_decode_max_abs_errors": errors,
            "full_block_tolerance": APPLICATION_TOLERANCE}
