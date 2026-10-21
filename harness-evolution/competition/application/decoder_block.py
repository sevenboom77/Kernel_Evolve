"""A fixed-weight decoder computation block with source-bound operator substitution.

This is an architecture benchmark, not a pretrained language model. Projections
and attention matmuls accumulate in FP32. Operator inputs, residuals and the
final output have explicit FP16 rounding points shared by both implementations.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "harness-evolution/codex-rsi/operators/g001/manifest.json"
DEFAULT_MANIFEST_SHA256 = "a47283321704dcf990fdfc5b612e838b79a630dd95f1cf6edc641990d8b39f4f"
OPERATOR_HASHES = {
    "K04": "39b2320674e9f6877398950d3856d9deb5dc086d59eea4b7b4213a00ec02cc8f",
    "K06": "14ddd53dd56edd96e92fd0348fb14e619b9511393069b05b3d235d49ffebd2d6",
    "K10": "99ccb19096acf19e4be6d20079e67959747eaa093f5b8009695f60f3cd8005d2",
}
OPERATOR_TOLERANCES = {"K04": (1e-4, 1e-5), "K06": (0.002, 0.01), "K10": (0.002, 0.01)}
# Application output tolerance, declared independently of the immutable tasks.
APPLICATION_TOLERANCE = {"rtol": 0.005, "atol": 0.005}


class ContractError(ValueError):
    pass


class SourceMismatch(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceBindings:
    manifest: Path
    manifest_sha256: str
    operators: dict

    def receipt(self) -> dict:
        return {"manifest": str(self.manifest), "manifest_sha256": self.manifest_sha256,
                "operators": self.operators}


def verify_sources(manifest: Path = DEFAULT_MANIFEST,
                   expected_sha256: str = DEFAULT_MANIFEST_SHA256) -> SourceBindings:
    """Verify bytes and containment without importing any candidate module."""
    manifest = Path(manifest).resolve()
    try:
        if sha256_file(manifest) != expected_sha256:
            raise SourceMismatch("operator manifest SHA256 mismatch")
        data = json.loads(manifest.read_text())
        if data.get("kind") != "source-bound-verified-kernel-delivery":
            raise SourceMismatch("operator manifest kind mismatch")
        entries = {}
        for row in data["operators"]:
            key = row["task_id"]
            if key not in OPERATOR_HASHES:
                continue
            if key in entries:
                raise SourceMismatch(f"duplicate {key} operator")
            path = (manifest.parent / row["source"]).resolve()
            if not path.is_relative_to(manifest.parent):
                raise SourceMismatch(f"{key} source escapes bundle")
            expected = OPERATOR_HASHES[key]
            if row.get("source_file_sha256") != expected or sha256_file(path) != expected:
                raise SourceMismatch(f"{key} source SHA256 mismatch")
            if not row.get("verify_job_id"):
                raise SourceMismatch(f"{key} missing original verification identity")
            entries[key] = {"path": str(path), "sha256": expected,
                            "candidate_id": row["candidate_id"],
                            "verify_job_id": row["verify_job_id"]}
        if set(entries) != set(OPERATOR_HASHES):
            raise SourceMismatch("required K04/K06/K10 source missing")
        return SourceBindings(manifest, expected_sha256, entries)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SourceMismatch(f"source bundle unreadable: {exc}") from exc


@dataclass(frozen=True)
class DecoderConfig:
    batch: int = 1
    sequence: int = 64
    heads: int = 8
    head_dim: int = 64
    mlp_dim: int = 1024

    @property
    def embedding(self) -> int:
        return self.heads * self.head_dim

    def validate(self) -> None:
        if any(type(v) is not int for v in asdict(self).values()):
            raise ContractError("shape dimensions must be exact integers")
        if not (1 <= self.batch <= 2 and 8 <= self.heads <= 32
                and 64 <= self.head_dim <= 128 and self.head_dim % 2 == 0):
            raise ContractError("K10 requires B=1..2, H=8..32, even D=64..128")
        if not 33 <= self.sequence <= 1024:
            raise ContractError("K04 and K10 require sequence=33..1024")
        if self.batch * self.heads * self.sequence > 2048:
            raise ContractError("K04 flattened rows exceed 2048")
        if self.batch * self.sequence > 2048 or not 33 <= self.mlp_dim <= 11008:
            raise ContractError("K06 requires R<=2048 and MLP dimension=33..11008")


@dataclass
class DecoderCase:
    config: DecoderConfig
    seed: int
    x: torch.Tensor
    weights: dict[str, torch.Tensor]
    cos: torch.Tensor
    sin: torch.Tensor
    mask: torch.Tensor

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"x": self.x, "cos": self.cos, "sin": self.sin, "mask": self.mask,
                **{f"weight.{k}": v for k, v in self.weights.items()}}

    def fingerprint(self) -> str:
        h = hashlib.sha256(json.dumps({"config": asdict(self.config), "seed": self.seed},
                                     sort_keys=True).encode())
        for name, tensor in sorted(self.tensors().items()):
            h.update(name.encode())
            h.update(str(tensor.dtype).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def make_case(config: DecoderConfig, seed: int = 20261005) -> DecoderCase:
    config.validate()
    if type(seed) is not int or seed < 0:
        raise ContractError("seed must be a nonnegative exact integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    e, f = config.embedding, config.mlp_dim
    x = (torch.randn(config.batch, config.sequence, e, generator=generator) * 0.25).half()
    shapes = {"q": (e, e), "k": (e, e), "v": (e, e), "o": (e, e),
              "gate": (e, f), "up": (e, f), "down": (f, e)}
    weights = {name: torch.randn(*shape, generator=generator) / math.sqrt(shape[0])
               for name, shape in shapes.items()}
    inv = 10000.0 ** (-torch.arange(0, config.head_dim, 2).float() / config.head_dim)
    angles = torch.arange(config.sequence).float()[:, None] * inv[None, :]
    causal = torch.ones(config.sequence, config.sequence, dtype=torch.bool).tril()
    mask = causal.expand(config.batch * config.heads, -1, -1).reshape(-1, config.sequence).contiguous()
    return DecoderCase(config, seed, x, weights, angles.cos(), angles.sin(), mask)


class ReferenceOperators:
    """Equations and rounding match the original K04/K06/K10 contracts."""

    @staticmethod
    def softmax(x, mask):
        xm = torch.where(mask, x, torch.full_like(x, torch.finfo(x.dtype).min))
        exponent = torch.exp(xm - xm.max(-1, keepdim=True).values)
        out = exponent / exponent.sum(-1, keepdim=True)
        return torch.where(mask & mask.any(-1, keepdim=True), out, torch.zeros_like(out))

    @staticmethod
    def silu_mul(a, b):
        af, bf = a.float(), b.float()
        return (af * torch.sigmoid(af) * bf).half()

    @staticmethod
    def rope(x, cos, sin):
        even, odd = x[..., 0::2].float(), x[..., 1::2].float()
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        return torch.stack((even * c - odd * s, even * s + odd * c), -1).reshape_as(x).half()


class CandidateOperators:
    """GPU-only loader; executed bytes are exactly those checked in the receipt."""

    def __init__(self, bindings: SourceBindings):
        if not torch.cuda.is_initialized():
            raise ContractError("candidate loading requires an admitted GPU accounting job")
        self.models = {}
        for key, entry in bindings.operators.items():
            path = Path(entry["path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise SourceMismatch(f"{key} source changed before load")
            module = types.ModuleType(f"decoder_candidate_{key}_{entry['sha256'][:12]}")
            module.__file__ = str(path)
            sys.modules[module.__name__] = module
            try:
                # Inheriting this application's future annotations would alter
                # the unchanged kernel's tl.constexpr annotations at compile time.
                exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
            except BaseException:
                sys.modules.pop(module.__name__, None)
                raise
            self.models[key] = module.ModelNew().eval()

    def softmax(self, x, mask):
        return self.models["K04"](x, mask)[0]

    def silu_mul(self, a, b):
        return self.models["K06"](a, b)[0]

    def rope(self, x, cos, sin):
        return self.models["K10"](x, cos, sin)[0]


class DecoderBlock(torch.nn.Module):
    def __init__(self, case: DecoderCase, operators=None):
        super().__init__()
        case.config.validate()
        self.config = case.config
        self.operators = operators or ReferenceOperators()
        for name, tensor in case.weights.items():
            self.register_buffer(f"w_{name}", tensor)
        for name in ("cos", "sin", "mask"):
            self.register_buffer(name, getattr(case, name))

    def validate_structure(self, x) -> None:
        c = self.config
        if x.shape != (c.batch, c.sequence, c.embedding) or x.dtype != torch.float16 or not x.is_contiguous():
            raise ContractError("input requires contiguous FP16 [B,S,H*D]")
        if x.device != self.w_q.device:
            raise ContractError("input and weights must share a device")

    def validate_input(self, x) -> None:
        self.validate_structure(x)
        if not torch.isfinite(x).all().item():
            raise ContractError("input must be finite")

    def forward(self, x):
        # Numerical validation is deliberately outside the timed/compiled graph.
        self.validate_structure(x)
        return self._forward(x, None)

    def forward_with_trace(self, x):
        self.validate_input(x)
        trace = {}
        return self._forward(x, trace), trace

    def _forward(self, x, trace):
        c = self.config
        shape = (c.batch, c.sequence, c.heads, c.head_dim)
        q = (x.float() @ self.w_q).half().reshape(shape)
        k = (x.float() @ self.w_k).half().reshape(shape)
        v = (x.float() @ self.w_v).half().reshape(shape)
        rq = self.operators.rope(q, self.cos, self.sin)
        rk = self.operators.rope(k, self.cos, self.sin)
        scores = ((rq.permute(0, 2, 1, 3).float() @ rk.permute(0, 2, 3, 1).float())
                  / math.sqrt(c.head_dim)).reshape(-1, c.sequence).contiguous()
        probability = self.operators.softmax(scores, self.mask)
        attention = (probability.reshape(c.batch, c.heads, c.sequence, c.sequence)
                     @ v.permute(0, 2, 1, 3).float()).permute(0, 2, 1, 3).reshape_as(x)
        residual = (x.float() + attention @ self.w_o).half()
        gate = (residual.float() @ self.w_gate).half().reshape(-1, c.mlp_dim)
        up = (residual.float() @ self.w_up).half().reshape(-1, c.mlp_dim)
        gated = self.operators.silu_mul(gate, up)
        out = (residual.float() + gated.reshape(c.batch, c.sequence, c.mlp_dim).float() @ self.w_down).half()
        if trace is not None:
            trace.update({"rope_q": {"inputs": (q, self.cos, self.sin), "output": rq},
                          "rope_k": {"inputs": (k, self.cos, self.sin), "output": rk},
                          "softmax": {"inputs": (scores, self.mask), "output": probability},
                          "silu_mul": {"inputs": (gate, up), "output": gated}})
        return out


def _tensor(t, shape, dtype, bound, label):
    if t.shape != shape or t.dtype != dtype or not t.is_contiguous():
        raise ContractError(f"{label} shape/dtype/contiguity mismatch")
    if not torch.isfinite(t).all().item() or (bound is not None and t.abs().max().item() > bound):
        raise ContractError(f"{label} requires finite values with max_abs<={bound}")


def validate_operator_trace(trace) -> None:
    """Validate every invocation's domain before timing; never relax task tolerances."""
    for name in ("rope_q", "rope_k"):
        x, cos, sin = trace[name]["inputs"]
        b, s, h, d = x.shape
        if not (1 <= b <= 2 and 1 <= s <= 1024 and 8 <= h <= 32 and 64 <= d <= 128 and d % 2 == 0):
            raise ContractError("K10 domain mismatch")
        _tensor(x, (b, s, h, d), torch.float16, 30, "K10 x")
        for table in (cos, sin):
            _tensor(table, (s, d // 2), torch.float32, 1.000001, "K10 table")
        torch.testing.assert_close(cos.square() + sin.square(), torch.ones_like(cos), rtol=1e-5, atol=1e-5)
    scores, mask = trace["softmax"]["inputs"]
    r, d = scores.shape
    if not (1 <= r <= 2048 and 33 <= d <= 8192):
        raise ContractError("K04 domain mismatch")
    _tensor(scores, (r, d), torch.float32, 80, "K04 scores")
    _tensor(mask, (r, d), torch.bool, None, "K04 mask")
    a, b = trace["silu_mul"]["inputs"]
    r, d = a.shape
    if not (1 <= r <= 2048 and 33 <= d <= 11008):
        raise ContractError("K06 domain mismatch")
    for t in (a, b):
        _tensor(t, (r, d), torch.float16, 30, "K06 input")
