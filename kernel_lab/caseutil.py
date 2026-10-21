"""Deterministic input generation shared by every task's ``cases.py``.

Two rules matter for the integrity of the lab:

* every case's inputs are a pure function of ``(case_id, run_seed)``; the worker
  never sees the run seed for the final (reserved) verification, so it cannot
  pre-compute answers for those inputs;
* a *family* names the numerical regime (zeros, huge, repeated, tied, ...) and is
  applied consistently, so coverage claims are machine-checkable.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch

GENERIC_FAMILIES = (
    "plain", "zeros", "tiny", "large", "uniform", "mixed_sign", "repeated",
    "quantised", "extreme",
)


def derive_seed(case: dict, run_seed: int | None = None) -> int:
    if run_seed is None:
        if "seed" in case:
            return int(case["seed"])
        run_seed = 0
    blob = f"{run_seed}:{case['case_id']}:{case.get('family','')}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little") % (2**31 - 1)


def generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) % (2**63 - 1))
    return g


def randn(shape, gen, dtype=torch.float32) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, dtype=dtype)


def uniform(shape, gen, lo=-1.0, hi=1.0, dtype=torch.float32) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype).uniform_(lo, hi, generator=gen)


def draw(shape, gen: torch.Generator, family: str = "plain", abs_max: float = 1.0, dtype=torch.float32):
    """Draw one tensor in a named numerical regime."""
    return apply_family(torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype), family, gen, abs_max)


def apply_family(t: torch.Tensor, family: str, gen: torch.Generator, abs_max: float) -> torch.Tensor:
    """Map an already-drawn tensor into the requested numerical regime."""
    if family in ("plain", "random"):
        return t
    if family == "zeros":
        return torch.zeros_like(t)
    if family == "tiny":
        return t * 1e-3
    if family == "large":
        return (t * (abs_max / 6.0)).clamp(-abs_max, abs_max)
    if family == "uniform":
        return uniform(t.shape, gen, -1.0, 1.0, t.dtype)
    if family == "mixed_sign":
        sign = torch.where(torch.rand(t.shape, generator=gen) < 0.5, -1.0, 1.0)
        return (t.abs() * sign).to(t.dtype)
    if family == "repeated":
        return (t * 4.0).round() / 4.0
    if family == "quantised":
        return (t * 16.0).round() / 16.0
    if family == "extreme":
        return uniform(t.shape, gen, -abs_max, abs_max, t.dtype)
    if family == "alternating":
        sign = torch.ones_like(t)
        sign[..., 1::2] = -1.0
        return ((t.abs() + 0.25) * sign).to(t.dtype)
    raise ValueError(f"unknown family {family!r}")


def offsets_family(n: int, s: int, family: str, gen: torch.Generator) -> torch.Tensor:
    """Build a monotone int32 CSR offset vector of length s+1 with offsets[0]=0, offsets[-1]=n."""
    if n < 0 or s < 1:
        raise ValueError("bad segment geometry")
    if s == 1:
        vec = torch.tensor([0, n], dtype=torch.int64)
    elif family == "uniform":
        cuts = torch.sort(torch.randint(0, n + 1, (s - 1,), generator=gen, dtype=torch.int64)).values
        vec = torch.cat([torch.zeros(1, dtype=torch.int64), cuts, torch.full((1,), n, dtype=torch.int64)])
    elif family == "skewed":
        raw = torch.rand(s - 1, generator=gen) ** 3.0
        cuts = torch.sort((raw * n).to(torch.int64)).values
        vec = torch.cat([torch.zeros(1, dtype=torch.int64), cuts, torch.full((1,), n, dtype=torch.int64)])
    elif family == "one_long":
        # every row lands in segment 0; segments 1..s-1 are exactly empty
        vec = torch.cat([torch.zeros(1, dtype=torch.int64),
                         torch.full((s - 1,), n, dtype=torch.int64),
                         torch.full((1,), n, dtype=torch.int64)])
    elif family == "empty_heavy":
        # draw cuts from a coarse grid so collisions (empty segments) are common
        grid = max(1, n // 16)
        cuts = torch.sort(torch.randint(0, grid + 1, (s - 1,), generator=gen, dtype=torch.int64) * 16).values
        vec = torch.cat([torch.zeros(1, dtype=torch.int64), cuts, torch.full((1,), n, dtype=torch.int64)])
    elif family == "single_row_segments":
        cuts = torch.arange(1, s, dtype=torch.int64).clamp(max=n)
        vec = torch.cat([torch.zeros(1, dtype=torch.int64), cuts[: s - 1],
                         torch.full((1,), n, dtype=torch.int64)])
    else:
        raise ValueError(f"unknown offsets family {family!r}")
    vec = vec.clamp(0, n)
    vec = torch.cummax(vec, 0).values  # guarantee monotone after clamping
    vec[0] = 0
    vec[-1] = n
    return vec.to(torch.int32)


def mask_family(shape, family: str, gen: torch.Generator) -> torch.Tensor:
    r, d = shape
    if family == "all_true":
        return torch.ones(r, d, dtype=torch.bool)
    if family == "all_false":
        return torch.zeros(r, d, dtype=torch.bool)
    if family == "single_true":
        m = torch.zeros(r, d, dtype=torch.bool)
        m[torch.arange(r), torch.randint(0, d, (r,), generator=gen)] = True
        return m
    if family == "sparse":
        return torch.rand(r, d, generator=gen) < 0.1
    if family == "dense_keep":
        return torch.rand(r, d, generator=gen) < 0.9
    if family == "ragged":
        m = torch.zeros(r, d, dtype=torch.bool)
        for i in range(r):
            keep = int(torch.randint(1, d + 1, (1,), generator=gen).item())
            perm = torch.randperm(d, generator=gen)[:keep]
            m[i, perm] = True
        return m
    if family == "mixed_rows":
        m = torch.ones(r, d, dtype=torch.bool)
        m[0] = False
        if r > 1:
            m[1] = True
        if r > 2:
            m[2, :] = False
            m[2, d // 2] = True
        return m
    raise ValueError(f"unknown mask family {family!r}")


def indices_family(v: int, length: int, family: str, gen: torch.Generator) -> torch.Tensor:
    if family == "random":
        return torch.randint(0, v, (length,), generator=gen, dtype=torch.int64)
    if family == "single":
        return torch.full((length,), int(torch.randint(0, v, (1,), generator=gen).item()), dtype=torch.int64)
    if family == "repeated":
        base = torch.randint(0, v, (max(1, length // 8),), generator=gen, dtype=torch.int64)
        return base[torch.randint(0, base.numel(), (length,), generator=gen)]
    if family == "boundary":
        idx = torch.zeros(length, dtype=torch.int64)
        if length > 0:
            idx[0] = 0
        if length > 1:
            idx[1] = v - 1
        if length > 2:
            idx[2:] = torch.randint(0, v, (length - 2,), generator=gen, dtype=torch.int64)
        return idx
    if family == "monotone":
        return torch.sort(torch.randint(0, v, (length,), generator=gen, dtype=torch.int64)).values
    raise ValueError(f"unknown indices family {family!r}")


def topk_family(shape, family: str, gen: torch.Generator, abs_max: float) -> torch.Tensor:
    r, d = shape
    if family == "plain":
        return torch.randn(r, d, generator=gen)
    if family == "ties":
        # few distinct magnitudes -> guaranteed ties around the k-th boundary
        base = (torch.randn(r, d, generator=gen) * 2.0).round()
        return base
    if family == "all_equal":
        return torch.full((r, d), 0.5)
    if family == "sorted_asc":
        return torch.arange(d, dtype=torch.float32).unsqueeze(0).repeat(r, 1)
    if family == "sorted_desc":
        return torch.arange(d, dtype=torch.float32).flip(0).unsqueeze(0).repeat(r, 1)
    if family == "two_levels":
        return (torch.randn(r, d, generator=gen).round() * 10.0)
    if family == "extreme":
        return uniform((r, d), gen, -abs_max, abs_max)
    raise ValueError(f"unknown topk family {family!r}")


def to_device(inputs: tuple, device) -> tuple:
    return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs)


def describe_case(case: dict) -> dict:
    out: dict[str, Any] = {"case_id": case["case_id"], "family": case.get("family")}
    out.update({f"dim_{k}": v for k, v in case.get("shape", {}).items()})
    if case.get("static"):
        out.update({f"static_{k}": v for k, v in case["static"].items()})
    return out
