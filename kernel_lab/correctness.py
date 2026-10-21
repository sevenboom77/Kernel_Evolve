"""Correctness checking.

Deliberately *not* one loose ``allclose`` for every task.  Each task declares the
checks it needs in ``spec["checks"]``:

``inputs_unmodified``   the caller's tensors must be bit-identical afterwards
``outputs_not_aliased`` no output may share storage with an input
``output_contiguous``   the output must be a materialised contiguous tensor
``topk_property``       K16: values are exactly the sorted top-k, indices unique
``cos_sin_contract``    K10: the generated cos/sin satisfy cos^2 + sin^2 == 1
``labels_in_range``     K13 input contract
``indices_in_range``    K14 input contract
``offsets_contract``    K15 input contract (monotone, pinned ends)
``std_positive``        K19 input contract
"""
from __future__ import annotations

from typing import Any

import torch


def _as_tuple(obj) -> tuple:
    if isinstance(obj, torch.Tensor):
        return (obj,)
    if isinstance(obj, (tuple, list)):
        return tuple(obj)
    raise TypeError(f"forward() must return a tensor or a tuple of tensors, got {type(obj)!r}")


def tensor_diff(expected: torch.Tensor, actual: torch.Tensor, rtol: float, atol: float) -> dict:
    """Elementwise comparison with the first offending coordinate."""
    if expected.shape != actual.shape:
        return {"ok": False, "reason": f"shape {tuple(actual.shape)} != expected {tuple(expected.shape)}"}
    if expected.dtype != actual.dtype:
        return {"ok": False, "reason": f"dtype {actual.dtype} != expected {expected.dtype}"}
    e = expected.to(torch.float64)
    a = actual.to(torch.float64)
    diff = (a - e).abs()
    tol = atol + rtol * e.abs()
    bad = diff > tol
    n_bad = int(bad.sum().item())
    out: dict[str, Any] = {
        "ok": n_bad == 0,
        "n_bad": n_bad,
        "n_total": int(expected.numel()),
        "max_abs_err": float(diff.max().item()) if expected.numel() else 0.0,
        "max_tol": float(tol.max().item()) if expected.numel() else 0.0,
    }
    if n_bad:
        flat = int(bad.flatten().nonzero()[0].item())
        out["first_bad_flat"] = flat
        out["expected"] = float(e.flatten()[flat].item())
        out["actual"] = float(a.flatten()[flat].item())
        if expected.dim() > 0:
            out["first_bad_index"] = list(torch.unravel_index(torch.tensor(flat), expected.shape))
            out["first_bad_index"] = [int(i) for i in out["first_bad_index"]]
    return out


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    try:
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    except Exception:
        return False


def check_input_contract(task_id: str, spec: dict, inputs: tuple) -> list[str]:
    problems: list[str] = []
    checks = set(spec.get("checks", []))
    if task_id == "K10" and "cos_sin_contract" in checks:
        cos, sin = inputs[1], inputs[2]
        err = float(((cos.to(torch.float64) ** 2 + sin.to(torch.float64) ** 2) - 1.0).abs().max().item())
        if err > 1e-5:
            problems.append(f"cos/sin contract violated: max |cos^2+sin^2-1| = {err:.3e}")
    if task_id == "K13" and "labels_in_range" in checks:
        labels, c = inputs[1], inputs[0].shape[-1]
        if int(labels.min()) < 0 or int(labels.max()) >= c:
            problems.append("labels out of range")
    if task_id == "K14" and "indices_in_range" in checks:
        idx, v = inputs[1], inputs[0].shape[0]
        if int(idx.min()) < 0 or int(idx.max()) >= v:
            problems.append("indices out of range")
    if task_id == "K15" and "offsets_contract" in checks:
        offs = inputs[1].to(torch.int64)
        n = inputs[0].shape[0]
        if offs.numel() < 2:
            problems.append("offsets must have length S+1 >= 2")
        else:
            if int(offs[0]) != 0 or int(offs[-1]) != n:
                problems.append(f"offsets must run 0..N (got {int(offs[0])}..{int(offs[-1])}, N={n})")
            if bool((torch.diff(offs) < 0).any()):
                problems.append("offsets must be monotonically non-decreasing")
    if task_id == "K19" and "std_positive" in checks:
        if float(inputs[2].min()) <= 0:
            problems.append("std must be strictly positive")
    return problems


def check_outputs(task_id: str, spec: dict, inputs: tuple, expected: tuple,
                  actual: tuple, rtol: float, atol: float) -> dict:
    """Full output verification for one case."""
    problems: list[str] = []
    detail: dict[str, Any] = {}
    checks = set(spec.get("checks", []))

    if len(actual) != len(expected):
        return {"ok": False, "problems": [f"arity {len(actual)} != {len(expected)}"], "detail": {}}

    for i, (exp, act) in enumerate(zip(expected, actual)):
        if not isinstance(act, torch.Tensor):
            problems.append(f"output[{i}] is not a tensor: {type(act)!r}")
            continue
        d = tensor_diff(exp, act, rtol, atol)
        detail[f"out{i}"] = d
        if not d.get("ok", False):
            problems.append(f"output[{i}] mismatch: {d}")
        if "outputs_not_aliased" in checks:
            for j, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor) and _same_storage(inp, act):
                    problems.append(f"output[{i}] aliases input[{j}]")
        if "output_contiguous" in checks and not act.is_contiguous():
            problems.append(f"output[{i}] is not contiguous")

    if task_id == "K04" and len(actual) >= 1 and len(inputs) >= 2:
        mask = inputs[1]
        dead = ~mask.any(dim=-1)
        if bool(dead.any()):
            if float(actual[0][dead].abs().max().item()) != 0.0:
                problems.append("fully masked rows must be exactly zero")
    if task_id == "K16" and len(actual) >= 2:
        problems.extend(_check_topk(inputs[0], actual[0], actual[1], int(actual[0].shape[-1])))

    return {"ok": not problems, "problems": problems, "detail": detail}


def _check_topk(x: torch.Tensor, values: torch.Tensor, indices: torch.Tensor, k: int) -> list[str]:
    """Verify the top-k *property* rather than index equality with torch.topk."""
    problems: list[str] = []
    r, d = x.shape
    if values.shape != (r, k) or indices.shape != (r, k):
        return [f"top-k output shapes {tuple(values.shape)}/{tuple(indices.shape)} != ({r},{k})"]
    if indices.dtype != torch.int64:
        problems.append(f"indices dtype {indices.dtype} != int64")
    idx = indices.to(torch.int64)
    out_of_range = bool(((idx < 0) | (idx >= d)).any())
    if out_of_range:
        # never feed an out-of-range index to gather: the check must report, not raise
        return problems + ["top-k indices out of range"]
    for row in range(min(r, 8)):
        if idx[row].unique().numel() != k:
            problems.append(f"row {row}: duplicate indices in top-k")
            break
    gathered = x.gather(1, idx)
    if not torch.equal(gathered, values):
        problems.append("values do not equal x gathered at indices")
    ref = torch.sort(x, dim=-1, descending=True).values[:, :k]
    if not torch.equal(values, ref):
        problems.append("values are not the exact descending top-k of the row")
    if values.shape[1] > 1 and bool((values[:, 1:] > values[:, :-1]).any()):
        problems.append("values are not in descending order")
    return problems


def compare_call(tp, case: dict, reference_out: tuple, candidate_out: tuple,
                 inputs: tuple) -> dict:
    """Compare a candidate call against the oracle for one case."""
    rtol, atol = tp.tolerance
    return check_outputs(tp.task_id, tp.spec, inputs, reference_out, candidate_out, rtol, atol)


def call_module(module, inputs: tuple) -> tuple:
    out = module(*inputs)
    return _as_tuple(out)


def snapshot_inputs(inputs: tuple) -> list:
    return [t.detach().clone() if isinstance(t, torch.Tensor) else t for t in inputs]


def inputs_unchanged(before: list, after: tuple) -> bool:
    for b, a in zip(before, after):
        if isinstance(b, torch.Tensor):
            if not torch.equal(b, a):
                return False
        elif b != a:
            return False
    return True
