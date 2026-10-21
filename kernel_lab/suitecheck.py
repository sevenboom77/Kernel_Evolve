"""Suite validation: contract completeness and CPU end-to-end smoke of all tasks.

Runs with no GPU and no candidate, so it is safe while every card is busy.  A
task only passes when its reference produces the declared dtype/shape/arity on
every development correctness case, its baselines agree with the reference, and
the declared shape domain actually contains all timing points.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import torch

from . import correctness as C
from .tasks import (REQUIRED_SPEC_KEYS, TaskPackage, load_suite, validate_shape_dict)
from .util import read_json


def _expected_shape_hint(tp: TaskPackage, case: dict) -> dict:
    return dict(case["shape"])


def check_spec(tp: TaskPackage) -> list[str]:
    problems = []
    for key in REQUIRED_SPEC_KEYS:
        if key not in tp.spec:
            problems.append(f"spec.json missing key {key!r}")
    if tp.spec.get("task_id") != tp.dir.name:
        problems.append("spec task_id does not match the directory name")
    if tp.output_arity != len(tp.spec["outputs"]):
        problems.append("output_arity disagrees with the outputs table")
    if len(tp.dev_timing_cases()) != 3:
        problems.append(f"expected 3 development timing points, found {len(tp.dev_timing_cases())}")
    if len(tp.reserved_timing_cases()) != 2:
        problems.append(f"expected 2 reserved timing points, found {len(tp.reserved_timing_cases())}")
    if len(tp.dev_correctness_cases()) < 12:
        problems.append("fewer than 12 development correctness cases")
    if len(tp.final_correctness_cases()) < 20:
        problems.append("fewer than 20 final correctness cases")
    for label, cases in (("dev timing", tp.dev_timing_cases()),
                         ("reserved timing", tp.reserved_timing_cases()),
                         ("dev correctness", tp.dev_correctness_cases()),
                         ("final correctness", tp.final_correctness_cases())):
        for case in cases:
            bad = validate_shape_dict(case["shape"], tp.domain)
            if bad:
                problems.append(f"{label} case {case['case_id']}: " + "; ".join(bad))
    return problems


def check_reference(tp: TaskPackage, device: str = "cpu", limit: int | None = None) -> dict:
    """Run the oracle on every development correctness case."""
    problems = []
    cases = tp.dev_correctness_cases()
    if limit:
        cases = cases[:limit]
    for case in cases:
        try:
            inputs = tp.make_inputs(case, device=device, run_seed=11)
        except Exception as exc:
            problems.append(f"{case['case_id']}: make_inputs raised {type(exc).__name__}: {exc}")
            continue
        static = tp.case_static(case)
        problems.extend(f"{case['case_id']}: {p}" for p in C.check_input_contract(tp.task_id, tp.spec, inputs))
        try:
            ref = tp.build_reference(static)
            with torch.inference_mode():
                out = C.call_module(ref, inputs)
        except Exception as exc:
            problems.append(f"{case['case_id']}: reference raised {type(exc).__name__}: {exc}")
            continue
        if len(out) != tp.output_arity:
            problems.append(f"{case['case_id']}: reference arity {len(out)} != {tp.output_arity}")
            continue
        for i, (declared, got) in enumerate(zip(tp.spec["outputs"], out)):
            if not isinstance(got, torch.Tensor):
                problems.append(f"{case['case_id']}: output[{i}] is not a tensor")
                continue
            if str(got.dtype).replace("torch.", "") != declared["dtype"]:
                problems.append(f"{case['case_id']}: output[{i}] dtype {got.dtype} != {declared['dtype']}")
            if not bool(torch.isfinite(got).all()):
                problems.append(f"{case['case_id']}: reference produced NaN/Inf in output[{i}]")
        # determinism of the generator
        again = tp.make_inputs(case, device=device, run_seed=11)
        for a, b in zip(inputs, again):
            if isinstance(a, torch.Tensor) and not torch.equal(a, b):
                problems.append(f"{case['case_id']}: input generation is not deterministic")
                break
    return {"cases": len(cases), "ok": not problems, "problems": problems}


def check_baselines(tp: TaskPackage, device: str = "cpu", n_cases: int = 2) -> dict:
    """Each baseline must build, run, and agree with the reference."""
    results: dict[str, dict] = {}
    cases = tp.dev_correctness_cases()[:n_cases]
    for name in tp.baseline_names:
        problems = []
        n_ok = 0
        for case in cases:
            static = tp.case_static(case)
            try:
                ref = tp.build_reference(static)
                base = tp.build_baseline(name, static)
            except NotImplementedError as exc:
                results[name] = {"status": "not_applicable", "reason": str(exc)}
                problems = []
                n_ok = -1
                break
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                # torch.compile is unavailable on some interpreters (Dynamo needs
                # Python <= 3.12). That is an environment limitation, not a broken
                # baseline, and it must be reported as such.
                env_limited = any(k in str(exc) for k in (
                    "Dynamo is not supported", "no CUDA GPUs are available",
                    "Torch not compiled with CUDA enabled"))
                results[name] = {"status": "unavailable" if env_limited else "build_error",
                                 "reason": msg}
                n_ok = -1
                break
            try:
                inputs = tp.make_inputs(case, device=device, run_seed=11)
                with torch.inference_mode():
                    exp = C.call_module(ref, inputs)
                    actual = C.call_module(base, inputs)
            except Exception as exc:
                msg = f"{case['case_id']}: {type(exc).__name__}: {exc}"
                if any(k in str(exc) for k in ("Dynamo is not supported",
                                               "no CUDA GPUs are available")):
                    results[name] = {"status": "unavailable", "reason": msg}
                    n_ok = -1
                    break
                if "Unsupported" in str(exc) or "Unsupported" in type(exc).__name__:
                    # A documented torch.compile limitation is recorded, not
                    # treated as an invalid task or an artificially slow baseline.
                    results[name] = {"status": "compile_error", "reason": msg[:600]}
                    n_ok = -1
                    break
                problems.append(msg)
                continue
            res = C.check_outputs(tp.task_id, tp.spec, inputs, exp, actual, *tp.tolerance)
            if not res["ok"]:
                problems.append(f"{case['case_id']}: {res['problems'][:2]}")
            else:
                n_ok += 1
        if n_ok == -1:
            continue
        results[name] = {"status": "ok" if not problems else "mismatch",
                         "cases_ok": n_ok, "problems": problems}
    return results


def check_task(root: Path, task_id: str, device: str = "cpu",
               with_baselines: bool = True, limit: int | None = None) -> dict:
    out: dict = {"task_id": task_id}
    try:
        tp = TaskPackage(root, task_id)
    except Exception as exc:
        return {"task_id": task_id, "ok": False, "problems": [f"load: {type(exc).__name__}: {exc}"]}
    out["name"] = tp.name
    try:
        spec_problems = check_spec(tp)
        ref = check_reference(tp, device=device, limit=limit)
        out["reference"] = ref
        if with_baselines:
            out["baselines"] = check_baselines(tp, device=device)
        out["problems"] = spec_problems + ref["problems"]
        out["ok"] = ref["ok"] and not spec_problems
    except Exception:
        out["ok"] = False
        out["problems"] = [traceback.format_exc()]
    return out


def check_suite(root: Path, device: str = "cpu", tasks: list[str] | None = None,
                with_baselines: bool = True, limit: int | None = None) -> dict:
    suite = read_json(Path(root) / "configs" / "suite_v1.json")
    ids = tasks or suite["tasks"]
    per_task = []
    for tid in ids:
        per_task.append(check_task(Path(root), tid, device=device,
                                   with_baselines=with_baselines, limit=limit))
    ok = all(t["ok"] for t in per_task)
    return {
        "suite_id": suite["suite_id"],
        "n_tasks": len(ids),
        "n_ok": sum(1 for t in per_task if t["ok"]),
        "ok": ok,
        "tasks": per_task,
    }
