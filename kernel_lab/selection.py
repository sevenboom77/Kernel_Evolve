"""Champion promotion — the only component allowed to change a task's winner.

The rule is deliberately arithmetic, not a matter of opinion:

``shape_speedup``  strongest baseline latency / candidate latency
``task_speedup``   geometric mean of the shape speedups (equal weight per shape)
``promote``        task_speedup at least ``min_gain`` better than the incumbent
                   AND no development shape more than ``max_slowdown`` behind the
                   strongest baseline for that shape.
"""
from __future__ import annotations

import math

from .util import geometric_mean


def speedup(baseline_us: float | None, candidate_us: float | None) -> float | None:
    if not baseline_us or not candidate_us or candidate_us <= 0:
        return None
    return float(baseline_us) / float(candidate_us)


def task_speedup(shape_speedups: dict) -> float | None:
    return geometric_mean(list(shape_speedups.values()))


def strongest_per_shape(baseline_latencies: dict, qualified: set[str]) -> dict:
    """Pick the fastest *qualified* baseline for every shape independently."""
    out: dict[str, dict] = {}
    for label, per_shape in baseline_latencies.items():
        if label not in qualified:
            continue
        for case_id, us in (per_shape or {}).items():
            if us is None or us <= 0:
                continue
            cur = out.get(case_id)
            if cur is None or us < cur["latency_us"]:
                out[case_id] = {"label": label, "latency_us": float(us)}
    return out


def evaluate_candidate(candidate_us: dict, strongest: dict) -> dict:
    """Per-shape speedups plus the geometric-mean task speedup."""
    shape_speedups: dict[str, float] = {}
    for case_id, us in candidate_us.items():
        base = strongest.get(case_id)
        if base is None:
            continue
        s = speedup(base["latency_us"], us)
        if s is not None:
            shape_speedups[case_id] = s
    return {"shape_speedups": shape_speedups, "task_speedup": task_speedup(shape_speedups),
            "strongest": strongest}


def promote(current: dict | None, challenger: dict, *, min_gain: float = 0.02,
            max_slowdown: float = 0.05) -> tuple[bool, str]:
    """Decide whether ``challenger`` replaces the incumbent champion."""
    new_task = challenger.get("task_speedup")
    if new_task is None:
        return False, "no comparable shapes against a qualified baseline"
    for case_id, s in challenger["shape_speedups"].items():
        if s < 1.0 - max_slowdown:
            return False, f"shape {case_id} is {100*(1-s):.1f}% slower than the strongest baseline"
    if current is None:
        return True, "no incumbent; first correct candidate sets the bar"
    old = current.get("task_speedup")
    if old is None:
        return True, "incumbent has no task speedup"
    if new_task < old * (1.0 + min_gain):
        return False, (f"task speedup {new_task:.4f} is not >= {(1+min_gain):.2f}x the incumbent "
                       f"{old:.4f}")
    return True, f"task speedup improves {old:.4f} -> {new_task:.4f}"


def required_speedup_for_verification() -> dict:
    return {"bootstrap_lower_bound": 1.05, "per_shape_median_floor": 0.95}
