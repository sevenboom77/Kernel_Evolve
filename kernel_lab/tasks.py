"""Task-package loading and validation.

A task package is a frozen directory ``tasks/<ID>/`` holding ``spec.json``,
``reference.py``, ``cases.py``, ``baselines.py`` and ``task.md``.  Nothing in
this module ever touches a GPU: it is safe to run while every card is busy.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from .util import read_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_SPEC_KEYS = (
    "schema_version", "task_id", "name", "group", "backend", "description",
    "inputs", "outputs", "output_arity", "accumulation_dtype", "layout",
    "mutates_inputs", "domain", "dev_shapes", "reserved_shapes", "static_params",
    "finite_input_abs_max", "rtol", "atol", "max_candidate_attempts",
    "stagnation_attempts", "checks", "dev_correctness_cases",
    "final_correctness_cases", "dev_timing_cases", "reserved_timing_cases",
)

_module_counter = 0


def load_module(path: str | Path, alias: str):
    """Import a standalone .py file under a private module name."""
    global _module_counter
    path = Path(path).resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _module_counter += 1
    name = f"_kernellab_{alias}_{_module_counter}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


class TaskPackage:
    """One frozen task contract plus lazy access to its PyTorch modules."""

    def __init__(self, root: Path, task_id: str):
        self.root = Path(root)
        self.dir = self.root / "tasks" / task_id
        self.spec_path = self.dir / "spec.json"
        if not self.spec_path.is_file():
            raise FileNotFoundError(self.spec_path)
        self.spec: dict[str, Any] = read_json(self.spec_path)
        self.task_id = self.spec["task_id"]
        self._mods: dict[str, Any] = {}

    # ---------------------------------------------------------------- basics
    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def group(self) -> str:
        return self.spec["group"]

    @property
    def static_params(self) -> dict:
        return dict(self.spec["static_params"])

    @property
    def dev_shapes(self) -> list[dict]:
        return [dict(s) for s in self.spec["dev_shapes"]]

    @property
    def reserved_shapes(self) -> list[dict]:
        return [dict(s) for s in self.spec["reserved_shapes"]]

    @property
    def domain(self) -> dict:
        return dict(self.spec["domain"])

    @property
    def output_arity(self) -> int:
        return int(self.spec["output_arity"])

    @property
    def tolerance(self) -> tuple[float, float]:
        return float(self.spec["rtol"]), float(self.spec["atol"])

    def module(self, which: str):
        if which not in self._mods:
            self._mods[which] = load_module(self.dir / f"{which}.py", f"{self.task_id}_{which}")
        return self._mods[which]

    # ----------------------------------------------------------------- cases
    def case_static(self, case: dict) -> dict:
        return self.module("cases").case_static(case)

    def make_inputs(self, case: dict, device="cpu", run_seed: int | None = None):
        return self.module("cases").make_inputs(case, device=device, run_seed=run_seed)

    def dev_correctness_cases(self) -> list[dict]:
        return [dict(c) for c in self.module("cases").DEV_CORRECTNESS]

    def final_correctness_cases(self) -> list[dict]:
        return [dict(c) for c in self.module("cases").FINAL_CORRECTNESS]

    def dev_timing_cases(self) -> list[dict]:
        return [dict(c) for c in self.module("cases").DEV_TIMING]

    def reserved_timing_cases(self) -> list[dict]:
        return [dict(c) for c in self.module("cases").RESERVED_TIMING]

    # ------------------------------------------------------------ factories
    def build_reference(self, static: dict):
        return self.module("reference").Model(**static)

    def build_baseline(self, name: str, static: dict):
        return self.module("baselines").build(name, static)

    @property
    def baseline_names(self) -> tuple[str, ...]:
        return tuple(self.module("baselines").NAMES)


def timing_override_for(protocol, task_id: str) -> dict:
    """Shape overrides for one task, from the run protocol."""
    ov = ((protocol or {}).get("timing_shapes") or {}).get(task_id) or {}
    return {str(k): dict(v) for k, v in ov.items() if not str(k).startswith("_")}


def apply_timing_override(cases, task_id: str, protocol) -> list[dict]:
    """Replace the shape of named timing cases.

    The task contract is frozen, so the shapes a run measures are overridden here
    rather than edited in tasks/<id>/cases.py. case_id is preserved, so every
    downstream key (measurements, selections, reports) is unaffected.
    """
    ov = timing_override_for(protocol, task_id)
    out = []
    for c in cases:
        c = dict(c)
        if c["case_id"] in ov:
            c["shape"] = dict(ov[c["case_id"]])
        out.append(c)
    return out


def validate_timing_override(root: str | Path, protocol) -> list[str]:
    """Every override must name a real case and stay inside the task domain.

    Without this an override could silently move a timing point outside the
    declared domain, and the resulting number would be about a shape the task
    never promised to support.
    """
    root = Path(root)
    problems: list[str] = []
    overrides = (protocol or {}).get("timing_shapes") or {}
    suite = read_json(root / "configs" / "suite_v1.json")
    known = set(suite["tasks"])
    for tid, ov in overrides.items():
        if tid not in known:
            problems.append("%s: not a task in this suite" % tid)
            continue
        tp = TaskPackage(root, tid)
        by_id = {c["case_id"]: c for c in (tp.dev_timing_cases() + tp.reserved_timing_cases())}
        for cid, shape in ov.items():
            if str(cid).startswith("_"):
                continue
            if cid not in by_id:
                problems.append("%s.%s: no such timing case" % (tid, cid))
                continue
            bad = validate_shape_dict(dict(shape), tp.domain)
            if bad:
                problems.append("%s.%s: %s" % (tid, cid, "; ".join(bad)))
    return problems


def load_suite(root: str | Path | None = None) -> dict[str, TaskPackage]:
    root = Path(root or PROJECT_ROOT)
    suite = read_json(root / "configs" / "suite_v1.json")
    return {tid: TaskPackage(root, tid) for tid in suite["tasks"]}


def validate_shape_dict(shape: dict, domain: dict) -> list[str]:
    """Return the list of dimensions that fall outside the declared domain."""
    problems = []
    for key, value in shape.items():
        if key not in domain:
            problems.append(f"{key} not in domain")
            continue
        rng = domain[key]
        if rng == "__same__":
            continue
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        lo, hi = rng
        if isinstance(value, str) or value < lo or value > hi:
            problems.append(f"{key}={value} outside [{lo},{hi}]")
    for key, rng in domain.items():
        if rng == "__same__":
            continue
        if key not in shape and not isinstance(rng, str):
            # dimension is implied by the case generator, not necessarily present
            continue
    return problems
