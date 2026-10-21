"""Child-process entry points for the evaluator.

Every GPU-side stage runs in a fresh process so that

* a candidate can never observe the oracle's expected outputs (it only ever
  receives input tensors),
* an illegal memory access or a wedged context dies with the child instead of
  corrupting the queue,
* the oracle, the correctness run and the timing run each get a clean CUDA
  context and a clean import graph.

Invoked as ``python -m kernel_lab.children --spec <job.json>``.  The child always
writes its JSON result to the path given in the spec, even on failure.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from . import timing
from .tasks import TaskPackage


def _write(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, default=str)
    tmp.replace(path)


def _load_candidate(candidate_dir: str):
    path = Path(candidate_dir) / "kernel.py"
    if not path.is_file():
        raise FileNotFoundError(f"no kernel.py in {candidate_dir}")
    sys.path.insert(0, str(Path(candidate_dir).resolve()))
    name = f"_candidate_{abs(hash(str(path)))%10**8}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ModelNew"):
        raise AttributeError("kernel.py does not define ModelNew")
    return mod.ModelNew


def _set_precision(device: str = "cuda:0"):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        timing.configure(use_cuda_events=False)
        return
    timing.configure(use_cuda_events=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _save_case(out_dir: Path, sub: str, case_id: str, payload: dict) -> None:
    d = out_dir / sub
    d.mkdir(parents=True, exist_ok=True)
    torch.save(payload, d / f"{case_id}.pt")


def _mem_stats() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
        "free_at_start": int(torch.cuda.mem_get_info()[0]),
        "total": int(torch.cuda.mem_get_info()[1]),
    }


# --------------------------------------------------------------------------- #
# oracle
# --------------------------------------------------------------------------- #
def run_oracle(spec: dict) -> dict:
    tp = TaskPackage(spec["root"], spec["task_id"])
    out_dir = Path(spec["out_dir"])
    device = spec.get("device", "cuda:0")
    manifest = []
    t0 = time.time()
    for case in spec["cases"]:
        inputs = tp.make_inputs(case, device=device, run_seed=spec.get("run_seed"))
        static = tp.case_static(case)
        ref = tp.build_reference(static)
        with torch.inference_mode():
            expected = ref(*inputs)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
        _save_case(out_dir, "inputs", case["case_id"], {"inputs": list(inputs), "case": case})
        _save_case(out_dir, "expected", case["case_id"], {"outputs": list(expected), "case": case})
        manifest.append({
            "case_id": case["case_id"],
            "shape": case["shape"],
            "family": case.get("family"),
            "input_shapes": [list(t.shape) for t in inputs],
            "input_dtypes": [str(t.dtype) for t in inputs],
            "output_shapes": [list(t.shape) for t in expected],
            "output_dtypes": [str(t.dtype) for t in expected],
            "static": static,
        })
    _write(out_dir / "manifest.json", {"task_id": spec["task_id"], "cases": manifest,
                                       "seconds": time.time() - t0})
    return {"ok": True, "n_cases": len(manifest), "seconds": time.time() - t0}


# --------------------------------------------------------------------------- #
# candidate correctness
# --------------------------------------------------------------------------- #
def run_check(spec: dict) -> dict:
    device = spec.get("device", "cuda:0")
    _set_precision(device)
    in_dir = Path(spec["inputs_dir"])
    out_dir = Path(spec["out_dir"])
    results = []
    static_by_case = spec.get("static_by_case") or {}
    if spec.get("baseline"):
        tp = TaskPackage(spec["root"], spec["task_id"])
        name = spec["baseline"]
        ModelNew = None
        factory = (lambda static: tp.build_baseline(name, static))
    else:
        ModelNew = _load_candidate(spec["candidate_dir"])
    for case in spec["cases"]:
        cid = case["case_id"]
        rec = {"case_id": cid, "status": "ok", "error": None, "seconds": None}
        t0 = time.time()
        try:
            blob = torch.load(in_dir / f"{cid}.pt", weights_only=False)
            inputs = tuple(blob["inputs"])
            static = dict(static_by_case.get(cid, {}))
            model = factory(static) if ModelNew is None else ModelNew(**static)
            before = [t.detach().clone() if isinstance(t, torch.Tensor) else t for t in inputs]
            with torch.inference_mode():
                out = model(*inputs)
            if isinstance(out, torch.Tensor):
                out = (out,)
            out = tuple(out)
            # The parent cannot see cross-process mutation or storage aliasing, so
            # both are detected here, inside the candidate's own process.
            rec["inputs_mutated"] = not all(
                (not isinstance(b, torch.Tensor)) or torch.equal(b, a)
                for b, a in zip(before, inputs))
            aliased = []
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    for j, inp in enumerate(inputs):
                        if isinstance(inp, torch.Tensor):
                            try:
                                if o.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr():
                                    aliased.append([i, j])
                            except Exception:
                                pass
            rec["aliased"] = aliased
            _save_case(out_dir, "outputs", cid, {"outputs": out, "case": case})
            rec["n_outputs"] = len(out)
        except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else RuntimeError) as exc:
            rec.update(status="oom", error=f"{type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()
        except Exception as exc:
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc()[-4000:])
        rec["seconds"] = time.time() - t0
        results.append(rec)
    _write(out_dir / "run.json", {"results": results, "memory": _mem_stats()})
    return {"ok": True, "n_cases": len(results),
            "n_error": sum(1 for r in results if r["status"] != "ok")}


# --------------------------------------------------------------------------- #
# timing: candidate + baselines, interleaved, one clean process
# --------------------------------------------------------------------------- #
def make_labeled_ctors(tp: TaskPackage, names, static: dict,
                       compile_times: dict, baseline_status: dict) -> list:
    """Build one constructor per baseline label for one static configuration.

    Every label must call the implementation that was built for *it*. A bare
    ``lambda **kw: mod`` closed over the loop variable made every label return
    the last successful build (reproduced in the 2026-10-03 audit,
    docs/audit_20261003/baseline_binding_cpu_reproduction.json). The default
    argument below binds each lambda to its own module, and ``baseline_status``
    gains an ``identity`` entry so a label can never silently stand for an
    aliased implementation.
    """
    ctors = []
    for name in names:
        try:
            t0 = time.time()
            mod = tp.build_baseline(name, static)
            compile_times[f"build::{name}"] = time.time() - t0
            ctors.append((name, (lambda _mod=mod, **kw: _mod)))
            rec = baseline_status.setdefault(name, {"status": "ok"})
            if rec.get("status") == "ok":
                rec["identity"] = _impl_identity(mod)
        except NotImplementedError as exc:
            baseline_status[name] = {"status": "not_applicable", "reason": str(exc)}
        except Exception as exc:
            baseline_status[name] = {"status": "unavailable",
                                     "reason": f"{type(exc).__name__}: {exc}"}
    return ctors


def _impl_identity(obj) -> str:
    """A short, log-safe name for the object a label actually built."""
    cls = obj if isinstance(obj, type) else type(obj)
    return str(getattr(cls, "__qualname__", repr(cls)))


def run_timing(spec: dict) -> dict:
    device = spec.get("device", "cuda:0")
    _set_precision(device)
    tp = TaskPackage(spec["root"], spec["task_id"])
    buffers = int(spec.get("buffers", 2))
    out: dict = {"cases": {}, "baseline_status": {}, "memory": {}, "compile_times": {}}
    ws_map: dict = {}
    fp_map: dict = {}

    ctors0 = []
    if spec.get("candidate_dir"):
        ctors0.append(("candidate", _load_candidate(spec["candidate_dir"])))

    def make_ctors(static):
        ctors = list(ctors0)
        ctors.extend(make_labeled_ctors(tp, spec.get("baselines", []), static,
                                        out["compile_times"], out["baseline_status"]))
        return ctors

    for case in spec["cases"]:
        cid = case["case_id"]
        static = tp.case_static(case)
        # distinct buffers so a "remember the last answer" kernel cannot look fast
        input_sets = [tp.make_inputs(case, device=device, run_seed=(spec.get("run_seed") or 0) + 1000 * b)
                      for b in range(max(1, buffers))]
        ws_map[cid] = timing.estimate_working_set(input_sets[0])
        ctors = make_ctors(static)
        fns: dict[str, object] = {}
        errors: dict[str, str] = {}
        error_traces: dict[str, str] = {}
        first_out = None
        for label, ctor in ctors:
            try:
                engines = []
                for buf in input_sets:
                    model = ctor(**static)
                    engines.append(model)
                state = {"n": 0}

                def make_call(engines=engines, input_sets=input_sets, state=state):
                    def call():
                        i = state["n"] % len(engines)
                        state["n"] += 1
                        with torch.inference_mode():
                            return engines[i](*input_sets[i])
                    return call
                # time the first call separately: JIT / torch.compile warm-up
                t0 = time.time()
                first_out = make_call()()
                timing.sync()
                out["compile_times"][f"first_call::{cid}::{label}"] = time.time() - t0
                fns[label] = make_call()
            except Exception as exc:
                errors[label] = f"{type(exc).__name__}: {exc}"
                error_traces[label] = traceback.format_exc()[-2500:]
                out["baseline_status"].setdefault(label, {"status": "unavailable",
                                                          "reason": errors[label]})
        # What one call must touch, read off a real call.
        fp_map[cid] = timing.footprint_bytes(input_sets[0], first_out)
        if "candidate" not in fns and ctors0:
            out["cases"][cid] = {"status": "timing_error", "errors": errors,
                                 "error_traces": error_traces}
            continue
        if not fns:
            out["cases"][cid] = {"status": "timing_error", "errors": errors,
                                 "error_traces": error_traces}
            continue
        try:
            res = timing.measure_interleaved(
                fns, warmup=int(spec.get("warmup", 20)), batches=int(spec.get("batches", 30)),
                target_ms=float(spec.get("target_batch_ms", 20.0)),
                order_seed=spec.get("order_seed"))
            out["cases"][cid] = {"status": "ok", "measurements": res, "errors": errors,
                                 "error_traces": error_traces}
        except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else RuntimeError) as exc:
            out["cases"][cid] = {"status": "oom", "errors": {**errors, "candidate": str(exc)}}
        except Exception as exc:
            out["cases"][cid] = {"status": "timing_error",
                                 "errors": {**errors, "candidate": f"{type(exc).__name__}: {exc}"},
                                 "error_traces": {**error_traces,
                                                  "candidate": traceback.format_exc()[-2500:]}}
    # Attach the working set of every case: without it a launch-bound shape is
    # indistinguishable from a genuinely fast one.
    for cid, blk in out["cases"].items():
        blk["working_set_bytes"] = ws_map.get(cid)
        # The footprint decides whether a shape can be measured at
        # all, so it travels with the case.
        blk["footprint_bytes"] = fp_map.get(cid)
    try:
        _pr = torch.cuda.get_device_properties(device)
        _l2 = 0
        for _attr in ("l2_cache_size", "L2_cache_size"):   # torch renamed it
            _l2 = int(getattr(_pr, _attr, 0) or 0)
            if _l2:
                break
        out["device"] = {"name": _pr.name, "l2_bytes": _l2,
                         "multiprocessor_count": int(_pr.multi_processor_count)}
    except Exception as exc:   # a device probe must never fail a run
        out["device"] = {"error": type(exc).__name__ + ": " + str(exc)}
    out["memory"] = _mem_stats()
    out["precision"] = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    _write(spec["out"], out)
    return {"ok": True, "cases": list(out["cases"])}


MODES = {"oracle": run_oracle, "check": run_check, "timing": run_timing}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    mode = spec["mode"]
    out_path = spec.get("result_json") or spec.get("out") or str(Path(args.spec).with_suffix(".result.json"))
    try:
        res = MODES[mode](spec)
        _write(out_path, {"ok": True, "mode": mode, **res})
        return 0
    except BaseException as exc:
        _write(out_path, {"ok": False, "mode": mode, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-8000:]})
        return 1


if __name__ == "__main__":
    sys.exit(main())
