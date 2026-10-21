"""The evaluator: the only component that may touch a GPU.

One evaluator process owns exactly one physical card.  It leases a job from the
queue, runs each stage in a fresh child process, compares candidate output
against an oracle produced in *another* process, and records measurements and
the champion decision.  It never calls a model.
"""
from __future__ import annotations

import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch

from . import correctness as C
from . import gpuinfo
from . import selection
from . import tasks
from . import timing
from .events import EventLog
from .paths import RunPaths
from .queue import GpuLock, Queue
from .tasks import TaskPackage
from .util import read_json, sha256_file, utcnow, write_json

def classify_child_failure(message: str, traceback_text: str = "") -> str:
    """Map a child-process exception onto the candidate state machine."""
    blob = message + "\n" + traceback_text
    if "OutOfMemory" in blob or "CUDA out of memory" in blob:
        return "oom"
    if any(k in blob for k in ("CompilationError", "ptxas", "InductorError", "CompileError",
                               "SyntaxError", "triton.compiler", "Unsupported GPU")):
        return "compile_fail"
    return "crashed"


TERMINAL_STATUSES = ("complete", "incorrect", "compile_fail", "invalid_source",
                     "timeout", "oom", "env_error", "contaminated", "crashed")


class ChildFailure(Exception):
    def __init__(self, status: str, message: str, detail: dict | None = None):
        super().__init__(message)
        self.status = status
        self.detail = detail or {}


class Evaluator:
    def __init__(self, run_id: str, gpu: str, root: str | Path, python_exe: str | None = None,
                 owner: str | None = None, poll_s: float = 2.0, log_level: str = "info",
                 protocol: dict | None = None, device: str = "cuda:0"):
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.gpu = str(gpu)
        self.python_exe = python_exe or sys.executable
        # `device` is what children are told to use; `gpu` is the physical card
        # whose exclusive lock we hold.  Tests run the whole pipeline with
        # device="cpu" so that no card is disturbed.
        self.device = device
        self.is_cuda = str(device).startswith("cuda")

        if self.is_cuda:
            # Pin ourselves: the evaluator must never create a CUDA
            # context on a card it is not responsible for. Without this
            # the parent lands on GPU 0 and quietly pollutes a card
            # somebody else may be using.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.owner = owner or f"{os.uname().nodename}:{os.getpid()}"
        self.poll_s = poll_s
        self.paths = RunPaths(self.root, run_id).ensure()
        self.events = EventLog(self.paths.events, run_id=run_id, source=f"evaluator[{self.gpu}]",
                               level=log_level)
        self.q = Queue(self.paths.db, events=self.events)
        self.protocol = protocol or read_json(self.paths.protocol) if self.paths.protocol.is_file() else (protocol or {})
        self.physical_key = str(gpu)
        self._shutdown = False

    def _warmup(self) -> int:
        return int(self.protocol.get("timing", {}).get("warmup_iters", 20))

    def _batches(self) -> int:
        return int(self.protocol.get("timing", {}).get("batches_per_shape", 30))

    def _timing_cases(self, tp, phase: str = "dev") -> list[dict]:
        """The shapes this run actually measures for one task.

        The contract in tasks/<id>/cases.py is frozen; a run may override the
        shapes it measures through the protocol, which is how a task whose
        declared shapes fit inside L2 can still be measured honestly.
        """
        base = (tp.dev_timing_cases() if phase == "dev"
                else tp.dev_timing_cases() + tp.reserved_timing_cases())
        return tasks.apply_timing_override(base, tp.task_id, self.protocol)

    # ------------------------------------------------------------------ utils
    def _child_env(self, task_id: str, candidate_id: str) -> dict:
        env = os.environ.copy()
        if self.is_cuda:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        code_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join([str(self.root), code_root, env.get("PYTHONPATH", "")])
        ns = self.paths.work / "cache" / task_id / str(candidate_id)
        env["TRITON_CACHE_DIR"] = str(ns / "triton")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(ns / "inductor")
        env["TORCH_EXTENSIONS_DIR"] = str(ns / "torch_ext")
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["TRITON_F32_DEFAULT"] = "ieee"
        return env

    def _run_child(self, spec: dict, timeout: float, task_id: str, candidate_id: str) -> dict:
        job_dir = self.paths.work / "jobs" / f"{spec['mode']}_{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True, exist_ok=True)
        spec = dict(spec)
        spec["root"] = str(self.root)
        spec.setdefault("result_json", str(job_dir / "result.json"))
        if "out" not in spec and spec["mode"] == "timing":
            spec["out"] = str(job_dir / "timing.json")
        spec_path = job_dir / "spec.json"
        write_json(spec_path, spec)
        env = self._child_env(task_id, candidate_id)
        t0 = time.time()
        proc = subprocess.Popen(
            [self.python_exe, "-m", "kernel_lab.children", "--spec", str(spec_path)],
            cwd=str(self.root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, text=True)
        try:
            out, err = proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            out, err = proc.communicate()
        wall = time.time() - t0
        (job_dir / "stdout.log").write_text(out or "")
        (job_dir / "stderr.log").write_text(err or "")
        payload = None
        rp = Path(spec["result_json"])
        if rp.is_file():
            try:
                payload = read_json(rp)
            except Exception:
                payload = None
        self.events.emit("child_finished", source="evaluator", mode=spec["mode"],
                         wall_s=round(wall, 3), timed_out=timed_out, rc=proc.returncode)
        if timed_out:
            raise ChildFailure("timeout", f"{spec['mode']} exceeded {timeout:.0f}s",
                               {"stderr_tail": (err or "")[-1500:], "wall_s": wall})
        if payload is None:
            raise ChildFailure("crashed", f"{spec['mode']} child produced no result",
                               {"rc": proc.returncode, "stderr_tail": (err or "")[-2500:]})
        if not payload.get("ok", False):
            raise ChildFailure("crashed", f"{spec['mode']} child failed: {payload.get('error')}",
                               {"traceback": payload.get("traceback", "")[-2500:], "wall_s": wall})
        payload["wall_s"] = wall
        payload["job_dir"] = str(job_dir)
        return payload

    # ---------------------------------------------------------------- oracle
    def ensure_oracle(self, task_id: str, phase: str, cases: list[dict]) -> Path:
        tp = TaskPackage(self.root, task_id)
        out_dir = self.paths.work / "oracle" / task_id / phase
        manifest = out_dir / "manifest.json"
        if manifest.is_file():
            try:
                have = {c["case_id"] for c in read_json(manifest)["cases"]}
                if have == {c["case_id"] for c in cases}:
                    return out_dir
            except Exception:
                pass
        run_seed = int(self.protocol.get("seeds", {}).get(phase, 0))
        spec = {"mode": "oracle", "task_id": task_id, "cases": cases, "run_seed": run_seed,
                "out_dir": str(out_dir), "device": self.device,
                "result_json": str(out_dir / "oracle_run.json")}
        self._run_child(spec, timeout=float(self.protocol.get("timeouts", {}).get("correctness_s", 60)) * 4,
                        task_id=task_id, candidate_id=f"oracle_{phase}")
        return out_dir

    # -------------------------------------------------------------- baselines
    def evaluate_baselines(self, task_id: str) -> dict:
        tp = TaskPackage(self.root, task_id)
        cases = tp.dev_correctness_cases()
        n_qual = self.protocol.get("baseline_qualification_cases")
        if n_qual:
            cases = cases[: int(n_qual)]
        oracle = self.ensure_oracle(task_id, "dev", cases)
        names = list(tp.baseline_names)
        qualification: dict[str, dict] = {}
        latencies: dict[str, dict] = {}
        static_by_case = {c["case_id"]: tp.case_static(c) for c in cases}

        for name in names:
            out_dir = self.paths.work / "baseline_check" / task_id / name
            spec = {"mode": "check", "task_id": task_id, "baseline": name, "cases": cases,
                    "inputs_dir": str(oracle / "inputs"), "out_dir": str(out_dir),
                    "static_by_case": static_by_case, "device": self.device}
            try:
                res = self._run_child(spec, timeout=float(self.protocol.get("timeouts", {}).get("correctness_s", 60)) * 3,
                                      task_id=task_id, candidate_id=f"baseline_{name}")
            except ChildFailure as exc:
                qualification[name] = {"status": exc.status, "reason": str(exc)}
                continue
            verdict = self._compare_outputs(tp, cases, oracle, out_dir)
            qualification[name] = {"status": "ok" if verdict["ok"] else "incorrect",
                                   "detail": verdict}

        qualified = [n for n, v in qualification.items() if v["status"] == "ok"]
        timing_cases = self._timing_cases(tp)
        if qualified:
            # The baseline-only interleave gets the same frozen order seed as
            # every other timing run, so no baseline label keeps a positional
            # advantage when the strongest yardstick is picked (audit 4.4).
            seed = int(self.protocol.get("seeds", {}).get("dev", 0))
            spec = {"mode": "timing", "task_id": task_id, "candidate_dir": None,
                    "cases": timing_cases, "baselines": qualified,
                    "device": self.device, "warmup": self._warmup(), "batches": self._batches(),
                    "buffers": 2,
                    "run_seed": seed,
                    "order_seed": seed + 100003,
                    "out": str(self.paths.work / "baseline_timing" / f"{task_id}.json")}
            try:
                res = self._run_child(spec, timeout=float(self.protocol.get("timeouts", {}).get("job_s", 300)),
                                      task_id=task_id, candidate_id="baseline_timing")
                data = read_json(Path(spec["out"]))
                for cid, blk in data["cases"].items():
                    if blk.get("status") != "ok":
                        continue
                    for label, m in blk["measurements"].items():
                        latencies.setdefault(label, {})[cid] = m["gpu_us_median"]
                        self.q.record_measurement(
                            self.run_id, "baseline_job", task_id, "baseline", cid, label,
                            "baseline", m["gpu_us_median"], m.get("host_us_median"), m)
                for label, st in data.get("baseline_status", {}).items():
                    if st.get("status") != "ok" and label in qualification:
                        qualification[label]["status"] = st["status"]
                        qualification[label]["reason"] = st.get("reason")
                qualified = [n for n, v in qualification.items() if v["status"] == "ok"]
            except ChildFailure as exc:
                self.events.emit("baseline_timing_failed", level="warn", task_id=task_id,
                                 message=str(exc))

        summary = {}
        for name in names:
            info = qualification.get(name, {"status": "not_run"})
            self.q.set_baseline(self.run_id, task_id, name, info.get("status", "unknown"),
                                info.get("reason") or json.dumps(info.get("detail", {}))[:900],
                                latencies.get(name, {}), info)
            summary[name] = {"status": info.get("status"), "shapes": len(latencies.get(name, {}))}
        strongest = selection.strongest_per_shape(latencies, set(qualified))
        self.q.set_baseline(self.run_id, task_id, "__strongest__", "ok", None, strongest, {})
        return {"qualified": qualified, "summary": summary, "strongest": strongest}

    def _ensure_baselines(self, task_id: str) -> dict:
        have = self.q.baselines(self.run_id, task_id)
        if "__strongest__" in have and have["__strongest__"]["latencies"]:
            return {"qualified": [k for k, v in have.items()
                                  if v["status"] == "ok" and k != "__strongest__"],
                    "strongest": have["__strongest__"]["latencies"]}
        return self.evaluate_baselines(task_id)

    # ------------------------------------------------------------ comparison
    def _compare_outputs(self, tp: TaskPackage, cases: list[dict], oracle: Path,
                         out_dir: Path) -> dict:
        per_case = []
        problems = []
        for case in cases:
            cid = case["case_id"]
            op = out_dir / "outputs" / f"{cid}.pt"
            if not op.is_file():
                per_case.append({"case_id": cid, "ok": False, "reason": "no output produced"})
                continue
            try:
                exp_blob = torch.load(oracle / "expected" / f"{cid}.pt", weights_only=False)
                in_blob = torch.load(oracle / "inputs" / f"{cid}.pt", weights_only=False)
                act_blob = torch.load(op, weights_only=False)
            except Exception as exc:
                per_case.append({"case_id": cid, "ok": False, "reason": f"load: {exc}"})
                continue
            inputs = tuple(in_blob["inputs"])
            res = C.check_outputs(tp.task_id, tp.spec, inputs, tuple(exp_blob["outputs"]),
                                  tuple(act_blob["outputs"]), *tp.tolerance)
            per_case.append({"case_id": cid, "ok": res["ok"],
                             "problems": res["problems"][:4], "detail": res["detail"]})
            if not res["ok"]:
                problems.append({"case_id": cid, "problems": res["problems"][:3]})
        n_ok = sum(1 for c in per_case if c["ok"])
        return {"ok": n_ok == len(cases), "passed": n_ok, "total": len(cases),
                "first_failures": problems[:5], "per_case": per_case}

    # ------------------------------------------------------------- candidate
    def evaluate_correctness(self, task_id: str, candidate_dir: str, candidate_id: str,
                             phase: str = "dev", job_id: str = "direct") -> dict:
        """Oracle -> isolated candidate run -> parent-side comparison.

        Split out from the full evaluation because it is the stage the
        adversarial tests exercise: it is where a bad candidate must be stopped.
        """
        tp = TaskPackage(self.root, task_id)
        cases = tp.dev_correctness_cases() if phase == "dev" else tp.final_correctness_cases()
        oracle = self.ensure_oracle(task_id, phase, cases)
        t0 = time.time()
        out_dir = self.paths.work / "candidate_check" / task_id / str(candidate_id) / phase
        static_by_case = {c["case_id"]: tp.case_static(c) for c in cases}
        spec = {"mode": "check", "task_id": task_id, "candidate_dir": candidate_dir,
                "cases": cases, "inputs_dir": str(oracle / "inputs"), "out_dir": str(out_dir),
                "static_by_case": static_by_case, "device": self.device}
        try:
            self._run_child(spec, timeout=float(self.protocol.get("timeouts", {}).get("correctness_s", 60)) * 3,
                            task_id=task_id, candidate_id=str(candidate_id))
        except ChildFailure as exc:
            status = exc.status
            if status == "crashed" and "OutOfMemory" in str(exc):
                status = "oom"
            return {"status": status, "error": str(exc), "detail": exc.detail,
                    "phase": phase, "seconds": time.time() - t0}

        run_info = read_json(out_dir / "run.json")
        mutated = [r["case_id"] for r in run_info["results"] if r.get("inputs_mutated")]
        aliased = [r["case_id"] for r in run_info["results"] if r.get("aliased")]
        if mutated or aliased:
            return {"status": "incorrect", "phase": phase, "seconds": time.time() - t0,
                    "error": f"inputs_mutated={mutated[:5]} aliased_outputs={aliased[:5]}",
                    "correctness": {"ok": False, "passed": 0, "total": len(cases),
                                    "reason": "input mutation / output aliasing"}}
        errs = [r for r in run_info["results"] if r["status"] != "ok"]
        if len(errs) == len(cases):
            msg = errs[0].get("error") or ""
            tb = errs[0].get("traceback") or ""
            return {"status": classify_child_failure(msg, tb), "error": msg,
                    "traceback": tb[-4000:], "phase": phase, "seconds": time.time() - t0}
        verdict = self._compare_outputs(tp, cases, oracle, out_dir)
        if not verdict["ok"]:
            return {"status": "incorrect", "correctness": verdict, "phase": phase,
                    "seconds": time.time() - t0,
                    "error": f"{verdict['passed']}/{verdict['total']} correctness cases passed"}
        return {"status": "correct", "correctness": verdict, "phase": phase,
                "seconds": time.time() - t0}

    def evaluate_timing(self, task_id: str, candidate_dir: str, candidate_id: str,
                        phase: str = "dev") -> dict:
        tp = TaskPackage(self.root, task_id)
        bl = self._ensure_baselines(task_id)
        timing_cases = self._timing_cases(tp, phase)
        # The measurement order re-draws from the frozen protocol seed plus a
        # fixed offset, so the order is determined by the protocol yet stays
        # independent from the input-generation seed.
        order_seed = int(self.protocol.get("seeds", {}).get(phase, 0)) + 100003
        tspec = {"mode": "timing", "task_id": task_id, "candidate_dir": candidate_dir,
                 "cases": timing_cases, "baselines": bl["qualified"], "device": self.device,
                 "warmup": self._warmup(), "batches": self._batches(),
                 "buffers": 3 if phase == "final" else 2,
                 "run_seed": int(self.protocol.get("seeds", {}).get(phase, 0)),
                 "order_seed": order_seed,
                 "out": str(self.paths.work / "candidate_timing" / task_id / str(candidate_id) / f"{phase}.json")}
        try:
            self._run_child(tspec, timeout=float(self.protocol.get("timeouts", {}).get("job_s", 300)),
                            task_id=task_id, candidate_id=str(candidate_id))
        except ChildFailure as exc:
            return {"status": "oom" if "OutOfMemory" in str(exc) else exc.status, "error": str(exc)}
        data = read_json(Path(tspec["out"]))
        per_shape: dict[str, float] = {}
        raw: dict[str, dict] = {}
        for case_id, blk in data["cases"].items():
            if blk.get("status") != "ok" or "candidate" not in blk["measurements"]:
                continue
            per_shape[case_id] = blk["measurements"]["candidate"]["gpu_us_median"]
            raw[case_id] = blk["measurements"]
        if not per_shape:
            return {"status": "env_error", "error": "no timing samples collected"}
        # Pair the candidate against baselines measured in the *same* interleaved
        # run. Latencies from an earlier job were taken under a different CPU/GPU
        # contention state and a different clock state, so pairing across jobs
        # would produce a speedup that is an artefact of the machine, not the code.
        paired: dict[str, dict] = {}
        qualified = set(bl["qualified"])
        for case_id, per_label in raw.items():
            for label, meas in per_label.items():
                if label == "candidate" or label not in qualified:
                    continue
                us = meas["gpu_us_median"]
                cur = paired.get(case_id)
                if cur is None or us < cur["latency_us"]:
                    paired[case_id] = {"label": label, "latency_us": us}
        launch_ws = {cid: (blk.get("working_set_bytes") or 0)

                     for cid, blk in data["cases"].items()}

        # The cross-shape diagnostic compares shapes on equal terms, so it uses
        # the state-robust estimator: one machine-state episode inside the job
        # must not inflate one shape's median past another's and hide the
        # launch-bound signature (2026-10-03 audit). Headline numbers stay the
        # plain medians recorded above.
        robust_times = {}
        for case_id, per_label in raw.items():
            samples = (per_label.get("candidate") or {}).get("gpu_us_samples") or []
            robust_times[case_id] = timing.state_robust_median(samples) or per_shape.get(case_id)
        launch_bound, launch_bound_note = timing.flag_launch_bound(robust_times, launch_ws)

        # Shapes that cannot resolve kernel work are a property of the task,
        # so this is decided here next to contamination and travels with the
        # result into promotion and the report.
        below_floor, floor_note = timing.measurement_floor(
            per_shape, data, self.protocol.get("timing") or {})
        footprints = {c: (b.get("footprint_bytes") or 0)
                      for c, b in data["cases"].items()}

        return {"status": "ok", "candidate_us": per_shape, "raw": raw,
                "strongest_baseline": paired, "qualified_baselines": bl["qualified"],
                "unpaired_baseline_reference": bl["strongest"],

                        "working_set_bytes": launch_ws,

                        "below_measurement_floor": below_floor,
                        "measurement_floor_note": floor_note,
                        "footprint_bytes": footprints,
                        "device_l2_bytes": int(((data.get("device") or {}).get("l2_bytes")) or 0),
                        "launch_bound": launch_bound,

                        "launch_bound_note": launch_bound_note,
                "baseline_status": data.get("baseline_status", {}),
                "compile_times": data.get("compile_times", {}),
                "memory": data.get("memory", {}), "precision": data.get("precision", {})}

    def evaluate_candidate(self, job, phase: str = "dev") -> dict:
        task_id, cid, cand_dir = job["task_id"], job["candidate_id"], job["snapshot_dir"]
        t_start = time.time()
        corr = self.evaluate_correctness(task_id, cand_dir, cid, phase=phase, job_id=job["job_id"])
        if corr["status"] != "correct":
            return dict(corr, seconds=time.time() - t_start)
        tm = self.evaluate_timing(task_id, cand_dir, cid, phase=phase)
        if tm.get("status") != "ok":
            return {"status": tm.get("status", "env_error"), "error": tm.get("error"),
                    "phase": phase, "correctness": corr.get("correctness"),
                    "seconds": time.time() - t_start}
        outcome = selection.evaluate_candidate(tm["candidate_us"], tm["strongest_baseline"])
        for case_id, m in tm["raw"].items():
            for label, meas in m.items():
                self.q.record_measurement(self.run_id, job["job_id"], task_id,
                                          str(cid) if label == "candidate" else "baseline",
                                          case_id, label, phase, meas["gpu_us_median"],
                                          meas.get("host_us_median"), meas)
        return {
            "status": "complete", "phase": phase,
            "correctness": corr.get("correctness"),
            "candidate_us": tm["candidate_us"], "strongest_baseline": tm["strongest_baseline"],
            "unpaired_baseline_reference": tm.get("unpaired_baseline_reference"),
"working_set_bytes": tm.get("working_set_bytes"),
"launch_bound": tm.get("launch_bound"),
"launch_bound_note": tm.get("launch_bound_note"),
"below_measurement_floor": tm.get("below_measurement_floor"),
"measurement_floor_note": tm.get("measurement_floor_note"),
"footprint_bytes": tm.get("footprint_bytes"),
"device_l2_bytes": tm.get("device_l2_bytes"),
            "shape_speedups": outcome["shape_speedups"], "task_speedup": outcome["task_speedup"],
            "baseline_status": tm["baseline_status"], "compile_times": tm["compile_times"],
            "memory": tm["memory"], "precision": tm["precision"], "raw": tm["raw"],
            "seconds": time.time() - t_start,
            "gpu_uuid": self.gpu, "host": os.uname().nodename,
        }

    # ------------------------------------------------------------- promotion
    def maybe_promote(self, task_id: str, candidate_id: str, source_hash: str,
                      result: dict, contaminated: bool = False) -> tuple[bool, str]:
        """Decide whether this candidate becomes the champion.

        A below-the-floor note (cache residency / launch overhead) is a
        diagnostic, not a veto: it travels in the result payload and the report
        must carry it, but small operators and launch-bound shapes are real
        workloads this project optimises (2026-10-03 audit).
        """
        if contaminated and not self.protocol.get("allow_contaminated_promotion", False):
            return False, ("measurement was contaminated by a foreign process on the card; "
                           "a speedup taken under interference cannot promote a champion")
        incumbent = self.q.champion(self.run_id, task_id)
        challenger = {"task_speedup": result.get("task_speedup"),
                      "shape_speedups": result.get("shape_speedups", {})}
        ok, why = selection.promote(incumbent, challenger,
                                    min_gain=float(self.protocol.get("promotion_min_gain", 0.02)),
                                    max_slowdown=float(self.protocol.get("promotion_max_slowdown", 0.05)))
        if ok:
            self.q.set_champion(self.run_id, task_id, candidate_id, source_hash,
                                result["task_speedup"], result["shape_speedups"])
        return ok, why

    # ------------------------------------------------------------------ serve
    def stop(self, *_sig):
        self._shutdown = True

    def serve(self, max_jobs: int | None = None, idle_exit_s: float | None = None) -> dict:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        handled = 0
        last_work = time.time()
        self.q.recover_expired(self.run_id, grace_s=0.0)
        self.events.emit("evaluator_started", gpu=self.gpu, owner=self.owner,
                         max_jobs=max_jobs, run_id=self.run_id)
        while not self._shutdown:
            job = self.q.claim(self.run_id, self.owner, self.gpu)
            if job is None:
                if idle_exit_s is not None and time.time() - last_work > idle_exit_s:
                    break
                if max_jobs is not None and handled >= max_jobs:
                    break
                # Exit once the run is drained. Waiting forever for work that may
                # never come leaves a process holding a CUDA context on a shared
                # card, which is exactly the interference this lab measures
                # against. Pass --keep-alive to park instead.
                if not self.protocol.get("keep_alive") and not self.q.has_pending(self.run_id):
                    self.events.emit("evaluator_drained", handled=handled, gpu=self.gpu)
                    break
                time.sleep(self.poll_s)
                continue
            last_work = time.time()
            with GpuLock(self.root, self.physical_key):
                try:
                    status, result = self._execute(job)
                except Exception as exc:  # never let one job kill the loop
                    status, result = "crashed", {"status": "crashed",
                                                 "error": f"{type(exc).__name__}: {exc}"}
            self.q.finish(job, status, result)
            self.q.spend(self.run_id, job["task_id"],
                         gpu_seconds=float(result.get("seconds", 0.0) or 0.0))
            write_json(self.paths.results / f"{job['task_id']}_{job['job_id']}.json",
                       {"job": dict(job), "status": status, "result": result})
            handled += 1
            if max_jobs is not None and handled >= max_jobs:
                break
        self.events.emit("evaluator_stopped", handled=handled, gpu=self.gpu)
        return {"handled": handled, "gpu": self.gpu}

    def _gpu_snapshot(self) -> dict:
        if not self.is_cuda:
            return {"available": False, "device": "cpu"}
        # own_pid is essential: the evaluator holds a CUDA context, so without it
        # we would report ourselves as an interfering process on every job.
        return gpuinfo.snapshot(self.gpu, own_pid=os.getpid())

    def _execute(self, job) -> tuple[str, dict]:
        self.events.emit("job_started", task_id=job["task_id"], job_id=job["job_id"],
                         kind=job["kind"], candidate_id=job["candidate_id"])
        before = self._gpu_snapshot()
        status, result = self._dispatch(job)
        after = self._gpu_snapshot()
        result["gpu_state_before"] = before
        result["gpu_state_after"] = after
        result["contaminated"] = gpuinfo.contaminated(before, after)
        if result["contaminated"]:
            result["contamination_note"] = (
                "other processes or foreign utilisation were present on "
                f"{self.gpu} during this measurement; the latency is indicative only")
            self.events.emit("measurement_contaminated", level="warn", task_id=job["task_id"],
                             job_id=job["job_id"],
                             other_pids=before.get("other_pids", []) + after.get("other_pids", []))
            if job["kind"] == "verify":
                # A final verification that ran next to a foreign process
                # cannot certify a gain (2026-10-03 audit): the number is
                # indicative, so the verdict is INCONCLUSIVE, never complete.
                result["status"] = "inconclusive"
                result["reason"] = ("final verification was contaminated by a foreign "
                                    "process on the card; re-verify on an exclusive card")
                status = "inconclusive"
        # Promotion happens here, after contamination is known, because a speedup
        # measured next to somebody else's job is not evidence about the kernel.
        if job["kind"] == "candidate" and result.get("status") == "complete":
            promoted, why = self.maybe_promote(job["task_id"], job["candidate_id"],
                                               job["source_hash"], result,
                                               contaminated=result["contaminated"])
            result["promoted"] = promoted
            result["promotion_reason"] = why
            self.q.update_candidate_status(self.run_id, job["task_id"], job["candidate_id"],
                                           "promoted" if promoted else "no_gain")
        return status, result

    def _dispatch(self, job) -> tuple[str, dict]:
        kind = job["kind"]
        if kind == "baseline":
            res = self.evaluate_baselines(job["task_id"])
            return "complete", {"status": "complete", "baselines": res}
        if kind == "candidate":
            res = self.evaluate_candidate(job, phase=job["phase"] or "dev")
            status = res.get("status", "crashed")
            return (status if status in TERMINAL_STATUSES else "crashed"), res
        if kind == "verify":
            res = self.verify_task(job["task_id"], job["candidate_id"], job["snapshot_dir"],
                                   job_id=job["job_id"])
            return (res.get("status", "crashed"), res)
        return "env_error", {"status": "env_error", "error": f"unknown job kind {kind}"}

    # -------------------------------------------------------------- final
    def verify_task(self, task_id: str, candidate_id: str, candidate_dir: str,
                    rounds: int | None = None, job_id: str = "direct") -> dict:
        tp = TaskPackage(self.root, task_id)
        rounds = rounds or int(self.protocol.get("final_verification_rounds", 5))
        cases = tp.final_correctness_cases()
        oracle = self.ensure_oracle(task_id, "final", cases)
        static_by_case = {c["case_id"]: tp.case_static(c) for c in cases}
        out_dir = self.paths.work / "verify" / task_id / str(candidate_id)
        spec = {"mode": "check", "task_id": task_id, "candidate_dir": candidate_dir,
                "cases": cases, "inputs_dir": str(oracle / "inputs"), "out_dir": str(out_dir),
                "static_by_case": static_by_case, "device": self.device}
        try:
            self._run_child(spec, timeout=float(self.protocol.get("timeouts", {}).get("correctness_s", 60)) * 3,
                            task_id=task_id, candidate_id=str(candidate_id))
        except ChildFailure as exc:
            return {"status": "crashed", "error": str(exc)}
        verdict = self._compare_outputs(tp, cases, oracle, out_dir)
        if not verdict["ok"]:
            return {"status": "incorrect", "correctness": verdict}

        bl = self._ensure_baselines(task_id)
        timing_cases = self._timing_cases(tp, "final")
        required_case_ids = [c["case_id"] for c in timing_cases]
        all_rounds = []          # full measurement dicts, for the backend flag and the record
        all_rounds_us = []       # per-round medians, for the statistics
        failed_rounds = []
        for r in range(rounds):
            tspec = {"mode": "timing", "task_id": task_id, "candidate_dir": candidate_dir,
                     "cases": timing_cases, "baselines": bl["qualified"], "device": self.device,
                     "warmup": self._warmup(), "batches": self._batches(), "buffers": 3,
                     "run_seed": 900000 + 7919 * r,
                     "order_seed": 900000 + 7919 * r + 100003,
                     "out": str(self.paths.work / "verify_timing" / task_id / str(candidate_id) / f"round{r}.json")}
            try:
                self._run_child(tspec, timeout=float(self.protocol.get("timeouts", {}).get("job_s", 300)),
                                task_id=task_id, candidate_id=str(candidate_id))
            except ChildFailure as exc:
                failed_rounds.append({"round": r, "error": str(exc)})
                self.events.emit("verify_round_failed", level="warn", task_id=task_id,
                                 round=r, message=str(exc))
                continue
            data = read_json(Path(tspec["out"]))
            round_rows: dict = {}
            round_us: dict = {}
            for case_id, blk in data["cases"].items():
                if blk.get("status") != "ok":
                    continue
                for label, m in blk["measurements"].items():
                    round_rows.setdefault(label, {})[case_id] = m
                    round_us.setdefault(label, {})[case_id] = m["gpu_us_median"]
            all_rounds.append(round_rows)
            all_rounds_us.append(round_us)

        stats = verification_statistics(all_rounds_us)
        # The backend flag must come from the real measurements: a placeholder
        # `for m in [{}]` loop made it read "cpu" on every run (2026-10-03 audit).
        stats["timing_backend"] = ("cuda_events" if all_rounds and any(
            m.get("timing_backend") == "cuda_events"
            for rows in all_rounds for per in rows.values() for m in per.values())
            else "cpu")

        # Full pre-registered protocol or INCONCLUSIVE: every round and every
        # timing point (3 dev + 2 reserved) must carry a candidate and a
        # qualified baseline measurement. Missing anything is reported as
        # incomplete, never silently shrunk to the subset that ran
        # (2026-10-03 audit, item 4.3.3).
        gaps = verify_round_gaps(all_rounds_us, required_case_ids, rounds, failed_rounds)
        if gaps:
            return {"status": "inconclusive", "correctness": verdict, "rounds": len(all_rounds),
                    "expected_rounds": rounds, "incomplete": gaps, "statistics": stats,
                    "baselines": bl["qualified"],
                    "reason": "final verification is incomplete; missing rounds or shapes"}

        # The raw final measurements are recorded like the candidate path
        # records its own, so the report pairs candidate and baseline from the
        # same verify round instead of stitching jobs (2026-10-03 audit, 4.3.4).
        for i, rows in enumerate(all_rounds):
            for label, per in rows.items():
                for case_id, m in per.items():
                    self.q.record_measurement(
                        self.run_id, job_id, task_id,
                        str(candidate_id) if label == "candidate" else "baseline",
                        case_id, label, "final", m["gpu_us_median"],
                        m.get("host_us_median"), {**m, "verify_round": i})

        if not self.is_cuda:
            return {"status": "inconclusive", "correctness": verdict, "rounds": len(all_rounds),
                    "statistics": stats, "baselines": bl["qualified"],
                    "reason": "verification ran on a non-CUDA device; never a verified gain"}
        status = "complete"
        if stats.get("task_speedup_ci_low") is None or stats["task_speedup_ci_low"] <= 1.05:
            status = "inconclusive"
        if stats.get("min_shape_median_speedup", 0) < 0.95:
            status = "inconclusive"
        return {"status": status, "correctness": verdict, "rounds": len(all_rounds),
                "statistics": stats, "baselines": bl["qualified"]}


def verify_round_gaps(rounds_us: list[dict], required_case_ids: list[str],
                      expected_rounds: int, failed_rounds: list | None = None) -> list[str]:
    """What is missing from a final verification.

    Formal acceptance pre-registers the round count and every timing point
    (3 dev + 2 reserved). A missing round or shape makes the verification
    INCONCLUSIVE; the denominator is never shrunk to whatever ran
    (2026-10-03 audit, item 4.3.3).
    """
    gaps: list[str] = []
    for f in (failed_rounds or [])[:3]:
        gaps.append("timing round %s failed: %s" % (f.get("round"), f.get("error", "?")))
    if len(rounds_us) != expected_rounds:
        gaps.append("%d/%d timing rounds completed" % (len(rounds_us), expected_rounds))
    for i, rows in enumerate(rounds_us):
        cand = rows.get("candidate") or {}
        for cid in required_case_ids:
            if cid not in cand:
                gaps.append("round %d: no candidate measurement for %s" % (i, cid))
                break
            if not any(cid in per for lb, per in rows.items() if lb != "candidate"):
                gaps.append("round %d: no qualified baseline measurement for %s" % (i, cid))
                break
    return gaps


def verification_statistics(rounds: list[dict], n_boot: int = 2000, seed: int = 12345) -> dict:
    """Paired per-round task speedups and a bootstrap CI over rounds.

    Rounds (not individual calls) are the independent unit: 30 batches inside one
    round share a clock state and a cache state.
    """
    per_round_task = []
    per_shape_medians: dict[str, list[float]] = {}
    per_shape_cand: dict[str, list[float]] = {}
    per_shape_base: dict[str, list[float]] = {}
    for rows in rounds:
        cand = rows.get("candidate", {})
        speedups = {}
        for case_id, cus in cand.items():
            best = None
            for label, per_shape in rows.items():
                if label == "candidate":
                    continue
                us = per_shape.get(case_id)
                if us and us > 0 and (best is None or us < best):
                    best = us
            if best and cus > 0:
                speedups[case_id] = best / cus
                per_shape_medians.setdefault(case_id, []).append(best / cus)
                per_shape_cand.setdefault(case_id, []).append(cus)
                per_shape_base.setdefault(case_id, []).append(best)
        ts = selection.task_speedup(speedups)
        if ts is not None:
            per_round_task.append(ts)
    out = {"rounds_used": len(per_round_task),
           "per_round_task_speedup": per_round_task,
           "shape_speedup_medians": {k: statistics.median(v) for k, v in per_shape_medians.items()},
           "candidate_us_medians": {k: statistics.median(v) for k, v in per_shape_cand.items()},
           "paired_baseline_us_medians": {k: statistics.median(v) for k, v in per_shape_base.items()}}
    if not per_round_task:
        return out
    out["task_speedup_mean"] = statistics.fmean(per_round_task)
    out["min_shape_median_speedup"] = min(out["shape_speedup_medians"].values()) if out["shape_speedup_medians"] else 0
    rng = random.Random(seed)
    boots = []
    n = len(per_round_task)
    for _ in range(n_boot):
        boots.append(statistics.fmean([per_round_task[rng.randrange(n)] for _ in range(n)]))
    boots.sort()
    out["task_speedup_ci_low"] = boots[int(0.025 * n_boot)]
    out["task_speedup_ci_high"] = boots[int(0.975 * n_boot) - 1]
    return out
