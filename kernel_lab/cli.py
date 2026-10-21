"""``python -m kernel_lab <subcommand>`` - every subcommand prints JSON on stdout.

Logs and progress go to stderr so that a DSH session can parse stdout directly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

from . import correctness as C
from . import dsh_adapter, envprobe, reporting, static_checks, suitecheck
from .events import EventLog
from .evaluator import Evaluator
from .paths import run_paths
from .queue import Queue
from .tasks import TaskPackage, load_suite, validate_timing_override
from .util import read_json, sha256_dir, sha256_file, utcnow, write_json

ROOT = Path(__file__).resolve().parent.parent


def _json_feedback(value):
    """Keep non-finite diagnostics explicit without invalid JSON on stdout."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _json_feedback(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_feedback(item) for item in value]
    return value


def _emit(obj) -> int:
    json.dump(_json_feedback(obj), sys.stdout, indent=1, default=str, allow_nan=False)
    sys.stdout.write("\n")
    return 0 if (not isinstance(obj, dict) or obj.get("ok", True)) else 1


def _python() -> str:
    v = os.environ.get("KL_PYTHON")
    if v:
        return v
    v = ROOT / ".venv" / "bin" / "python"
    return str(v) if v.is_file() else sys.executable


def _existing_environment(run_id: str) -> dict:
    """The environment probe for this run, if one has been written yet."""
    p = run_paths(run_id).environment
    return read_json(p) if p.is_file() else {}


def _timing_shapes_data() -> dict:
    p = ROOT / "configs" / "timing_shapes.json"
    return read_json(p) if p.is_file() else {}


def _timing_shapes() -> dict:
    """Shape overrides declared in configs/timing_shapes.json, task-keyed."""
    return {k: v for k, v in (_timing_shapes_data().get("tasks") or {}).items()}


def _protocol(run_id: str, enlarged_shapes: bool = False) -> dict:
    suite = read_json(ROOT / "configs" / "suite_v1.json")
    proto = {
        "schema_version": 1,
        "run_id": run_id,
        "created": utcnow(),
        "suite_id": suite["suite_id"],
        "seeds": {"dev": 20261002, "final": 8123671},
        "timeouts": {
            "job_s": suite["gpu_rules"]["candidate_job_timeout_s"],
            "compile_s": suite["gpu_rules"]["compile_timeout_s"],
            "correctness_s": suite["gpu_rules"]["correctness_timeout_s"],
        },
        "timing": suite["gpu_rules"],
        "promotion_min_gain": suite["budgets"]["promotion_min_relative_gain"],
        "promotion_max_slowdown": suite["budgets"]["promotion_max_slowdown_vs_baseline"],
        "final_verification_rounds": suite["gpu_rules"]["final_verification_rounds"],
        "baseline_selection": suite["baseline_selection"],
        "total_source_candidates_cap": suite["budgets"]["total_source_candidates_cap"],
        "precision": {"matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                      "fp16_dot": "fp32 accumulation"},
        "frozen_components": {
            "suite_v1.json": sha256_file(ROOT / "configs" / "suite_v1.json"),
            "tasks": sha256_dir(ROOT / "tasks"),
            # The evaluator code itself is part of the frozen protocol: the
            # 2026-10-03 audit's repairs make old and new evaluator versions
            # produce different numbers, so a run must record which one it used.
            "kernel_lab": sha256_dir(ROOT / "kernel_lab"),
        },
        "target_hardware": envprobe.target_hardware(_existing_environment(run_id)),
        "timing_shapes": {},
    }
    # Enlarged shapes are an explicit per-run opt-in and a separate protocol:
    # the default run measures the declared task shapes (2026-10-03 audit).
    if enlarged_shapes:
        proto["timing_shapes"] = _timing_shapes()
        proto["timing_shapes_protocol_id"] = (
            _timing_shapes_data().get("protocol_id") or "enlarged-shapes")
    return proto


def _ensure_run(run_id: str, protocol_file: str | None = None,
                enlarged_shapes: bool = False) -> tuple:
    paths = run_paths(run_id).ensure()
    # The environment probe must land before the protocol is frozen: the
    # protocol's target_hardware records which card actually measured the
    # numbers (stage A freezes GPU metadata with the run).
    if not paths.environment.is_file():
        env = envprobe.probe(smoke=False)
        write_json(paths.environment, env)
    if not paths.protocol.is_file():
        if protocol_file:
            proto = read_json(protocol_file)
            proto.setdefault("run_id", run_id)
            proto.setdefault("created", utcnow())
            proto.setdefault("suite_id", read_json(ROOT / "configs" / "suite_v1.json")["suite_id"])
            proto.setdefault("frozen_components", {
                "suite_v1.json": sha256_file(ROOT / "configs" / "suite_v1.json"),
                "tasks": sha256_dir(ROOT / "tasks"),
                "kernel_lab": sha256_dir(ROOT / "kernel_lab")})
            write_json(paths.protocol, proto)
        else:
            write_json(paths.protocol, _protocol(run_id, enlarged_shapes=enlarged_shapes))
    proto = read_json(paths.protocol)
    if enlarged_shapes and not (proto.get("timing_shapes") or {}):
        print("warning: --timing-shapes ignored: the run protocol already exists "
              "without enlarged shapes; start a new run to opt in", file=sys.stderr)
    log = EventLog(paths.events, run_id=run_id, source="cli")
    q = Queue(paths.db, events=log)
    q.create_run(run_id, proto["suite_id"], proto["frozen_components"]["tasks"])
    return paths, q, proto


# --------------------------------------------------------------------------- #
def cmd_doctor(a) -> int:
    env = envprobe.probe(python_exe=a.python, smoke=not a.no_cuda_smoke)
    if a.output:
        write_json(a.output, env)
    for line in envprobe.report_lines(env):
        print(line, file=sys.stderr)
    return _emit({"ok": bool(env.get("gpus_total")), "output": a.output, **{
        "gpus_total": env.get("gpus_total"),
        "gpus_busy_at_probe": env.get("gpus_busy_at_probe"),
        "shared_machine": env.get("shared_machine"),
        "cuda_smoke_ok": env.get("cuda_smoke", {}).get("ok"),
        "dsh": env.get("dsh", {}).get("version"),
        "driver": env.get("driver_version"),
    }})


def cmd_suite_check(a) -> int:
    res = suitecheck.check_suite(ROOT, tasks=a.tasks, with_baselines=not a.no_baselines)
    if a.output:
        write_json(a.output, res)
    return _emit(res)


def cmd_baseline(a) -> int:
    paths, q, proto = _ensure_run(a.run, getattr(a, "protocol_file", None),
                                  enlarged_shapes=getattr(a, "timing_shapes", False))
    ids = a.tasks or read_json(ROOT / "configs" / "suite_v1.json")["groups"][a.group]
    out = {}
    for tid in ids:
        job = q.enqueue(a.run, tid, "baseline", phase="dev", priority=10,
                        payload={"reason": "frozen baseline qualification"})
        out[tid] = job
    return _emit({"ok": True, "run": a.run, "queued": len(ids), "jobs": out})


def cmd_serve(a) -> int:
    paths, q, proto = _ensure_run(a.run, getattr(a, "protocol_file", None),
                                  enlarged_shapes=getattr(a, "timing_shapes", False))
    _problems = validate_timing_override(ROOT, proto)
    if _problems:
        return _emit({"ok": False, "error": "invalid timing_shapes in the run protocol",
                      "problems": _problems})
    if getattr(a, "keep_alive", False) and not proto.get("keep_alive"):
        proto = dict(proto)
        proto["keep_alive"] = True
        write_json(paths.protocol, proto)
    ev = Evaluator(run_id=a.run, gpu=a.gpu, root=ROOT, python_exe=a.python or _python(),
                   poll_s=a.poll, log_level=a.log_level, protocol=proto)
    res = ev.serve(max_jobs=a.max_jobs, idle_exit_s=a.idle_exit)
    return _emit({"ok": True, **res})


def cmd_submit(a) -> int:
    paths, q, proto = _ensure_run(a.run, getattr(a, "protocol_file", None),
                                  enlarged_shapes=getattr(a, "timing_shapes", False))
    if q.is_frozen(a.run):
        return _emit({"ok": False, "error": "run is frozen; no new optimisation candidates"})
    tp = TaskPackage(ROOT, a.task)
    cand_dir = Path(a.candidate_dir).resolve()
    checks = static_checks.check_snapshot(cand_dir)
    if not checks["ok"]:
        return _emit({"ok": False, "status": "INVALID_SOURCE", "findings": checks["findings"]})

    budget = q.budget(a.run, a.task)
    cap = int(tp.spec["max_candidate_attempts"])
    total = q.total_source_candidates(a.run)
    cap_total = int(proto["total_source_candidates_cap"])
    if budget["source_candidates"] >= cap:
        return _emit({"ok": False, "status": "BUDGET_EXHAUSTED",
                      "used": budget["source_candidates"], "cap": cap})
    if total >= cap_total:
        return _emit({"ok": False, "status": "SUITE_BUDGET_EXHAUSTED", "used": total, "cap": cap_total})

    rows = q.candidates(a.run, a.task)
    stagnant = 0
    for r in reversed(rows):
        if r["status"] == "promoted":
            break
        stagnant += 1
    if rows and stagnant >= int(tp.spec["stagnation_attempts"]):
        return _emit({"ok": False, "status": "STAGNATION_STOP",
                      "consecutive_non_improving": stagnant})

    source_hash = sha256_dir(cand_dir)
    seq = len(rows) + 1
    candidate_id = a.candidate_id or f"c{seq:03d}-{source_hash[:8]}"
    snap = paths.snapshots / a.task / candidate_id
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(cand_dir, snap)
    for p in snap.rglob("*"):
        if p.is_file():
            p.chmod(0o444)

    proposal = {}
    pj = cand_dir / "proposal.json"
    if pj.is_file():
        try:
            proposal = read_json(pj)
        except Exception as exc:
            proposal = {"unparsable": str(exc)}
    parent_hash = proposal.get("parent_hash") or a.parent_hash

    q.record_candidate(a.run, a.task, candidate_id, source_hash, parent_hash, proposal, "pending")
    phase = a.phase
    job = q.enqueue(a.run, a.task, "candidate", candidate_id=candidate_id,
                    source_hash=source_hash, snapshot_dir=str(snap), phase=phase,
                    priority=a.priority, max_attempts=a.max_attempts,
                    payload={"proposal": proposal})
    q.spend(a.run, a.task, source_candidates=1, variants=int(proposal.get("launch_variants", 1) or 1))
    return _emit({"ok": True, "job_id": job, "candidate_id": candidate_id,
                  "source_hash": source_hash, "snapshot": str(snap), "phase": phase,
                  "budget_used": budget["source_candidates"] + 1, "budget_cap": cap})


def cmd_job(a) -> int:
    paths, q, proto = _ensure_run(a.run)
    job = q.job(a.id)
    if job is None:
        return _emit({"ok": False, "error": "no such job"})
    res = q.result(a.id)
    return _emit({"ok": True, "job": dict(job),
                  "status": job["status"],
                  "result": (res or {}).get("payload"),
                  "has_result": res is not None})


def cmd_task_status(a) -> int:
    paths, q, proto = _ensure_run(a.run)
    tp = TaskPackage(ROOT, a.task)
    champ = q.champion(a.run, a.task)
    bl = q.baselines(a.run, a.task)
    budget = q.budget(a.run, a.task)
    rows = q.candidates(a.run, a.task)
    stagnant = 0
    for r in reversed(rows):
        if r["status"] == "promoted":
            break
        stagnant += 1
    return _emit({
        "ok": True, "run": a.run, "task_id": a.task, "name": tp.name,
        "champion": champ,
        "budget": {**budget, "max_candidates": tp.spec["max_candidate_attempts"],
                   "remaining": max(0, tp.spec["max_candidate_attempts"] - budget["source_candidates"])},
        "stagnation": {"consecutive_non_improving": stagnant,
                       "stop_after": tp.spec["stagnation_attempts"]},
        "baselines": {k: {"status": v["status"], "shapes": len(v["latencies"])}
                      for k, v in bl.items() if k != "__strongest__"},
        "strongest_baseline_per_shape": bl.get("__strongest__", {}).get("latencies", {}),
        "candidates": [{"candidate_id": r["candidate_id"], "status": r["status"],
                        "hash": (r["source_hash"] or "")[:12]} for r in rows[-12:]],
    })


def cmd_freeze(a) -> int:
    paths, q, proto = _ensure_run(a.run)
    out = {}
    suite = read_json(ROOT / "configs" / "suite_v1.json")
    ids = a.tasks or suite["tasks"]
    for tid in ids:
        tp = TaskPackage(ROOT, tid)
        champ = q.champion(a.run, tid)
        bl = q.baselines(a.run, tid)
        rec = {
            "task_id": tid, "frozen_at": utcnow(),
            "champion": champ,
            "baselines": {k: {"status": v["status"], "latencies": v["latencies"]}
                          for k, v in bl.items() if k != "__strongest__"},
            "strongest_per_shape": bl.get("__strongest__", {}).get("latencies", {}),
            "hashes": {
                "spec": sha256_file(tp.spec_path),
                "reference": sha256_file(tp.dir / "reference.py"),
                "cases": sha256_file(tp.dir / "cases.py"),
                "baselines": sha256_file(tp.dir / "baselines.py"),
            },
            "protocol_hash": proto["frozen_components"]["suite_v1.json"],
        }
        write_json(paths.frozen / f"{tid}_frozen.json", rec)
        out[tid] = {"champion": champ["candidate_id"] if champ else None,
                    "has_champion": champ is not None}
    q.set_frozen(a.run, True)
    return _emit({"ok": True, "run": a.run, "frozen_tasks": len(ids), "tasks": out})


def cmd_verify(a) -> int:
    paths, q, proto = _ensure_run(a.run)
    suite = read_json(ROOT / "configs" / "suite_v1.json")
    ids = a.tasks or suite["tasks"]
    queued = {}
    for tid in ids:
        champ = q.champion(a.run, tid)
        if champ is None:
            continue
        job = q.enqueue(a.run, tid, "verify", candidate_id=champ["candidate_id"],
                        source_hash=champ["source_hash"],
                        snapshot_dir=str(paths.snapshots / tid / champ["candidate_id"]),
                        phase="final", priority=5)
        queued[tid] = job
    return _emit({"ok": True, "run": a.run, "queued": len(queued), "jobs": queued,
                  "skipped_no_champion": [t for t in ids if t not in queued]})


def cmd_report(a) -> int:
    res = reporting.write_report(a.run, ROOT, a.output_md, a.output_csv)
    return _emit({"ok": True, **res})


def cmd_resume(a) -> int:
    paths, q, proto = _ensure_run(a.run)
    moved = q.recover_expired(a.run, grace_s=0.0)
    counts: dict[str, int] = {}
    for j in q.list_jobs(a.run, limit=10000):
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return _emit({"ok": True, "run": a.run, "requeued": moved,
                  "job_status_counts": counts,
                  "frozen": q.is_frozen(a.run)})


def cmd_dsh(a) -> int:
    paths, q, proto = _ensure_run(a.run)
    disc = dsh_adapter.discover()
    if a.action == "discover":
        return _emit({"ok": disc["found"], **disc})
    if a.action == "digest":
        return _emit({"ok": True, **dsh_adapter.task_digest(ROOT, a.run, a.task)})
    if a.action == "prompt":
        ctx = {}
        if a.task:
            ctx["TASK_DIGEST"] = dsh_adapter.task_digest(ROOT, a.run, a.task)
        if a.context:
            ctx.update(read_json(a.context))
        text = dsh_adapter.render_prompt(Path(a.template), ctx)
        outp = paths.dsh / (a.out or f"prompt_{Path(a.template).stem}.md")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text)
        return _emit({"ok": True, "prompt": str(outp), "chars": len(text)})
    if a.action == "launch":
        res = dsh_adapter.launch(a.run, ROOT, Path(a.template), profile=a.profile,
                                 extra_args=a.extra, timeout=a.timeout)
        return _emit(res)
    return _emit({"ok": False, "error": f"unknown dsh action {a.action}"})


def cmd_tasks(a) -> int:
    suite = read_json(ROOT / "configs" / "suite_v1.json")
    out = []
    for tid in suite["tasks"]:
        tp = TaskPackage(ROOT, tid)
        out.append({"task_id": tid, "name": tp.name, "group": tp.group,
                    "arity": tp.output_arity, "dev_shapes": len(tp.dev_shapes),
                    "reserved_shapes": len(tp.reserved_shapes),
                    "dev_cases": len(tp.dev_correctness_cases()),
                    "final_cases": len(tp.final_correctness_cases()),
                    "max_candidates": tp.spec["max_candidate_attempts"]})
    return _emit({"ok": True, "suite": suite["suite_id"], "n": len(out), "tasks": out})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m kernel_lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor"); d.add_argument("--output"); d.add_argument("--python")
    d.add_argument("--no-cuda-smoke", action="store_true"); d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("suite-check"); s.add_argument("--suite", default="configs/suite_v1.json")
    s.add_argument("--tasks", nargs="*"); s.add_argument("--output")
    s.add_argument("--no-baselines", action="store_true"); s.set_defaults(func=cmd_suite_check)

    t = sub.add_parser("tasks"); t.set_defaults(func=cmd_tasks)

    b = sub.add_parser("baseline"); b.add_argument("--run", required=True)
    b.add_argument("--tasks", nargs="*"); b.add_argument("--group", default="core")
    b.add_argument("--protocol-file")
    b.add_argument("--timing-shapes", action="store_true",
                   help="opt this new run in to the enlarged shapes in "
                        "configs/timing_shapes.json (a separate protocol; ignored "
                        "if the run already exists)")
    b.set_defaults(func=cmd_baseline)

    sv = sub.add_parser("serve-evaluator"); sv.add_argument("--run", required=True)
    sv.add_argument("--gpu", required=True); sv.add_argument("--python")
    sv.add_argument("--poll", type=float, default=2.0); sv.add_argument("--max-jobs", type=int)
    sv.add_argument("--idle-exit", type=float); sv.add_argument("--log-level", default="info")
    sv.add_argument("--protocol-file")
    sv.add_argument("--timing-shapes", action="store_true",
                   help="opt this new run in to the enlarged shapes (see baseline)")
    sv.add_argument("--keep-alive", action="store_true")
    sv.set_defaults(func=cmd_serve)

    su = sub.add_parser("submit"); su.add_argument("--run", required=True)
    su.add_argument("--task", required=True); su.add_argument("--candidate-dir", required=True)
    su.add_argument("--candidate-id"); su.add_argument("--parent-hash")
    su.add_argument("--phase", default="dev", choices=["dev", "final"])
    su.add_argument("--priority", type=int, default=100)
    su.add_argument("--max-attempts", type=int, default=2)
    su.add_argument("--protocol-file")
    su.add_argument("--timing-shapes", action="store_true",
                   help="opt this new run in to the enlarged shapes (see baseline)")
    su.set_defaults(func=cmd_submit)

    j = sub.add_parser("job"); j.add_argument("--run", required=True)
    j.add_argument("--id", required=True); j.set_defaults(func=cmd_job)

    ts = sub.add_parser("task-status"); ts.add_argument("--run", required=True)
    ts.add_argument("--task", required=True); ts.set_defaults(func=cmd_task_status)

    f = sub.add_parser("freeze"); f.add_argument("--run", required=True)
    f.add_argument("--tasks", nargs="*"); f.set_defaults(func=cmd_freeze)

    v = sub.add_parser("verify"); v.add_argument("--run", required=True)
    v.add_argument("--tasks", nargs="*"); v.set_defaults(func=cmd_verify)

    r = sub.add_parser("report"); r.add_argument("--run", required=True)
    r.add_argument("--output-md"); r.add_argument("--output-csv"); r.set_defaults(func=cmd_report)

    rs = sub.add_parser("resume"); rs.add_argument("--run", required=True)
    rs.set_defaults(func=cmd_resume)

    dsh = sub.add_parser("dsh"); dsh.add_argument("action", choices=["discover", "digest", "prompt", "launch"])
    dsh.add_argument("--run", required=True); dsh.add_argument("--task")
    dsh.add_argument("--template"); dsh.add_argument("--context"); dsh.add_argument("--out")
    dsh.add_argument("--profile", default="headless"); dsh.add_argument("--extra", nargs="*")
    dsh.add_argument("--timeout", type=float); dsh.set_defaults(func=cmd_dsh)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
