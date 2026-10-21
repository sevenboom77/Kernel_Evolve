"""Report generation for a run.

Every task keeps a row, including the ones that won nothing: the denominator of
the report is the fixed suite, never the subset that happened to succeed.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from . import envprobe, selection
from .queue import Queue
from .tasks import TaskPackage
from .util import geometric_mean, read_json

STATUS_LABEL = {
    "verified_gain": "VERIFIED_GAIN",
    "correct_no_gain": "CORRECT_NO_GAIN",
    "inconclusive": "INCONCLUSIVE",
    "incorrect": "INCORRECT",
    "timeout": "TIMEOUT",
    "unsupported_env": "UNSUPPORTED_ENV",
    "not_finished": "NOT_FINISHED",
}


def _fmt(x, nd=2):
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def collect(run_id: str, root: Path) -> dict:
    suite = read_json(Path(root) / "configs" / "suite_v1.json")
    q = Queue(Path(root) / "artifacts" / run_id / "queue.sqlite")
    rows = []
    for tid in suite["tasks"]:
        tp = TaskPackage(root, tid)
        champ = q.champion(run_id, tid)
        bl = q.baselines(run_id, tid)
        strongest = bl.get("__strongest__", {}).get("latencies", {})
        budget = q.budget(run_id, tid)
        cands = q.candidates(run_id, tid)
        jobs = q.list_jobs(run_id, tid, limit=500)
        all_meas = q.measurements(run_id, tid)

        # -------- verification evidence -----------------------------------
        # Only the actual verify job result for the *current* champion's
        # candidate/hash counts. A freeze file, or a queued-but-unfinished
        # job, is not evidence that verification passed (2026-10-03 audit).
        verify = None
        verify_job_id = None
        if champ is not None:
            for j in jobs:                      # newest first
                if (j["kind"] == "verify"
                        and j["candidate_id"] == champ["candidate_id"]
                        and (j["source_hash"] or "") == (champ["source_hash"] or "")):
                    res = q.result(j["job_id"])
                    if res is not None:
                        verify = res.get("payload") or {}
                        verify_job_id = j["job_id"]
                        break

        # -------- development detail --------------------------------------
        # Only the current champion's own job, with its baselines from the
        # same interleaved run. Taking the minimum across every candidate and
        # job could stitch together an implementation that never existed
        # (2026-10-03 audit, item 4.3.5). With no champion, fall back to the
        # newest candidate job so a measured-but-rejected candidate is still
        # shown with its own pairing.
        detail_job_id = None
        if champ is not None:
            cand_meas = [m for m in all_meas
                         if m["phase"] == "dev" and m["label"] == "candidate"
                         and str(m["candidate_id"]) == str(champ["candidate_id"])]
            if cand_meas:
                detail_job_id = max(cand_meas, key=lambda m: m["id"])["job_id"]
        if detail_job_id is None:
            cand_meas = [m for m in all_meas
                         if m["phase"] == "dev" and m["label"] == "candidate"]
            if cand_meas:
                detail_job_id = max(cand_meas, key=lambda m: m["id"])["job_id"]
        dev_candidate: dict = {}
        dev_by_label: dict = {}
        if detail_job_id is not None:
            for m in all_meas:
                if m["job_id"] != detail_job_id:
                    continue
                if m["label"] == "candidate":
                    dev_candidate.setdefault(m["case_id"], []).append(m["latency_us"])
                else:
                    dev_by_label.setdefault(m["label"], {})[m["case_id"]] = m["latency_us"]
            dev_candidate = {k: statistics.median(v) for k, v in dev_candidate.items()}
        dev_qualified = {k for k, v in bl.items() if k != "__strongest__" and v["status"] == "ok"}
        dev_best = selection.strongest_per_shape(dev_by_label, dev_qualified)
        if not dev_best and not dev_candidate and strongest:
            # No candidate job yet: the frozen baseline map is still a
            # measured baseline (2026-10-03 audit: the detail section used to
            # claim "no usable baseline" while counting the task in the
            # deployment summary).
            dev_best = {cid: {"label": "frozen baseline", "latency_us": info["latency_us"]}
                        for cid, info in strongest.items()
                        if isinstance(info, dict) and info.get("latency_us")}

        # -------- verified final detail (paired within verify rounds) -----
        # speedup here is the median of the *per-round paired* ratios — the
        # same statistic the verification gate uses — not the ratio of two
        # separately-taken medians (2026-10-03 audit).
        final_detail = {"baseline": {}, "candidate": {}, "shape_speedup_medians": {}}
        if verify is not None and verify.get("status") == "complete":
            st = verify.get("statistics") or {}
            final_detail["candidate"] = st.get("candidate_us_medians") or {}
            final_detail["shape_speedup_medians"] = st.get("shape_speedup_medians") or {}
            final_detail["baseline"] = {
                cid: {"label": "paired per round", "latency_us": us}
                for cid, us in (st.get("paired_baseline_us_medians") or {}).items()}

        correct_candidates = [c for c in cands if c["status"] in ("promoted", "no_gain")]
        contaminated = False
        below_floor = False
        floor_note = ""
        launch_bound = False
        launch_bound_note = ""
        for j in jobs:
            payload = (q.result(j["job_id"]) or {}).get("payload") or {}
            if payload.get("contaminated"):
                contaminated = True
            if payload.get("below_measurement_floor"):
                below_floor = True
                floor_note = payload.get("measurement_floor_note") or ""
            if payload.get("launch_bound"):
                launch_bound = True
                launch_bound_note = payload.get("launch_bound_note") or ""
        pending = any(j["status"] in ("queued", "running") for j in jobs)

        # Status rules, in priority order. A development champion is never
        # upgraded to VERIFIED_GAIN without passing the independent final check.
        status = "not_finished"
        verified = False
        if verify is not None:
            st = verify.get("status")
            status = {"complete": "verified_gain", "inconclusive": "inconclusive",
                      "incorrect": "incorrect", "timeout": "timeout",
                      "crashed": "inconclusive"}.get(st, "inconclusive")
            verified = status == "verified_gain"
            if verified and (verify.get("contaminated") or contaminated):
                # A verification that ran next to a foreign process cannot
                # certify a gain (2026-10-03 audit).
                status = "inconclusive"
                verified = False
        elif champ is not None:
            gain = (champ.get("task_speedup") or 0) > 1.0
            status = "not_finished" if gain else "correct_no_gain"
        elif contaminated and correct_candidates:
            # Correct and measured next to a foreign job: that number cannot
            # promote anyone, and calling it a plain "no gain" would hide why.
            # Cache-residency and launch-overhead conditions do NOT flip the
            # status (2026-10-03 audit): they are notes carried in the row.
            status = "inconclusive"
        elif correct_candidates and not pending:
            # A correct candidate was measured and never promoted (e.g. one
            # shape slower than the strongest baseline): that is
            # correct_no_gain, not "not finished" (2026-10-03 audit).
            status = "correct_no_gain"
        if not bl:
            status = "not_finished"
            verified = False

        # Shape counts: what this run actually measured, by timing-case id.
        dev_ids = {c["case_id"] for c in tp.dev_timing_cases()}
        res_ids = {c["case_id"] for c in tp.reserved_timing_cases()}
        seen = (set(dev_candidate) | set(dev_best) | set(strongest)
                | set(final_detail["candidate"]))
        n_dev_seen = len((seen & dev_ids))
        n_res_seen = len(set(final_detail["candidate"]) & res_ids)

        rows.append({
            "task_id": tid, "name": tp.name, "group": tp.group, "status": status,
            "contaminated": contaminated, "launch_bound": launch_bound,
            "launch_bound_note": launch_bound_note,
            "below_floor": below_floor, "floor_note": floor_note,
            "champion": champ["candidate_id"] if champ else None,
            "champion_task_speedup": champ["task_speedup"] if champ else None,
            "baseline_available": bool(strongest) or bool(dev_best),
            "strongest_baseline_dev": dev_best, "strongest_baseline_final": strongest,
            "candidate_dev_us": dev_candidate,
            "detail_phase": ("dev (current champion's own job)" if champ is not None
                             else ("dev (last candidate's own job; candidate rejected)"
                                   if dev_candidate else "baseline qualification only")),
            "final_detail": final_detail,
            "final_task_speedup": ((verify or {}).get("statistics") or {}).get("task_speedup_mean")
                                  if verified else None,
            "verify": verify, "verify_job_id": verify_job_id, "verified": verified,
            "verify_incomplete": ((verify or {}).get("incomplete") or []),
            "n_dev_shapes_measured": n_dev_seen,
            "n_reserved_shapes_measured": n_res_seen,
            "candidates_used": budget["source_candidates"],
            "candidate_rows": len(cands),
            "jobs": len(jobs),
            "peak_memory": max((j.get("payload", {}) or {}).get("memory", {}).get("max_memory_allocated", 0)
                               for j in (q.result(j["job_id"]) or {} for j in jobs) if j) if jobs else 0,
        })
    env_path = Path(root) / "artifacts" / run_id / "environment.json"
    proto_path = Path(root) / "artifacts" / run_id / "protocol.json"
    env = read_json(env_path) if env_path.is_file() else {}
    _th = (read_json(proto_path).get("target_hardware") if proto_path.is_file() else None) or {}
    # An empty placeholder (written before the environment probe) must not
    # defeat the fallback: the report always shows the hardware that measured
    # the numbers (2026-10-03 audit).
    target = _th if _th.get("gpu_name") else envprobe.target_hardware(env)
    _proto = read_json(proto_path) if proto_path.is_file() else {}
    return {"run_id": run_id, "suite_id": suite["suite_id"], "rows": rows,
            "timing_shapes": _proto.get("timing_shapes") or {},
            "timing_shapes_protocol_id": _proto.get("timing_shapes_protocol_id"),
            "budgets": suite["budgets"], "gpu_rules": suite["gpu_rules"],
            "target_hardware": target}


def write_csv(collected: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "name", "group", "status", "champion", "champion_task_speedup",
              "verified_task_speedup",
              "candidates_used", "n_dev_shapes", "n_reserved_shapes", "verified_low_ci",
              "peak_memory_bytes", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in collected["rows"]:
            v = r.get("verify") or {}
            st = (v.get("statistics") or {})
            w.writerow({
                "task_id": r["task_id"], "name": r["name"], "group": r["group"],
                "status": STATUS_LABEL.get(r["status"], r["status"]),
                "champion": r["champion"] or "",
                "champion_task_speedup": _fmt(r["champion_task_speedup"], 4),
                "verified_task_speedup": _fmt(r.get("final_task_speedup"), 4),
                "candidates_used": r["candidates_used"],
                "n_dev_shapes": r.get("n_dev_shapes_measured", 0),
                "n_reserved_shapes": r.get("n_reserved_shapes_measured", 0),
                "verified_low_ci": _fmt(st.get("task_speedup_ci_low"), 4),
                "peak_memory_bytes": r["peak_memory"] or "",
                "notes": "",
            })
    return path


def render_markdown(collected: dict, root: Path) -> str:
    rows = collected["rows"]
    n = len(rows)
    verified = [r for r in rows if r["status"] == "verified_gain"]
    core = [r for r in rows if r["group"] == "core"]
    core_verified = [r for r in core if r["status"] == "verified_gain"]
    deployable = [r for r in rows if r.get("baseline_available")]
    unavailable = [r["task_id"] for r in rows if not r.get("baseline_available")]
    # Deployment summary: only candidates that passed independent verification
    # contribute their verified speedup; every other task keeps its baseline
    # and is counted at 1.0. Development numbers never appear here
    # (2026-10-03 audit, item 4.3.6).
    gain_geomean = geometric_mean([
        (r.get("final_task_speedup") or 1.0) if r["status"] == "verified_gain" else 1.0
        for r in deployable]) if deployable else None

    out = [f"# kernel-lab report - run `{collected['run_id']}`", ""]
    th = collected.get("target_hardware") or {}
    if th.get("gpu_name"):
        sm = f"{th.get('multiprocessor_count')} SM, " if th.get("multiprocessor_count") else ""
        out.append(f"- measured on: **{th['gpu_name']}** (cc {th.get('compute_capability')}, "
                   f"{sm}"
                   f"{(th.get('total_memory_bytes') or 0) / 2**30:.0f} GiB, "
                   f"driver {th.get('driver_version')}, x{th.get('gpu_count')})")
        out.append("- these numbers belong to that hardware; they do not transfer to a "
                   "different GPU generation")
    out.append(f"- suite: `{collected['suite_id']}` with a fixed denominator of {n} tasks")
    out.append(f"- VERIFIED_GAIN: **{len(verified)}/{n}** overall, {len(core_verified)}/{len(core)} in the core 10")
    out.append(f"- tasks with a usable baseline: {len(deployable)}/{n}")
    out.append(f"- deployment geometric-mean speedup over the {len(deployable)} tasks with a usable "
               f"baseline: {_fmt(gain_geomean, 3)} (only independently verified gains count; "
               f"every other task keeps its baseline and is counted at 1.0)")
    if unavailable:
        out.append(f"- tasks without a usable baseline (unavailable, excluded from the geomean): "
                   f"{', '.join(unavailable)}")
    _ov = collected.get("timing_shapes") or {}
    if _ov:
        _pid = collected.get("timing_shapes_protocol_id")
        out.append("- enlarged timing shapes opted in%s: "
                   % (f" (separate protocol `{_pid}`)" if _pid else "")
                   + ", ".join("%s (%d shapes)" % (t, len(v)) for t, v in sorted(_ov.items()))
                   + "; this run is a separate protocol from the main track and its numbers "
                     "must not be summarised together with original-shape runs")
    out.append("")
    out.append("| task | group | status | champion | dev speedup | verified speedup | 95% CI low | "
               "min shape median | candidates | peak mem MiB | shared-card interference | "
               "launch-bound | measurable |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        v = r.get("verify") or {}
        st = (v.get("statistics") or (v.get("result") or {}).get("statistics") or {})
        out.append("| {tid} | {grp} | {st} | {ch} | {ts} | {vs} | {ci} | {ms} | {nc} | {mem} | {cont} | {lb} | {fl} |".format(
            tid=r["task_id"], grp=r["group"], st=STATUS_LABEL.get(r["status"], r["status"]),
            ch=r["champion"] or "-", ts=_fmt(r["champion_task_speedup"], 3),
            vs=_fmt(r.get("final_task_speedup"), 3),
            ci=_fmt(st.get("task_speedup_ci_low"), 3),
            ms=_fmt(st.get("min_shape_median_speedup"), 3),
            nc=r["candidates_used"], mem=_fmt((r["peak_memory"] or 0) / 2**20, 0),
            cont="yes" if r.get("contaminated") else "no",
            lb="yes" if r.get("launch_bound") else "no",
            fl=("-" if not r.get("candidate_rows") else
                ("yes" if not r.get("below_floor") else "no"))))
    out.append("")
    out.append("## Per-shape detail")
    out.append("")
    for r in rows:
        out.append(f"### {r['task_id']} {r['name']} - {STATUS_LABEL.get(r['status'], r['status'])}")
        if r.get("champion") and not r.get("verified"):
            out.append("")
            out.append("_development champion only: not yet re-measured by independent final "
                       "verification, so no gain is claimed_")
        if r.get("verify_incomplete"):
            out.append("")
            out.append("_final verification is incomplete (INCONCLUSIVE; the denominator is not "
                       "shrunk): " + "; ".join(r["verify_incomplete"]) + "_")
        # Show the verified (final) pair when it exists, otherwise the development
        # pair; never mix a baseline from one phase with a candidate from another.
        # The final map holds within-round paired medians and the speedup column
        # is the same statistic the verification gate uses (median of per-round
        # paired ratios); the dev map comes from one candidate job only.
        if r.get("verified"):
            fd = r.get("final_detail") or {}
            pairs, cand_map = fd.get("baseline") or {}, fd.get("candidate") or {}
            speedups = fd.get("shape_speedup_medians") or {}
            phase = "final (median of paired verification rounds)"
        else:
            pairs, cand_map = r["strongest_baseline_dev"], r["candidate_dev_us"]
            speedups = {cid: selection.speedup(base.get("latency_us"), cand_map.get(cid))
                        for cid, base in pairs.items()}
            phase = r.get("detail_phase") or "dev"
        if pairs:
            out.append("")
            out.append(f"_phase: {phase}_")
            if r.get("floor_note") or r.get("launch_bound_note"):
                notes = [x for x in (r.get("floor_note"), r.get("launch_bound_note")) if x]
                joined = "; ".join(n[len("measurement note: "):] if n.startswith("measurement note: ")
                                   else n for n in notes)
                out.append("")
                out.append(f"_measurement note: {joined}_")
            out.append("")
            out.append("| shape | strongest baseline | baseline us | candidate us | speedup |")
            out.append("|---|---|---|---|---|")
            for case_id, base in sorted(pairs.items()):
                cus = cand_map.get(case_id)
                out.append(f"| {case_id} | {base.get('label')} | {_fmt(base.get('latency_us'))} | "
                           f"{_fmt(cus)} | {_fmt(speedups.get(case_id), 3)} |")
        elif r.get("baseline_available"):
            out.append("")
            out.append("_baselines were measured but no candidate was paired in this phase_")
        else:
            out.append("")
            out.append("_no usable baseline measured (environment or applicability)_")
        out.append("")
    out.append("## Notes")
    out.append("")
    out.append("- Speedups are relative to the fastest *tolerance-satisfying* baseline for that "
               "shape, not to eager alone.")
    out.append("- A task without a verified gain keeps its baseline as the deployment choice; "
               "no task is deleted from this denominator.")
    out.append("- Operator-level speedups are not model-level speedups.")
    out.append("- \"shared-card interference = yes\" means another process was computing on the "
               "same physical card while the numbers were taken; those latencies are indicative "
               "and must be re-measured on an exclusive card before being called a gain.")
    out.append("- \"measurable = no\" is a note, not a verdict: the whole footprint fits in "
               "the device L2 cache (the number reflects cache bandwidth) or the measured "
               "time is at the host launch cost (the host is the bottleneck). Such gains "
               "still count — the note says what actually got faster. \"-\" means the task "
               "was never measured.")
    return "\n".join(out) + "\n"


def write_report(run_id: str, root: Path, out_md: Path | None = None,
                 out_csv: Path | None = None) -> dict:
    collected = collect(run_id, root)
    md = render_markdown(collected, root)
    out_md = Path(out_md or (Path(root) / "artifacts" / run_id / "final_report.md"))
    out_csv = Path(out_csv or (Path(root) / "artifacts" / run_id / "final_table.csv"))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    write_csv(collected, out_csv)
    return {"markdown": str(out_md), "csv": str(out_csv),
            "n_tasks": len(collected["rows"]),
            "n_verified_gain": sum(1 for r in collected["rows"] if r["status"] == "verified_gain")}
