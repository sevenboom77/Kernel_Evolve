"""GPU state sampling.

Timing on a shared card is only meaningful if we also record what else was
running.  Every measurement therefore carries a before/after snapshot: clock,
power, temperature, utilisation and the set of *other* PIDs on the same physical
device.  A run with foreign processes is flagged contaminated and must not be
presented as a clean number.  Our own PID and its children are excluded, and a
process nvidia-smi refuses to name still counts as foreign.

The evaluator holds a CUDA context of its own, so its PID (and its worker
children) legitimately appear in nvidia-smi process lists.  Counting those as
interference would flag literally every measurement, so they are excluded by
walking /proc rather than by guessing.
"""
from __future__ import annotations

import os
import subprocess

QUERY = ("index,uuid,utilization.gpu,memory.used,memory.total,power.draw,"
         "temperature.gpu,clocks.current.sm,clocks.max.sm")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return ""


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def sample() -> list[dict]:
    out = _run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"])
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            continue
        idx, uuid, util, mused, mtot, pwr, temp, csm, cmax = parts
        rows.append({"index": int(idx), "uuid": uuid, "util": _num(util),
                     "mem_used_mib": _num(mused), "mem_total_mib": _num(mtot),
                     "power_w": _num(pwr), "temp_c": _num(temp),
                     "sm_clock_mhz": _num(csm), "sm_clock_max_mhz": _num(cmax)})
    return rows


def compute_procs() -> list[dict]:
    out = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits"])
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            # nvidia-smi prints "N/A" for a process it is not allowed to name.
            # That is still somebody using the card, so keep it as unknown
            # rather than dropping the evidence.
            pid = None
        rows.append({"pid": pid, "used_memory_mib": _num(parts[1]), "uuid": parts[2]})
    return rows


def own_process_tree(root: int | None = None) -> set[int]:
    """root (default: this process) plus every live descendant, read from /proc."""
    root = os.getpid() if root is None else root
    tree = {root}
    entries: list[tuple[int, int]] = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return tree
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                rest = fh.read().rsplit(b") ", 1)[-1].split()
            entries.append((int(name), int(rest[1])))
        except (OSError, IndexError, ValueError):
            continue
    changed = True
    while changed:
        changed = False
        for pid, ppid in entries:
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def classify(procs_on_card: list[dict], own_pids) -> dict:
    """Split one card process list into ours and everybody else."""
    own = set(own_pids or ())
    foreign, mine, unknown = [], [], 0
    for p in procs_on_card:
        pid = p.get("pid")
        if pid is None:
            unknown += 1
            foreign.append(-1)        # placeholder for an unnameable process
        elif pid in own:
            mine.append(pid)
        else:
            foreign.append(pid)
    return {"foreign_pids": sorted(foreign), "own_pids": sorted(mine),
            "foreign_processes": len(foreign), "own_processes": len(mine),
            "unknown_processes": unknown,
            "visible_processes": len(foreign) + len(mine)}


def snapshot(uuid: str | int | None = None, own_pid: int | None = None,
             own_tree: set[int] | None = None) -> dict:
    """State of one card, separating our own processes from everybody else."""
    gpus = sample()
    procs = compute_procs()
    target = None
    for g in gpus:
        if uuid is None or str(g["uuid"]) == str(uuid) or str(g["index"]) == str(uuid):
            target = g
            break
    if target is None:
        return {"available": False, "requested": uuid}
    own = own_tree if own_tree is not None else own_process_tree(own_pid)
    on_card = [p for p in procs if p["uuid"] == target["uuid"]]
    cls = classify(on_card, own)
    foreign = set(cls["foreign_pids"])
    return {**target, "available": True, **cls,
            "other_processes": cls["foreign_processes"],
            "other_pids": cls["foreign_pids"],
            "own_pids_present": cls["own_pids"],
            "other_memory_mib": sum(p["used_memory_mib"] or 0 for p in on_card
                                    if p["pid"] in foreign)}


def contaminated(before: dict, after: dict) -> bool:
    """Was anybody else computing on this card while we measured?

    Deliberately based on process evidence only.  Utilisation is a one-second
    average that still reads high just after our own workers exit, so treating it
    as interference flags our own residue; a PID we are not allowed to name is
    counted as foreign, so a job we cannot enumerate is still caught.
    """
    for snap in (before, after):
        if snap.get("available") and snap.get("foreign_processes", 0) > 0:
            return True
    return False
