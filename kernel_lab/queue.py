"""Transactional job queue, budgets and per-physical-card GPU locks.

Two independent mechanisms cooperate:

* a SQLite row lease decides *which* job a given evaluator may run and lets a
  crashed evaluator's jobs be recovered;
* an OS ``flock`` on a per-physical-card lock file guarantees that at most one
  measuring process is on a card at any instant, even across evaluator restarts.

A MIG instance belonging to a physical card maps onto the *same* lock file: two
MIG slices are not two independent cards for timing purposes.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .events import EventLog
from .util import utcnow

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs(
    run_id TEXT PRIMARY KEY, created TEXT, suite_id TEXT, protocol_hash TEXT,
    status TEXT DEFAULT 'open', frozen INTEGER DEFAULT 0, notes TEXT);
CREATE TABLE IF NOT EXISTS jobs(
    job_id TEXT PRIMARY KEY, run_id TEXT, task_id TEXT, kind TEXT,
    candidate_id TEXT, source_hash TEXT, snapshot_dir TEXT, static_json TEXT,
    phase TEXT, status TEXT, priority INTEGER DEFAULT 100,
    created TEXT, updated TEXT, lease_owner TEXT, lease_expires REAL,
    gpu_uuid TEXT, attempts INTEGER DEFAULT 0, max_attempts INTEGER DEFAULT 2,
    payload_json TEXT);
CREATE TABLE IF NOT EXISTS results(
    job_id TEXT PRIMARY KEY, run_id TEXT, task_id TEXT, candidate_id TEXT,
    status TEXT, payload_json TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS measurements(
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, job_id TEXT, task_id TEXT,
    candidate_id TEXT, case_id TEXT, label TEXT, phase TEXT, latency_us REAL,
    host_us REAL, payload_json TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS champions(
    run_id TEXT, task_id TEXT, candidate_id TEXT, source_hash TEXT,
    task_speedup REAL, shape_speedups_json TEXT, updated TEXT,
    PRIMARY KEY(run_id, task_id));
CREATE TABLE IF NOT EXISTS candidates(
    run_id TEXT, task_id TEXT, candidate_id TEXT, source_hash TEXT,
    parent_hash TEXT, proposal_json TEXT, status TEXT, created TEXT,
    PRIMARY KEY(run_id, task_id, candidate_id));
CREATE TABLE IF NOT EXISTS baselines(
    run_id TEXT, task_id TEXT, label TEXT, status TEXT, reason TEXT,
    latencies_json TEXT, qualification_json TEXT, updated TEXT,
    PRIMARY KEY(run_id, task_id, label));
CREATE TABLE IF NOT EXISTS budgets(
    run_id TEXT, task_id TEXT, source_candidates INTEGER DEFAULT 0,
    variants INTEGER DEFAULT 0, gpu_seconds REAL DEFAULT 0,
    wall_seconds REAL DEFAULT 0, updated TEXT,
    PRIMARY KEY(run_id, task_id));
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, kind TEXT,
    payload_json TEXT);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(run_id, status, priority, created);
"""

TERMINAL = ("complete", "incorrect", "compile_fail", "invalid_source", "timeout",
            "oom", "env_error", "contaminated", "crashed")


class Queue:
    def __init__(self, db_path: str | Path, events: EventLog | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=60, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.events = events

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()

    # ------------------------------------------------------------------ runs
    def create_run(self, run_id: str, suite_id: str, protocol_hash: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO runs(run_id, created, suite_id, protocol_hash) VALUES(?,?,?,?)",
            (run_id, utcnow(), suite_id, protocol_hash))

    def run_row(self, run_id: str):
        cur = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return cur.fetchone()

    def set_frozen(self, run_id: str, frozen: bool = True) -> None:
        self.conn.execute("UPDATE runs SET frozen=?, status=? WHERE run_id=?",
                          (1 if frozen else 0, "frozen" if frozen else "open", run_id))

    def is_frozen(self, run_id: str) -> bool:
        row = self.run_row(run_id)
        return bool(row and row["frozen"])

    # ------------------------------------------------------------------ jobs
    def enqueue(self, run_id: str, task_id: str, kind: str, *, candidate_id: str | None = None,
                source_hash: str | None = None, snapshot_dir: str | None = None,
                static: dict | None = None, phase: str = "dev", priority: int = 100,
                max_attempts: int = 2, payload: dict | None = None) -> str:
        job_id = uuid.uuid4().hex[:16]
        now = utcnow()
        self.conn.execute(
            "INSERT INTO jobs(job_id, run_id, task_id, kind, candidate_id, source_hash,"
            " snapshot_dir, static_json, phase, status, priority, created, updated,"
            " max_attempts, payload_json) VALUES(?,?,?,?,?,?,?,?,?, 'queued', ?,?,?,?,?)",
            (job_id, run_id, task_id, kind, candidate_id, source_hash, snapshot_dir,
             json.dumps(static or {}), phase, priority, now, now, max_attempts,
             json.dumps(payload or {})))
        self._emit("job_enqueued", run_id=run_id, job_id=job_id, task_id=task_id,
                   kind=kind, candidate_id=candidate_id, phase=phase)
        return job_id

    def claim(self, run_id: str, owner: str, gpu_uuid: str, lease_s: float = 900.0):
        """Atomically take the highest-priority queued job. Returns a Row or None."""
        now = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self.conn.execute(
                "SELECT * FROM jobs WHERE run_id=? AND status='queued'"
                " ORDER BY priority ASC, created ASC LIMIT 1", (run_id,))
            row = cur.fetchone()
            if row is None:
                self.conn.execute("COMMIT")
                return None
            self.conn.execute(
                "UPDATE jobs SET status='running', lease_owner=?, lease_expires=?,"
                " gpu_uuid=?, attempts=attempts+1, updated=? WHERE job_id=? AND status='queued'",
                (owner, now + lease_s, gpu_uuid, utcnow(), row["job_id"]))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        cur = self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],))
        claimed = cur.fetchone()
        self._emit("job_claimed", run_id=run_id, job_id=claimed["job_id"],
                   task_id=claimed["task_id"], kind=claimed["kind"], owner=owner,
                   gpu_uuid=gpu_uuid, attempt=claimed["attempts"])
        return claimed

    def heartbeat(self, job_id: str, owner: str, lease_s: float = 900.0) -> bool:
        cur = self.conn.execute(
            "UPDATE jobs SET lease_expires=?, updated=? WHERE job_id=? AND lease_owner=? AND status='running'",
            (time.time() + lease_s, utcnow(), job_id, owner))
        return cur.rowcount > 0

    def finish(self, job: sqlite3.Row, status: str, result: dict) -> None:
        """Idempotently commit the result and both job/candidate terminal states."""
        payload = json.dumps(result, default=str)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT INTO results(job_id, run_id, task_id, candidate_id, status, payload_json, created)"
                " VALUES(?,?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET"
                " status=excluded.status, payload_json=excluded.payload_json, created=excluded.created",
                (job["job_id"], job["run_id"], job["task_id"], job["candidate_id"],
                 status, payload, utcnow()))
            self.conn.execute("UPDATE jobs SET status=?, updated=? WHERE job_id=?",
                              (status, utcnow(), job["job_id"]))
            if job["kind"] == "candidate" and job["candidate_id"]:
                # Promotion is decided by the evaluator. Failures caught by its
                # serve loop, or controller cancellations, still end a candidate.
                # A verify job must never replace the search candidate's state.
                self.conn.execute(
                    "UPDATE candidates SET status=CASE "
                    "WHEN ?='complete' AND status IN ('promoted','no_gain') THEN status "
                    "ELSE ? END WHERE run_id=? AND task_id=? AND candidate_id=?",
                    (status, status, job["run_id"], job["task_id"], job["candidate_id"]))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self._emit("job_finished", run_id=job["run_id"], job_id=job["job_id"],
                   task_id=job["task_id"], status=status)

    def release(self, job: sqlite3.Row, reason: str) -> None:
        """Return a job to the queue without recording a verdict."""
        row = self.conn.execute("SELECT attempts, max_attempts FROM jobs WHERE job_id=?",
                                (job["job_id"],)).fetchone()
        if row and row["attempts"] >= row["max_attempts"]:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute("UPDATE jobs SET status='lost', updated=? WHERE job_id=?",
                                  (utcnow(), job["job_id"]))
                if job["kind"] == "candidate" and job["candidate_id"]:
                    self.update_candidate_status(job["run_id"], job["task_id"],
                                                 job["candidate_id"], "lost")
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self._emit("job_lost", run_id=job["run_id"], job_id=job["job_id"], reason=reason)
        else:
            self.conn.execute(
                "UPDATE jobs SET status='queued', lease_owner=NULL, lease_expires=NULL, updated=?"
                " WHERE job_id=?", (utcnow(), job["job_id"]))
            self._emit("job_requeued", run_id=job["run_id"], job_id=job["job_id"], reason=reason)

    def recover_expired(self, run_id: str, grace_s: float = 0.0) -> list[str]:
        """Requeue jobs whose lease expired (e.g. after an evaluator crash)."""
        now = time.time() - grace_s
        cur = self.conn.execute(
            "SELECT * FROM jobs WHERE run_id=? AND status='running' AND lease_expires IS NOT NULL"
            " AND lease_expires < ?", (run_id, now))
        moved = []
        for row in cur.fetchall():
            self.release(row, "lease_expired")
            moved.append(row["job_id"])
        return moved

    def job(self, job_id: str):
        cur = self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        return cur.fetchone()

    def result(self, job_id: str):
        cur = self.conn.execute("SELECT * FROM results WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        if row is None:
            return None
        out = dict(row)
        out["payload"] = json.loads(out.pop("payload_json"))
        return out

    def list_jobs(self, run_id: str, task_id: str | None = None, limit: int = 100):
        sql = "SELECT * FROM jobs WHERE run_id=?"
        args: list[Any] = [run_id]
        if task_id:
            sql += " AND task_id=?"
            args.append(task_id)
        sql += " ORDER BY created DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # ------------------------------------------------------------- budgets
    def budget(self, run_id: str, task_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM budgets WHERE run_id=? AND task_id=?",
                                (run_id, task_id)).fetchone()
        if row is None:
            return {"source_candidates": 0, "variants": 0, "gpu_seconds": 0.0, "wall_seconds": 0.0}
        return {k: row[k] for k in ("source_candidates", "variants", "gpu_seconds", "wall_seconds")}

    def spend(self, run_id: str, task_id: str, *, source_candidates: int = 0,
              variants: int = 0, gpu_seconds: float = 0.0, wall_seconds: float = 0.0) -> None:
        self.conn.execute(
            "INSERT INTO budgets(run_id, task_id, source_candidates, variants, gpu_seconds,"
            " wall_seconds, updated) VALUES(?,?,?,?,?,?,?)"
            " ON CONFLICT(run_id, task_id) DO UPDATE SET"
            " source_candidates=source_candidates+excluded.source_candidates,"
            " variants=variants+excluded.variants,"
            " gpu_seconds=gpu_seconds+excluded.gpu_seconds,"
            " wall_seconds=wall_seconds+excluded.wall_seconds, updated=excluded.updated",
            (run_id, task_id, source_candidates, variants, gpu_seconds, wall_seconds, utcnow()))

    def total_source_candidates(self, run_id: str) -> int:
        row = self.conn.execute("SELECT COALESCE(SUM(source_candidates),0) AS n FROM budgets"
                                " WHERE run_id=?", (run_id,)).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------ baselines
    def set_baseline(self, run_id: str, task_id: str, label: str, status: str,
                     reason: str | None = None, latencies: dict | None = None,
                     qualification: dict | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO baselines(run_id, task_id, label, status, reason,"
            " latencies_json, qualification_json, updated) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, task_id, label, status, reason,
             json.dumps(latencies or {}, default=str),
             json.dumps(qualification or {}, default=str), utcnow()))

    def baselines(self, run_id: str, task_id: str) -> dict:
        rows = self.conn.execute(
            "SELECT * FROM baselines WHERE run_id=? AND task_id=?", (run_id, task_id)).fetchall()
        out = {}
        for r in rows:
            out[r["label"]] = {
                "status": r["status"], "reason": r["reason"],
                "latencies": json.loads(r["latencies_json"] or "{}"),
                "qualification": json.loads(r["qualification_json"] or "{}"),
            }
        return out

    # ----------------------------------------------------------- candidates
    def record_candidate(self, run_id: str, task_id: str, candidate_id: str, source_hash: str,
                         parent_hash: str | None, proposal: dict, status: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO candidates(run_id, task_id, candidate_id, source_hash,"
            " parent_hash, proposal_json, status, created) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, task_id, candidate_id, source_hash, parent_hash,
             json.dumps(proposal, default=str), status, utcnow()))

    def has_pending(self, run_id: str) -> bool:
        """Is there still work queued or in flight for this run?"""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE run_id=? AND status IN ('queued','running')",
            (run_id,)).fetchone()
        return bool(row and row[0])
    def update_candidate_status(self, run_id: str, task_id: str, candidate_id: str,
                                status: str) -> None:
        self.conn.execute(
            "UPDATE candidates SET status=? WHERE run_id=? AND task_id=? AND candidate_id=?",
            (status, run_id, task_id, candidate_id))

    def candidates(self, run_id: str, task_id: str):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM candidates WHERE run_id=? AND task_id=? ORDER BY created",
            (run_id, task_id)).fetchall()]

    # ------------------------------------------------------------ champions
    def champion(self, run_id: str, task_id: str):
        row = self.conn.execute("SELECT * FROM champions WHERE run_id=? AND task_id=?",
                                (run_id, task_id)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["shape_speedups"] = json.loads(out.pop("shape_speedups_json"))
        return out

    def set_champion(self, run_id: str, task_id: str, candidate_id: str, source_hash: str,
                     task_speedup: float, shape_speedups: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO champions(run_id, task_id, candidate_id, source_hash,"
            " task_speedup, shape_speedups_json, updated) VALUES(?,?,?,?,?,?,?)",
            (run_id, task_id, candidate_id, source_hash, task_speedup,
             json.dumps(shape_speedups, default=str), utcnow()))
        self._emit("champion_updated", run_id=run_id, task_id=task_id,
                   candidate_id=candidate_id, task_speedup=task_speedup)

    def record_measurement(self, run_id: str, job_id: str, task_id: str, candidate_id: str,
                           case_id: str, label: str, phase: str, latency_us: float,
                           host_us: float | None, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO measurements(run_id, job_id, task_id, candidate_id, case_id, label,"
            " phase, latency_us, host_us, payload_json, created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, job_id, task_id, candidate_id, case_id, label, phase, latency_us,
             host_us, json.dumps(payload, default=str), utcnow()))

    def measurements(self, run_id: str, task_id: str | None = None):
        sql = "SELECT * FROM measurements WHERE run_id=?"
        args = [run_id]
        if task_id:
            sql += " AND task_id=?"
            args.append(task_id)
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def _emit(self, event_kind: str, **fields) -> None:
        rec = {"ts": utcnow(), "kind": event_kind}
        rec.update(fields)
        self.conn.execute("INSERT INTO events(ts, run_id, kind, payload_json) VALUES(?,?,?,?)",
                          (rec["ts"], fields.get("run_id"), event_kind, json.dumps(rec, default=str)))
        if self.events is not None:
            self.events.emit(event_kind, source="queue", **fields)


class GpuLock:
    """Exclusive OS lock covering one *physical* card, shared by every run.

    The lock file lives under ``<project root>/.locks/`` keyed by the physical
    card identity (its UUID), so two runs — or a run and a later calibration —
    serialise on the same lock instead of each guarding its own private
    directory. It only binds processes that take it: foreign jobs on the host
    are outside this contract and are handled by interference detection.
    """

    def __init__(self, root: Path, physical_key: str):
        self.path = Path(root) / ".locks" / f"gpu-{physical_key}.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "w")
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        self.fh.write(f"pid={os.getpid()} ts={utcnow()}\n")
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            with contextlib.suppress(Exception):
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None
        return False
