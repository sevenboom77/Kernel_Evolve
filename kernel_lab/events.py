from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .util import append_jsonl, utcnow

_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}


class EventLog:
    """Append-only structured event stream; JSONL on disk, human text on stderr."""

    def __init__(self, path: Path | None, run_id: str | None = None, source: str = "evaluator", level: str = "info"):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.source = source
        self.level = level

    def emit(self, event_kind: str, level: str = "info", **fields: Any) -> dict:
        """Append one event. ``event_kind`` is positional to leave ``kind`` free."""
        rec = {"ts": utcnow(), "run_id": self.run_id, "source": self.source,
               "kind": event_kind, "level": level}
        rec.update(fields)
        if self.path is not None:
            append_jsonl(self.path, rec)
        if _LEVELS.get(level, 20) >= _LEVELS.get(self.level, 20):
            extra = " ".join(f"{k}={v}" for k, v in fields.items() if k != "message")
            print(f"[{rec['ts']}] {self.source}/{event_kind}: {fields.get('message','')} {extra}".rstrip(),
                  file=sys.stderr, flush=True)
        return rec
