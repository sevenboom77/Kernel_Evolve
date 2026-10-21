from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root / "artifacts" / self.run_id

    @property
    def db(self) -> Path:
        return self.run_dir / "queue.sqlite"

    @property
    def events(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def protocol(self) -> Path:
        return self.run_dir / "protocol.json"

    @property
    def environment(self) -> Path:
        return self.run_dir / "environment.json"

    @property
    def results(self) -> Path:
        return self.run_dir / "results"

    @property
    def snapshots(self) -> Path:
        return self.run_dir / "snapshots"

    @property
    def frozen(self) -> Path:
        return self.run_dir / "frozen"

    @property
    def dsh(self) -> Path:
        return self.run_dir / "dsh"

    @property
    def work(self) -> Path:
        return self.run_dir / "work"

    def ensure(self) -> "RunPaths":
        for p in (self.run_dir, self.results, self.snapshots, self.frozen, self.dsh, self.work):
            p.mkdir(parents=True, exist_ok=True)
        return self


def run_paths(run_id: str, root: Path | None = None) -> RunPaths:
    return RunPaths(root=(root or PROJECT_ROOT), run_id=run_id)
