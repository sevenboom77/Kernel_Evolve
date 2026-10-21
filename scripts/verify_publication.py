"""Verify the published workbench, frozen K12 inputs and vendored AGH source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def check_file(path: Path, expected: str, size: int | None = None) -> None:
    if not path.is_file():
        raise ValueError(f"Missing: {path.relative_to(ROOT)}")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Size mismatch: {path.relative_to(ROOT)}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path.relative_to(ROOT)}")


def main() -> None:
    workbench = json.loads((ROOT / "demo/review-manifest.json").read_text(encoding="utf-8"))
    for record in workbench["files"]:
        check_file(ROOT / "demo" / record["path"], record["sha256"], record["bytes"])
    frozen = json.loads((ROOT / "harness-evolution/competition/application/native-small-m-reproduction-inputs-v1.json").read_text(encoding="utf-8"))
    for record in frozen.values():
        check_file(ROOT / record["relative_path"], record["sha256"])
    vendor = json.loads((ROOT / "third_party/AGH_SNAPSHOT.json").read_text(encoding="utf-8"))
    for record in vendor["files"]:
        check_file(ROOT / "third_party/agnes-harness" / record["path"], record["sha256"], record["bytes"])
    print(f"PASS: {len(workbench['files'])} workbench files, {len(frozen)} frozen K12 inputs, {len(vendor['files'])} AGH source files")
    print("No model calls or GPU measurements were performed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
