"""Adversarial source checks.

These cannot stop a determined attacker inside a trusted development workflow,
but they do catch the failure modes the KernelBench evaluation notes warn about:
reading the oracle's answers, tampering with the timer or with torch, caching a
computed output keyed on one input buffer, and pushing work onto an untimed
stream.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

BANNED_MODULES = {
    "reference",
    "baselines",
    "kernel_lab.correctness",
    "kernel_lab.timing",
    "kernel_lab.evaluator",
    "kernel_lab.queue",
}

BANNED_SNIPPETS = [
    ("torch.cuda.synchronize", "candidate must not synchronise; the harness owns the timer"),
    ("events.record", "candidate must not create its own CUDA events"),
    ("torch.cuda.Event", "candidate must not create its own CUDA events"),
    ("torch.cuda.Stream", "candidate must not move work onto its own stream"),
    ("graph_pool_handle", "CUDA graph capture is out of scope for the main track"),
    ("manual_seed", "candidate must not reseed the global generator"),
    ("set_device", "device selection belongs to the evaluator"),
    ("cuda.set_per_process_memory_fraction", "candidate must not restrict device memory"),
    ("os.environ", "candidate must not touch the process environment"),
    ("subprocess", "candidate must not spawn processes"),
    ("globals()[", "dynamic global access is not allowed"),
    ("exec(", "dynamic execution is not allowed"),
]

BANNED_PATTERNS = [
    (r"(?<![\w.])open\s*\(", "candidate must not read files; inputs arrive as tensors"),
    (r"(?<![\w.])eval\s*\(", "dynamic execution is not allowed"),
    (r"\bpathlib\b", "candidate must not read files; inputs arrive as tensors"),
]

REQUIRED_ENTRY = "ModelNew"


def check_source(path: str | Path) -> dict:
    """Static review of one candidate snapshot. Returns findings, never raises."""
    path = Path(path)
    findings: list[dict] = []
    try:
        src = path.read_text()
    except OSError as exc:
        return {"ok": False, "findings": [{"level": "error", "message": f"unreadable: {exc}"}]}

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"ok": False, "findings": [{"level": "error", "message": f"syntax error: {exc}"}]}

    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    if REQUIRED_ENTRY not in classes:
        findings.append({"level": "error", "message": f"kernel.py must define class {REQUIRED_ENTRY}"})

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in BANNED_MODULES:
                    findings.append({"level": "error",
                                     "message": f"import of forbidden module {alias.name}"})
        elif isinstance(node, ast.ImportFrom):
            if node.module in BANNED_MODULES:
                findings.append({"level": "error",
                                 "message": f"import from forbidden module {node.module}"})
        elif isinstance(node, ast.Attribute):
            if node.attr in ("_C", "_cuda", "torch"):
                pass

    for snippet, why in BANNED_SNIPPETS:
        if snippet in src:
            findings.append({"level": "error", "message": f"{why} (found {snippet!r})"})

    # Word-boundary patterns: `model.eval()` and `torch.eval` must not be flagged.
    for pattern, why in BANNED_PATTERNS:
        if re.search(pattern, src):
            findings.append({"level": "error", "message": why})

    return {"ok": not any(f["level"] == "error" for f in findings),
            "findings": findings,
            "bytes": len(src),
            "lines": src.count("\n") + 1}


def check_snapshot(candidate_dir: str | Path) -> dict:
    candidate_dir = Path(candidate_dir)
    kernel = candidate_dir / "kernel.py"
    if not kernel.is_file():
        return {"ok": False, "findings": [{"level": "error", "message": "kernel.py is missing"}]}
    result = check_source(kernel)
    proposal = candidate_dir / "proposal.json"
    if not proposal.is_file():
        result["findings"].append(
            {"level": "warn", "message": "proposal.json is missing; the change is not explained"})
    return result
