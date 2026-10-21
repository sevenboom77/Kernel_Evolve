"""DSH integration.

The adapter never replaces DSH: it prepares the run directory, renders the
prompts, invokes the DSH command line the installation actually supports, keeps
the raw event stream, and translates project state into the short summaries an
agent works from.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .queue import Queue
from .tasks import TaskPackage
from .util import read_json, utcnow, write_json

HEADLESS_CLI = Path.home() / ".dsh" / "profiles" / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js"


def _node_supports_import_meta_main(node: str) -> bool:
    """``bin.js`` self-executes via ``import.meta.main``, which only exists on Node >= 24.

    On an older Node the CLI loads, prints nothing and exits 0, so this check is
    the difference between "the harness ran" and "nothing happened".
    """
    try:
        out = subprocess.run([node, "--input-type=module", "-e",
                              "console.log(import.meta.main === true)"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() == "true"
    except Exception:
        return False


def find_node() -> str | None:
    candidates: list[str] = []
    exe = shutil.which("node")
    if exe:
        candidates.append(exe)
    nvm = Path.home() / ".nvm" / "versions" / "node"
    if nvm.is_dir():
        for v in sorted(nvm.glob("v*"), key=lambda p: [int(x) for x in p.name.lstrip("v").split(".")],
                        reverse=True):
            candidates.append(str(v / "bin" / "node"))
    for c in candidates:
        if _node_supports_import_meta_main(c):
            return c
    return None


def headless_usage(node: str) -> str:
    try:
        out = subprocess.run([node, str(HEADLESS_CLI), "--profile", "headless", "--help"],
                             capture_output=True, text=True, timeout=60)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        return f"<help failed: {exc}>"


def discover() -> dict:
    """Locate a usable DSH entry point and record its real capabilities."""
    info: dict = {"found": False, "invocations": [], "flags": {}}
    exe = shutil.which("dsh")
    if exe:
        info["invocations"].append([exe])
    if HEADLESS_CLI.is_file():
        info["found"] = True
        pkg = HEADLESS_CLI.parent.parent / "package.json"
        if pkg.is_file():
            try:
                info["version"] = json.loads(pkg.read_text()).get("version")
            except Exception:
                pass
        node = find_node()
        info["node"] = node
        info["system_node"] = shutil.which("node")
        info["node_ok"] = node is not None
        info["invocations"].append(([node] if node else ["node"]) + [str(HEADLESS_CLI),
                                                                     "--profile", "headless"])
        if node:
            usage = headless_usage(node)
            info["headless_help"] = usage.strip()
            # The shipped headless bundle is a one-shot runner; record what it
            # actually accepts instead of assuming the documented flag set.
            info["flags"] = {
                "--json": "usage" in usage and "--json" in usage,
                "--session-id": "--session-id" in usage,
                "task_as_argv": "task text" in usage or "[task...]" in usage,
            }
    prof = Path.home() / ".dsh" / "profiles"
    if prof.is_dir():
        info["profiles"] = sorted(d.name for d in prof.iterdir()
                                  if d.is_dir() and d.name != "node_modules")
    return info


def render_prompt(template_path: Path, context: dict) -> str:
    text = Path(template_path).read_text()
    for key, value in context.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=1, ensure_ascii=False, default=str)
        text = text.replace("{{" + key + "}}", str(value))
    return text


def task_digest(root: Path, run_id: str, task_id: str, champion_info: dict | None = None) -> dict:
    """Compact task state for a worker: no raw measurement arrays."""
    tp = TaskPackage(root, task_id)
    q = Queue(Path(root) / "artifacts" / run_id / "queue.sqlite")
    budget = q.budget(run_id, task_id)
    bl = q.baselines(run_id, task_id)
    return {
        "task_id": task_id,
        "name": tp.name,
        "task_package": str(tp.dir),
        "spec": str(tp.spec_path),
        "task_md": str(tp.dir / "task.md"),
        "output_arity": tp.output_arity,
        "static_params": tp.static_params,
        "domain": tp.domain,
        "dev_shapes": tp.dev_shapes,
        "tolerance": {"rtol": tp.tolerance[0], "atol": tp.tolerance[1]},
        "budget": {**budget, "max_candidates": tp.spec["max_candidate_attempts"]},
        "qualified_baselines": [k for k, v in bl.items()
                                if k != "__strongest__" and v["status"] == "ok"],
        "baseline_status": {k: v["status"] for k, v in bl.items() if k != "__strongest__"},
        "strongest_baseline_per_shape": bl.get("__strongest__", {}).get("latencies", {}),
        "champion": champion_info or q.champion(run_id, task_id),
    }


def launch(run_id: str, root: Path, prompt_path: Path, profile: str = "headless",
           extra_args: list[str] | None = None, timeout: float | None = None) -> dict:
    """Run one DSH headless session and keep everything it emitted.

    This installation ships a one-shot runner: the task is passed as arguments,
    reasoning streams to stderr and the final message to stdout. There is no
    ``--json`` event stream, so both streams are archived verbatim and the mode
    actually used is recorded rather than assumed.
    """
    root = Path(root)
    disc = discover()
    if not disc.get("node"):
        return {"ok": False, "error": "no Node >= 24 found; the DSH CLI cannot execute "
                                      "(bin.js self-executes through import.meta.main)",
                "discovery": disc}
    prompt = Path(prompt_path).read_text().strip()
    argv = [disc["node"], str(HEADLESS_CLI), "--profile", profile]
    use_stdin = False
    if disc.get("flags", {}).get("--json"):
        argv.append("--json")
    if disc.get("flags", {}).get("task_as_argv", True) and len(prompt) < 100_000:
        argv.append(prompt)
    else:
        use_stdin = True
    if extra_args:
        argv += extra_args
    out_dir = root / "artifacts" / run_id / "dsh"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().replace(":", "").replace("-", "")[:15]
    stdout_p = out_dir / f"final_{stamp}.txt"
    stderr_p = out_dir / f"reasoning_{stamp}.log"
    meta = {"argv": argv[:4] + [f"<{len(prompt)} char prompt>"], "mode": "argv",
            "prompt": str(prompt_path), "started": utcnow(),
            "stdout": str(stdout_p), "stderr": str(stderr_p),
            "profile": profile, "node": disc["node"], "dsh_version": disc.get("version"),
            "json_events": bool(disc.get("flags", {}).get("--json"))}
    rc = None
    with open(stdout_p, "wb") as fout, open(stderr_p, "wb") as ferr:
        if use_stdin:
            meta["mode"] = "stdin"
            with open(prompt_path, "rb") as fin:
                proc = subprocess.run(argv, stdin=fin, stdout=fout, stderr=ferr,
                                      cwd=str(root), timeout=timeout)
        else:
            proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=fout, stderr=ferr,
                                  cwd=str(root), timeout=timeout)
        rc = proc.returncode
    meta["returncode"] = rc
    meta["finished"] = utcnow()
    meta["final_message"] = stdout_p.read_text().strip()[:2000]
    write_json(out_dir / f"session_{stamp}.json", meta)
    return {"ok": rc == 0 and bool(meta["final_message"]), **meta}
