"""Environment detection (`doctor`).

Records what actually exists rather than what a note claims: driver, card
identity, utilisation at probe time, the interpreter and its torch/triton
versions, whether Triton can really launch on this card, and the DSH install.
A probe that runs while other users are active records the interference instead
of pretending the machine is free.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .util import utcnow, write_json

SMOKE_SRC = r'''
import json, sys, time
import torch, triton, triton.language as tl

@triton.jit
def _dbl(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    tl.store(y_ptr + o, tl.load(x_ptr + o, mask=m) * 2.0, mask=m)

def main():
    out = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "triton": triton.__version__, "device_count": torch.cuda.device_count(),
           "devices": [], "smoke_ok": False}
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out["devices"].append({"index": i, "name": p.name,
                               "compute_capability": f"{p.major}.{p.minor}",
                               "total_memory": p.total_memory,
                               "multi_processor_count": p.multi_processor_count})
    dev = "cuda:0"
    x = torch.arange(4096, dtype=torch.float32, device=dev)
    y = torch.empty_like(x)
    _dbl[(triton.cdiv(4096, 1024),)](x, y, 4096, BLOCK=1024)
    torch.cuda.synchronize()
    out["smoke_ok"] = bool(torch.equal(y, x * 2))
    out["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    out["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    free, total = torch.cuda.mem_get_info()
    out["free_memory_at_probe"] = free
    out["util_at_probe"] = None
    print(json.dumps(out))

main()
'''


def _nvidia_smi(query: str) -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return []
    keys = [k.strip() for k in query.split(",")]
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(keys):
            continue
        rows.append(dict(zip(keys, parts)))
    return rows


def _dsh_info() -> dict:
    info: dict = {"found": False}
    for cand in ("dsh",):
        p = shutil.which(cand)
        if p:
            info["path"] = p
            info["found"] = True
    if not info["found"]:
        cli = Path.home() / ".dsh" / "profiles" / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js"
        if cli.is_file():
            info.update({"found": True, "path": str(cli), "launch": f"node {cli}"})
            pkg = cli.parent.parent / "package.json"
            if pkg.is_file():
                try:
                    info["version"] = json.loads(pkg.read_text())["version"]
                except Exception:
                    pass
    prof = Path.home() / ".dsh" / "profiles"
    if prof.is_dir():
        info["profiles"] = sorted([d.name for d in prof.iterdir()
                                   if d.is_dir() and d.name != "node_modules"])
        for name in info["profiles"]:
            pj = prof / name / "package.json"
            if pj.is_file():
                try:
                    info.setdefault("profile_bundles", {})[name] = json.loads(pj.read_text()) \
                        .get("dsh", {}).get("profile", {}).get("bundles", [])
                except Exception:
                    pass
    nd = Path.home() / ".dsh" / "profiles" / "node_modules" / "@deepseek-ai"
    if nd.is_dir():
        info["subagent_packages"] = sorted([d.name for d in nd.iterdir()
                                            if "subagent" in d.name])
    return info


def probe(python_exe: str | None = None, smoke: bool = True, timeout: float = 300.0) -> dict:
    python_exe = python_exe or sys.executable
    gpus = _nvidia_smi("index,uuid,name,memory.total,memory.used,utilization.gpu,compute_cap,"
                       "driver_version,temperature.gpu,power.draw,power.limit,mig.mode.current")
    busy = [g for g in gpus if float(g.get("utilization.gpu", 0) or 0) > 5.0]
    env = {
        "probed_at": utcnow(),
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()}",
        "kernel": platform.version(),
        "python": sys.version,
        "python_exe": python_exe,
        "cpus": os.cpu_count(),
        "project_root": str(Path(__file__).resolve().parent.parent),
        "nvidia_smi_gpus": gpus,
        "gpus_total": len(gpus),
        "gpus_busy_at_probe": len(busy),
        "busy_uuids": [g.get("uuid") for g in busy],
        "shared_machine": len(busy) > 0,
        "dsh": _dsh_info(),
        "cuda_smoke": {"ran": False},
    }
    if gpus:
        env["driver_version"] = gpus[0].get("driver_version")
        env["cuda_version_reported"] = _cuda_version()
    if smoke:
        src = Path(env["project_root"]) / "artifacts" / "doctor_smoke.py"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(SMOKE_SRC)
        cmd = [python_exe, str(src)]
        v = Path(env["project_root"]) / ".venv" / "bin" / "python"
        if v.is_file():
            cmd = [str(v), str(src)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            line = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
            env["cuda_smoke"] = {"ran": True, "cmd": cmd,
                                 "ok": bool(line) and json.loads(line[-1]).get("smoke_ok"),
                                 "rc": proc.returncode,
                                 "stderr_tail": proc.stderr[-1200:]}
            if line:
                env["cuda_smoke"]["detail"] = json.loads(line[-1])
        except Exception as exc:
            env["cuda_smoke"] = {"ran": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return env


def _cuda_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def target_hardware(env: dict) -> dict:
    """Summarise the GPU that measurements on this machine actually come from.

    Recorded into every protocol and report so a number can never be read
    without knowing which card produced it.
    """
    gpus = env.get("nvidia_smi_gpus") or []
    det = (env.get("cuda_smoke") or {}).get("detail") or {}
    devices = det.get("devices") or []
    out = {
        "gpu_count": env.get("gpus_total") or len(gpus),
        "gpu_name": None,
        "compute_capability": None,
        "multiprocessor_count": None,
        "total_memory_bytes": None,
        "driver_version": env.get("driver_version"),
        "mig_mode": None,
        "uuids": [g.get("uuid") for g in gpus if g.get("uuid")],
        "shared_at_probe": bool(env.get("shared_machine")),
        "probed_at": env.get("probed_at"),
    }
    if gpus:
        out["gpu_name"] = gpus[0].get("name")
        out["compute_capability"] = gpus[0].get("compute_cap")
        out["mig_mode"] = gpus[0].get("mig.mode.current")
        try:
            out["total_memory_bytes"] = int(float(gpus[0].get("memory.total")) * 2 ** 20)
        except (TypeError, ValueError):
            pass
    if devices:
        out["multiprocessor_count"] = devices[0].get("multi_processor_count")
        out["compute_capability"] = devices[0].get("compute_capability") or out["compute_capability"]
        out["total_memory_bytes"] = devices[0].get("total_memory") or out["total_memory_bytes"]
    return out


def report_lines(env: dict) -> list[str]:
    lines = [
        f"host                : {env['hostname']} ({env['platform']})",
        f"python              : {env.get('python_exe')}",
        f"CPUs                : {env.get('cpus')}",
        f"GPUs                : {env.get('gpus_total')} (busy at probe: {env.get('gpus_busy_at_probe')})",
        f"driver              : {env.get('driver_version')}",
        f"shared machine      : {env.get('shared_machine')}",
        f"triton smoke        : {env['cuda_smoke'].get('ok')}",
        f"DSH                 : {env.get('dsh', {}).get('version') or env.get('dsh', {}).get('path')}",
        f"DSH profiles        : {env.get('dsh', {}).get('profiles')}",
        f"DSH subagent pkgs   : {env.get('dsh', {}).get('subagent_packages')}",
    ]
    det = env.get("cuda_smoke", {}).get("detail") or {}
    if det:
        lines.append(f"torch/triton        : {det.get('torch')} / {det.get('triton')} (cuda {det.get('torch_cuda')})")
        for d in det.get("devices", []):
            lines.append(f"  gpu {d['index']}: {d['name']} cc{d['compute_capability']} "
                         f"{d['total_memory']/2**30:.0f}GiB {d['multi_processor_count']} SM")
    return lines
