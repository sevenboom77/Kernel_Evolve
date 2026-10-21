"""Same six cached-decoder backends with capacity for all twelve static graphs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
from pathlib import Path

import torch._dynamo
import benchmark_cached_decoder as frozen

FROZEN_V1_SHA256 = "681a3302de194c1829904d25c57203ca2f2e4169a79a45f9c77abb998f59dac5"
FROZEN_DECODER_SHA256 = "307044f4c238b095a83eae774e072cca6962c332b54772464497abb8c9b58bbe"


@contextlib.contextmanager
def compiler_scope():
    # Four configurations and three projection implementations produce twelve
    # variants of the same Python frame; Torch 2.5's default limit is eight.
    with torch._dynamo.config.patch(cache_size_limit=32, accumulated_cache_size_limit=256):
        yield


def make_protocol(rounds=30, warmup=20, max_wall_seconds=900):
    result = frozen.make_protocol(rounds, warmup, max_wall_seconds)
    result.update(name="source-bound-cached-decoder-v2", compiler_specialization_limit=32,
                  compiler_accumulated_specialization_limit=256)
    return result


def make_plan(bindings, protocol):
    for path, expected in ((frozen.APP / "benchmark_cached_decoder.py", FROZEN_V1_SHA256),
                           (frozen.APP / "cached_decoder.py", FROZEN_DECODER_SHA256)):
        if frozen.sha256_file(path) != expected:
            raise frozen.BenchmarkError("frozen cached-decoder V1 source changed")
    result = frozen.make_plan(bindings, protocol)
    package = result["source_package"]
    package["application_sources"][str(Path(__file__).resolve().relative_to(frozen.ROOT))] = frozen.sha256_file(Path(__file__))
    result["source_package_sha256"] = hashlib.sha256(frozen.canonical(package)).hexdigest()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--gpu", type=int, choices=range(8), default=1)
    p.add_argument("--run-id", default="agh-hackathon-cached-decoder-v2")
    p.add_argument("--accounting-queue", type=Path)
    p.add_argument("--registry-reference", type=Path)
    p.add_argument("--registry-sha256")
    p.add_argument("--budget-argv-json", type=Path)
    p.add_argument("--budget-argv-sha256")
    p.add_argument("--max-wall-seconds", type=float, default=900)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--warmup", type=int, default=20)
    a = p.parse_args(argv)
    if a.out.exists() or not math.isfinite(a.max_wall_seconds) or not 0 < a.max_wall_seconds <= 3600:
        p.error("fresh output and finite wall interval required")
    if not 4 <= a.rounds <= 1000 or not 0 <= a.warmup <= 1000:
        p.error("rounds/warmup out of range")
    if a.execute and any(getattr(a, key) is None for key in
                         ("accounting_queue", "registry_reference", "registry_sha256", "budget_argv_json", "budget_argv_sha256")):
        p.error("execute requires official queue, registered policy and bound live budget argv")
    report = make_plan(frozen.verify_sources(), make_protocol(a.rounds, a.warmup, a.max_wall_seconds))
    if not a.execute:
        frozen.atomic_json(a.out, report)
        return 0
    report["budget_bindings"] = {"registry": str(a.registry_reference.resolve()), "registry_sha256": a.registry_sha256,
                                 "argv": str(a.budget_argv_json.resolve()), "argv_sha256": a.budget_argv_sha256}
    checker = frozen.BudgetChecker(a.budget_argv_json, a.budget_argv_sha256)
    try:
        frozen.validate_registry(a.registry_reference, a.registry_sha256, a.accounting_queue)
        report["last_budget_snapshot"] = checker.check()
        frozen.atomic_json(a.out, report)
        with compiler_scope():
            frozen.run_gpu(a, frozen.verify_sources(), report, checker)
        return 0
    except BaseException as exc:
        report.update(status="BUDGET_STOPPED" if isinstance(exc, frozen.BudgetStopped) else "FAILED",
                      error_type=type(exc).__name__, error=str(exc)[:500], completed_at=frozen.utcnow())
        frozen.atomic_json(a.out, report)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
