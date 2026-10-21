"""Paired merged projections with six mandatory library baselines and raw samples."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

import torch
import benchmark_decoder_v3 as original
from benchmark_decoder_v3 import (
    ApplicationJob, BenchmarkError, BudgetChecker, BudgetStopped, NumericGpuLock,
    atomic_json, canonical, clean_snapshot, cuda_completion, gpuinfo, measure_paired,
    paired_gain, timing, utcnow, validate_registry, wall_limit,
)
from packed_cached_decoder import PackedCachedDecoder, validate_packed_trace
from cached_decoder import (
    APPLICATION_TOLERANCE, CASE_LABELS, FIXED_CONFIGS, OPERATOR_TOLERANCES,
    CachedCandidateOperators, CachedConfig, CachedDecoder, compare_outputs,
    make_case, sha256_file, verify_sources,
)

LIBRARY_BACKENDS = ("fp32_reference", "fp16_library", "compiled_fp32", "compiled_fp16", "packed_fp16_library", "compiled_packed_fp16")
REQUESTED_BACKENDS = (*LIBRARY_BACKENDS, "candidate", "compiled_candidate")
ORIGINAL_V3_SHA256 = "d5d38b76c3ab25d4257f1042e19397bda373dbab90d76830e237f9eaf488c685"


def make_protocol(rounds=30, warmup=20, max_wall_seconds=900):
    return {"schema_version": 1, "name": "source-bound-packed-cached-decoder-v1", "seed": 20261005,
            "cases": [asdict(c) for c in FIXED_CONFIGS], "case_labels": list(CASE_LABELS),
            "operator_tolerances": {k: list(v) for k, v in OPERATOR_TOLERANCES.items()},
            "full_block_output_tolerance": APPLICATION_TOLERANCE,
            "requested_backends": list(REQUESTED_BACKENDS),
            "strong_baseline_required": list(LIBRARY_BACKENDS),
            "strong_baseline_policy": "minimum of all six qualified library latencies including identically packed projections within each paired round; missing baseline forbids victory",
            "compile_mode": "default", "compile_fullgraph": True, "compile_dynamic": False,
            "compilation_failure_policy": "record unavailable/incorrect; no eager fallback",
            "fp16_reduced_precision_reduction": False, "tf32": False,
            "rounding": "FP16 weights and seven FP32-accumulating projections with FP16 outputs; attention output FP16 before output projection; residuals FP16",
            "timing": {"backend": "cuda_events", "paired_rounds": rounds, "warmup": warmup,
                       "order_seed": 20261005, "target_batch_ms": 20.0, "max_iters": 256,
                       "jit_compile_upload_validation_timed": False, "cuda_graphs": False,
                       "weight_input_policy": "identical fixed input, FP16 weight bytes and immutable KV prefix"},
            "max_wall_seconds": max_wall_seconds,
            "compiler_specialization_limit": 32,
            "accumulated_compiler_specialization_limit": 256,
            "logical_projection_count": 7, "candidate_physical_projection_count": 4,
            "packing": "original FP16 Q/K/V and gate/up weights concatenated by output column outside timing, inside official accounting span; slices explicitly contiguous",
            "failure_examples": ["context32 rejected", "odd head dimension rejected", "source mismatch rejected",
                                 "K12 input bound exceeded rejected", "nonfinite input rejected", "contaminated GPU refused"]}


def source_package(bindings, protocol):
    if sha256_file(APP / 'benchmark_cached_decoder.py') != '681a3302de194c1829904d25c57203ca2f2e4169a79a45f9c77abb998f59dac5':
        raise BenchmarkError('original cached benchmark source changed')
    original.verify_base_sources()
    if sha256_file(APP / "benchmark_decoder_v3.py") != ORIGINAL_V3_SHA256:
        raise BenchmarkError("immutable V3 accounting/timing source changed")
    sources = {str(path.relative_to(ROOT)): sha256_file(path) for path in
               (*original.BASE_SOURCE_HASHES, APP / "benchmark_decoder_v3.py",
                APP / "cached_decoder.py", APP / "benchmark_cached_decoder.py",
                APP / "packed_cached_decoder.py", Path(__file__))}
    value = {"application_sources": sources,
             "operator_manifest_sha256s": [r["sha256"] for r in bindings.manifests],
             "operator_sources": {key: row["sha256"] for key, row in bindings.operators.items()},
             "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest()}
    return value, hashlib.sha256(canonical(value)).hexdigest()


def make_plan(bindings, protocol):
    package, sha = source_package(bindings, protocol)
    return {"schema_version": 1, "kind": "source-bound-packed-cached-decoder-application", "status": "GPU_PENDING",
            "created_at": utcnow(), "pretrained_model": False, "text_quality_measured": False,
            "architecture": "single query token + seven logical/four physical candidate projections + adjacent RoPE + immutable KV prefix + causal attention + gated MLP + residuals",
            "operator_sources": bindings.receipt(), "source_package": package, "source_package_sha256": sha,
            "protocol": protocol, "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest(),
            "measurements": None, "harness_performance_causality_proven": False,
            "limits": ["Fixed random weights; no pretrained text quality or model claim.",
                       "Application evidence cannot promote the primary AGH series.",
                       "Repeated fixed weights may fit cache; all paired raw samples retained.",
                       "Any missing predeclared library baseline forbids a strong-baseline victory."]}


def strong_gain(samples, candidate):
    if any(name not in samples for name in (*LIBRARY_BACKENDS, candidate)):
        raise BenchmarkError("all predeclared library baselines and candidate required")
    vectors = [samples[name]["gpu_us_samples"] for name in (*LIBRARY_BACKENDS, candidate)]
    count = len(vectors[0])
    if count < 4 or any(len(v) != count for v in vectors):
        raise BenchmarkError("paired baseline samples incomplete")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for values in vectors for x in values):
        raise BenchmarkError("paired samples must be finite positive numbers")
    minima = [min(values) for values in zip(*vectors[:-1])]
    ratios = [a / b for a, b in zip(minima, vectors[-1])]
    rng = random.Random(20261005)
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=count)) for _ in range(2000))
    return {"metric": "median of within-round strongest-library/candidate latency ratios",
            "required_library_backends": list(LIBRARY_BACKENDS), "paired_rounds": count,
            "strong_baseline_us_samples": minima, "paired_ratio_samples": ratios,
            "median": statistics.median(ratios), "bootstrap_95_percent_interval": [bootstrap[49], bootstrap[1949]],
            "bootstrap_seed": 20261005, "bootstrap_resamples": 2000,
            "victory_proven": bootstrap[49] > 1.0}


def close_tuple(actual, reference):
    if not isinstance(actual, tuple) or len(actual) != 3:
        raise AssertionError("decoder must return output and two new caches")
    for left, right in zip(actual, reference):
        torch.testing.assert_close(left, right, **APPLICATION_TOLERANCE, equal_nan=False)
        if not torch.isfinite(left).all().item():
            raise AssertionError("nonfinite decode output/cache")


def compile_block(block, x, reference, *, sync):
    start = time.monotonic()
    try:
        fn = torch.compile(block, mode="default", fullgraph=True, dynamic=False)
        actual = fn(x)
        sync()
        close_tuple(actual, reference)
        return lambda: fn(x), {"status": "AVAILABLE", "mode": "default", "fullgraph": True,
                              "dynamic": False, "compile_warmup_seconds": time.monotonic() - start}
    except BudgetStopped:
        raise
    except Exception as exc:
        return None, {"status": "INCORRECT" if isinstance(exc, AssertionError) else "UNAVAILABLE",
                      "error_type": type(exc).__name__, "compiler_diagnostic": str(exc)[:4000],
                      "mode": "default", "fullgraph": True, "dynamic": False,
                      "compile_warmup_seconds": time.monotonic() - start}


def run_gpu(args, bindings, report, checker):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with wall_limit(args.max_wall_seconds), NumericGpuLock(ROOT, args.gpu):
        preflight = clean_snapshot(args.gpu)
        checker.check()
        with ApplicationJob(args.accounting_queue, args.run_id, report["protocol_sha256"], report["source_package_sha256"],
                            args.gpu, args.max_wall_seconds) as job:
            report.update(status="RUNNING", measurements={},
                          accounting={"queue": str(Path(args.accounting_queue).resolve()), "run_id": args.run_id,
                                      "job_id": job.job["job_id"], "gpu_physical_index": args.gpu,
                                      "gpu_span_semantics": "official claim-to-finish conservative wall time"},
                          gpu_snapshots=[preflight])
            job.result["report_path"] = str(args.out.resolve())
            with cuda_completion(), torch.inference_mode():
                if not torch.cuda.is_available():
                    raise BenchmarkError("CUDA unavailable after accounting admission")
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                torch.set_float32_matmul_precision("highest")
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
                props = torch.cuda.get_device_properties(device)
                report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
                                         "device_name": props.name, "device_total_memory": props.total_memory,
                                         "tf32": False, "fp16_reduced_precision_reduction": False,
                                         "pid": os.getpid(), "physical_gpu": args.gpu}
                operators = CachedCandidateOperators(bindings)
                def checkpoint():
                    report["last_budget_snapshot"] = checker.check()
                    report["gpu_snapshots"].append(clean_snapshot(args.gpu))
                    if not job.queue.heartbeat(job.job["job_id"], job.job["lease_owner"], args.max_wall_seconds + 120):
                        raise BenchmarkError("application accounting lease lost")
                for index, (label, config) in enumerate(zip(CASE_LABELS, report["protocol"]["cases"])):
                    checkpoint()
                    case = make_case(CachedConfig(**config), report["protocol"]["seed"])
                    row = {"config": config, "input_weights_cache_sha256": case.fingerprint(), "status": "RUNNING",
                           "backend_qualification": {}, "samples": {}}
                    report["measurements"][label] = row
                    x = case.x.to(device)
                    original_x = x.clone()
                    blocks = {"fp32_reference": CachedDecoder(case),
                              "fp16_library": CachedDecoder(case, projection="fp16_library"),
                              "packed_fp16_library": PackedCachedDecoder(case, projection="fp16_library"),
                              "candidate": PackedCachedDecoder(case, operators, projection="candidate")}
                    blocks = {name: value.to(device).eval() for name, value in blocks.items()}
                    before = {name: {key: value.clone() for key, value in block.named_buffers()}
                              for name, block in blocks.items()}
                    reference, reference_trace = blocks["fp32_reference"].forward_with_trace(x)
                    row["reference_correctness"] = compare_outputs(reference, reference, reference_trace)
                    fns = {"fp32_reference": lambda: blocks["fp32_reference"](x)}
                    row["backend_qualification"]["fp32_reference"] = {"status": "AVAILABLE"}
                    for name in ("fp16_library", "packed_fp16_library", "candidate"):
                        try:
                            actual, trace = blocks[name].forward_with_trace(x)
                            if isinstance(blocks[name], PackedCachedDecoder):
                                validate_packed_trace(blocks[name], trace)
                            row[f"{name}_correctness"] = compare_outputs(reference, actual, trace)
                            fns[name] = lambda name=name: blocks[name](x)
                            row["backend_qualification"][name] = {"status": "AVAILABLE"}
                        except BudgetStopped:
                            raise
                        except Exception as exc:
                            row["backend_qualification"][name] = {"status": "INCORRECT" if isinstance(exc, AssertionError) else "UNAVAILABLE",
                                                                   "error_type": type(exc).__name__, "diagnostic": str(exc)[:4000]}
                    for name, source in (("compiled_fp32", "fp32_reference"), ("compiled_fp16", "fp16_library"),
                                         ("compiled_packed_fp16", "packed_fp16_library"), ("compiled_candidate", "candidate")):
                        checkpoint()
                        if source not in fns:
                            row["backend_qualification"][name] = {"status": "UNAVAILABLE", "reason": "source backend failed correctness"}
                            continue
                        fn, qualification = compile_block(blocks[source], x, reference, sync=torch.cuda.synchronize)
                        row["backend_qualification"][name] = qualification
                        if fn is not None:
                            fns[name] = fn
                    atomic_json(args.out, report)
                    timing.configure(True)
                    measure_paired(fns, report["protocol"]["timing"]["warmup"], report["protocol"]["timing"]["paired_rounds"],
                                   report["protocol"]["timing"]["order_seed"] + index, checkpoint, partial=row["samples"])
                    checkpoint()
                    if not torch.equal(x, original_x):
                        raise AssertionError("timed input mutated")
                    for name, block in blocks.items():
                        for key, value in block.named_buffers():
                            if not torch.equal(value, before[name][key]):
                                raise AssertionError("timed weights or KV prefix mutated")
                    row["candidate_gains"] = {}
                    for name in ("candidate", "compiled_candidate"):
                        if name not in fns:
                            row["candidate_gains"][name] = None
                            continue
                        gains = {"against_fp32_reference": paired_gain(row["samples"], "fp32_reference", name),
                                 "against_fp16_library": paired_gain(row["samples"], "fp16_library", name) if "fp16_library" in fns else None,
                                 "against_strong_library": strong_gain(row["samples"], name) if all(k in fns for k in LIBRARY_BACKENDS) else None}
                        row["candidate_gains"][name] = gains
                    row.update(status="MEASURED_CLEAN", strong_baseline_complete=all(k in fns for k in LIBRARY_BACKENDS))
                    atomic_json(args.out, report)
                checkpoint()
                report.update(status="MEASURED_CLEAN", completed_at=utcnow(),
                              all_strong_baselines_complete=all(row["strong_baseline_complete"] for row in report["measurements"].values()))
                atomic_json(args.out, report)
                job.result.update(report_sha256=sha256_file(args.out), measured_cases=len(report["measurements"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu", type=int, choices=range(8), default=1)
    parser.add_argument("--run-id", default="agh-hackathon-packed-cached-decoder-v1")
    parser.add_argument("--accounting-queue", type=Path)
    parser.add_argument("--registry-reference", type=Path)
    parser.add_argument("--registry-sha256")
    parser.add_argument("--budget-argv-json", type=Path)
    parser.add_argument("--budget-argv-sha256")
    parser.add_argument("--max-wall-seconds", type=float, default=900)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output must be fresh; retain previous samples and failures")
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 3600:
        parser.error("finite wall interval required")
    if not 4 <= args.rounds <= 1000 or not 0 <= args.warmup <= 1000:
        parser.error("rounds/warmup out of range")
    if args.execute and any(getattr(args, key) is None for key in
                            ("accounting_queue", "registry_reference", "registry_sha256", "budget_argv_json", "budget_argv_sha256")):
        parser.error("execute requires official queue, registered policy and bound live budget argv")
    bindings = verify_sources()
    report = make_plan(bindings, make_protocol(args.rounds, args.warmup, args.max_wall_seconds))
    if not args.execute:
        atomic_json(args.out, report)
        return 0
    report["budget_bindings"] = {"registry": str(args.registry_reference.resolve()), "registry_sha256": args.registry_sha256,
                                 "argv": str(args.budget_argv_json.resolve()), "argv_sha256": args.budget_argv_sha256}
    checker = BudgetChecker(args.budget_argv_json, args.budget_argv_sha256)
    try:
        validate_registry(args.registry_reference, args.registry_sha256, args.accounting_queue)
        report["last_budget_snapshot"] = checker.check()
        atomic_json(args.out, report)
        with torch._dynamo.config.patch(cache_size_limit=32, accumulated_cache_size_limit=256):
            run_gpu(args, bindings, report, checker)
        return 0
    except BaseException as exc:
        report.update(status="BUDGET_STOPPED" if isinstance(exc, BudgetStopped) else "FAILED",
                      error_type=type(exc).__name__, error=str(exc)[:500], completed_at=utcnow())
        atomic_json(args.out, report)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
