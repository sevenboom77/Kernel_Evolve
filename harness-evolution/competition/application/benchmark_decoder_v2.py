"""Whole compiled candidate decoder, paired with three unchanged V1 backends."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

import torch
from kernel_lab import gpuinfo, timing
from kernel_lab.queue import GpuLock, Queue, TERMINAL
from kernel_lab.util import utcnow
from decoder_block import (
    APPLICATION_TOLERANCE, DEFAULT_MANIFEST, DEFAULT_MANIFEST_SHA256,
    OPERATOR_HASHES, OPERATOR_TOLERANCES, CandidateOperators, ContractError,
    DecoderBlock, DecoderConfig, make_case, sha256_file, validate_operator_trace,
    verify_sources,
)


class BenchmarkError(RuntimeError):
    pass


class BudgetStopped(BenchmarkError):
    pass


BASE_SOURCE_HASHES = {
    APP / "decoder_block.py": "12e7ddd84eefe91f072248b672d2fb2f9ed89c8382a4df872cfab618c9730d61",
    APP / "benchmark_decoder.py": "ed05c682490411ae12d20efc4e132951f7815ca889412f794fe611d2f79370b9",
}


def verify_base_sources():
    for path, expected in BASE_SOURCE_HASHES.items():
        if sha256_file(path) != expected:
            raise BenchmarkError("frozen V1 application source changed")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def read_bound(path: Path, expected_sha256: str):
    try:
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise BenchmarkError("SHA256 binding mismatch")
        return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"bound JSON unavailable: {type(exc).__name__}") from exc


def validate_registry(path, expected_sha256, queue_path):
    registry = read_bound(path, expected_sha256)
    paths = registry.get("gpu_queue_paths")
    if not isinstance(paths, list) or str(Path(queue_path).resolve()) not in paths:
        raise BenchmarkError("application accounting queue is not registered")
    return registry


def validate_budget(snapshot):
    try:
        canonical(snapshot)
        seconds = snapshot.get("gpu_seconds")
        if (snapshot.get("coverage_complete") is not True
                or snapshot.get("should_stop") is not False
                or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("coverage/stop/total")
    except (ValueError, TypeError, AttributeError) as exc:
        raise BudgetStopped("budget coverage unavailable, stale, or stopped") from exc
    return snapshot


class BudgetChecker:
    def __init__(self, argv_file, argv_sha256):
        self.path, self.sha256 = Path(argv_file), argv_sha256
        self.last = None

    def check(self):
        try:
            argv = read_bound(self.path, self.sha256)
            if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
                raise BenchmarkError("invalid budget argv")
            result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
            if result.returncode != 0:
                raise BenchmarkError("budget command failed")
            value = json.loads(result.stdout, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            self.last = validate_budget(value)
            return self.last
        except (OSError, ValueError, subprocess.TimeoutExpired, BenchmarkError) as exc:
            raise BudgetStopped(f"budget check refused: {str(exc) if isinstance(exc, BenchmarkError) else type(exc).__name__}") from exc


class NumericGpuLock(GpuLock):
    """Use the evaluator's numeric project lock, fail instead of waiting forever."""
    def __init__(self, root, gpu):
        if type(gpu) is not int or not 0 <= gpu <= 7:
            raise BenchmarkError("GPU index must be an exact integer in 0..7")
        super().__init__(root, str(gpu))

    def __enter__(self):
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.fh.close()
            self.fh = None
            raise BenchmarkError("physical GPU lock is already held") from exc
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"pid={os.getpid()} ts={utcnow()}\n")
        self.fh.flush()
        return self


class ApplicationJob:
    """Append one fresh official job only after all prior app jobs terminate."""
    def __init__(self, db, run_id, protocol_hash, source_hash, gpu, max_wall_seconds):
        self.db, self.run_id = Path(db), run_id
        self.protocol_hash, self.source_hash = protocol_hash, source_hash
        self.gpu, self.max_wall_seconds = gpu, max_wall_seconds
        self.queue, self.job, self.result = None, None, {}

    def __enter__(self):
        self.queue = Queue(self.db)
        try:
            # The check and enqueue share one writer transaction. A second app
            # process then sees this queued job and cannot admit a second run.
            self.queue.conn.execute("BEGIN IMMEDIATE")
            try:
                if self.queue.run_row(self.run_id) is not None:
                    raise BenchmarkError("application append requires a fresh run ID")
                for row in self.queue.conn.execute("SELECT kind,task_id,status FROM jobs"):
                    if row["kind"] != "application" or row["task_id"] != "APP_DECODER":
                        raise BenchmarkError("accounting queue contains a nonapplication job")
                    if row["status"] not in (*TERMINAL, "lost"):
                        raise BenchmarkError("prior application jobs are not all terminal")
                self.queue.create_run(self.run_id, "decoder-application-v2", self.protocol_hash)
                self.queue.enqueue(self.run_id, "APP_DECODER", "application",
                                   source_hash=self.source_hash, max_attempts=1, phase="application")
                self.queue.conn.execute("COMMIT")
            except BaseException:
                self.queue.conn.execute("ROLLBACK")
                raise
            self.job = self.queue.claim(self.run_id, f"application:{os.getpid()}", str(self.gpu),
                                        lease_s=self.max_wall_seconds + 120)
            if self.job is None:
                raise BenchmarkError("application job admission failed")
            self.result = {"protocol_sha256": self.protocol_hash, "source_package_sha256": self.source_hash,
                           "accounting": "conservative official claim-to-finish wall span"}
            return self
        except BaseException:
            self.queue.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            status = "complete" if exc is None else "crashed"
            if exc is not None:
                self.result.update(error_type=exc_type.__name__, budget_stop=isinstance(exc, BudgetStopped))
            self.queue.finish(self.job, status, self.result)
        finally:
            self.queue.close()
        return False


def clean_snapshot(gpu):
    snapshot = gpuinfo.snapshot(gpu)
    if not snapshot.get("available"):
        raise BenchmarkError("GPU state unavailable")
    if type(snapshot.get("foreign_processes")) is not int or snapshot["foreign_processes"] != 0:
        raise BenchmarkError("GPU measurement contaminated by foreign or unknown processes")
    return snapshot


def make_protocol(rounds=30, warmup=20, order_seed=20261005, max_wall_seconds=900):
    cases = [DecoderConfig(sequence=64), DecoderConfig(sequence=128),
             DecoderConfig(sequence=33, mlp_dim=33),
             DecoderConfig(batch=2, sequence=128, head_dim=128)]
    return {"schema_version": 2, "name": "fixed-random-decoder-v2", "seed": 20261005,
            "cases": [asdict(c) for c in cases],
            "case_labels": ["normal-s64", "normal-s128", "min-s33-mlp33", "boundary-b2-d128-rows2048"],
            "operator_tolerances": {k: list(v) for k, v in OPERATOR_TOLERANCES.items()},
            "full_block_output_tolerance": APPLICATION_TOLERANCE,
            "compile_baseline": True, "compile_mode": "default", "compile_fullgraph": True,
            "compile_candidate_fullgraph": True,
            "requested_backends": ["eager", "candidate", "compiled", "compiled_candidate"],
            "compilation_failure_policy": "record unavailable/incorrect; no eager fallback",
            "optimization": "whole candidate compilation; unchanged decoder math and operator source bytes",
            "timing": {"backend": "cuda_events", "paired_rounds": rounds, "warmup": warmup,
                       "order_seed": order_seed, "target_batch_ms": 20.0, "max_iters": 256,
                       "jit_compile_upload_validation_timed": False,
                       "weight_input_policy": "same fixed weights and input in every implementation",
                       "cuda_graphs": False},
            "max_wall_seconds": max_wall_seconds,
            "failure_examples": ["sequence32 rejected", "rows>2048 rejected", "odd head_dim rejected",
                                 "nonfinite input rejected", "source mismatch rejected", "contaminated GPU refused"]}


def source_package(bindings, protocol):
    verify_base_sources()
    entries = {str(p.relative_to(ROOT)): sha256_file(p) for p in
               (APP / "decoder_block.py", APP / "benchmark_decoder.py", APP / "benchmark_decoder_v2.py")}
    inputs = {"application_sources": entries, "manifest_sha256": bindings.manifest_sha256,
              "operator_sources": {k: v["sha256"] for k, v in bindings.operators.items()},
              "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest()}
    return inputs, hashlib.sha256(canonical(inputs)).hexdigest()


def make_plan(bindings, protocol):
    package, package_hash = source_package(bindings, protocol)
    return {"schema_version": 2, "status": "GPU_PENDING", "created_at": utcnow(),
            "pretrained_model": False, "text_quality_measured": False,
            "architecture": "QKV projections + adjacent-pair RoPE + causal softmax + attention/output projection + residual + gated SiLU MLP + residual",
            "operator_sources": bindings.receipt(), "source_package": package,
            "source_package_sha256": package_hash, "protocol": protocol,
            "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest(),
            "generation_environment": {"torch": torch.__version__, "python": sys.version.split()[0]},
            "measurements": None,
            "limits": ["Fixed random weights, no normalization, KV cache, tokenizer or pretrained text model.",
                       "Past operator gains do not establish a full-block gain.",
                       "CPU Dynamo math capture does not prove CUDA Triton capture, correctness or performance.",
                       "Repeated fixed inputs can fit cache; all raw paired samples must be reported."]}


def compare_outputs(reference, candidate, reference_trace, candidate_trace):
    validate_operator_trace(reference_trace)
    validate_operator_trace(candidate_trace)
    diagnostics = {}
    # Each substituted operator is also checked on its own actual inputs. Later
    # stages may differ slightly because earlier FP16 values are rounded.
    from decoder_block import ReferenceOperators
    for name, entry in candidate_trace.items():
        key = "K10" if name.startswith("rope") else "K04" if name == "softmax" else "K06"
        fn = ReferenceOperators.rope if key == "K10" else ReferenceOperators.softmax if key == "K04" else ReferenceOperators.silu_mul
        expected = fn(*entry["inputs"])
        rtol, atol = OPERATOR_TOLERANCES[key]
        torch.testing.assert_close(entry["output"], expected, rtol=rtol, atol=atol, equal_nan=False)
        if key == "K04":
            mask = entry["inputs"][1]
            if not torch.equal(entry["output"][~mask], torch.zeros_like(entry["output"][~mask])):
                raise AssertionError("K04 masked values must be exactly zero")
        diagnostics[name] = {"task_id": key, "max_abs_error": (entry["output"].float() - expected.float()).abs().max().item(),
                             "rtol": rtol, "atol": atol}
    torch.testing.assert_close(candidate, reference, **APPLICATION_TOLERANCE, equal_nan=False)
    if not torch.isfinite(candidate).all().item():
        raise AssertionError("nonfinite decoder output")
    return {"operators": diagnostics, "full_block_max_abs_error": (reference.float() - candidate.float()).abs().max().item(),
            "full_block_tolerance": APPLICATION_TOLERANCE}


def measure_paired(fns, warmup, rounds, order_seed, checkpoint, partial=None):
    """Same CUDA-event batches as kernel-lab; checkpoints are outside timed calls."""
    samples = partial if partial is not None else {}
    iters = {}
    for label, fn in fns.items():
        checkpoint()
        for _ in range(warmup):
            fn()
        timing.sync()
        checkpoint()
        iters[label], calibration_ms = timing.calibrate(fn, target_ms=20.0, max_iters=256)
        samples[label] = {"timing_backend": "cuda_events", "iters_per_batch": iters[label],
                          "calibration_batch_ms": calibration_ms, "warmup": warmup,
                          "gpu_us_samples": [], "host_us_samples": [], "batch_label_order": [],
                          "host_time_note": timing.HOST_TIME_NOTE}
    rng = random.Random(order_seed)
    for _ in range(rounds):
        checkpoint()
        order = list(fns)
        rng.shuffle(order)
        for label in order:
            checkpoint()
            gpu_ms, host_ms = timing._run_batch(fns[label], iters[label])
            row = samples[label]
            row["gpu_us_samples"].append(gpu_ms * 1000 / iters[label])
            row["host_us_samples"].append(host_ms * 1000 / iters[label])
            row["batch_label_order"].append(order[:])
        checkpoint()
    for row in samples.values():
        row.update(gpu_us_median=statistics.median(row["gpu_us_samples"]),
                   host_us_median=statistics.median(row["host_us_samples"]),
                   gpu_state_split=timing.state_split_record(row["gpu_us_samples"]),
                   batches=len(row["gpu_us_samples"]))
    return samples


def paired_gain(samples, baseline, candidate="candidate", seed=20261005):
    a, b = samples[baseline]["gpu_us_samples"], samples[candidate]["gpu_us_samples"]
    ratios = [x / y for x, y in zip(a, b)]
    rng = random.Random(seed)
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return {"metric": "median of within-round full-block baseline/candidate latency ratios",
            "paired_ratio_samples": ratios, "median": statistics.median(ratios),
            "bootstrap_95_percent_interval": [bootstrap[49], bootstrap[1949]],
            "bootstrap_seed": seed, "bootstrap_resamples": 2000}


def compile_candidate(block, x, reference, *, sync):
    """Compile a whole block, check its real output, and never return fallback."""
    started = time.monotonic()
    try:
        compiled = torch.compile(block, mode="default", fullgraph=True)
        actual = compiled(x)
        sync()
        torch.testing.assert_close(actual, reference, **APPLICATION_TOLERANCE, equal_nan=False)
        if not torch.isfinite(actual).all().item():
            raise AssertionError("nonfinite compiled candidate output")
        receipt = {"status": "AVAILABLE", "fullgraph": True, "mode": "default",
                   "compile_warmup_seconds": time.monotonic() - started,
                   "full_block_max_abs_error": (actual.float() - reference.float()).abs().max().item(),
                   "full_block_tolerance": APPLICATION_TOLERANCE}
        return lambda: compiled(x), receipt
    except BudgetStopped:
        raise
    except Exception as exc:
        return None, {"status": "INCORRECT" if isinstance(exc, AssertionError) else "UNAVAILABLE",
                      "error_type": type(exc).__name__, "fullgraph": True, "mode": "default",
                      "compile_warmup_seconds": time.monotonic() - started}


@contextlib.contextmanager
def wall_limit(seconds):
    def stop(signum, frame):
        raise BudgetStopped("finite application wall deadline or interrupt")
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old.items():
            signal.signal(sig, handler)


@contextlib.contextmanager
def cuda_completion():
    """Do not close the official span while queued CUDA work is still active."""
    try:
        yield
    finally:
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()


def run_gpu(args, bindings, report, checker):
    # No CUDA API, candidate import or CUDA tensor before the official claim.
    verify_base_sources()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with wall_limit(args.max_wall_seconds), NumericGpuLock(ROOT, args.gpu):
        preflight = clean_snapshot(args.gpu)
        checker.check()
        with ApplicationJob(args.accounting_queue, args.run_id, report["protocol_sha256"],
                            report["source_package_sha256"], args.gpu, args.max_wall_seconds) as job:
            report.update(status="RUNNING", accounting={"queue": str(Path(args.accounting_queue).resolve()),
                          "run_id": args.run_id, "job_id": job.job["job_id"], "gpu_physical_index": args.gpu,
                          "gpu_span_semantics": "official claim-to-finish conservative wall time"}, measurements={})
            job.result["report_path"] = str(Path(args.out).resolve())
            report["gpu_snapshots"] = [preflight]
            with cuda_completion(), torch.inference_mode():
                if not torch.cuda.is_available():
                    raise BenchmarkError("CUDA unavailable after accounting admission")
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.set_float32_matmul_precision("highest")
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
                props = torch.cuda.get_device_properties(device)
                report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                         "device_name": props.name, "device_total_memory": props.total_memory,
                                         "python": sys.version.split()[0], "tf32": False,
                                         "pid": os.getpid(), "physical_gpu": args.gpu}
                operators = CandidateOperators(bindings)
                def checkpoint():
                    report["last_budget_snapshot"] = checker.check()
                    report["gpu_snapshots"].append(clean_snapshot(args.gpu))
                    if not job.queue.heartbeat(job.job["job_id"], job.job["lease_owner"], args.max_wall_seconds + 120):
                        raise BenchmarkError("application accounting lease lost")
                for index, (label, config) in enumerate(zip(report["protocol"]["case_labels"], report["protocol"]["cases"])):
                    checkpoint()
                    case = make_case(DecoderConfig(**config), report["protocol"]["seed"])
                    row = {"input_weights_sha256": case.fingerprint(), "config": config, "status": "RUNNING",
                           "samples": {}, "compiled_baseline": {"status": "PENDING"},
                           "compiled_candidate": {"status": "PENDING"}}
                    report["measurements"][label] = row
                    x = case.x.to(device)
                    original = x.clone()
                    eager = DecoderBlock(case).to(device).eval()
                    candidate = DecoderBlock(case, operators=operators).to(device).eval()
                    # Materialize eager output first, before candidate code sees inputs.
                    reference, reference_trace = eager.forward_with_trace(x)
                    eager_weights = {k: v.clone() for k, v in eager.named_buffers()}
                    candidate_weights = {k: v.clone() for k, v in candidate.named_buffers()}
                    actual, candidate_trace = candidate.forward_with_trace(x)
                    row["correctness"] = compare_outputs(reference, actual, reference_trace, candidate_trace)
                    for block, before in ((eager, eager_weights), (candidate, candidate_weights)):
                        for key, tensor in block.named_buffers():
                            if not torch.equal(tensor, before[key]):
                                raise AssertionError("weights/tables mutated")
                    if not torch.equal(x, original):
                        raise AssertionError("input mutated")
                    fns = {"eager": lambda: eager(x), "candidate": lambda: candidate(x)}
                    checkpoint()
                    compile_start = time.monotonic()
                    try:
                        compiled = torch.compile(eager, mode="default", fullgraph=True)
                        compiled_out = compiled(x)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(compiled_out, reference, **APPLICATION_TOLERANCE, equal_nan=False)
                        fns["compiled"] = lambda: compiled(x)
                        row["compiled_baseline"] = {"status": "AVAILABLE", "compile_warmup_seconds": time.monotonic() - compile_start}
                    except BudgetStopped:
                        raise
                    except Exception as exc:
                        row["compiled_baseline"] = {"status": "INCORRECT" if isinstance(exc, AssertionError) else "UNAVAILABLE", "error_type": type(exc).__name__,
                                                    "compile_warmup_seconds": time.monotonic() - compile_start}
                    checkpoint()
                    compiled_candidate, row["compiled_candidate"] = compile_candidate(
                        candidate, x, reference, sync=torch.cuda.synchronize)
                    if compiled_candidate is not None:
                        fns["compiled_candidate"] = compiled_candidate
                    # Compilation work is excluded from timed calls, but stays
                    # inside the official span and bounded checkpoint loop.
                    atomic_json(Path(args.out), report)
                    timing.configure(True)
                    measure_paired(fns, report["protocol"]["timing"]["warmup"],
                                   report["protocol"]["timing"]["paired_rounds"],
                                   report["protocol"]["timing"]["order_seed"] + index,
                                   checkpoint, partial=row["samples"])
                    checkpoint()
                    if not torch.equal(x, original):
                        raise AssertionError("timed input mutated")
                    for block, before in ((eager, eager_weights), (candidate, candidate_weights)):
                        for key, tensor in block.named_buffers():
                            if not torch.equal(tensor, before[key]):
                                raise AssertionError("timed weights/tables mutated")
                    row["speedups"] = {"against_eager": paired_gain(row["samples"], "eager")}
                    row["speedups"]["against_compiled"] = paired_gain(row["samples"], "compiled") if "compiled" in fns else None
                    row["compiled_candidate_speedups"] = {
                        "against_eager": paired_gain(row["samples"], "eager", "compiled_candidate")
                            if "compiled_candidate" in fns else None,
                        "against_compiled": paired_gain(row["samples"], "compiled", "compiled_candidate")
                            if "compiled_candidate" in fns and "compiled" in fns else None,
                        "against_candidate": paired_gain(row["samples"], "candidate", "compiled_candidate")
                            if "compiled_candidate" in fns else None,
                    }
                    row["status"] = "MEASURED_CLEAN"
                    atomic_json(Path(args.out), report)
                checkpoint()
                report.update(status="MEASURED_CLEAN", completed_at=utcnow())
                atomic_json(Path(args.out), report)
                job.result.update(report_sha256=sha256_file(Path(args.out)), measured_cases=len(report["measurements"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="requires external authorization/registry and live budget observer")
    parser.add_argument("--out", type=Path, required=True, help="fresh JSON receipt path")
    parser.add_argument("--gpu", type=int, choices=range(8), default=4)
    parser.add_argument("--run-id", default="agh-hackathon-decoder-v2")
    parser.add_argument("--operator-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--operator-manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--accounting-queue", type=Path)
    parser.add_argument("--registry-reference", type=Path)
    parser.add_argument("--registry-sha256")
    parser.add_argument("--budget-argv-json", type=Path)
    parser.add_argument("--budget-argv-sha256")
    parser.add_argument("--max-wall-seconds", type=float, default=900)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 3600:
        parser.error("max-wall-seconds must be finite in (0,3600]")
    if args.rounds < 4 or args.rounds > 1000 or args.warmup < 0 or args.warmup > 1000:
        parser.error("rounds must be 4..1000 and warmup 0..1000")
    if args.out.exists():
        parser.error("output must be fresh; preserve previous receipts")
    if args.execute and any(getattr(args, field) is None for field in
                            ("accounting_queue", "registry_reference", "registry_sha256", "budget_argv_json", "budget_argv_sha256")):
        parser.error("execute requires accounting queue, registered policy SHA and budget argv SHA")
    bindings = verify_sources(args.operator_manifest, args.operator_manifest_sha256)
    protocol = make_protocol(args.rounds, args.warmup, max_wall_seconds=args.max_wall_seconds)
    report = make_plan(bindings, protocol)
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
        run_gpu(args, bindings, report, checker)
        return 0
    except BaseException as exc:
        report.update(status="BUDGET_STOPPED" if isinstance(exc, BudgetStopped) else "FAILED",
                      error_type=type(exc).__name__, error=str(exc)[:500], completed_at=utcnow())
        atomic_json(args.out, report)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
