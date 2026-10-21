"""GPU timing under the frozen protocol.

Rules implemented here (spec section 7.3):

* the primary metric is the GPU elapsed time of a *complete* operator call,
  measured with CUDA events and an explicit synchronise;
* compilation, JIT and warm-up are excluded from steady state and reported
  separately;
* batch sizes are auto-calibrated so each batch runs for roughly 20-100 ms, then
  30 batches are collected per shape per version;
* the host-side call time is recorded separately and never mixed into the GPU
  number;
* inputs are rotated between buffers so that a constant-tensor trick is visible;
* CUDA graphs are not used in the main track.
"""
from __future__ import annotations

import random
import statistics
import time
from typing import Callable

import torch

DEFAULT_WARMUP = 20
DEFAULT_BATCHES = 30
TARGET_BATCH_MS = (20.0, 100.0)

HOST_TIME_NOTE = ("host wall time of the submission loop (includes backpressure while the "
                  "queue is full); the final device wait after the last call is excluded "
                  "and is not part of the host number")


_USE_CUDA_EVENTS = True


def configure(use_cuda_events: bool) -> None:
    """Select the timing backend.

    CUDA events are mandatory for every real measurement; the CPU backend exists
    only so the *pipeline* can be exercised (queue, verdicts, adversarial
    candidates) without occupying a card.  Measurements taken with it are marked
    ``timing_backend="cpu"`` and are never used to declare a verified gain.
    """
    global _USE_CUDA_EVENTS
    _USE_CUDA_EVENTS = bool(use_cuda_events)


def using_cuda_events() -> bool:
    return _USE_CUDA_EVENTS


def sync() -> None:
    """Synchronise the GPU, or no-op on the CPU test backend."""
    if _USE_CUDA_EVENTS:
        torch.cuda.synchronize()


_sync = sync


def _ev():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _run_batch(fn: Callable[[], None], iters: int) -> tuple[float, float]:
    if not _USE_CUDA_EVENTS:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1000.0, (time.perf_counter() - t0) * 1000.0
    start, end = _ev()
    _sync()
    start.record()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    end.record()
    _sync()
    return start.elapsed_time(end), (t1 - t0) * 1000.0


def calibrate(fn: Callable[[], None], target_ms: float = 50.0, max_iters: int = 4096) -> tuple[int, float]:
    """Pick the inner iteration count so one batch lasts about ``target_ms``."""
    for _ in range(3):
        fn()
    _sync()
    iters = 1
    ms = 0.0
    for _ in range(16):
        ms, _ = _run_batch(fn, iters)
        if ms >= target_ms or iters >= max_iters:
            break
        scale = max(2.0, target_ms / max(ms, 1e-4))
        iters = min(max_iters, max(iters + 1, int(iters * scale)))
    return iters, ms


def measure(fn: Callable[[], None], warmup: int = DEFAULT_WARMUP,
            batches: int = DEFAULT_BATCHES, target_ms: tuple[float, float] = TARGET_BATCH_MS,
            max_iters: int = 4096) -> dict:
    """Steady-state measurement of one callable at one shape."""
    if warmup:
        for _ in range(warmup):
            fn()
        _sync()
    iters, calib_ms = calibrate(fn, target_ms=target_ms[0], max_iters=max_iters)
    gpu_us: list[float] = []
    host_us: list[float] = []
    for _ in range(batches):
        gpu_ms, host_ms = _run_batch(fn, iters)
        gpu_us.append(gpu_ms * 1000.0 / iters)
        host_us.append(host_ms * 1000.0 / iters)
    gpu_sorted = sorted(gpu_us)
    return {
        "iters_per_batch": iters,
        "calibration_batch_ms": calib_ms,
        "timing_backend": "cuda_events" if _USE_CUDA_EVENTS else "cpu",
        "batches": batches,
        "warmup": warmup,
        "host_time_note": HOST_TIME_NOTE,
        "gpu_us_samples": gpu_us,
        "host_us_samples": host_us,
        "gpu_us_median": statistics.median(gpu_us),
        "gpu_us_mean": statistics.fmean(gpu_us),
        "gpu_us_min": gpu_sorted[0],
        "gpu_us_p10": gpu_sorted[max(0, int(0.1 * len(gpu_sorted)) - 1)],
        "host_us_median": statistics.median(host_us),
        "host_state_split": state_split_record(host_us),
        "gpu_state_split": state_split_record(gpu_us),
    }


def measure_rotating(build_call: Callable[[int], Callable[[], None]], buffers: int,
                     **kw) -> dict:
    """Measure with the callable switching input buffer every batch."""
    if buffers <= 1:
        return measure(build_call(0), **kw)
    calls = [build_call(i) for i in range(buffers)]
    state = {"n": 0}

    def rotating():
        i = state["n"] % buffers
        state["n"] += 1
        calls[i]()

    return measure(rotating, **kw)


def measure_interleaved(fns: dict, warmup: int = DEFAULT_WARMUP, batches: int = DEFAULT_BATCHES,
                        target_ms: float = 20.0, max_iters: int = 4096,
                        order_seed: int | None = None) -> dict:
    """Measure several callables round-robin so no label enjoys a private cache.

    Each label is calibrated separately, then one batch per label is run per
    round. This is the protocol used for candidate-versus-baseline comparison.

    With ``order_seed`` the within-round label order is re-drawn every round
    from that seed (the frozen protocol supplies it), so no label keeps a
    positional advantage across the rounds; the order actually used is
    recorded in ``batch_label_order``. Without a seed the insertion order is
    kept, which is what the CPU pipeline tests use.
    """
    labels = list(fns)
    iters: dict[str, int] = {}
    for lb in labels:
        for _ in range(warmup):
            fns[lb]()
        _sync()
        iters[lb], _ = calibrate(fns[lb], target_ms=target_ms, max_iters=max_iters)
    rng = random.Random(order_seed) if order_seed is not None else None
    orders: list[list[str]] = []
    samples: dict[str, dict] = {lb: {"gpu_us_samples": [], "host_us_samples": []} for lb in labels}
    for _ in range(batches):
        order = list(labels)
        if rng is not None:
            rng.shuffle(order)
        orders.append(order)
        for lb in order:
            gpu_ms, host_ms = _run_batch(fns[lb], iters[lb])
            samples[lb]["gpu_us_samples"].append(gpu_ms * 1000.0 / iters[lb])
            samples[lb]["host_us_samples"].append(host_ms * 1000.0 / iters[lb])
    out = {}
    for lb in labels:
        gpu = samples[lb]["gpu_us_samples"]
        host = samples[lb]["host_us_samples"]
        gs = sorted(gpu)
        out[lb] = {
            "timing_backend": "cuda_events" if _USE_CUDA_EVENTS else "cpu",
            "iters_per_batch": iters[lb],
            "batches": batches,
            "warmup": warmup,
            "host_time_note": HOST_TIME_NOTE,
            "batch_label_order": [list(o) for o in orders],
            "gpu_us_samples": gpu,
            "host_us_samples": host,
            "gpu_us_median": statistics.median(gpu),
            "gpu_us_mean": statistics.fmean(gpu),
            "gpu_us_min": gs[0],
            "gpu_us_p10": gs[max(0, int(0.1 * len(gs)) - 1)],
            "host_us_median": statistics.median(host),
            "host_state_split": state_split_record(host),
            "gpu_state_split": state_split_record(gpu),
        }
    return out


def estimate_working_set(inputs: tuple, extra_copies: int = 1) -> int:
    total = 0
    for t in inputs:
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total * (extra_copies + 2)


def half_medians(values: list[float]) -> tuple[float | None, float | None]:
    """Median of the first and second half of a sample series."""
    if not values or len(values) < 4:
        return None, None
    mid = len(values) // 2
    return statistics.median(values[:mid]), statistics.median(values[mid:])


def state_robust_median(values: list[float]) -> float | None:
    """One machine-state episode must not inflate a shape past another.

    Host/GPU state episodes show up as one half of the batch series being
    systematically slower. Taking the *minimum* of the two half-medians
    estimates the clean state, which is what cross-shape diagnostics (like
    the launch-bound check) need to compare shapes on equal terms. This is a
    diagnostic estimator; the headline numbers stay the plain medians.
    """
    if not values:
        return None
    a, b = half_medians(values)
    if a is None:
        return statistics.median(values)
    return min(a, b)


def state_split_record(values: list[float]) -> dict:
    """Half-split diagnostics so a state shift inside a job is visible."""
    a, b = half_medians(values)
    if a is None:
        return {"half_medians": None, "state_shift_flag": False}
    ratio = max(a, b) / max(min(a, b), 1e-9)
    return {"half_medians": [a, b], "state_shift_ratio": ratio,
            "state_shift_flag": ratio > 1.25}


LAUNCH_BOUND_WS_RATIO = 4.0
LAUNCH_BOUND_TIME_RATIO = 1.3


def flag_launch_bound(times_us: dict, working_set: dict, ws_ratio: float = LAUNCH_BOUND_WS_RATIO,
                      time_ratio: float = LAUNCH_BOUND_TIME_RATIO) -> tuple:
    """Note a measurement dominated by per-call overhead rather than GPU work.

    If two shapes differ by at least ws_ratio in bytes touched yet finish within
    time_ratio of each other, the GPU cannot have been the bottleneck on the
    larger one: the time is launch/dispatch overhead that does not scale with
    work. This is a property of the shape, not of the candidate. The caller
    records the note next to the numbers — reducing per-call overhead is one of
    the optimisations this project targets for small operators, so the gain
    still counts; the note says what actually got faster (launch cost rather
    than kernel work) and does not invalidate the measurement.
    """
    ids = [c for c in times_us if c in working_set and times_us.get(c)]
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            wa, wb = working_set[a], working_set[b]
            if not wa or not wb:
                continue
            big, small = (a, b) if wa >= wb else (b, a)
            ratio = working_set[big] / working_set[small]
            if ratio < ws_ratio:
                continue
            tr = times_us[big] / times_us[small]
            if tr <= time_ratio:
                pairs.append((big, small, ratio, tr))
    if not pairs:
        return False, ""
    detail = "; ".join("%s vs %s: %.1fx the bytes in %.2fx the time" % (b, s, r, t)
                       for b, s, r, t in pairs)
    return True, ("measurement note: per-call overhead, not GPU work, dominates these shapes ("
                  + detail + "); a speedup here reflects reduced launch/dispatch cost "
                  "rather than kernel work")


LAUNCH_FLOOR_MARGIN = 2.0


def footprint_bytes(inputs, outputs=None) -> int:
    """Bytes one call must touch: every input plus every freshly written output.

    This is a lower bound on the traffic a correct kernel has to move, so it is
    also the denominator of the honest bandwidth of a measurement.
    """
    total = 0
    for t in list(inputs or ()) + list(outputs or ()):
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


def measurement_floor(times_us: dict, cases: dict, rules: dict = None) -> tuple:
    """Note shapes on which a speedup number carries a cache/launch caveat.

    Two device-derived conditions are recorded next to any number taken on
    these shapes (2026-10-03 audit: these are diagnostics, not validity
    verdicts — small operators and hot caches are real workloads, and cutting
    per-call overhead is one of the optimisations this project targets):

    * footprint below the L2 size - repeated calls reuse the same buffers, so the
      traffic never leaves the cache and the number describes cache bandwidth. On
      an RTX 4090 (75.5 MB of L2) a 33 MB working set measured 4003 GB/s, four
      times the card's HBM bandwidth.
    * per-call time within a small factor of the host launch cost - the host may
      be the bottleneck and the clock then reads submission cost. This is bimodal
      in practice: a submission-limited shape measures 1.00x its host cost,
      while a genuinely GPU-bound 134 MB shape measured 9.8x. An empty kernel
      already costs about 20 us to launch on this card.

    The floor is the device's L2 size, not a tuned constant; min_working_set_bytes
    can raise it but cannot lower it.
    """
    rules = rules or {}
    declared = int(rules.get("min_working_set_bytes") or 0)
    margin = float(rules.get("launch_floor_margin") or LAUNCH_FLOOR_MARGIN)
    device = (cases or {}).get("device") or {}
    l2 = int(device.get("l2_bytes") or 0)
    floor = max(declared, l2)
    problems = []
    for cid, blk in sorted(((cases or {}).get("cases") or {}).items()):
        if blk.get("status") != "ok":
            continue
        fp = blk.get("footprint_bytes") or 0
        gpu = times_us.get(cid) or 0
        host = ((blk.get("measurements") or {}).get("candidate") or {}).get("host_us_median") or 0
        if fp and floor and fp < floor:
            problems.append("%s touches %.1f MB, inside the %.1f MB L2 cache"
                            % (cid, fp / 1e6, floor / 1e6))
        if gpu and host and gpu < margin * host:
            problems.append("%s takes %.1f us per call against a %.1f us host launch cost, "
                            "so the host may be the bottleneck and the clock reads submission cost"
                            % (cid, gpu, host))
    if not problems:
        return False, ""
    shown = problems[:3]
    if len(problems) > len(shown):
        shown.append("and %d more" % (len(problems) - len(shown)))
    return True, ("measurement note: these shapes sit at or below the L2/launch floor: "
                  + "; ".join(shown)
                  + ". A gain measured here reflects cache residency or per-call launch "
                    "overhead; the report carries this note next to the number")
