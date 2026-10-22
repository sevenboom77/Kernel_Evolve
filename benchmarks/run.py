"""Standalone operator correctness checks and local CUDA-event measurements."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def constants(path):
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"FINAL_CORRECTNESS", "DEV_TIMING", "RESERVED_TIMING"}:
                    out[target.id] = ast.literal_eval(node.value)
    return out


def plan(kind):
    if kind == "small-m":
        cases = constants(ROOT / "tasks/K12/cases.py")
        spec = json.loads((ROOT / "tasks/K12/spec.json").read_text())
        return {"operator": kind, "source": "operators/small_m/kernel.py",
                "correctness_cases": cases["FINAL_CORRECTNESS"],
                "timing_cases": cases["DEV_TIMING"] + cases["RESERVED_TIMING"],
                "tolerance": {key: spec[key] for key in ("rtol", "atol")}}
    return {"operator": kind, "source": "operators/graph/csr_gather_sum.py",
            "cases": [{"nodes": 1024, "targets": n, "degree_pattern": [0, 1, 8, 9, 32, 33, 129, 300]}
                      for n in (8, 64, 256)],
            "feature_width": 128, "tolerance": {"rtol": 0.0003, "atol": 0.00001}}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def measure(torch, functions, rounds, iterations):
    # Compilation and warmup are completed before the paired rounds.
    for fn in functions.values():
        for _ in range(20):
            fn()
    torch.cuda.synchronize()
    samples = {key: [] for key in functions}
    rng = random.Random(421)
    for _ in range(rounds):
        order = list(functions)
        rng.shuffle(order)
        for key in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                functions[key]()
            end.record()
            end.synchronize()
            samples[key].append(start.elapsed_time(end) * 1000 / iterations)
    return {"microseconds": samples, "median_us": {key: statistics.median(x) for key, x in samples.items()}}


def execute(args, value):
    import torch
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(421)
    device = f"cuda:{args.gpu}"
    operator = module(ROOT / value["source"], "public_operator")
    results = []
    with torch.inference_mode():
        if args.operator == "small-m":
            cases = module(ROOT / "tasks/K12/cases.py", "public_cases")
            reference = module(ROOT / "tasks/K12/reference.py", "public_reference").Model()
            candidate = operator.ModelNew()
            for case in value["correctness_cases"]:
                inputs = cases.make_inputs(case, device=device, run_seed=8123671)
                torch.testing.assert_close(candidate(*inputs)[0], reference(*inputs)[0], **value["tolerance"])
            for case in value["timing_cases"]:
                inputs = cases.make_inputs(case, device=device, run_seed=421)
                a, b = inputs
                fns = {"candidate": lambda: candidate(*inputs)[0], "torch_mm_fp32_reduction": lambda: torch.mm(a, b)}
                torch.testing.assert_close(fns["candidate"](), reference(*inputs)[0], **value["tolerance"])
                torch.testing.assert_close(fns["torch_mm_fp32_reduction"](), reference(*inputs)[0], **value["tolerance"])
                results.append({"case": case, **measure(torch, fns, args.rounds, args.iterations)})
        else:
            for case in value["cases"]:
                n = case["targets"]
                degree = torch.tensor((case["degree_pattern"] * ((n + 7) // 8))[:n], device=device, dtype=torch.int64)
                offsets = torch.cat((degree.new_zeros(1), degree.cumsum(0)))
                sources = torch.randint(0, case["nodes"], (int(offsets[-1].item()),), device=device)
                features = torch.randn(case["nodes"], 128, device=device)
                target = torch.repeat_interleave(torch.arange(n, device=device), degree)
                schedule = operator.build_schedule(offsets)
                def reference():
                    out = features.new_zeros((n, 128))
                    return out.index_add_(0, target, features[sources])
                fns = {"candidate": lambda: operator.aggregate(features, sources, offsets, schedule),
                       "torch_gather_index_add": reference}
                torch.testing.assert_close(fns["candidate"](), reference(), **value["tolerance"])
                results.append({"case": case, **measure(torch, fns, args.rounds, args.iterations)})
    return {"device": torch.cuda.get_device_name(args.gpu), "rounds": args.rounds,
            "iterations_per_sample": args.iterations, "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operator", choices=["small-m", "graph"], required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.rounds < 1 or args.iterations < 1:
        p.error("rounds and iterations must be positive")
    if args.out.exists():
        p.error("choose a new output file")
    value = plan(args.operator)
    value["stage"] = "plan"
    if args.execute:
        value["measurement"] = execute(args, value)
        value["stage"] = "measured"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(args.out)


if __name__ == "__main__":
    main()
