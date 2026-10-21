"""Separate strong controls; retain the qualified original graph source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA256 = "1e519186b97381754b4297a79ff50329a891ee310c83dadadd091f842fb62b39"
STRONG_BASELINES = ("segment_reduce", "index_add", "sparse_mm", "compiled_segment_reduce", "compiled_index_add")


def original():
    path = HERE / "citation_graph.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Qualified original graph source changed")
    name = "kernelevolve_citation_graph_original_R1"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    module = sys.modules[name]
    if Path(module.__file__).resolve() != path:
        raise ValueError("Original graph module belongs to another path")
    return module


def prepare_adjacency(graph, *, device, node_begin=0, node_end=None):
    node_end = graph.features.shape[0] if node_end is None else node_end
    if not 0 <= node_begin < node_end <= graph.features.shape[0]:
        raise ValueError("Valid original target-node window required")
    offsets = graph.offsets[node_begin:node_end + 1] - graph.offsets[node_begin]
    sources = graph.sources[graph.offsets[node_begin]:graph.offsets[node_end]]
    return torch.sparse_csr_tensor(
        torch.tensor(offsets, dtype=torch.int64, device=device),
        torch.tensor(sources.copy(), dtype=torch.int64, device=device),
        torch.ones(sources.size, dtype=torch.float32, device=device),
        size=(node_end - node_begin, graph.features.shape[0]), device=device,
    )


def sparse_message_passing(inputs, weights, adjacency):
    if isinstance(weights[0][0], np.ndarray):
        weights = original().prepare_weights(weights, device=inputs.features.device)
    hidden = inputs.features
    begin, end = inputs.node_begin, inputs.node_end
    for self_weight, neighbor_weight, bias in weights:
        total = torch.sparse.mm(adjacency, hidden)
        mean = total / inputs.degrees.clamp(min=1).unsqueeze(1)
        updated = torch.relu(hidden[begin:end] @ self_weight + mean @ neighbor_weight + bias) + hidden[begin:end]
        hidden = updated if begin == 0 and end == hidden.shape[0] else torch.cat((hidden[:begin], updated, hidden[end:]))
    return hidden[begin:end].clone()


class DenseModel(torch.nn.Module):
    def __init__(self, inputs, weights, aggregate):
        super().__init__()
        self.inputs, self.weights, self.aggregate = inputs, weights, aggregate
        self.computation = original().message_passing

    def forward(self):
        return self.computation(self.inputs, self.weights, self.aggregate)


class SparseModel(torch.nn.Module):
    def __init__(self, inputs, weights, adjacency):
        super().__init__()
        self.inputs, self.weights, self.adjacency = inputs, weights, adjacency

    def forward(self):
        return sparse_message_passing(self.inputs, self.weights, self.adjacency)


def prepare_backend(name, inputs, weights, *, adjacency=None, candidate_source=None, compile_factory=None):
    if name not in (*STRONG_BASELINES, "candidate"):
        raise ValueError(f"Unknown declared backend: {name}")
    base = original()
    weights = base.prepare_weights(weights, device=inputs.features.device) if isinstance(weights[0][0], np.ndarray) else weights
    compiled = name.startswith("compiled_")
    receipt = {"backend": name, "compiled_requested": compiled, "status": "PREPARED_UNQUALIFIED", "fallback_used": False}
    if name == "sparse_mm":
        if adjacency is None:
            raise ValueError("Direct sparse baseline requires the original adjacency")
        model = SparseModel(inputs, weights, adjacency)
    elif name == "candidate":
        if candidate_source is None:
            raise ValueError("Candidate requires the unchanged source-bound K15")
        model = DenseModel(inputs, weights, base.tiled_aggregation(inputs, base.SourceBoundCSR(candidate_source)))
    else:
        aggregate_name = "torch.segment_reduce" if "segment_reduce" in name else "torch.index_add_"
        model = DenseModel(inputs, weights, base.stock_aggregation(aggregate_name, inputs))
    model.eval()
    if compiled:
        try:
            factory = torch.compile if compile_factory is None else compile_factory
            model = factory(model, mode="default", fullgraph=True, dynamic=False)
        except Exception as exc:
            receipt.update(status="UNAVAILABLE", error_type=type(exc).__name__, diagnostic=str(exc)[:4000])
            return None, receipt
    # Lazy compiler or numerical failures are still unqualified until the caller
    # actually executes the model and preserves the original correctness result.
    return model, receipt


def strong_gain(samples):
    if any(name not in samples for name in (*STRONG_BASELINES, "candidate")):
        return None
    vectors = [samples[name] for name in (*STRONG_BASELINES, "candidate")]
    if any(len(v) != 30 for v in vectors) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                               or not math.isfinite(x) or x <= 0 for v in vectors for x in v):
        raise ValueError("All30paired samples must be finite positive values")
    minimum = [min(row) for row in zip(*vectors[:-1])]
    ratios = [baseline / candidate for baseline, candidate in zip(minimum, vectors[-1])]
    generator = random.Random(20261006)
    bootstrap = sorted(statistics.median(generator.choices(ratios, k=30)) for _ in range(2000))
    return {"required_library_backends": list(STRONG_BASELINES), "paired_rounds": 30,
            "strong_baseline_samples": minimum, "paired_ratio_samples": ratios,
            "median": statistics.median(ratios), "bootstrap_95_percent_interval": [bootstrap[49], bootstrap[1949]],
            "bootstrap_seed": 20261006, "bootstrap_resamples": 2000, "victory_proven": bootstrap[49] > 1}


def cpu_plan(graph, *, archive_sha256):
    base = original()
    plan = base.cpu_plan(graph, archive_sha256=archive_sha256, operator_sha256=base.CSR_SHA256)
    plan.update(protocol="real-citation-graph-strong-controls-v2", distinct_from_R1=True,
                original_R1_source_sha256=ORIGINAL_SHA256, original_R1_results_preserved=True,
                strong_baselines_required=list(STRONG_BASELINES),
                requested_backends=[*STRONG_BASELINES, "candidate"],
                direct_sparse_pipeline_avoids_dense_message_materialization=True,
                compiled_control_options={"mode": "default", "fullgraph": True, "dynamic": False},
                compilation_failure_policy="retain unavailable or incorrect; no eager fallback or broad strong victory",
                strong_gain="per-round minimum of all five qualified libraries divided by candidate",
                GPU_execution_supported_by_this_entrypoint=False)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refuse to overwrite an existing plan")
    base = original()
    graph = base.load_graph_archive(args.archive, args.archive_sha256)
    plan = cpu_plan(graph, archive_sha256=args.archive_sha256)
    plan["strong_controls_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"status": plan["status"], "strong_baselines": plan["strong_baselines_required"], "GPU_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
