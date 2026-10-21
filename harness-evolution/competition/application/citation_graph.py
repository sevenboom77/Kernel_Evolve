"""CPU preparation for source-bound CSR on the real OGB citation graph.

No CUDA, network or OGB-package imports. Archive downloading is a separate,
explicit operation. Node order, directed edges and duplicate citations remain.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np

CSR_SHA256 = "a0a474db012746a5b9c571c9d2cbad86e08b127456773e950b0c5806ed8dae24"
OFFICIAL_DOC = "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv"
ARCHIVE_MEMBERS = (
    "arxiv/raw/node-feat.csv.gz", "arxiv/raw/edge.csv.gz",
    "arxiv/raw/num-node-list.csv.gz", "arxiv/raw/num-edge-list.csv.gz",
)


@dataclass(frozen=True)
class GraphCSR:
    features: np.ndarray
    sources: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True)
class Tile:
    node_begin: int
    node_end: int
    edge_begin: int
    edge_end: int


@dataclass(frozen=True)
class TensorInputs:
    features: object
    sources: object
    offsets: object
    destinations: object
    degrees: object
    node_begin: int
    node_end: int
    tiles: tuple


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def incoming_csr(features: np.ndarray, edges: np.ndarray) -> GraphCSR:
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError("Node features must be a nonempty matrix")
    if not np.isfinite(features).all():
        raise ValueError("Node features must be finite")
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Edges must be an integer [edge_count, 2] matrix")
    nodes = features.shape[0]
    if edges.size and (edges.min() < 0 or edges.max() >= nodes):
        raise ValueError("Edge endpoint is outside the original node range")
    order = np.argsort(edges[:, 1], kind="stable")
    sources = np.array(edges[order, 0], dtype=np.int64, copy=True)
    counts = np.bincount(edges[:, 1], minlength=nodes)
    offsets = np.concatenate((np.array([0], dtype=np.int64), counts.cumsum(dtype=np.int64)))
    copied_features = np.array(features, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(copied_features).all():
        raise ValueError("Features overflow finite FP32 storage")
    for array in (copied_features, sources, offsets):
        array.flags.writeable = False
    return GraphCSR(copied_features, sources, offsets)


def load_graph_archive(
    path: Path, expected_sha256: str, *, expected_nodes: int = 169343,
    expected_edges: int = 1166243, expected_features: int = 128,
) -> GraphCSR:
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError("Dataset archive SHA256 mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for member in ARCHIVE_MEMBERS:
            if names.count(member) != 1:
                raise ValueError(f"Missing or duplicated required raw member: {member}")
            if archive.getinfo(member).file_size > 1024 * 1024 * 1024:
                raise ValueError("Raw compressed member exceeds declared size bound")

        def read(member: str, dtype, ndmin: int):
            with archive.open(member) as raw, gzip.GzipFile(fileobj=raw) as stream:
                return np.loadtxt(stream, delimiter=",", dtype=dtype, ndmin=ndmin)

        node_count = read(ARCHIVE_MEMBERS[2], np.int64, 1)
        edge_count = read(ARCHIVE_MEMBERS[3], np.int64, 1)
        if node_count.tolist() != [expected_nodes] or edge_count.tolist() != [expected_edges]:
            raise ValueError("Original dataset node/edge counts mismatch")
        features = read(ARCHIVE_MEMBERS[0], np.float32, 2)
        edges = read(ARCHIVE_MEMBERS[1], np.int64, 2)
    if features.shape != (expected_nodes, expected_features):
        raise ValueError("Original node feature dimensions mismatch")
    if edges.shape != (expected_edges, 2):
        raise ValueError("Original directed edge dimensions mismatch")
    return incoming_csr(features, edges)


def make_tiles(offsets: np.ndarray, *, max_rows: int = 32768, max_segments: int = 2048) -> list[Tile]:
    if max_rows < 1 or max_segments < 1:
        raise ValueError("Tile domain limits must be positive")
    if offsets.ndim != 1 or offsets.size < 2 or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("Offsets must be an integer vector with at least one segment")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("Offsets must start at zero and be monotonic")
    if np.any(np.diff(offsets) > max_rows):
        raise ValueError("A single node exceeds the original CSR row domain")
    tiles = []
    begin = 0
    nodes = offsets.size - 1
    while begin < nodes:
        end = begin + 1
        while end < nodes and end - begin < max_segments and offsets[end + 1] - offsets[begin] <= max_rows:
            end += 1
        tiles.append(Tile(begin, end, int(offsets[begin]), int(offsets[end])))
        begin = end
    return tiles


def sum_reference(graph: GraphCSR) -> np.ndarray:
    """Independent FP64 oracle, intended for small CPU qualification fixtures."""
    result = np.zeros((graph.offsets.size - 1, graph.features.shape[1]), dtype=np.float64)
    for node, (begin, end) in enumerate(zip(graph.offsets[:-1], graph.offsets[1:])):
        if end > begin:
            result[node] = graph.features[graph.sources[begin:end]].sum(axis=0, dtype=np.float64)
    return result


def _case_plan(graph: GraphCSR, case_id: str, begin: int, end: int, *, synthetic: bool = False) -> dict:
    offsets = graph.offsets[begin:end + 1] - graph.offsets[begin]
    tiles = make_tiles(offsets)
    degrees = np.diff(offsets)
    return {
        "case_id": case_id, "synthetic_boundary_fixture": synthetic,
        "node_id_begin": begin, "node_id_end": end, "num_nodes": end - begin,
        "num_edges": int(offsets[-1]), "feature_width": graph.features.shape[1],
        "empty_nodes": int(np.count_nonzero(degrees == 0)), "max_degree": int(degrees.max()),
        "tiles": [vars(tile) for tile in tiles],
        "zero_edge_tiles": sum(tile.edge_begin == tile.edge_end for tile in tiles),
        "operator_domain": {"max_rows": 32768, "max_segments": 2048, "feature_width": 128},
    }


def boundary_fixture() -> GraphCSR:
    features = np.arange(5 * 128, dtype=np.float32).reshape(5, 128) / 1024
    return incoming_csr(features, np.array([[0, 1], [0, 1], [2, 1], [1, 4]], dtype=np.int64))


def fixed_weights() -> tuple:
    generator = np.random.default_rng(202610061)
    layers = []
    for _ in range(2):
        matrices = []
        for _ in range(2):
            matrix = generator.uniform(-1, 1, (128, 128)).astype(np.float32)
            matrix *= np.float32(0.125) / np.abs(matrix).sum(axis=0, keepdims=True)
            matrices.append(matrix)
        bias = generator.uniform(-0.001, 0.001, 128).astype(np.float32)
        layers.append((*matrices, bias))
    return tuple(layers)


def prepare_weights(weights: tuple, *, device) -> tuple:
    import torch
    return tuple(tuple(torch.as_tensor(array, device=device) for array in layer) for layer in weights)


def prepare_tensors(graph: GraphCSR, *, device, node_begin: int = 0, node_end: int | None = None) -> TensorInputs:
    import torch
    node_end = graph.features.shape[0] if node_end is None else node_end
    if graph.features.shape[1] != 128 or not 0 <= node_begin < node_end <= graph.features.shape[0]:
        raise ValueError("Original 128-dimensional features and a valid target window required")
    scale = max(1.0, float(np.max(np.abs(graph.features))))
    features = torch.tensor(graph.features / scale, device=device)
    offsets = graph.offsets[node_begin:node_end + 1] - graph.offsets[node_begin]
    sources = graph.sources[graph.offsets[node_begin]:graph.offsets[node_end]]
    degrees = np.diff(offsets)
    destinations = np.repeat(np.arange(node_end - node_begin, dtype=np.int64), degrees)
    tiles = tuple((tile, torch.tensor(
        offsets[tile.node_begin:tile.node_end + 1] - offsets[tile.node_begin],
        dtype=torch.int32, device=device,
    )) for tile in make_tiles(offsets))
    return TensorInputs(
        features, torch.tensor(sources.copy(), device=device),
        torch.tensor(offsets, dtype=torch.int64, device=device),
        torch.tensor(destinations, dtype=torch.int64, device=device),
        torch.tensor(degrees, dtype=torch.float32, device=device),
        node_begin, node_end, tiles,
    )


def stock_aggregation(name: str, inputs: TensorInputs):
    import torch
    if name == "torch.segment_reduce":
        return lambda messages, offsets: torch.segment_reduce(messages, "sum", offsets=offsets, axis=0)
    if name == "torch.index_add_":
        def aggregate(messages, offsets):
            output = messages.new_zeros((inputs.node_end - inputs.node_begin, 128))
            return output.index_add_(0, inputs.destinations, messages)
        return aggregate
    raise ValueError(f"Unknown aggregation backend: {name}")


class SourceBoundCSR:
    """Use the unchanged verified source; no CPU or numerical fallback."""
    def __init__(self, source: Path):
        self.source = Path(source).resolve()
        if sha256(self.source) != CSR_SHA256:
            raise ValueError("Original CSR source SHA256 mismatch")
        self._model = None

    def __call__(self, messages, offsets):
        import torch
        if messages.device.type != "cuda" or offsets.device != messages.device:
            raise ValueError("Original candidate requires CUDA inputs on the same device")
        if (messages.dtype != torch.float32 or offsets.dtype != torch.int32
                or messages.ndim != 2 or messages.shape[1] != 128
                or not 1 <= messages.shape[0] <= 32768 or not 2 <= offsets.numel() <= 2049
                or not messages.is_contiguous() or not offsets.is_contiguous()):
            raise ValueError("Candidate inputs exceed the original verified CSR domain")
        if self._model is None:
            if sha256(self.source) != CSR_SHA256:
                raise ValueError("Original CSR source SHA256 changed before execution")
            spec = importlib.util.spec_from_file_location("source_bound_K15_real_graph", self.source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._model = module.ModelNew()
        return self._model(messages, offsets)[0]


def tiled_aggregation(inputs: TensorInputs, candidate: SourceBoundCSR):
    import torch
    def aggregate(messages, offsets):
        outputs = []
        for tile, local_offsets in inputs.tiles:
            if tile.edge_begin == tile.edge_end:
                # Analytic empty block shared by the application contract; no
                # unsupported N=0 operator launch and no baseline fallback.
                outputs.append(messages.new_zeros((tile.node_end - tile.node_begin, 128)))
            else:
                outputs.append(candidate(messages[tile.edge_begin:tile.edge_end], local_offsets))
        return torch.cat(outputs, dim=0)
    return aggregate


def message_passing(inputs: TensorInputs, weights: tuple, aggregate):
    import torch
    if isinstance(weights[0][0], np.ndarray):
        weights = prepare_weights(weights, device=inputs.features.device)
    hidden = inputs.features
    begin, end = inputs.node_begin, inputs.node_end
    for self_weight, neighbor_weight, bias in weights:
        messages = hidden.index_select(0, inputs.sources)
        total = aggregate(messages, inputs.offsets)
        mean = total / inputs.degrees.clamp(min=1).unsqueeze(1)
        updated = torch.relu(hidden[begin:end] @ self_weight + mean @ neighbor_weight + bias) + hidden[begin:end]
        hidden = updated if begin == 0 and end == hidden.shape[0] else torch.cat((hidden[:begin], updated, hidden[end:]))
    # Return independently owned storage even for a selected target window.
    return hidden[begin:end].clone()


def message_passing_reference(graph: GraphCSR, weights: tuple, *, node_begin: int = 0, node_end: int | None = None) -> np.ndarray:
    from scipy.sparse import csr_matrix
    node_end = graph.features.shape[0] if node_end is None else node_end
    offsets = graph.offsets[node_begin:node_end + 1] - graph.offsets[node_begin]
    sources = graph.sources[graph.offsets[node_begin]:graph.offsets[node_end]]
    adjacency = csr_matrix((np.ones(sources.size, dtype=np.float64), sources, offsets),
                           shape=(node_end - node_begin, graph.features.shape[0]))
    scale = max(1.0, float(np.max(np.abs(graph.features))))
    # Common FP32 conditioning precedes the independent FP64 computation.
    hidden = (graph.features / scale).astype(np.float64)
    degrees = np.maximum(1, np.diff(offsets)).astype(np.float64)[:, None]
    for self_weight, neighbor_weight, bias in weights:
        mean = (adjacency @ hidden) / degrees
        updated = np.maximum(0, hidden[node_begin:node_end] @ self_weight.astype(np.float64)
                             + mean @ neighbor_weight.astype(np.float64) + bias.astype(np.float64)) + hidden[node_begin:node_end]
        hidden = hidden.copy()
        hidden[node_begin:node_end] = updated
    return hidden[node_begin:node_end].copy()


def cpu_plan(graph: GraphCSR, *, archive_sha256: str, operator_sha256: str) -> dict:
    if graph.features.shape[1] != 128:
        raise ValueError("This application fixes feature width at the original 128 dimensions")
    return {
        "schema_version": 1, "status": "GPU_PENDING", "dataset": "ogbn-arxiv",
        "dataset_license": "ODC-BY", "attribution_url": OFFICIAL_DOC,
        "archive_sha256": archive_sha256, "operator_sha256": operator_sha256,
        "edge_semantics": "original directed citations, duplicates retained, no added edges",
        "case_selection": "full graph and fixed original node IDs; no degree or timing selection",
        "cases": [
            _case_plan(graph, "arxiv-full", 0, graph.features.shape[0]),
            _case_plan(graph, "arxiv-first-2048", 0, min(2048, graph.features.shape[0])),
            _case_plan(boundary_fixture(), "boundary-fixture", 0, 5, synthetic=True),
        ],
        "computation": {
            "layers": 2, "dtype": "float32", "weight_seed": 202610061,
            "trained_weights": False, "classification_accuracy_claim": False,
            "operations": "gather, incoming sum, degree mean, seeded projections, ReLU and residual",
            "feature_conditioning": "divide by max(1, maximum absolute original feature)",
            "window_boundary": "update only declared target nodes; external source features stay fixed each layer",
            "weight_conditioning": "each projection column absolute sum 0.125; bias bounded by 0.001",
            "tf32": False,
        },
        "baselines": ["torch.segment_reduce", "torch.index_add_"],
        "numerical_oracle": "independent FP64 accumulation",
        "tolerance": {"rtol": 0.0003, "atol": 0.00001},
        "paired_timing_rounds": 30, "gpu_used": False, "model_api_calls": 0,
        "timings": {}, "rsi_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--operator-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite an existing plan")
    if sha256(args.operator_source) != CSR_SHA256:
        parser.error("Original CSR source SHA256 mismatch")
    graph = load_graph_archive(args.archive, args.archive_sha256)
    result = cpu_plan(graph, archive_sha256=args.archive_sha256, operator_sha256=CSR_SHA256)
    result["source_sha256"] = sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": [
        {k: case[k] for k in ("case_id", "num_nodes", "num_edges", "max_degree", "empty_nodes")}
        | {"tile_count": len(case["tiles"])} for case in result["cases"]
    ], "gpu_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
