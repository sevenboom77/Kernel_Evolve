"""New application candidate with every frozen strong library control retained."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
STRONG_SHA = '07a50890a51e60efeb65a9ae98f6404b4a67dd3b417f941ea0f82296a01679b9'
SPLIT_SHA = '045663fab1a66b8b24944234babb65d27dc5827149debb4899d08081097fa937'
STRONG_BASELINES = ('segment_reduce', 'index_add', 'sparse_mm',
                    'compiled_segment_reduce', 'compiled_index_add')


def source_bound(name, path, expected):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen graph control/candidate source changed')
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
        raise ValueError('Graph source module belongs to another path')
    return module


def frozen_controls():
    return source_bound('frozen_graph_strong_controls_for_fused',
                        HERE / 'citation_graph_strong_controls.py', STRONG_SHA)


def original():
    return frozen_controls().original()


def split_candidate(path):
    if Path(path).resolve() != HERE / 'citation_graph_split_csr.py':
        raise ValueError('Only the independently source-bound new graph candidate is accepted')
    return source_bound('frozen_graph_split_csr_R1', path, SPLIT_SHA)


def prepare_adjacency(*args, **kwargs):
    return frozen_controls().prepare_adjacency(*args, **kwargs)


def prepare_backend(name, inputs, weights, *, adjacency=None, candidate_source=None,
                    compile_factory=None):
    if name != 'candidate':
        return frozen_controls().prepare_backend(name, inputs, weights, adjacency=adjacency,
            candidate_source=candidate_source, compile_factory=compile_factory)
    if candidate_source is None:
        raise ValueError('New fused graph candidate source required')
    candidate = split_candidate(candidate_source)
    weights = original().prepare_weights(weights, device=inputs.features.device)
    model = candidate.GraphModel(inputs, weights).eval()
    try:
        factory = torch.compile if compile_factory is None else compile_factory
        model = factory(model, mode='default', fullgraph=True, dynamic=False)
    except Exception as exc:
        return None, {'backend': name, 'compiled_requested': True, 'status': 'UNAVAILABLE',
            'error_type': type(exc).__name__, 'diagnostic': str(exc)[:4000], 'fallback_used': False}
    return model, {'backend': name, 'compiled_requested': True,
        'status': 'PREPARED_UNQUALIFIED', 'fallback_used': False,
        'candidate_source_sha256': SPLIT_SHA, 'source_author': 'Codex',
        'original_K15_source_or_verified_domain_reused': False,
        'gather_and_reduction_fused': True, 'candidate_launches_per_layer': 4}


def strong_gain(samples):
    return frozen_controls().strong_gain(samples)


def cpu_plan(*args, **kwargs):
    plan = frozen_controls().cpu_plan(*args, **kwargs)
    plan.update(protocol='real-citation-graph-split-CSR-controls-v1',
        original_K15_operator_sha256=original().CSR_SHA256,
        operator_sha256=SPLIT_SHA, candidate_source_author='Codex',
        new_candidate_domain={'dtype': 'float32', 'feature_width': 128,
            'maximum_source_nodes': 169343, 'maximum_edges': 1166243,
            'maximum_target_segments': 169343, 'maximum_degree': 13155},
        original_K15_verification_is_lineage_only=True,
        new_candidate_qualification='actual aggregation plus complete computation on all three fixed cases',
        candidate_launches_per_layer=4, candidate_materializes_messages=False,
        primary_AGH_RSI_evidence=False, schedule_preparation='once outside paired inference timing; included in accounted wall span', candidate_compiled=True)
    return plan
