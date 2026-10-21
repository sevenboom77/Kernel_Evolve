"""Input generation for K12 (small_m_gemm).

Every case is a pure function of (case_id, run_seed). The shape/family skeleton
is visible to an optimisation worker; the concrete seed used for the reserved
final verification lives in the run protocol, not in this package.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from kernel_lab import caseutil as cu

SPEC = json.loads((Path(__file__).with_name("spec.json")).read_text())
TASK_ID = SPEC["task_id"]
STATIC_SHAPE_KEYS = []


def static_params():
    return dict(SPEC["static_params"])


def case_static(case):
    sp = static_params()
    sp.update(case.get("static", {}))
    for key in STATIC_SHAPE_KEYS:
        if key in case.get("shape", {}):
            sp[key] = case["shape"][key]
    return sp


def make_inputs(case, device="cpu", run_seed=None):
    """Build the input tuple for one case on ``device``."""
    shape = dict(case["shape"])
    family = case.get("family", "plain")
    ap = SPEC["shape_params"]
    abs_max = float(SPEC["finite_input_abs_max"])
    gen = cu.generator(cu.derive_seed(case, run_seed))
    if ap:
        shape.update({k: v for k, v in ap.items() if k not in shape})
        family = case.get("family", "plain")
    m, n, k = shape["M"], shape["N"], shape["K"]
    a = cu.draw((m, k), gen, family, abs_max, torch.float16)
    b = cu.draw((k, n), gen, family, abs_max, torch.float16)
    inputs = (a, b)
    return cu.to_device(tuple(inputs), device)


DEV_CORRECTNESS = [{'case_id': 'd00', 'shape': {'M': 1, 'N': 128, 'K': 128}, 'family': 'plain'}, {'case_id': 'd01', 'shape': {'M': 1, 'N': 128, 'K': 128}, 'family': 'tiny'}, {'case_id': 'd02', 'shape': {'M': 1, 'N': 4096, 'K': 4096}, 'family': 'large'}, {'case_id': 'd03', 'shape': {'M': 1, 'N': 4096, 'K': 4096}, 'family': 'plain'}, {'case_id': 'd04', 'shape': {'M': 4, 'N': 4096, 'K': 4096}, 'family': 'tiny'}, {'case_id': 'd05', 'shape': {'M': 4, 'N': 4096, 'K': 4096}, 'family': 'large'}, {'case_id': 'd06', 'shape': {'M': 8, 'N': 11008, 'K': 4096}, 'family': 'plain'}, {'case_id': 'd07', 'shape': {'M': 8, 'N': 11008, 'K': 4096}, 'family': 'tiny'}, {'case_id': 'd08', 'shape': {'M': 2, 'N': 11008, 'K': 4096}, 'family': 'large'}, {'case_id': 'd09', 'shape': {'M': 2, 'N': 11008, 'K': 4096}, 'family': 'plain'}, {'case_id': 'd10', 'shape': {'M': 8, 'N': 8192, 'K': 4096}, 'family': 'tiny'}, {'case_id': 'd11', 'shape': {'M': 8, 'N': 8192, 'K': 4096}, 'family': 'large'}]
FINAL_CORRECTNESS = [{'case_id': 'f00', 'shape': {'M': 1, 'N': 129, 'K': 257}, 'family': 'plain'}, {'case_id': 'f01', 'shape': {'M': 1, 'N': 129, 'K': 257}, 'family': 'tiny'}, {'case_id': 'f02', 'shape': {'M': 1, 'N': 129, 'K': 257}, 'family': 'large'}, {'case_id': 'f03', 'shape': {'M': 1, 'N': 11008, 'K': 11008}, 'family': 'mixed_sign'}, {'case_id': 'f04', 'shape': {'M': 1, 'N': 11008, 'K': 11008}, 'family': 'plain'}, {'case_id': 'f05', 'shape': {'M': 1, 'N': 11008, 'K': 11008}, 'family': 'tiny'}, {'case_id': 'f06', 'shape': {'M': 2, 'N': 11008, 'K': 4096}, 'family': 'large'}, {'case_id': 'f07', 'shape': {'M': 2, 'N': 11008, 'K': 4096}, 'family': 'mixed_sign'}, {'case_id': 'f08', 'shape': {'M': 2, 'N': 11008, 'K': 4096}, 'family': 'plain'}, {'case_id': 'f09', 'shape': {'M': 8, 'N': 8192, 'K': 4096}, 'family': 'tiny'}, {'case_id': 'f10', 'shape': {'M': 8, 'N': 8192, 'K': 4096}, 'family': 'large'}, {'case_id': 'f11', 'shape': {'M': 8, 'N': 8192, 'K': 4096}, 'family': 'mixed_sign'}, {'case_id': 'f12', 'shape': {'M': 4, 'N': 4097, 'K': 2049}, 'family': 'plain'}, {'case_id': 'f13', 'shape': {'M': 4, 'N': 4097, 'K': 2049}, 'family': 'tiny'}, {'case_id': 'f14', 'shape': {'M': 4, 'N': 4097, 'K': 2049}, 'family': 'large'}, {'case_id': 'f15', 'shape': {'M': 8, 'N': 11008, 'K': 4096}, 'family': 'mixed_sign'}, {'case_id': 'f16', 'shape': {'M': 8, 'N': 11008, 'K': 4096}, 'family': 'plain'}, {'case_id': 'f17', 'shape': {'M': 8, 'N': 11008, 'K': 4096}, 'family': 'tiny'}, {'case_id': 'f18', 'shape': {'M': 1, 'N': 4096, 'K': 4096}, 'family': 'large'}, {'case_id': 'f19', 'shape': {'M': 1, 'N': 4096, 'K': 4096}, 'family': 'mixed_sign'}, {'case_id': 'f20', 'shape': {'M': 1, 'N': 4096, 'K': 4096}, 'family': 'plain'}]
DEV_TIMING = [{'case_id': 't0', 'shape': {'M': 1, 'N': 4096, 'K': 4096}, 'family': 'plain'}, {'case_id': 't1', 'shape': {'M': 4, 'N': 4096, 'K': 4096}, 'family': 'plain'}, {'case_id': 't2', 'shape': {'M': 8, 'N': 11008, 'K': 4096}, 'family': 'plain'}]
RESERVED_TIMING = [{'case_id': 'r0', 'shape': {'M': 2, 'N': 11008, 'K': 4096}, 'family': 'plain'}, {'case_id': 'r1', 'shape': {'M': 8, 'N': 8192, 'K': 4096}, 'family': 'plain'}]
