"""Input generation for K19 (resize_normalize).

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
STATIC_SHAPE_KEYS = ['Ho', 'Wo']


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
    b, c, h, w = shape["B"], shape["C"], shape["H"], shape["W"]
    x = cu.draw((b, c, h, w), gen, family, abs_max, torch.float32)
    mean = (0.5 * cu.randn((c,), gen)).to(torch.float32)
    std = (torch.rand(c, generator=gen, dtype=torch.float32) * 1.5 + 0.5)
    inputs = (x, mean, std)
    return cu.to_device(tuple(inputs), device)


DEV_CORRECTNESS = [{'case_id': 'd00', 'shape': {'B': 1, 'C': 3, 'H': 16, 'W': 16, 'Ho': 8, 'Wo': 8}, 'family': 'plain'}, {'case_id': 'd01', 'shape': {'B': 1, 'C': 3, 'H': 16, 'W': 16, 'Ho': 8, 'Wo': 8}, 'family': 'large'}, {'case_id': 'd02', 'shape': {'B': 1, 'C': 3, 'H': 33, 'W': 47, 'Ho': 17, 'Wo': 23}, 'family': 'uniform'}, {'case_id': 'd03', 'shape': {'B': 1, 'C': 3, 'H': 33, 'W': 47, 'Ho': 17, 'Wo': 23}, 'family': 'plain'}, {'case_id': 'd04', 'shape': {'B': 1, 'C': 3, 'H': 480, 'W': 640, 'Ho': 224, 'Wo': 224}, 'family': 'large'}, {'case_id': 'd05', 'shape': {'B': 1, 'C': 3, 'H': 480, 'W': 640, 'Ho': 224, 'Wo': 224}, 'family': 'uniform'}, {'case_id': 'd06', 'shape': {'B': 4, 'C': 3, 'H': 224, 'W': 224, 'Ho': 112, 'Wo': 112}, 'family': 'plain'}, {'case_id': 'd07', 'shape': {'B': 4, 'C': 3, 'H': 224, 'W': 224, 'Ho': 112, 'Wo': 112}, 'family': 'large'}, {'case_id': 'd08', 'shape': {'B': 1, 'C': 3, 'H': 720, 'W': 1280, 'Ho': 384, 'Wo': 384}, 'family': 'uniform'}, {'case_id': 'd09', 'shape': {'B': 1, 'C': 3, 'H': 720, 'W': 1280, 'Ho': 384, 'Wo': 384}, 'family': 'plain'}, {'case_id': 'd10', 'shape': {'B': 2, 'C': 3, 'H': 113, 'W': 97, 'Ho': 57, 'Wo': 41}, 'family': 'large'}, {'case_id': 'd11', 'shape': {'B': 2, 'C': 3, 'H': 113, 'W': 97, 'Ho': 57, 'Wo': 41}, 'family': 'uniform'}]
FINAL_CORRECTNESS = [{'case_id': 'f00', 'shape': {'B': 1, 'C': 3, 'H': 16, 'W': 16, 'Ho': 8, 'Wo': 8}, 'family': 'plain'}, {'case_id': 'f01', 'shape': {'B': 1, 'C': 3, 'H': 16, 'W': 16, 'Ho': 8, 'Wo': 8}, 'family': 'large'}, {'case_id': 'f02', 'shape': {'B': 1, 'C': 3, 'H': 16, 'W': 16, 'Ho': 8, 'Wo': 8}, 'family': 'uniform'}, {'case_id': 'f03', 'shape': {'B': 2, 'C': 3, 'H': 113, 'W': 97, 'Ho': 57, 'Wo': 41}, 'family': 'mixed_sign'}, {'case_id': 'f04', 'shape': {'B': 2, 'C': 3, 'H': 113, 'W': 97, 'Ho': 57, 'Wo': 41}, 'family': 'plain'}, {'case_id': 'f05', 'shape': {'B': 2, 'C': 3, 'H': 113, 'W': 97, 'Ho': 57, 'Wo': 41}, 'family': 'large'}, {'case_id': 'f06', 'shape': {'B': 1, 'C': 3, 'H': 720, 'W': 1280, 'Ho': 384, 'Wo': 384}, 'family': 'uniform'}, {'case_id': 'f07', 'shape': {'B': 1, 'C': 3, 'H': 720, 'W': 1280, 'Ho': 384, 'Wo': 384}, 'family': 'mixed_sign'}, {'case_id': 'f08', 'shape': {'B': 1, 'C': 3, 'H': 720, 'W': 1280, 'Ho': 384, 'Wo': 384}, 'family': 'plain'}, {'case_id': 'f09', 'shape': {'B': 1, 'C': 3, 'H': 65, 'W': 259, 'Ho': 129, 'Wo': 37}, 'family': 'large'}, {'case_id': 'f10', 'shape': {'B': 1, 'C': 3, 'H': 65, 'W': 259, 'Ho': 129, 'Wo': 37}, 'family': 'uniform'}, {'case_id': 'f11', 'shape': {'B': 1, 'C': 3, 'H': 65, 'W': 259, 'Ho': 129, 'Wo': 37}, 'family': 'mixed_sign'}, {'case_id': 'f12', 'shape': {'B': 3, 'C': 3, 'H': 127, 'W': 129, 'Ho': 63, 'Wo': 65}, 'family': 'plain'}, {'case_id': 'f13', 'shape': {'B': 3, 'C': 3, 'H': 127, 'W': 129, 'Ho': 63, 'Wo': 65}, 'family': 'large'}, {'case_id': 'f14', 'shape': {'B': 3, 'C': 3, 'H': 127, 'W': 129, 'Ho': 63, 'Wo': 65}, 'family': 'uniform'}, {'case_id': 'f15', 'shape': {'B': 4, 'C': 3, 'H': 224, 'W': 224, 'Ho': 112, 'Wo': 112}, 'family': 'mixed_sign'}, {'case_id': 'f16', 'shape': {'B': 4, 'C': 3, 'H': 224, 'W': 224, 'Ho': 112, 'Wo': 112}, 'family': 'plain'}, {'case_id': 'f17', 'shape': {'B': 4, 'C': 3, 'H': 224, 'W': 224, 'Ho': 112, 'Wo': 112}, 'family': 'large'}, {'case_id': 'f18', 'shape': {'B': 1, 'C': 3, 'H': 480, 'W': 640, 'Ho': 224, 'Wo': 224}, 'family': 'uniform'}, {'case_id': 'f19', 'shape': {'B': 1, 'C': 3, 'H': 480, 'W': 640, 'Ho': 224, 'Wo': 224}, 'family': 'mixed_sign'}, {'case_id': 'f20', 'shape': {'B': 1, 'C': 3, 'H': 480, 'W': 640, 'Ho': 224, 'Wo': 224}, 'family': 'plain'}]
DEV_TIMING = [{'case_id': 't0', 'shape': {'B': 1, 'C': 3, 'H': 480, 'W': 640, 'Ho': 224, 'Wo': 224}, 'family': 'plain'}, {'case_id': 't1', 'shape': {'B': 4, 'C': 3, 'H': 224, 'W': 224, 'Ho': 112, 'Wo': 112}, 'family': 'plain'}, {'case_id': 't2', 'shape': {'B': 1, 'C': 3, 'H': 720, 'W': 1280, 'Ho': 384, 'Wo': 384}, 'family': 'plain'}]
RESERVED_TIMING = [{'case_id': 'r0', 'shape': {'B': 2, 'C': 3, 'H': 113, 'W': 97, 'Ho': 57, 'Wo': 41}, 'family': 'plain'}, {'case_id': 'r1', 'shape': {'B': 1, 'C': 3, 'H': 719, 'W': 1279, 'Ho': 359, 'Wo': 383}, 'family': 'plain'}]
