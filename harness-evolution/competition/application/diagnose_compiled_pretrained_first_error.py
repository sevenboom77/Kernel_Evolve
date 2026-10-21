"""Source-bound compiled diagnostic primitives; no GPU admission or timing here."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ORIGINAL_SOURCE_SHA256 = {
    "fused_welford_compiled.py": "dab446dfa18dd0bdb7be08bc02f292b90c245a7c35fd57b5f72142119436db77",
    "benchmark_pretrained_gpt2_welford_compiled.py": "877cc87d5d2896787a9ad3739d22a4583ee615a7fd03cbee1663659166a308d7",
    "benchmark_pretrained_gpt2.py": "f2152dcaf06bb643df35b50292b3fcd0399457a9771e7420c071860bd90ca1a9",
    "pretrained_gpt2.py": "eeab9a6743cf98390ebe9c23955f38cee8b4ac2da2d5309e32576d4a0089f0cc",
}


class HiddenLogits(torch.nn.Module):
    """Request native model states; never add hooks or alter arithmetic."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        output = self.model(input_ids=ids, use_cache=False, output_attentions=False,
                            output_hidden_states=True, return_dict=False)
        return output[0], output[1]


def tensor_error(actual, expected) -> dict:
    if actual.shape != expected.shape or not actual.numel():
        return {"status": "WRONG_SHAPE", "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    if actual.dtype != expected.dtype or actual.device != expected.device:
        return {"status": "WRONG_DTYPE_OR_DEVICE", "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
        return {"status": "NONFINITE", "shape": list(actual.shape)}
    delta = (actual.float() - expected.float()).abs()
    # Compare actual bytes, including signed zero, rather than numerical ==.
    equal = torch.equal(actual.detach().contiguous().view(torch.uint8),
                        expected.detach().contiguous().view(torch.uint8))
    return {"status": "FINITE", "shape": list(actual.shape), "dtype": str(actual.dtype),
            "max_absolute_error": delta.max().item(), "mean_squared_error": delta.square().mean().item(),
            "bitwise_equal": equal}


def assess_trace(original_compiled_logits, profiled_logits, actual_hidden, reference_hidden, *, expected_hidden_states=13) -> dict:
    if (expected_hidden_states < 2 or len(actual_hidden) != expected_hidden_states
            or len(reference_hidden) != expected_hidden_states):
        raise ValueError("Original hidden-state count must match on both paths")
    profile_error = tensor_error(profiled_logits, original_compiled_logits)
    fidelity = profile_error.get("bitwise_equal") is True
    errors = []
    for index, (actual, reference) in enumerate(zip(actual_hidden, reference_hidden)):
        stage = "after_final_layernorm" if index == expected_hidden_states - 1 else f"before_block_{index}"
        errors.append({"hidden_state_index": index, "stage": stage, **tensor_error(actual, reference)})
    valid = all(row["status"] == "FINITE" for row in errors)
    first = next(({"hidden_state_index": row["hidden_state_index"], "stage": row["stage"]}
                  for row in errors if row.get("bitwise_equal") is not True), None) if fidelity and valid else None
    return {
        "status": ("INVALID_HIDDEN_STATE_TRACE" if not valid else
                   "PROFILE_FAITHFUL_HIDDEN_STATES_RECORDED" if fidelity else "DIAGNOSTIC_PROFILE_CHANGED_OUTPUT"),
        "profile_final_logits_bitwise_equal": fidelity, "profile_final_logits_error": profile_error,
        "hidden_state_errors": errors, "first_bitwise_difference": first,
        "semantics": "positions before each block, followed by final normalized hidden state",
        "last_block_pre_final_layernorm_output_captured": False,
        "root_cause_proven": False, "performance_measured": False,
    }


def observe_backend(original_logits, ids, reference_logits, reference_hidden, *, qualify_logits,
                    checkpoint, expected_hidden_states=13, compile_factory=None) -> dict:
    """Call only from an admitted numerical job under its original wall limit.

    The caller supplies unchanged eager references, original correctness logic
    and live budget checks. This helper supplies no resource admission or timing.
    """
    import torch._inductor.config as config
    compile_factory = torch.compile if compile_factory is None else compile_factory
    options = {"mode": "default", "fullgraph": True, "dynamic": False}
    checkpoint()
    with config.patch({"emulate_precision_casts": True}), torch.inference_mode():
        original = compile_factory(original_logits, **options)
        original_output = original(ids)
        checkpoint()
        qualification = qualify_logits(original_output, reference_logits)
        profiled = compile_factory(HiddenLogits(original_logits.model), **options)
        profiled_output, actual_hidden = profiled(ids)
        checkpoint()
        trace = assess_trace(original_output, profiled_output, actual_hidden, reference_hidden,
                             expected_hidden_states=expected_hidden_states)
    return {"original_uninstrumented_qualification": qualification, "trace": trace,
            "compiler_options": options, "emulate_precision_casts": True,
            "resource_admission_required_from_caller": True, "performance_measured": False}


def cpu_plan() -> dict:
    sources = []
    for name, expected_sha256 in ORIGINAL_SOURCE_SHA256.items():
        path = HERE / name
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("The original compiler/model source changed")
        sources.append({"path": str(path), "sha256": actual_sha256})
    return {
        "schema_version": 1, "status": "GPU_PENDING", "source_bindings": sources,
        "cases": [{"case_id": label, "batch_size": batch, "sequence_length": length}
                  for label, batch, length in (("B1_S33", 1, 33), ("B1_S128", 1, 128),
                                                ("B1_S512", 1, 512), ("B2_S1024", 2, 1024))],
        "backends": ["compiled_fp16", "compiled_candidate"], "original_hidden_state_count": 13,
        "instrumentation_fidelity": "final output bytes equal original uninstrumented compiled logits",
        "unchanged_original_logit_tolerance": {"rtol": 0.005, "atol": 0.05},
        "unchanged_original_kernel_tolerance": {"rtol": 0.002, "atol": 0.01},
        "compiler_profile": {"emulate_precision_casts": True, "native_layernorm_both_backends": True,
                             "fullgraph": True, "dynamic": False, "mode": "default"},
        "GPU_used": False, "model_API_calls": 0, "timings": {}, "root_cause_proven": False,
        "GPU_execution_supported_by_this_entrypoint": False,
        "admission_required": "complete V19 primary closure, fresh carried accounting, clean GPU and original numeric lock",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refuse to overwrite an existing diagnostic plan")
    result = cpu_plan()
    result["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "GPU_used": False,
                      "root_cause_proven": False, "GPU_execution_supported_by_this_entrypoint": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
