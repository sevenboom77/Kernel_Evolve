# K19 - resize_normalize

NCHW bilinear resize to (Ho, Wo) with align_corners=False (PyTorch semantics), followed by per-channel (y - mean[c]) / std[c] in fp32.

## Contract

| field | value |
|---|---|
| task_id | `K19` |
| group | advanced |
| backend | triton |
| output arity | 1 |
| accumulation dtype | float32 |
| layout | contiguous |
| mutates inputs | false |
| rtol / atol | 0.0001 / 1e-05 |
| candidates per task | 12 |
| stagnation stop | 5 consecutive non-improving candidates |
| finite inputs only | true |

## Signature

```python
class ModelNew(torch.nn.Module):
    def __init__(self, **static_params): ...
    def forward(self, *inputs) -> tuple[torch.Tensor, ...]
```

Static parameters for this task: `{}`.
The argument order is exactly the input order below; outputs are returned in the
order of the output table.

## Inputs

| # | name | dtype | shape |
|---|---|---|---|
| 1 | `x` | `float32` | `BxCxHxW` |
| 2 | `mean` | `float32` | `C` |
| 3 | `std` | `float32` | `C` |

## Outputs

| # | name | dtype | shape |
|---|---|---|---|
| 1 | `y` | `float32` | `BxCxHoxWo` |

## Supported shape range

| dim | range |
|---|---|
| `B` | [1, 4] |
| `C` | [3, 3] |
| `H` | [16, 720] |
| `W` | [16, 1280] |
| `Ho` | [8, 384] |
| `Wo` | [8, 384] |

Development timing points (visible during the search):

- B=1, C=3, H=480, W=640, Ho=224, Wo=224
- B=4, C=3, H=224, W=224, Ho=112, Wo=112
- B=1, C=3, H=720, W=1280, Ho=384, Wo=384

Reserved timing points (measured only at final verification, never fed back):

- B=2, C=3, H=113, W=97, Ho=57, Wo=41
- B=1, C=3, H=719, W=1279, Ho=359, Wo=383

## Correctness coverage

- 12 development correctness cases and 21 final verification cases.
- families: ['large', 'plain', 'uniform'] (development), ['large', 'mixed_sign', 'plain', 'uniform'] (final)
- dtype, shape, layout and (where relevant) contiguity of every output are checked,
  not just a few sampled values.

## Rules

- The tensor maths must be implemented in Triton. Python may allocate outputs and
  query metadata; it may not wrap the reference, wrap an existing baseline, or
  read precomputed answers.
- Per-shape dispatch is allowed but every dispatch branch must be exercised by the
  tests; one submission is one candidate snapshot.
- Inputs must not be mutated and outputs must not alias inputs.
- `forward` must return a tuple, even for a single output.
- At most 4 launch configurations per source candidate; each counts against the
  variant budget and against GPU time.
- Do not change the semantics, dtype, tolerance or shape support of this task.

## Notes

std is strictly positive. The sampler must match PyTorch's align_corners=False rule (clamped source coordinates) at the borders.
- The checker also probes border pixels against PyTorch's align_corners=False sampler.
