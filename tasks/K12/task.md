# K12 - small_m_gemm

out = cast_fp16(A @ Bmat) for very small M with fp32 accumulation.

## Contract

| field | value |
|---|---|
| task_id | `K12` |
| group | advanced |
| backend | triton |
| output arity | 1 |
| accumulation dtype | float32 |
| layout | contiguous |
| mutates inputs | false |
| rtol / atol | 0.02 / 0.02 |
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
| 1 | `A` | `float16` | `MxK` |
| 2 | `Bmat` | `float16` | `KxN` |

## Outputs

| # | name | dtype | shape |
|---|---|---|---|
| 1 | `y` | `float16` | `MxN` |

## Supported shape range

| dim | range |
|---|---|
| `M` | [1, 8] |
| `N` | [128, 11008] |
| `K` | [128, 11008] |

Development timing points (visible during the search):

- M=1, N=4096, K=4096
- M=4, N=4096, K=4096
- M=8, N=11008, K=4096

Reserved timing points (measured only at final verification, never fed back):

- M=2, N=11008, K=4096
- M=8, N=8192, K=4096

## Correctness coverage

- 12 development correctness cases and 21 final verification cases.
- families: ['large', 'plain', 'tiny'] (development), ['large', 'mixed_sign', 'plain', 'tiny'] (final)
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

-
