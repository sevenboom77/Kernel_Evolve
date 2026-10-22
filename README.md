# KernelEvolve

### Fused GPU operators and an AGH optimization plugin

KernelEvolve connects a natural-language optimization request to executable GPU work. The public source contains reusable operator implementations, a native AGH plugin and the fixed local evaluation service used by that plugin. A standalone benchmark lets you call the operators directly and inspect numerical results and timing on your own GPU.

[![KernelEvolve walkthrough](presentation/rsi-loop.gif)](presentation/kernel-evolve.mp4)

[Watch the demonstration](presentation/kernel-evolve.mp4) · [Run the operators](BENCHMARKING.md) · [Connect AGH](docs/plugin-flow.md)

## The work starts with a concrete computation

GPU optimization has two practical questions: can several memory-heavy steps become one kernel, and does a proposed implementation still compute the required result? The source addresses both questions with two complementary implementations.

| Implementation | Computation | Main design choice |
|---|---|---|
| Fused CSR gather and reduction | Sum neighbor features for each graph node | Fuse source gathering with segment reduction; divide work by node degree |
| Small-M GEMM | Multiply matrices with very few output rows | A tiled Triton kernel with FP16 inputs/output and FP32 accumulation |

The graph kernel avoids constructing the full gathered edge-feature matrix. The Small-M kernel handles the low-row-count shape family through a single tiled launch. Their source files are independent of the model runtime, so they can be imported directly into Python applications.

## Inside the graph fusion

The graph input is stored in CSR form: `sources` identifies each edge's source node, while `offsets` marks each destination node's edge range. Features have width 128 and use FP32.

An unfused implementation first gathers all edge features and then reduces the rows for each destination. The fused implementation loads each source feature directly inside the reduction kernel.

The workload is partitioned into three groups:

- **Degree 0–8:** short-row kernels process several destination nodes together.
- **Degree 9–32:** a second short-row configuration handles medium rows.
- **Degree above 32:** 128-edge chunks compute partial sums in parallel; a merge kernel combines them.

`build_schedule(offsets)` creates the reusable degree schedule. `aggregate(features, sources, offsets, schedule)` runs the fused aggregation. `message_passing(...)` also connects the aggregation to neighbor averaging, feature projections, a bias, ReLU and a residual update.

```python
from operators.graph.csr_gather_sum import aggregate, build_schedule

schedule = build_schedule(offsets)
neighbor_sum = aggregate(features, sources, offsets, schedule)
```

Use contiguous CUDA tensors on the same device: FP32 node features and int64 CSR indices. The source implements the width-128 graph profile; its validation function specifies the supported sizes and layouts. Callers must also supply valid CSR data: offsets start at zero, are nondecreasing, end at `len(sources)`, and every source index is within the feature tensor. Reuse a schedule only with the offsets from which it was built; rebuild it after changing the graph.

## Inside Small-M

The matrix kernel accepts contiguous FP16 `A[M, K]` and `B[K, N]` and returns one FP16 output tensor as a tuple. Each program computes an output tile, loads masked blocks along K, accumulates with `tl.dot` in FP32 and stores the final FP16 result.

```python
from operators.small_m.kernel import ModelNew

y, = ModelNew()(a, b)
```

The packaged K12 task covers M values 1, 2, 4 and 8, including non-tile-aligned shapes and different input families. The reference, test cases and tolerances live beside the task definition.

## A first local run

Prepare Linux, Python 3.11 and an NVIDIA GPU with a CUDA-compatible driver:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install triton==3.1.0 numpy scipy
```

First inspect the planned inputs. These commands use the Python standard library:

```bash
python benchmarks/run.py --operator graph --out outputs/graph-plan.json
python benchmarks/run.py --operator small-m --out outputs/small-m-plan.json
```

Then execute correctness checks and CUDA-event timing:

```bash
python benchmarks/run.py --operator graph --execute --gpu 0 --out outputs/graph-gpu.json
python benchmarks/run.py --operator small-m --execute --gpu 0 --out outputs/small-m-gpu.json
```

Choose a new output file for each execution. Results contain the input settings, tolerance, GPU name, raw timing samples and medians. The graph example generates synthetic inputs with empty, short and high-degree rows. Small-M checks all 21 packaged correctness cases before measuring its five timing shapes.

The default measurement uses 20 warmup calls and 30 paired rounds, with 20 calls averaged per sample. Input creation, compilation and reusable graph scheduling happen before timing. [BENCHMARKING.md](BENCHMARKING.md) describes the comparison and measured boundaries.

## Let AGH work with the evaluator

The native plugin exposes five tools:

| Tool | Purpose |
|---|---|
| `kernel_get_task` | Read a numerical task and its requirements |
| `kernel_submit` | Submit a candidate implementation |
| `kernel_get_job` | Query its evaluation job |
| `kernel_get_task_status` | Inspect candidates and current selection |
| `kernel_read_feedback` | Read compilation, correctness and timing results |

The plugin calls the fixed `kernel_lab` service. That service queues submitted candidates, executes compilation and numerical checks, measures GPU time and writes structured feedback. It is the local implementation required to make the plugin runnable.

Use Node.js 24.10 or later and an AGH installation supporting native plugins. See [the plugin guide](docs/plugin-flow.md) for the compatible G031 source and installation commands. Configure the plugin with the Python interpreter and repository root:

```bash
python tools/configure_kernel_tools.py --run-id optimize_001
export KERNEL_TOOLS_CONFIG="$PWD/outputs/kernel-tools.json"
```

Export this setting in the terminal that starts AGH, then install and enable `plugins/kernel-workflow/`. Prepare the K12 baseline and run the evaluator in another terminal:

```bash
python -m kernel_lab baseline --run optimize_001 --tasks K12
python -m kernel_lab serve-evaluator --run optimize_001 --gpu 0 --keep-alive
```

A task can ask the agent to read K12, inspect the included kernel, submit a candidate and revise it using the returned numerical and timing feedback. The packaged task interface includes K12 Small-M and K19 resize/normalization. The graph-fusion example has a separate direct Python API and standalone benchmark.

See [the plugin guide](docs/plugin-flow.md) for the complete flow. Model credentials are configured in your AGH installation.

## What is implemented in this repository

The engineering work connects operator code with its practical use:

1. Tiled Small-M matrix multiplication with masked loads and FP32 accumulation.
2. Fused graph gathering/reduction with separate short-row and split-reduction paths.
3. Reusable degree scheduling and a graph message-passing integration function.
4. Independent references, correctness cases and CUDA-event comparison commands.
5. Five native AGH tools for task access, submission and feedback.
6. A fixed local evaluator with job state, numerical checks and candidate selection.

```text
operators/
  graph/csr_gather_sum.py    fused graph aggregation and message passing
  small_m/kernel.py         Small-M Triton implementation
plugins/kernel-workflow/    native AGH tools
benchmarks/run.py           direct correctness and timing entry point
kernel_lab/                fixed evaluator used by the plugin
tasks/K12/                 Small-M reference, cases and baselines
tasks/K19/                 resize/normalization task package
configs/                   local evaluation configuration
tools/                     plugin configuration helper
docs/                      agent/evaluator integration guide
presentation/              walkthrough video and preview
```

Start with `operators/` to read the GPU implementations. Follow `benchmarks/run.py` for direct use, or follow the [plugin data flow](docs/plugin-flow.md) for agent-driven evaluation.

## Demo and licensing

The [walkthrough video](presentation/kernel-evolve.mp4) presents the project in use. A standalone [video player](presentation/index.html) is also included.


Project licensing is in [LICENSE](LICENSE). Dependency roles and licensing are described in [DEPENDENCIES.md](DEPENDENCIES.md).

This source layout follows the updated project package. The two operator implementations and evaluation core are unchanged from the earlier repository; the direct benchmark and delivery layout are new. Historical experiments and the previously bundled AGH source remain available in [v0.1.0](https://github.com/sevenboom77/Kernel_Evolve/releases/tag/v0.1.0). See [the update notes](docs/update-20261022.md) for the comparison and validation scope.
