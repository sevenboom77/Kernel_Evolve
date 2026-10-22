# Running the operators

Use Linux, Python 3.11, PyTorch 2.5.1+cu121 and Triton 3.1.0. CUDA compilation and timing run on the NVIDIA GPU selected by `--gpu`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install triton==3.1.0 numpy scipy
```

## Inspect the inputs first

These commands use the Python standard library and write the selected cases and tolerances:

```bash
python benchmarks/run.py --operator small-m --out outputs/small-m-plan.json
python benchmarks/run.py --operator graph --out outputs/graph-plan.json
```

Choose a new output file for each run.

## Check correctness and measure

```bash
python benchmarks/run.py --operator small-m --execute --gpu 0 --out outputs/small-m-gpu.json
python benchmarks/run.py --operator graph --execute --gpu 0 --out outputs/graph-gpu.json
```

Small-M checks the 21 cases declared in `tasks/K12/cases.py`, then measures five shapes against FP16 `torch.mm` with FP32 reduction enabled. The independent FP32 arithmetic reference is used for numerical checks.

The graph command generates three synthetic CSR inputs at feature width 128. Each includes empty rows, short rows and high-degree rows. It compares fused gather/reduction with a PyTorch gather plus `index_add_` reference. This example uses generated inputs, so it can run without downloading a graph dataset.

Timing uses 20 warmup calls, then 30 paired rounds in shuffled order. Each sample averages 20 calls measured with CUDA events. `--rounds` and `--iterations` change these settings. Compilation, input preparation and the graph's reusable degree schedule are outside the measured calls. Candidate output allocation remains inside each measured call; the PyTorch graph baseline also allocates its output.

Output JSON includes the inputs, tolerance, device, per-round samples and medians. A correctness failure stops execution before producing a successful measurement report.

## Use the fixed evaluation service

The AGH plugin uses `kernel_lab` for asynchronous candidate evaluation:

```bash
python -m kernel_lab baseline --run optimize_001 --tasks K12
python -m kernel_lab serve-evaluator --run optimize_001 --gpu 0 --keep-alive
```

Run the evaluator in its own terminal. In another terminal, submit the included candidate directly:

```bash
python -m kernel_lab submit --run optimize_001 --task K12 --candidate-dir operators/small_m
python -m kernel_lab task-status --run optimize_001 --task K12
```

The same submission and query operations are exposed through the AGH plugin. See [plugin flow](docs/plugin-flow.md) for configuration. New jobs and their computed outputs are created under `artifacts/<run-id>/`; the plugin configuration is written separately to `outputs/kernel-tools.json`.
