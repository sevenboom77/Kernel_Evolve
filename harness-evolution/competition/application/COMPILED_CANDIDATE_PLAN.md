# Compiled candidate decoder: approved design and implementation plan

The V1 measurement improved over eager PyTorch and lost to compiled PyTorch in all four shapes. V2 tests whether whole-block compilation reduces Python dispatch gaps while retaining the exact K04/K06/K10 kernels and decoder equations. Root approved this scope and official Queue append accounting on 2026-10-05. No improvement is assumed before a new GPU measurement.

Create `benchmark_decoder_v2.py`, `tests/test_competition_decoder_compiled.py`, a V2 CPU receipt and command documentation. Preserve V1 sources, report, operator sources, manifest, four cases, seeds, FP32/FP16 boundaries, tolerances, full-block timing and strong compiled baseline. Compile the candidate with `torch.compile(..., mode="default", fullgraph=True)`. Record unavailable or incorrect compilation explicitly; never substitute eager execution under the compiled label. Pair eager, candidate, compiled and compiled_candidate in the same shuffled rounds. No packed projections or new kernel fusion are included in this version.

Use the already registered application queue with a fresh run/job. In one SQLite transaction, reject reused run IDs, any nonterminal prior job or any nonapplication job, then enqueue the new job using the official Queue API. Claim under the numeric project GPU lock before the first CUDA action and finish in `finally`, preserving previous rows/events/span IDs. Root supplies the SHA-bound live budget command and registry and authorizes GPU execution after the primary batch closes.

Implementation uses executing-plans and test-driven-development inline:

1. Write failing tests for V2 existence/plan provenance, actual one-graph Dynamo CPU capture and output equality on all four shapes, compilation refusal, incorrect output, interrupt propagation and append accounting conservation.
2. Run `/home/USER/anaconda3/envs/torch/bin/python -m pytest tests/test_competition_decoder_compiled.py -q`; retain the missing-implementation RED log.
3. Copy frozen V1 benchmark to the NEW V2 filename and make the minimal additions above. Reuse the frozen decoder math. Preserve explicit CUDA completion and budget checks.
4. Run new tests and the existing decoder/benchmark/Queue regression suites. CPU orchestration uses local substitutes only; actual Dynamo capture uses a CPU backend and does not prove Triton/Inductor GPU compilation.
5. Run py_compile, the real default V2 plan CLI and `git diff --check`; bind logs, unchanged source hashes and protocol into a CPU receipt. Commit only the new application files/tests/docs and provide root the actual execute argv. GPU status remains pending.

Acceptance requires same source-bound four inputs, no input/weight mutation, original operator gates plus the unchanged full-output gate, no silent compilation fallback, an explicit strong-baseline comparison, and official append-only accounting. Actual CUDA compilation, correctness and performance require a separate root-authorized measured run.
