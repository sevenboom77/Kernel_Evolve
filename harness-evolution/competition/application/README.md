# Fixed-operator decoder computation benchmarks

The sequence block below has completed its four-case, four-backend V3 run.
The separate cached decoder has completed six backends and an independent run
on a second physical GPU. The packed cached decoder has completed eight backends,
including two library implementations with the same merged projections. All
three use fixed random weights, not pretrained language models. Their reports
retain the strong-baseline losses.

For portable CPU plans, run `../reproduce_application.py`,
`../reproduce_cached_application.py`, or `../reproduce_packed_application.py`
with `--out` pointing to a fresh directory. Explicit `--execute` runs the
unchanged numerical benchmark on an independently accounted clean GPU; it does
not use the author's production policy or credentials. Packed measurement keeps
all six library baselines, four cases, thirty paired rounds and the 900-second
wall limit. The old sequence specification below describes only that block.

This executable block uses fixed random weights and input. It contains Q/K/V
projections, adjacent-pair RoPE, causal masked softmax, attention matmuls, an
output projection, two residual additions, and a gated SiLU MLP. It has no
pretrained weights, normalization, tokenizer, KV cache, or text-quality metric.

Only K04/K06/K10 are substituted. Their files are loaded unchanged from the
existing `codex-rsi/operators/g001/manifest.json`; the manifest SHA and each
source SHA are checked before loading. Projections and attention matmuls are
FP32, TF32 is disabled during GPU runs, and both implementations share the same
explicit FP16 rounding points. K06 multiplies in FP32 before its final FP16 cast.

The block domain is the intersection of the original operator contracts:
`B=1..2`, `H=8..32`, even `D=64..128`, `S>=33`, `B*H*S<=2048`, and
`MLP=33..11008`. Four cases are declared: `B1/S64/H8/D64/MLP1024`,
`B1/S128/H8/D64/MLP1024`, `B1/S33/H8/D64/MLP33`, and
`B2/S128/H8/D128/MLP1024`. CPU tests also exercise invalid dimensions,
nonfinite inputs, fully masked rows, source drift, and unchanged input/weights.

Run from the repository root with the project PyTorch environment:

```sh
python -m pytest tests/test_competition_decoder.py tests/test_competition_decoder_benchmark.py
python harness-evolution/competition/application/benchmark_decoder.py --out decoder-plan.json
```

The second command creates a source-bound `GPU_PENDING` plan. It does not query
the GPUs, import the candidate Triton modules, create a CUDA context, or record
latency. Use a fresh output path for every command; existing evidence is retained.

Real measurement requires prior authorization, a dedicated accounting queue
already registered in the current phase policy, a running budget collector
scanning all registered queues about every 0.5 seconds, and a SHA-bound JSON argv
file invoking the current phase `status` CLI. The original credential is not used
by this application. Root supplies the exact current policy and argv hashes.

```sh
python harness-evolution/competition/application/benchmark_decoder.py \
  --execute --gpu 4 --run-id agh-hackathon-decoder-v1 \
  --accounting-queue /data/lizhijun/agh-evo/codex-rsi-hackathon-v11-20261005/application/decoder/queue.sqlite \
  --registry-reference /absolute/current-phase-policy.json --registry-sha256 POLICY_SHA \
  --budget-argv-json /absolute/phase-status-argv.json --budget-argv-sha256 ARGV_SHA \
  --max-wall-seconds 900 --rounds 30 --warmup 20 --out /absolute/fresh-decoder-result.json
```

Those two SHA placeholders must be replaced with actual bound files; this is an
invocation template, not a record of GPU execution. The physical card is locked
through the evaluator's project `.locks/gpu-<numeric-index>.lock`. A foreign or
unnameable process prevents a clean measurement. No process on another user's
card is signaled. An official `Queue` job (`application/APP_DECODER`, one attempt)
is claimed before the first CUDA action and finished in `finally`. Claim-to-finish
wall time conservatively accounts for JIT, compilation, validation, and timing;
it is not CUDA execution time. SIGINT/SIGTERM, budget refusal, and failures finish
as `crashed`, preserving partial samples and a nonzero exit. The wall limit is
always finite (at most 3600 seconds), even when global caps are unlimited.

Correctness is checked for each substituted operator on its actual inputs with
the original tolerances: K04 `rtol=1e-4/atol=1e-5`, K06/K10
`rtol=0.002/atol=0.01`. Full-block output has a separately declared
`rtol=0.005/atol=0.005`. Masked probabilities must be exactly zero. Input, weights,
and positional/mask tables are checked for mutation before and after timing.

The eager and candidate blocks, and `torch.compile(fullgraph=True, mode='default')`
when numerically correct and supported, use identical tensors. Compilation,
warmup, input upload, source validation, and correctness checks are excluded
from steady-state latency. Each implementation is calibrated and measured in
30 paired rounds with shuffled order and CUDA events; the full call, including
all projections and attention/MLP work, is timed. Raw GPU and separate host
samples, orders, device state, fingerprints, source hashes, compilation time,
and split-half state diagnostics are retained. A compiled baseline failure is
reported as `UNAVAILABLE` or `INCORRECT`; it is never silently credited as a win.

Speedup is the median of within-round baseline/candidate latency ratios, with
2000 paired bootstrap resamples. Repeated fixed inputs can fit in cache. This
benchmark does not establish pretrained inference quality or AGH-version
causality, and historical individual operator speedups are never multiplied to
claim a full-block gain. No GPU performance is claimed by the CPU verification.

真实GPT-2 r2双卡实测已完成，见[全部四形状报告](pretrained-gpt2-gpu-verification-v1.json)。保留全部124439808个参数和12个原K02调用；只有B1S33通过，编译候选对普通FP16库2.020倍。其余三形状候选、所有compiled_fp16均数值失败，未计入收益；180 GPU与180主机样本完整保留。原full-logits容差rtol0.005/atol0.05未放宽，未宣称文本质量、全形状或强基线胜出。原r1设备预初始化/子进程收尾失败和全部GPU跨度保留；r2确认物理GPU0/3当前PID并正式关闭ownedRPC。下一数值诊断须单独登记，在V15主系列成本区间外开展。

单独精度诊断入口为`benchmark_pretrained_gpt2_precise.py`，只新增Inductor `emulate_precision_casts=True`，两编译后端共用同一开关；旧benchmark/K02源码、四形状、原容差和30配对不改。GPU是否有效仍待独立实测，旧失败不被替换。CPU实际编译已有显式转换舍入差异，只验证了运行元信息和作用域，不能当GPU精度恢复。`../reproduce_pretrained_precision.py`默认生成CPU计划，显式`--execute`仅在全新、原生产命名空间之外的本地官方Queue执行相同协议。核验入口`../export_pretrained_precision.py`独立复算，不改原validator和历史报告。

2026-10-06更新：精度诊断已完成，短形状相对强库1.000371倍，95%CI跨1，后三形状仍错误；上述待实测表述保留为旧准备时点。实际GPU数值定位确认参考接线四形状逐位相同、原K02局部48项通过，而完整模型仅1/4正确，旧结果保持。

新增 `fused_welford_pretrained.py` 和 `diagnose_pretrained_gpt2_welford.py` 将独立.cpp/.cu Welford修复接入全部12层，保留原权重/四输入/局部与完整logits容差；没有计时或强库胜出声明。实际作者GPU入口必须绑定当前V18完整八臂关闭和已闭资源端点，正在等待；不能用旧V16/V17完成文件准入。新源码已实际CPU编译加载，GPU正确性待实测。

团队独立复现入口 `../reproduce_pretrained_welford.py` 默认只生成本地原权重CPU计划，availability导入探测关闭后恢复，0CUDA初始化/枚举/网络/Queue，真实CLI及38项相关检查通过。显式--execute只使用新本地官方Queue/GPU锁/原900秒数值诊断，不依赖作者私有系列；独立输出不能当作作者V18闭账或AGH因果证据。所需兼容CUDA工具链用于从原.cpp/.cu源本地编译，作者.so和模型权重不在公开包中。

2026-10-06 实测更新：V18 八臂、160 个原生任务已关闭。独立 Welford 修复在全部四形状、48 项局部检查及完整 GPT-2 logits 上逐位一致，详见 `../../codex-rsi/reports/pretrained-fused-Welford-GPU-numerical-diagnosis-v18.json`。随后 `benchmark_pretrained_gpt2_welford.py` 按原四后端、30 配对、900 秒协议完成计时，详见 `pretrained-Welford-gpu-verification-v18-r1.json`：eager 对普通 FP16 库为 0.904–0.953 倍，未加速；编译库数值失败，编译候选因扩展加载路径无法完整追踪而不可用，强基线收益均为空。保留全部 240 GPU 与 240 主机样本及旧失败；下一编译接入实验另立源码与协议版本，不覆盖这些结果。
