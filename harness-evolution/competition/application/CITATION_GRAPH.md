# Real citation graph: two-device strong-control GPU verification

## Closed GPU result, 2026-10-06

The original tiled K15 integration and the first fused CSR implementation pass
numerical checks on both devices but lose to the required strong libraries.
Those negative measurements remain in their separate reports. The new
Codex-authored split/compiled CSR implementation wins for all three originally
fixed cases on two physical devices. GPU3/full ratio1.5904735003 has paired95%CI
[1.5893673785,1.5911333866]; GPU4/full ratio1.5920454866 has CI
[1.5905804004,1.5934099815]. The original2048-target window ratios are1.09425
and1.09396; the five-target boundary ratios are1.13119 and1.11542. All six lower
CI bounds exceed1. Complete numerical, storage and empty-row checks pass for
all six backends, with the original rtol3e-4/atol1e-5 and independentFP64 oracle.

[The public measurement report](../../codex-rsi/reports/citation-graph-split-compiled-actual-two-device-GPU-post-V19-r1.json)
retains all30paired GPU/host samples, qualified backends, per-round strongest
baseline samples, original protocol and source bindings. The strongest baseline
is the per-round minimum of five qualified libraries:segment_reduce,index_add,
sparse_mm,compiled_segment_reduce and compiled_index_add. Both physical-device
UUIDs and original jobs are bound. An independentCPU arithmetic audit also
recomputes all six ratios and2000-bootstrap CIs from the original raw reports;
that audit adds no GPU reproduction.

Static input topology is partitioned into small/medium/high-degree target
groups. High-degree sums use128-edge partials and a merge; small segments are
packed four at a time. Every target and edge remains represented once,
including duplicate edges and empty destinations. The entire fixed two-layer
candidate is compiled with fullgraph=true and no eager fallback. Group
construction and compilation occur outside paired timing, inside the accounted
application lifecycle. CUDA graphs are disabled. GPU measurements cover the
original two-layer event interval; host samples measure submission-loop cost
with the final device wait excluded. Neither number establishes a gain in
compile-inclusive total job time.

The model is a fixed, seeded, untrained scientific message-passing computation.
There is no claim about classification accuracy, pretrained text quality,
universal isolated-operator throughput, or causal AGH/RSI improvement. Original
K15 source and its previously qualified domain remain unchanged.

## Original CPU contract and earlier preparation evidence


This application uses the official [OGB ogbn-arxiv dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
under ODC-BY:169343nodes,1166243directed citation edges and actual128-dimensional
title/abstract features. Duplicate edges and empty destinations are retained.
No reverse edges or self loops are added. Fixed untrained projections define a
two-layer scientific message-passing computation; no classification accuracy is
claimed.

The source-bound candidate is the unchanged K15 CSR implementation from the
original H0 V18 DEV2 result, SHA
`a0a474db012746a5b9c571c9d2cbad86e08b127456773e950b0c5806ed8dae24`.
The graph is tiled in original node order into the verified N<=32768/S<=2048
domain. Complete graph, fixed target IDs[0,2048), and a labelled boundary fixture
remain declared regardless of future results. External source features remain
fixed for the target-window computation. Features are divided by
max(1,maximum absolute feature), projection columns have absolute sum0.125, and
bias magnitude is at most0.001. Every backend uses identical inputs and weights.

The CPU module prepares `torch.segment_reduce`, `index_add_`, an independent
FP64 oracle and a CUDA-only source adapter. The original rtol0.0003/atol0.00001
stays fixed. Ten CPU tests pass under the original CUDA/network/external-write
guard. The full archive was parsed and hash-bound; GPU numerical qualification,
original21-case cross-device reproduction and30-round application timing remain
pending. The earlier723.73× result compares with the original eligible Python
segment baseline; it does not establish a win against these stock libraries or
the complete graph computation.

Download the archive separately from the
[official registry](https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/master.csv)
at `https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip`. The observed
83058288-byte archive SHA256 is
`49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276`.
The archive and derived features are not embedded in the delivery package.

```sh
python harness-evolution/competition/application/citation_graph.py \
  --archive /path/to/arxiv.zip \
  --archive-sha256 49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276 \
  --operator-source harness-evolution/codex-rsi/operators/g000-v18-dev2/K15/kernel.py \
  --output /path/to/new-cpu-plan.json
```

The command reads only declared gzip members, never extracts archive paths,
refuses hash/count/domain mismatches and refuses to overwrite an existing plan.
It returns GPU_PENDING with no timings and makes no model API request.

`citation_graph_strong_controls.py` defines a separate V2 protocol preserving
this R1 source and results. It includes direct sparse.mm, which avoids expanding
all edge messages, and both compiled dense library paths. Its five required
libraries remain in the comparison; a missing or failed library prevents a
broad strong-baseline victory claim. Compiler construction and lazy execution
failures stay unavailable, without an eager fallback. Preparation returns
PREPARED_UNQUALIFIED until actual numerical qualification. The CPU entrypoint
prepares a plan; it provides no GPU admission or timing.

All22combined guarded CPU tests pass. Direct sparse.mm also passes aggregation
and two-layer numerical checks for all3actual predeclared cases, including the
full graph, with input immutability and exact empty rows. The compiler tests use
an injected identity factory or explicit failure; actual compiled/GPU numerical
qualification and performance remain pending.


独立图应用复现入口 [reproduce_citation_graph.py](../reproduce_citation_graph.py) 默认只读取团队提供的原始本地数据并生成CPU计划：

```sh
python harness-evolution/competition/reproduce_citation_graph.py \
  --archive /your/local/arxiv.zip --out /tmp/new-graph-plan
```

在团队自己的干净CUDA设备上，用另一个全新输出目录并显式加`--execute --gpu 0`，通过独立官方本地Queue、项目GPU锁、原900秒作业与原30配对轮运行同一个冻结benchmark。默认全三个固定案例、五种强库比较、数值/聚合/输入保持判据不变。数据不随包分发、不自动下载，原archive和候选源码hash不匹配会拒绝。作者原benchmark的私有历史晋级/谱系准入记录由独立本地验收范围代替，不生成作者闭账证据；本入口不会进入或修改作者主对照账本。

新7项加原契约共34CPU检查通过，无跳过；真实83058288字节本地archive的默认CLI也在禁止CUDA初始化/枚举、网络、范围外写入的原guard下通过，完整169343节点/1166243边/三案例计划正确。CPU执行double使用真实本地Queue与原registry/status子进程验证失败闭账，属于接线测试，不是GPU实测。原基准及候选源码字节保持，当前V21期间根agent未执行此独立GPU路径；新实际GPU复现及便携包解压验收仍待执行。


[图应用全量公开原样本](citation-graph-split-complete-public-samples-r1.json) 包含两设备/三固定案例/六后端的1080个GPU样本和1080个主机样本，数值与聚合资格、原输入hash、每轮五库最快值、配对比和置信区间全部保留；无私有绝对路径或模型请求正文。导出器对全部原样本独立重算中位数、2000次bootstrap及设备状态分半诊断，6项导出与篡改/缺项检查及真实原guard默认CLI通过；这是已有实测的公开导出，没有新增GPU测量。
