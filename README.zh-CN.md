# KernelEvolve 算子优化

[English](README.md) | 简体中文

围绕两类融合 GPU 算子，提供可运行的实现、基准测试和面向智能体的候选提交与反馈流程。

## 1. 项目介绍

GPU 计算中，中间结果搬运和大量小算子启动会带来额外开销。本项目针对不同的数据访问模式编写融合算子，并提供候选方案的检查、测量和反馈机制。

- **CSR 邻居聚合：**按节点度数划分调度，完成邻居特征的 gather 与 sum。
- **小 M 矩阵计算：**针对较小行数使用 Triton 实现，并以 FP32 累加。
- **优化工作流：**项目插件将 AGH 与评测服务连接，用于提交候选、读取结果和继续优化。

仓库包含项目算子、基准代码和插件；AGH 与 RSI 框架本身需要外部提供。图算子独立基准与插件评测任务是不同的运行路径。

## 2. 演示视频

[![优化闭环](presentation/rsi-loop.gif)](presentation/kernel-evolve.mp4)

[观看完整演示](presentation/kernel-evolve.mp4) · [打开展示页](presentation/index.html)

视频展示了算子优化及反馈过程。

## 3. 技术架构

```text
问题与候选算子
    │
    ├── 独立基准 ── 正确性与性能测量
    │
    └── AGH 插件 ── 评测服务 ── 任务状态与反馈
                           │
                       继续修改
```

| 组成 | 实现与入口 |
| --- | --- |
| 图聚合 | [csr_gather_sum.py](operators/graph/csr_gather_sum.py)：FP32、特征宽度 128，int64 索引和偏移；按度数 ≤8、9–32、>32 分别调度 |
| 小 M | [kernel.py](operators/small_m/kernel.py)：FP16 输入输出、FP32 累加，支持 M 为 1、2、4、8 |
| 评测后端 | `kernel_lab` 管理任务、提交和反馈 |
| 智能体工具 | [kernel-workflow](plugins/kernel-workflow) 将评测流程接入 AGH |

小 M 算子的 `ModelNew()(a, b)` 返回元组。测试设计和插件流程详见 [BENCHMARKING.md](BENCHMARKING.md)、[插件说明](docs/plugin-flow.md)。

## 4. 集成标准与使用

### 运行算子

建议环境为 Python 3.11、PyTorch 2.5.1（CUDA 12.1）和 Triton 3.1.0。先生成测试计划，再在兼容的 GPU 上执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install triton==3.1.0 numpy scipy
python benchmarks/run.py --operator graph --out outputs/graph-plan.json
python benchmarks/run.py --operator small-m --out outputs/small-m-plan.json
python benchmarks/run.py --operator graph --execute --gpu 0 --out outputs/graph-gpu.json
python benchmarks/run.py --operator small-m --execute --gpu 0 --out outputs/small-m-gpu.json
```

计划列出正确性与计时用例；执行需要 CUDA 设备，结果写入 `outputs`。

### 连接智能体

插件 `@kernel-lab/kernel-workflow` 提供 `kernel_get_task`、`kernel_submit`、`kernel_get_job`、`kernel_get_task_status`、`kernel_read_feedback`。评测后端与 AGH 运行时分开配置：

```bash
python tools/configure_kernel_tools.py --run-id optimize_001
export KERNEL_TOOLS_CONFIG="$PWD/outputs/kernel-tools.json"
python -m kernel_lab baseline --run optimize_001 --tasks K12
python -m kernel_lab serve-evaluator --run optimize_001 --gpu 0 --keep-alive
```

在单独终端保持评测服务运行，并让单独安装的 AGH 使用 [plugins/kernel-workflow](plugins/kernel-workflow)。插件工作流包含 K12、K19 任务，图算子另有独立基准。详见 [插件流程](docs/plugin-flow.md)、[依赖说明](DEPENDENCIES.md) 和 [LICENSE](LICENSE)。
