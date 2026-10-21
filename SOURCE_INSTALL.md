# KernelEvolve 源码安装与运行

命令从 `Kernel_Evolve` 仓库根目录执行。保存结果阅读使用 Python 标准库和现代浏览器；GPU 复测的原环境为 Linux、Python 3.11.10、PyTorch 2.5.1+cu121、Triton 3.1.0、CUDA 12.1 和 RTX 4090。其他显卡需验证兼容性及显存需求，单次复测使用一张设备。

## 1. 阅读结果与生成 CPU 复测计划

```bash
python3 -m http.server 8090 --bind 127.0.0.1 --directory demo
```

浏览器打开 `http://127.0.0.1:8090/`。页面展示保存记录，不会启动模型或 GPU 任务。Windows 可将 `python3` 换成 `python`。

图数据和历史源码归档作为 Release 附件提供，不进入 Git 历史。先获取图数据并核对 SHA256；已有文件若不匹配，脚本会停止而不会覆盖。K12 独立复测不需要图数据。

```bash
python3 scripts/fetch_release_assets.py --dataset
```

先完成第二节的依赖安装，再在另一个终端生成新的复测计划。图计划需要 Python 计算依赖，但不进行 GPU 计时：

```bash
.venv/bin/python -B harness-evolution/competition/reproduce_native_small_m_fp32_reduction.py --out /tmp/new-native-plan
.venv/bin/python -B harness-evolution/competition/reproduce_citation_graph.py --archive datasets/arxiv.zip --out /tmp/new-graph-plan --case all
```

输出目录采用新的路径，保存候选、输入和复测配置。

## 2. 安装 GPU 计算依赖

在具有 NVIDIA 驱动和对应 CUDA 运行条件的 Linux 主机建立 Python 3.11 环境：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install triton==3.1.0 numpy scipy
.venv/bin/python -c 'import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())'
```

确认 CUDA 可用后执行：

```bash
.venv/bin/python -B harness-evolution/competition/reproduce_native_small_m_fp32_reduction.py --out /tmp/new-native-gpu --execute --gpu 0
.venv/bin/python -B harness-evolution/competition/reproduce_citation_graph.py --archive datasets/arxiv.zip --out /tmp/new-graph-gpu --execute --gpu 0 --case all
```

执行入口遵守共享 GPU 锁和原数值/预算协议。输入、候选、正确性和性能输出分别保存，结果以当前设备实际测量为准。

## 3. 构建固定 AGH 源码

G031 完整固定源码已展开到 `third_party/agnes-harness/`，保留锁定文件和许可。原构建环境为 Node 24.21.0、pnpm 10.34.5；源码声明 Node 至少 24.10。构建器要求 `--output-dir` 为绝对路径，先在本仓库根记录路径：

```bash
KERNEL_EVOLVE_ROOT="$(pwd -P)"
cd "$KERNEL_EVOLVE_ROOT/third_party/agnes-harness"
pnpm install --frozen-lockfile
pnpm --filter @agnes/cli build:local --output-dir "$KERNEL_EVOLVE_ROOT/runtime-new"
cd "$KERNEL_EVOLVE_ROOT"
node runtime-new/agnes.mjs --help
python3 harness-evolution/competition/qualify_agh_runtime_smoke.py --runtime runtime-new --node "$(command -v node)" --out /tmp/new-AGH-startup-proof
```

历史 G000、G014、G031 的原始 `source.tar` 可以恢复到原路径，不影响浏览和构建当前源码：

```bash
python3 scripts/fetch_release_assets.py --sources
```

原包的预构建运行目录不随 Git 发布，当前机器应通过源码构建准备运行产物。

## 4. 配置新模型任务

专业工具插件位于 `harness-evolution/kernel-workflow-plugin/`，导出 `kernelWorkflowTools`，插件 ID 为 `ext:kernel-lab/kernel-workflow`。按固定 AGH 版本的账户及插件接口配置，原请求名为 `mimo-v2.6-pro`。模型通过服务 API 调用，无需本地加载模型权重。账户授权使用你自己的配置，参考[上游中文安装文档](third_party/agnes-harness/docs/guide/install.zh-CN.md)。

插件通过 `KERNEL_TOOLS_CONFIG` 环境变量读取你创建的 JSON 文件，至少包含：

```json
{
  "kl_python": "/absolute/path/Kernel_Evolve/.venv/bin/python",
  "kl_root": "/absolute/path/Kernel_Evolve",
  "run_id": "your-new-run-id"
}
```

这里的运行标识必须已经初始化；配置文件不会自动创建实验或启动评测队列。必须同时准备独立执行器来消费队列，模型才能取得真实反馈。可选预算和工具参数以插件 `index.mjs` 及 `kernel_lab` CLI 为准。

仓库提供 K12、K19 两个任务包及 K12、图应用独立复测入口。`configs/suite_v1.json` 保留历史 K01–K20 套件定义，完整重跑需补齐其他任务包。新任务需要定义计算契约、参考实现、输入、误差、基线和预算，并配置任务注册与执行环境。

已有候选的独立复测不需要模型调用。Windows 可查看工作台和阅读源码；完整评测沿用 Linux/CUDA 路径，不承诺 Windows 原生开箱即用。

## 5. 校验公开源码

```bash
python3 scripts/verify_publication.py
```

该命令检查公开工作台、冻结 K12 输入和展开的 G031 源码，不进行模型调用或 GPU 测量。公开整理范围与路径脱敏记录见 [docs/PUBLICATION_NOTES.md](docs/PUBLICATION_NOTES.md)。数据来源与条款见 [datasets/README.md](datasets/README.md)。
