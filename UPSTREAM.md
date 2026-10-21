# 上游源码与项目改动

KernelEvolve 基于 [Agnes Harness](https://github.com/AgnesAI-Labs/agnes-harness) 构建 GPU 算子优化工作流。

- `third_party/agnes-harness/` 直接展开自本项目交付的 G031 固定源码，保留原文件内容、锁文件、LICENSE、NOTICE 和组件许可，便于直接浏览与构建。
- G031 源码记录的提交为 `12e3b131e405295805caa3daa90f4c2b827cbed4`。这是项目交付版本的标识，不代表该提交必然存在于公开上游仓库。
- `third_party/AGH_SNAPSHOT.json` 记录展开文件的 SHA256，并绑定原 G031 源码归档的哈希。
- `demo/sources/G031-source.patch` 保留 G014 到 G031 的改动，用于查看错误诊断、可信候选记忆与受控接续等机制。
- G000、G014、G031 的原始 `source.tar` 以压缩附件随 [v0.1.0](https://github.com/sevenboom77/Kernel_Evolve/releases/tag/v0.1.0) 发布，使用 `python3 scripts/fetch_release_assets.py --sources` 获取并恢复到原路径。

专业工具插件位于 `harness-evolution/kernel-workflow-plugin/`。独立评测器位于 `kernel_lab/`，候选、任务及应用实现位于 `tasks/`、`demo/sources/` 和 `harness-evolution/competition/`。

Agnes Harness 保留 Apache-2.0 及其组件条款，插件包元数据声明 MIT。仓库没有新增覆盖全部代码、数据与材料的统一许可。第三方文件的许可应分别遵守。
