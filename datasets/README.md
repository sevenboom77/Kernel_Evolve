# 图计算数据

图应用使用 Open Graph Benchmark 的 **ogbn-arxiv**：169,343 个节点、1,166,243 条有向引用边，以及论文标题和摘要的 128 维特征。

数据归档按原始字节作为 [v0.1.0 发布附件](https://github.com/sevenboom77/Kernel_Evolve/releases/tag/v0.1.0) 提供，不放入 Git 历史。获取并校验：

```bash
python3 scripts/fetch_release_assets.py --dataset
```

脚本将文件保存为 `datasets/arxiv.zip`，同时核对文件长度和 SHA256。原始 SHA256 为：

```text
49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276
```

也可从 [OGB 官方来源](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) 下载对应归档，再核对上述哈希。原下载地址为 `https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip`。

数据遵循 OGB 所列 **ODC-BY** 来源条款，使用时请注明 Open Graph Benchmark 和 ogbn-arxiv。项目采用固定未训练权重进行计算验证，速度结果不代表分类准确率。
