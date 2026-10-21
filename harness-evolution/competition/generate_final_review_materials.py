"""Render nine slides from the completed October 9 review, using the original renderer."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def content(data):
    if data.get('release_id') != 'review-20261009-closed-r1' or data.get('rsi_performance_established') is not False:
        raise ValueError('closed review release required')
    path = Path(__file__).parent / 'generate_materials.py'
    spec = importlib.util.spec_from_file_location('original_review_slide_boxes', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    box = base.box
    green, muted, orange = base.GREEN, base.MUTED, base.ORANGE
    native = data['native']
    a, b = native['replicas']
    pair = data['last_complete_pair']
    bounds = lambda values: f'[{values[0]:.4f}, {values[1]:.4f}]'
    full = [next(c for c in replica['cases'] if c['case_id'] == 'arxiv-full') for replica in data['graph']['replicas']]
    return [
        ('KernelEvolve · 可验证的GPU优化', [
            box(120,250,1660,230,'从模型生成代码\n到正确性、失败修复和独立复测',67,green),
            box(120,535,1660,155,'AGH × mimo-v2.6-pro\n原生small-M生成闭环 + 真实引文图计算',39,muted),
            box(120,770,1660,110,'硕博组 · 2026-10-09闭合实测 · 报名信息待补',29,orange)],
            '0:00–0:20 GPU小批量任务和不规则图计算需要专门的实现。我们把AGH和模型生成、原GPU评测器、反馈修复及独立复测连接起来，让评审可以核查代码从哪里来、是否正确、比什么更快。'),
        ('AGH确实完成了生成、反馈和修复', [
            box(120,250,1650,180,'读任务 → 写源码 → 原生提交 → GPU反馈\n→ 修订 → 独立验证',45,green),
            box(120,465,1650,215,'原K12：write213 → receipt219 → submit244 → receipt248\n前两候选 crashed / compile_fail；第三候选完成验证\n源文件、原生事件、SQL、候选与验证身份相互绑定',32,muted),
            box(120,740,1650,135,'源码作者：AGH中的MiMo，原H0搜索\n此算子不是G031改进带来的收益',33,orange)],
            '0:20–0:45 这不是手工脚本伪造的模型生成。原生写入和工具回执连接到候选SQL、源文件快照与独立验证。前两个失败没有删除；第三候选通过验证，整段episode的预算耗尽也保留。算子来自H0与MiMo，不能归因于之后的G031。'),
        ('small-M：21/21正确，强库收益未认证', [
            box(120,245,1660,145,'GPU1  1.0339×  ·  95%CI '+bounds(a['strong_ci'])+'\nGPU4  1.0520×  ·  95%CI '+bounds(b['strong_ci']),38,green),
            box(120,440,1660,195,'相同FP16输入/输出，显式FP32中间归约\n两种直接/编译强库全部21/21；原五形状、五轮保留\n形状中位数约1.032–1.149×，首轮没有丢弃',34,muted),
            box(120,700,1660,190,'两卡均未通过原CI下限>1.05门\n原转换基线约4.31–4.38×，不能称为强库胜利\n默认FP16库19/21失败保留；不再补跑逼出正结果',32,orange)],
            '0:45–1:15 两张4090独立复测都通过原21个正确性案例。直接FP16库的默认归约有两个数值失败，因此另行声明关闭降精度归约的精度范围，保持原参考和容差。强库整体速度比分别1.0339和1.0520，原CI门未过；四倍只是相对原转换基线，不能作为强库主张。'),
        ('真实引文图：强库对照约1.59×', [
            box(120,240,1660,150,'169343节点 · 1166243边\n完整两层FP32图计算，五种合格库逐轮比较',41,green),
            box(120,450,1660,150,f'GPU4  {full[0]["speedup"]:.6f}×  ·  CI {bounds(full[0]["ci"])}\nGPU0  {full[1]["speedup"]:.6f}×  ·  CI {bounds(full[1]["ci"])}',36,green),
            box(120,665,1660,105,'全图 / 2048节点窗口 / 边界全部正确；每例30配对',33,muted),
            box(120,810,1660,80,'作者Codex；条件化输入、固定随机权重未训练，不证明分类精度',28,orange)],
            '1:15–1:45 真实ogbn-arxiv完整图计算在两张独立4090上稳定约1.59倍，相对每轮五个合格库中最快的值，完整耗时下降约37%。全图、窗口、边界均通过原数值门，GPU和主机各30配对原样本可查。此实现由Codex编写，权重未训练，不能声称模型分类精度或原生AGH生成。'),
        ('AGH改了什么：可信反馈与有界恢复', [
            box(120,240,1660,315,'① 原生剩余请求与候选预算进入模型上下文\n② 运行中的待完成作业有界续查，保留排队状态\n③ 可信DEV错误分类、搜索记忆与修复焦点\n④ 输出截断后丢弃半截工具，有界恢复\n⑤ G031首次数值错误立即进入既有修复焦点',34),
            box(120,610,1660,145,'原始反馈、来源认证、正确候选、数值约束保持\nG031：3633源码 / 175原运行文件 / 1587合成HTTP',33,muted),
            box(120,810,1660,70,'作者外层Codex；工程机制验证不等于搜索性能提升',29,orange)],
            '1:45–2:20 改进集中在模型收到什么可靠反馈、异步状态怎样恢复和失败后下一步怎样行动。G031让第一次可信数值错误就进入已有修复焦点，仍保留普通反馈、错误来源认证和四事实两千字节上限。实际本地HTTP验证包括原请求和候选上限，但它只证明机制，不能代替付费搜索性能实验。'),
        ('20任务对照：全部位置、负结果都保留', [
            box(120,240,1660,180,f'最近完整V50：G014 ↔ G030，共40位置\n整体比 {pair["speedup"]:.6f}，95%CI {bounds(pair["ci"])}\n原晋级门未过；accepted0，H0保持',37,green),
            box(120,485,1660,190,'后续V51父20终态：13最终评测 / 7ENV / 6正确\nGPU3持续忙，复验等待已正常中止并闭账\nG031子未启动；没有有效配对比，40位置不退款',34,orange),
            box(120,755,1660,110,'47754请求 / 34309历史unknown / 41原生SQL\nunknown为保守记录，不是确认账单',30,muted)],
            '2:20–2:50 最近完整两臂结果是V50，整体比约1.055但区间包含1，并没有晋级。随后G031筛选的父轨在忙卡复验启动被环境阻塞，已按own身份正常关闭，子轨未启动；因此不能计算G031配对收益，也不能将其判成优胜。失败位置和费用全部保留。'),
        ('正常、边界、失败：三种入口可核查', [
            box(120,240,1660,190,'正常：原21 GEMM案例 + 全图计算\n边界：小矩阵、2048节点窗口、图边界案例\n失败：crashed / compile_fail / 数值不合格',40,green),
            box(120,510,1660,180,'每个GPU作业前探测util=0与足够显存\n不结束外部进程，失败与未知消耗不删除\n新有限启动等待7个CPU检查；正式下一控制器接入待验',32,muted),
            box(120,780,1660,110,'演示是带日期的实测回放，点击不发起模型或GPU调用',30,orange)],
            '2:50–3:15 演示分别打开正常、边界和失败。所有库先通过原数值门才允许计时与性能主张。GPU执行先做新探测，不能为了取得空卡清理外部任务。我们也暴露了旧启动等待绕过复验期限的问题，新有限等待CPU验证已完成，但下一正式控制器接入还没有经过实测。'),
        ('复现：源码、样本和构建身份各自绑定', [
            box(120,240,1660,190,'默认CPU计划；显式--execute才启动GPU\n原源码、原参考、强基线、原样本与SQL可下载\n两设备复测与独立归档安装/构建/CLI已有记录',40,green),
            box(120,510,1660,150,'Python3.11 / PyTorch2.5.1+cu121 / Triton3.1\nRTX4090 · Node24 / pnpm10 · 无模型密钥随包',34,muted),
            box(120,745,1660,135,'最终ZIP的重新解压、daemon/HTTP与媒体统一验收仍在进行\n已有构建成功不能自动转移到新的运行产物',30,orange)],
            '3:15–3:40 默认复现只建立CPU计划，显式执行才使用GPU。原样本和SQL允许重新计算结论。G031归档已经独立安装构建并通过CLI，但21个构建文件与旧运行包不同，资格不能直接继承；最终ZIP仍须重新解压启动验证。演示无需密钥，新的模型任务使用复现者自己的凭据。'),
        ('有限研发，按时交付真实成果', [
            box(120,240,1660,190,'10-11 20:00 源码与功能冻结\n10-13 20:00 独立验收与交付候选\n10-14 18:00 最终技术包、视频、PPT和问答',42,green),
            box(120,520,1660,160,'最多两代候选、五配对；只保留一次干净筛选机会\n无法证明改进就冻结稳定版，公开负结果\nRSI是证据结论，不作为无限等待的交付条件',34,muted),
            box(120,805,1660,75,'报名身份、正式模型资格、公开链接和实际提交由用户完成',28,orange)],
            '3:40–4:00 最终作品的价值是可信的生成与修复、可复现的图计算收益以及可审查的AGH工程。研发明确封版，无法证明搜索进化就如实报告RSI未证明。比赛名次不作为无限运行的终点；技术交付和外部报名提交分别列明。'),
    ]


if __name__ == '__main__':
    path = Path(__file__).parent / 'generate_materials.py'
    spec = importlib.util.spec_from_file_location('original_final_review_renderer', path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.content = content
    renderer.main()
