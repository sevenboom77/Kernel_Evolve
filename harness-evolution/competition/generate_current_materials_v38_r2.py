"""Render current closed-work slides with the original presentation renderer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def renderer():
    path = Path(__file__).parent / 'generate_materials.py'
    spec = importlib.util.spec_from_file_location('closed_work_slide_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def content(data):
    latest = next((r for r in data.get('source_changes', []) if r['generation'] == 'G022'), None)
    ablation = data.get('closed_ablation', {})
    if (data.get('mode') != 'recorded' or data.get('rsi_performance_established') is not False
        or ablation.get('comparison', {}).get('campaign_revision') != 37 or latest is None
        or latest['actual_localhost_request_bodies'] != 408
        or latest['formal_source_performance_promoted'] is not False):
        raise ValueError('current recorded scope differs')
    r = renderer()
    def body(text, y=270, height=620, size=36, color=None):
        return r.box(120, y, 1660, height, text, size, color or r.WHITE)
    history = data['closed_history_v37']
    pairs = {s['comparison']['campaign_revision']: s for s in data['series_history'] if s.get('comparison')}
    pairs[37] = ablation
    rows = []
    for revision in (30, 32, 33, 36, 37):
        pair = pairs[revision]
        parent, child = pair['arms']
        rows.append(f"V{revision}  {pair['comparison']['candidate_generation']}：正确{parent['correct_count']}→{child['correct_count']}，"
            f"最终评测{parent['evaluation_complete_count']}→{child['evaluation_complete_count']} / 20\n"
            f"全任务比 {pair['development_geomean_ratio']:.4f}；95%CI [{pair['development_paired_ci_low']:.4f}, {pair['development_paired_ci_high']:.4f}]")
    graph = data['graph_application']
    graph_gains = '/'.join(f"{p['strong_gain']['median']:.4f}" for p in graph['replicas'])
    gemm = next(v for v in data['operator_variants'] if v['task_id'] == 'K12')
    app = data['whole_model_application_APP35']
    counts = app['backend_status_counts']
    return [
        ('KernelEvolve：持续改进与可复验证据',
         [body('20路原生AGH任务 → GPU验证 → 闭账轨迹诊断\n'
               '修改真实AGH源码 → 同条件复测 → 按原门接受或拒绝\n\n'
               '外层Codex驱动；性能晋级尚未通过\n'
               'H0保持incumbent，RSI未成立', size=42)],
         '作品面向硕博组。我们展示真实算子、开发者驱动的AGH源码迭代和可审计对照。源码作者是外层Codex；持续性能改进仍须实测证明。'),
        ('每一次修改都进入实际执行链',
         [body('mimo-v2.6-pro · 每组20个原生任务\n'
               '固定10个daemon worker；原24请求 / 四候选\n'
               '0/1/3/4隔离计时，GPU2辅助数值和编译\n\n'
               '源码commit → 独立归档 → 175文件运行包\n'
               '实际SDK请求体 → 新对照 → 原评分与关闭记录')],
         '累计token、美元、GPU时间和旧截止均已取消，但单任务预算、原数值门、计时隔离和历史失败不改变。当前主测量期间不加入辅助GPU任务。'),
        ('五组已闭账结果：拒绝也公开',
         [body('\n'.join(rows), height=490, size=26),
          body('V36缺K17仍保留20分母；V37两臂10/20正确，五代均未接受。', y=830, height=100, size=30, color=r.ORANGE)],
         '五组各40位置原贡献、置信区间和成本全部保留。V32父版还有一项缺失，V33子版18项最终评测及2环境错误，均没有重评分。V36正确11→10，全任务比1.0262、CI[0.9432,1.1172]；V37全任务比0.8977、CI[0.7362,1.0426]，没有性能晋级。'),
        ('G022：完整JIT调用约定反馈',
         [body('完整且绑定的Python缺失 / 重复 / 未知参数异常\n'
               '反馈JIT签名、位置/关键字参数、Config与launch约定\n'
               '普通function下标调用失败 → 澄清JIT与普通调用边界\n\n'
               '543项相关源码检查 · 408个实际本地SDK正文\n'
               '3629源码文件 · 175运行文件 · 原23工具')],
         'exact G014分支，作者外层Codex。四种正控、两种负控和父版对照验证最新下一HTTP工具结果中的提示；原请求25及候选5仍拒绝。V37的G021-r2已经闭账且未接受；G022的V38最终成绩待闭账，工程资格不等于搜索性能。'),
        ('已有可复验证据的算子与图计算',
         [body(f"small-M GEMM：原独立验证约{gemm['verified_speedup']:.3f}×\n"
               '指定五形状，原数值门与源码SHA保留\n\n'
               f'ogbn-arxiv全图：两设备相对强库 {graph_gains}×\n'
               '169343节点 / 1166243边，窗口与边界另测\n'
               '图投影固定、随机、未训练；不主张分类精度', size=36)],
         '单算子倍率不相乘推断整模型。图计算相对每轮五种合格库最快值约1.59倍，由Codex应用实现，不能证明AGH源代码进化。展示原形状、源码、verify身份和独立样本。'),
        ('APP35：五卡整模型后端的失败',
         [body(f"0/1/2/3/4五卡真实运行，{len(app['rows'])}个工况\n"
               f"编译FP16不正确 {counts['INCORRECT']}\n"
               f"编译候选不可用 {counts['UNAVAILABLE']}\n"
               f"普通eager后端正确可用 {counts['AVAILABLE']}\n\n"
               'GPU2只记辅助正确性，不计加速收益\n'
               '强编译基线未齐，尚无稳定整模型加速')],
         '使用真实预训练GPT-2。普通后端整模型倍率约0.955到1.059，收益不稳定；原输出误差、推理tensor版本计数错误和全部后端状态保留。应用结果不算AGH源码晋级。'),
        ('失败与消耗保留：可以独立审计',
         [body(f"V37闭账时点：{history['request_rows']}全历史请求\n"
               f"{history['unknown_all_history']}未知预留；GPU累计{history['GPU_seconds']:.3f}秒\n"
               '16份原生SQL / 5段应用历史完整继承\n\n'
               'V38独立登记200队列；原V37-r1未付费失败保留\n'
               '原10秒启动 / 5秒状态 / 2秒快照门通过')],
         '这些是闭账时点，不是实时余额。未知预留不退款，当前实验消费另计。V38预模型84项接线检查及两次真实MiMo HTTP200是资格，不能当最终性能。'),
        ('打开作品：源码、正常、边界与失败',
         [body('离线 / HTTP / 桌面 / 手机同一记录证据\n'
               '算子形状与独立验证；真实工具轨迹回放\n'
               'V37全40位置/原成本门；旧V36的K17缺失保留\n'
               'APP35全部工况、失败与GPU2辅助角色\n\n'
               '五份完整冻结源码和19份新增报告可下载')],
         '演示是已记录结果，不能被描述为实时生成。新解压目录的文件库存、所有新增报告及源码归档已经校验，离线和HTTP、桌面和手机验证通过；缺失或无效证据拒绝显示成功。'),
        ('下一步以原准入证据收敛',
         [body('目标：平均至少14/20正确，稳定提升至少10%\n'
               '三组预注册开发配对 + 独立保留与新任务\n'
               '正确性、置信区间、资源效率全部过门\n'
               '连续至少两代接受，才能证明持续进化\n\n'
               '当前：H0保持，RSI未成立\n'
               'G022 / V38本轮最终成绩待闭账')],
         '继续完成当前20路对照，两臂闭账后只用DEV轨迹选择新源码假设。没有排名承诺，不用应用收益替代源代码对照，也不把更多预算当作进化。'),
    ]


def generate(data_path, out):
    data_path, out = Path(data_path), Path(out)
    if out.exists() or out.is_symlink():
        raise ValueError('current slide output must be fresh')
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    provenance = json.loads((data_path.parent / 'current-provenance.json').read_text())
    if provenance['data_sha256'] != digest:
        raise ValueError('current snapshot data identity changed')
    data = json.loads(data_path.read_text())
    content(data)
    legacy = renderer()
    legacy.content = content
    saved = sys.argv
    try:
        sys.argv = ['generate_current_materials_v38_r2.py', '--data', str(data_path), '--out', str(out)]
        legacy.main()
    finally:
        sys.argv = saved
    manifest = out / 'materials-manifest.json'
    proof = json.loads(manifest.read_text())
    proof.update(status='CURRENT_NINE_SLIDES_RENDERED_VIDEO_NOT_UPDATED',
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_renderer_sha256=hashlib.sha256((Path(__file__).parent / 'generate_materials.py').read_bytes()).hexdigest(),
        current_engineering_generation='G022', closed_performance_revision=37,
        new_model_GPU_calls=0, RSI_proven=False, video_updated=False)
    manifest.write_text(json.dumps(proof, indent=2) + '\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.data, args.out)))
