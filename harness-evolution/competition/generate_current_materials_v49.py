"""Render the fixed closed V47 / G029 engineering snapshot with the original renderer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PARENT = 'e1dffe9a8eca77cb1fdd81c1a8b5842c8c36370e'
SOURCE = '34378fbc80c1f0060b4de27a9514bf94895a1fc5'


def renderer():
    path = Path(__file__).parent / 'generate_materials.py'
    spec = importlib.util.spec_from_file_location('closed_v47_original_slide_renderer', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def content(data):
    latest = next((x for x in data.get('source_changes', []) if x['generation'] == 'G029'), None)
    ablation = data.get('closed_ablation', {})
    arms = ablation.get('arms', [])
    ready = data.get('pending_source_measurement', {})
    if (data.get('mode') != 'recorded' or data.get('rsi_performance_established') is not False
        or ablation.get('comparison', {}).get('campaign_revision') != 47 or latest is None
        or latest.get('commit') != SOURCE or latest.get('measurement_parent_source_commit') != PARENT
        or latest.get('actual_localhost_request_bodies') != 1236
        or latest.get('formal_source_performance_promoted') is not False
        or latest.get('actual_restarted_output_limit_SDK_turn_tested') is not False
        or latest.get('failed_new_output_fixture_HTTP_bodies_retained') != 7
        or latest.get('full_Core_passed') != 2390 or len(arms) != 2
        or [x['evaluation_complete_count'] for x in arms] != [20, 19]
        or [x['correct_count'] for x in arms] != [10, 10]
        or arms[1]['missing_evaluation_task_ids'] != ['K16']
        or any(len(x['task_contributions']) != 20 for x in arms)
        or ready.get('phase_revision') != 48 or ready.get('CPU_checks') != 144
        or ready.get('registered_queues') != 250 or ready.get('performance_result_available') is not False):
        raise ValueError('current recorded scope differs')
    r = renderer()
    def body(text, y=270, height=620, size=36, color=None):
        return r.box(120, y, 1660, height, text, size, color or r.WHITE)
    pairs = {s['comparison']['campaign_revision']: s for s in data['series_history'] if s.get('comparison')}
    pairs[47] = ablation
    rows = []
    for revision in (44, 45, 46, 47):
        pair = pairs[revision]
        parent, child = pair['arms']
        rows.append(f"V{revision} {pair['comparison']['candidate_generation']}：正确{parent['correct_count']}→{child['correct_count']}，"
            f"评测{parent['evaluation_complete_count']}→{child['evaluation_complete_count']} / 20\n"
            f"全任务比 {pair['development_geomean_ratio']:.6f}；95%CI [{pair['development_paired_ci_low']:.6f}, {pair['development_paired_ci_high']:.6f}]")
    graph = data['graph_application']
    graph_gains = '/'.join(f"p{p['physical_gpu']} {p['strong_gain']['median']:.4f}" if 'physical_gpu' in p else f"{p['strong_gain']['median']:.4f}" for p in graph['replicas'])
    gemm = next(x for x in data['operator_variants'] if x['task_id'] == 'K12')
    app = data['whole_model_application_APP35']
    counts = app['backend_status_counts']
    return [
        ('KernelEvolve：闭账证据驱动源码迭代',
         [body('20路原生AGH任务 → GPU验证 → 全闭后诊断\n'
               '批量修改真实AGH → 同条件复测 → 原门接受或拒绝\n\n'
               '源码作者：外层Codex\n'
               'accepted0 · H0保持 · RSI未成立', size=42)],
         '硕博组作品。开发者驱动的真实源码迭代已经运行，持续性能改善仍未验收；不能把修复提示、工程检查或应用收益当成RSI。'),
        ('每次运行前都现场验证GPU利用率',
         [body('mimo-v2.6-pro · 每组20原生任务 / 10 daemon workers\n'
               'GPU0–4五评测器，每次GO与GPU作业新探util0\n'
               '剩余显存至少3072MiB，允许空闲进程驻留\n\n'
               '原24请求 / 四候选 / 4096输出 / 32K上下文\n'
               '源码commit → 独立归档 → 175文件运行包')],
         '累计token、美元、GPU时间和旧截止无上限。任务限额、原数值/评分门仍保持，不结束外部进程。这里描述当前实验协议；APP35的GPU2辅助角色是历史协议。'),
        ('四组原闭账：正向结果也须过完整门',
         [body('\n'.join(rows), height=480, size=25),
          body('V47：K16缺失，19最终评测 / 全20分母；GPU成本比1.341300。', y=820, height=110, size=29, color=r.ORANGE)],
         'V44的G026单组全任务收益约12.09%，但GPU资源门失败；V45重复没有保持，G027/V46和G028/V47也未过门。V47比1.003319，CI[0.966934,1.045184]跨1，未接受。缺失任务和失败成本全部保留，不能通过补跑或更换分母改写这一组。'),
        ('G029：输出截断后一次有界续接',
         [body('原生kernel回合，预算仍可用时允许一次续接\n'
               '半截tool参数全部丢弃，保留原费用与额度\n'
               '再次截断 / 取消 / 耗尽 / 普通工具仍终止\n\n'
               '2390 Core通过 · 1236 SDK正文 · 另七失败保留\n'
               '3633源码 / 175运行文件；SDK重启turn未测', size=34)],
         '作者外层Codex，工程父G028明确被拒绝，测量父仍exact G014。原31 SDK阶段加六新场景全部通过；重放/取消/作用域与活预算在真实SessionImpl验证，不能写成已实测重启SDK回合。G029付费对照在此快照没有未闭账性能成绩，不能由工程资格推导收益。'),
        ('算子与图计算：保留原适用范围',
         [body(f"历史small-M GEMM：原独立验证约{gemm['verified_speedup']:.3f}×\n"
               '仅指定五形状，原数值门与源码SHA保持\n\n'
               f'ogbn-arxiv全图：两设备相对强库 {graph_gains}×\n'
               '169343节点 / 1166243边，另有窗口和边界\n'
               '投影固定随机、未训练；不主张分类精度', size=35)],
         'GEMM来自历史H0搜索，不归因于G029。图应用约1.59倍由Codex实现，不证明AGH源代码进化。单算子倍率不相乘推断整模型；完整实际样本与验证身份可查。'),
        ('APP35：强整模型基线仍需完成',
         [body(f"真实预训练GPT-2五卡，{len(app['rows'])}个工况\n"
               f"编译FP16不正确 {counts['INCORRECT']}\n"
               f"编译候选不可用 {counts['UNAVAILABLE']}\n"
               f"普通后端正确可用 {counts['AVAILABLE']}\n\n"
               '此历史APP35的GPU2只记辅助正确性\n'
               '强编译基线未齐，尚无稳定整模型加速')],
         '全部12工况和失败保留。普通后端倍率约0.955到1.059，收益不稳定；不能忽略原logit误差、推理tensor版本计数问题或把辅助正确性计为加速。应用不代替AGH源对照。'),
        ('历史与登记时点：消耗不退款',
         [body('V47完整carry：45537请求 / 32757未知预留\n'
               '35份原生SQL / 245队列 / 五段APP历史\n'
               '未知、失败和原费用完整继承\n\n'
               'V48登记：144项CPU资格 / 250队列\n'
               '固定登记证据，无V48闭账性能结果')],
         '这些是原V47 carry和V48登记时点，不是实时余额。当前新增消耗之后完整接续，不能退还未知或删除失败。历史V37卡片仍明确标为历史；活动V48源码保持冻结，本材料不读取中间DEV成绩。'),
        ('现场打开：正常、边界、失败与来源',
         [body('离线 / HTTP / 桌面 / 手机同一记录展示\n'
               'V38–V47九组已闭，360个原任务位置\n'
               'V47全部40位置 / CI / K16缺失 / 原成本门\n'
               'G023–G029真实patch与工程报告，APP35全部失败\n\n'
               '42 CPU / 四视图 / 107相对证据下载已验证')],
         '演示是固定记录，不是现场模型生成。九组为V38、39、40、41、42、44、45、46、47。独立zip163文件与21固定报告已验；缺失或非法数据拒绝成功。最新完整源码便携包与PPT/视频的资格须在各自实际执行后声明。'),
        ('持续收益的验收目标保持',
         [body('平均至少14/20正确，稳定提升至少10%\n'
               '三组预注册开发配对 + 独立保留和新任务\n'
               '正确性、置信区间、资源效率全部通过\n'
               '连续至少两代接受，并完成强APP/跨设备\n\n'
               '当前accepted0 / H0保持 / RSI未成立\n'
               'G029工程完成；V48登记无闭账性能成绩')],
         '继续当前两组闭账后批量分析、修改真实AGH、工程验证再实测。不承诺比赛名次，不使用应用收益替代源码收益，不把更多消耗当成进化。'),
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
        sys.argv = ['generate_current_materials_v49.py', '--data', str(data_path), '--out', str(out)]
        legacy.main()
    finally:
        sys.argv = saved
    manifest = out / 'materials-manifest.json'
    proof = json.loads(manifest.read_text())
    proof.update(status='CURRENT_V47_G029_NINE_SLIDES_RENDERED_VIDEO_NOT_UPDATED',
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_renderer_sha256=hashlib.sha256((Path(__file__).parent / 'generate_materials.py').read_bytes()).hexdigest(),
        current_engineering_generation='G029', closed_performance_revision=47,
        new_model_GPU_calls=0, RSI_proven=False, video_updated=False)
    manifest.write_text(json.dumps(proof, indent=2) + '\n')
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(generate(a.data, a.out)))
