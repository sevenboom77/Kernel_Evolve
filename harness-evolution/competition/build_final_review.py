"""Build a fresh, offline review of the closed October 9 evidence.

This reads frozen reports and copies original bytes; it performs no search,
GPU evaluation, credential access or model request. New results need a new
explicit release snapshot instead of overwriting this evidence set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

REPORTS = {
    'native': ('final-native-small-M-two-device-accurate-FP16-original-protocol-r1.json', '7a89b1d5e87f9333a7f412bc39b9bcea290d6b9f1e37db2dc7ec650e939420cf'),
    'default_native': ('native-small-M-final-two-device-default-FP16-comparison-r1.json', 'e95d56a6ae73af527b92e4a51ec91b8af407a95d41dbbf4d3b611ab3a2c0d7cc'),
    'graph': ('final-independent-citation-graph-two-device-actual-reproduction-r1.json', '0c5bcfbc073c08bab18d0ba9de44a120993a3bc94fc153d12f82e2497af9159f'),
    'origin': ('original-native-MiMo-small-M-actual-source-trace-verification-r1.json', '0311c3cf19e66203b062fe453f74393d6b1c6b7efa9db0d91c0b2de0d7cbb247'),
    'attempt': ('v51-finite-first-screening-interrupted-registered-pool-r1.json', '80c91851653d5ac93e4655463106d12c6cdd69b02fbbadb2d48c684e3de744d3'),
    'history': ('v51-finite-first-screening-whole47754-exact-closed-history-r1.json', '836785b474e6ddc7fd513f547e3cc1fcd41fc9cbf8bcd77aa80afc83578c0f81'),
    'pair': ('g030-v50-original-two-arm-public-all40-r1.json', 'f90c7d031c5cffd19be5f6b5ca323f9d56dfe3879f1306b27f1ed88d9f82c953'),
    'sdk': ('g031-actual-source-full-SDK-qualification-r1.json', '86faa678d7d477c7939811ba02221f0c04d4002a57c01a39130a803918a568de'),
    'cost': ('final-six-independent-GPU-reproductions-closed-cost-appendix-r1.json', 'a6d2ca1bc1823203aba9d8a3f221511c2955648797b87a614f31a78e6f3abbd0'),
}
KEY = re.compile(rb'\bsk-[A-Za-z0-9_-]{32,100}\b')
GRAPH_LIBRARIES = {'segment_reduce', 'index_add', 'sparse_mm', 'compiled_segment_reduce', 'compiled_index_add'}
SHAPES = {'t0', 't1', 't2', 'r0', 'r1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(ref):
    path = Path(ref['path'])
    if not path.is_file() or path.is_symlink() or sha(path) != ref['sha256']:
        raise ValueError('original evidence identity changed: ' + str(path))
    return path


def load_reports(repo):
    parent = Path(repo) / 'harness-evolution/codex-rsi/reports'
    return {key: json.loads(bound({'path': str(parent / name), 'sha256': digest}).read_text())
            for key, (name, digest) in REPORTS.items()}


def make_snapshot(docs):
    native, graph, origin = docs['native'], docs['graph'], docs['origin']
    if graph['source_author'] != 'Codex' or graph['classification_accuracy_or_training_gain_claimed'] is not False:
        raise ValueError('graph author or untrained scope changed')
    if origin['chain']['model'] != 'mimo-v2.6-pro' or not origin['model_author_of_exported_kernel_confirmed_from_actual_write']:
        raise ValueError('native model provenance unavailable')
    native_rows = []
    for row in native['replicas']:
        qualifications = row['strong_libraries_21_case_qualification']
        if row['correctness_cases_passed'] != 21 or row['rounds'] != 5 or any(
                value['status'] != 'ok' or value['detail']['passed'] != 21
                for value in qualifications.values()) or len(qualifications) != 2:
            raise ValueError('native correctness or required library qualification incomplete')
        stat = row['comparisons']['all_qualified_libraries']['statistics']
        if set(stat['shape_speedup_medians']) != SHAPES or stat['rounds_used'] != 5:
            raise ValueError('native original shapes or rounds incomplete')
        failures = row['retained_default_FP16_library_failures']
        # Failed default libraries remain in the record and cannot be promoted.
        default_passed = [value['detail']['passed'] for value in failures.values()]
        if default_passed != [19, 19]:
            raise ValueError('original default library failure record changed')
        native_rows.append({'gpu': row['physical_gpu'], 'uuid': row['physical_gpu_uuid'],
            'strong_mean_speedup': stat['task_speedup_mean'],
            'strong_ci': [stat['task_speedup_ci_low'], stat['task_speedup_ci_high']],
            'original_baseline_mean_speedup': row['comparisons']['original_baselines']['statistics']['task_speedup_mean'],
            'shape_speedups': stat['shape_speedup_medians'],
            'candidate_us': stat['candidate_us_medians'], 'baseline_us': stat['paired_baseline_us_medians'],
            'default_library_cases_passed': default_passed[0], 'strong_library_cases_passed': 21})
    graph_rows = []
    for row in graph['replicas']:
        cases = []
        if {case['case_id'] for case in row['cases']} != {'arxiv-full', 'arxiv-first-2048', 'boundary-fixture'}:
            raise ValueError('graph original cases incomplete')
        for case in row['cases']:
            controls = case['backend_qualification']
            if set(controls) != GRAPH_LIBRARIES | {'candidate'} or any(
                    value['status'] != 'AVAILABLE' or value['timing_admitted'] is not True
                    for value in controls.values()):
                raise ValueError('graph library qualification incomplete')
            gain = case['strong_gain']
            if gain['paired_rounds'] != 30 or len(gain['paired_ratio_samples']) != 30:
                raise ValueError('graph original paired samples incomplete')
            cases.append({'case_id': case['case_id'], 'speedup': gain['median'],
                'ci': gain['bootstrap_95_percent_interval'], 'paired_rounds': 30,
                'qualified_library_count': len(GRAPH_LIBRARIES), 'victory_proven': gain['victory_proven']})
        graph_rows.append({'gpu': row['physical_gpu'], 'cases': cases})
    pair, attempt, history, sdk = (docs[key] for key in ('pair', 'attempt', 'history', 'sdk'))
    promotion = pair['original_promotion']
    if attempt['candidate']['model_tasks_started'] != 0 or attempt['paired_performance_ratio'] is not None:
        raise ValueError('frozen partial attempt must not be labelled as a completed pair')
    return {'schema_version': 1, 'release_id': 'review-20261009-closed-r1', 'mode': 'recorded',
        'evidence_date': '2026-10-09', 'rsi_performance_established': False,
        'application': {'status': 'THREE_GRAPH_CASES_TWO_4090_REPRODUCED'},
        'native': {'source_author': 'AGH using mimo-v2.6-pro, original H0',
            'source_sha256': origin['chain']['source_file_sha256'], 'correctness_cases': 21,
            'rounds': 5, 'strong_gain_certified': native['strong_full_task_original_gate_passed'],
            'strong_precision': native['strong_precision_control'], 'replicas': native_rows,
            'native_chain': origin['chain']},
        'graph': {'source_author': 'Codex', 'trained': False, 'classification_gain_claimed': False,
            'scope': '经条件化输入与固定随机权重的两层FP32图计算；全图、窗口和边界',
            'nodes': 169343, 'edges': 1166243, 'replicas': graph_rows},
        'engineering': {'generation': 'G031', 'source_author': 'Codex outer loop',
            'source_commit': sdk['source_commit'], 'actual_synthetic_HTTP_bodies': sdk['SDK']['actual_total_HTTP_bodies'],
            'external_model_calls_in_SDK': 0, 'paid_gain_proven': False,
            'implementation': sdk['implementation']},
        'last_complete_pair': {'revision': 50, 'candidate_generation': 'G030',
            'all_positions': 40, 'arms': [{'arm': arm['arm'], 'counts': arm['counts']} for arm in pair['arms']],
            'speedup': promotion['deployment_geomean_ratio'],
            'ci': [promotion['paired_ci_low'], promotion['paired_ci_high']], 'accepted': False},
        'latest_attempt': {'revision': 51, 'candidate_generation': 'G031',
            'candidate_started': False, 'paired_speedup': None, 'status': attempt['status'],
            'parent_counts': attempt['parent']['counts'], 'consumed_positions': 40,
            'remaining_positions': 160, 'refunds': 0},
        'history': {'requests': history['request_count'], 'unknown': history['unknown_all_history'],
            'native_SQL': history['native_SQL_copies'], 'new_independent_GPU_seconds': docs['cost']['new_independent_GPU_seconds']},
        'source_reports': {key: {'path': 'evidence/' + name, 'sha256': digest} for key, (name, digest) in REPORTS.items()},
        'limitations': ['RSI与G031搜索收益未证明；工程检查和合成HTTP不等于性能收益。',
            '图源码作者Codex，固定权重未训练；没有分类精度或整模型收益结论。',
            '只验证两个物理4090设备，未验证跨GPU架构。',
            '默认FP16库19/21失败全部保留；强库精度配置显式声明。',
            '报名身份、模型资格确认、公开链接和实际提交仍为外部依赖。']}


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def copy_bound(ref, out, relative, mappings):
    source = bound(ref)
    data = source.read_bytes()
    if KEY.search(data):
        raise ValueError('credential pattern refused in delivery source: ' + source.name)
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(data)
    mappings.append({'original': ref, 'portable_path': relative, 'sha256': sha(destination)})


def render(data):
    esc = html.escape
    report = lambda key: '<a download href="' + data['source_reports'][key]['path'] + '">原报告与全部失败</a>'
    ci = lambda values: f'[{values[0]:.6f}, {values[1]:.6f}]'
    native_rows = ''.join(f'<tr><td>GPU{r["gpu"]}</td><td>{r["strong_mean_speedup"]:.6f}×</td><td>{ci(r["strong_ci"])}</td><td>{r["original_baseline_mean_speedup"]:.6f}×</td></tr>' for r in data['native']['replicas'])
    shapes = ''.join(f'<tr><td>{shape}</td>' + ''.join(f'<td>{r["shape_speedups"][shape]:.4f}×</td><td>{r["candidate_us"][shape]:.2f} / {r["baseline_us"][shape]:.2f} μs</td>' for r in data['native']['replicas']) + '</tr>' for shape in ('t0', 't1', 't2', 'r0', 'r1'))
    graph_panels = ''
    for case_id in ('arxiv-full', 'arxiv-first-2048', 'boundary-fixture'):
        rows = ''.join(f'<tr><td>GPU{r["gpu"]}</td><td>{c["speedup"]:.6f}×</td><td>{ci(c["ci"])}</td><td>5 / 30</td></tr>' for r in data['graph']['replicas'] for c in r['cases'] if c['case_id'] == case_id)
        graph_panels += f'<div data-case="{case_id}" {"hidden" if case_id != "arxiv-full" else ""}><h3>{case_id}</h3><table><tr><th>设备</th><th>强库速度比</th><th>95% CI</th><th>合格库 / 配对轮</th></tr>{rows}</table></div>'
    links = ''.join('<li><a download href="'+v['path']+'">'+esc(k)+' · '+esc(Path(v['path']).name)+'</a></li>' for k,v in data['source_reports'].items())
    limits = ''.join('<li>'+esc(v)+'</li>' for v in data['limitations'])
    native = data['native']['native_chain']
    pair = data['last_complete_pair']
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KernelEvolve · 最终作品证据</title>
<style>:root{{color-scheme:dark;font-family:"Noto Sans CJK SC",system-ui,sans-serif;background:#0c1017;color:#edf2f7}}body{{max-width:1120px;margin:auto;padding:32px 24px 100px}}a{{color:#91c7ff}}nav{{display:flex;gap:18px;flex-wrap:wrap;margin:28px 0}}section{{border-top:1px solid #293544;padding:32px 0;scroll-margin-top:20px}}h1{{font-size:clamp(36px,5vw,60px);line-height:1.3;letter-spacing:-1px}}h2{{font-size:30px}}p,li{{font-size:18px;line-height:1.8}}.muted{{color:#a4b5c8}}.tag{{display:inline-block;border:1px solid #49683e;border-radius:30px;padding:6px 14px;color:#a9f06e}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{background:#151e2a;padding:24px;border-radius:12px}}.metric{{font-size:38px;color:#a9f06e;font-weight:650}}.notice{{border-left:4px solid #ffd28b;padding:10px 20px;background:#1c2028;color:#ffd28b}}table{{width:100%;border-collapse:collapse;margin:20px 0;display:block;overflow:auto}}td,th{{text-align:left;white-space:nowrap;border-bottom:1px solid #293544;padding:12px 18px}}button{{font:inherit;color:#edf2f7;background:#253448;padding:12px 20px;border:1px solid #5c748c;border-radius:6px;cursor:pointer}}button[aria-pressed=true]{{background:#355222;color:#c5ff92}}pre{{white-space:pre-wrap;word-break:break-word;background:#151e2a;padding:20px;line-height:1.6}}code{{font-family:monospace}}details{{padding:14px;background:#151e2a;margin:12px 0}}.scroll{{overflow:auto}}footer{{color:#a4b5c8}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}body{{padding:24px 16px}}h2{{font-size:25px}}td,th{{padding:10px}}}}[hidden]{{display:none!important}}</style>
<header id="intro"><span class="tag">硕博组 · 2026-10-09 闭合实测回放</span><h1>让自动生成的 GPU 代码<br>接受正确性与强基线检验</h1><p>KernelEvolve 使用 AGH 与 mimo-v2.6-pro 生成、提交、诊断和修复算子。代码来源、失败与独立复测共同构成结果。</p><p class="muted">当前页面展示已记录证据；点击不会发起模型请求或 GPU 作业。</p>
<div class="cards"><div class="card"><div class="metric">21 / 21</div>原生生成 small-M 正确性</div><div class="card"><div class="metric">1.59×</div>真实全图计算强库速度比</div><div class="card"><div class="metric">未证明</div>G031 搜索收益与 RSI</div></div></header>
<nav><a href="#native-chain">原生闭环</a><a href="#native">GEMM实测</a><a href="#graph">真实图计算</a><a href="#engineering">AGH改动</a><a href="#measurement">对照结论</a><a href="#failures">失败与边界</a><a href="#reproduction">复现入口</a><a href="#downloads">证据下载</a></nav>
<section id="native-chain"><h2>一次实际的生成、失败和修复</h2><p>AGH读取任务 → MiMo写源码 → 原生提交 → GPU反馈 → 修订 → 独立验证。前两候选 crashed / compile_fail 全部保留，第三候选 c003 完成验证。</p><pre>write {native['model_write_seq']} → receipt {native['write_result_seq']} → submit {native['submit_call_seq']} → receipt {native['submit_result_seq']}
run: {esc(native['run_id'])}
candidate: {native['candidate_id']} · verify: {native['verify_job_id']}
kernel SHA256: {native['source_file_sha256']}</pre><p>{report('origin')} · <a download href="sources/K12-kernel.py">MiMo原源码</a> · <a download href="raw/native-original/agh-events.jsonl">原生事件</a></p></section>
<section id="native"><h2>small-M GEMM：正确，收益有明确范围</h2><p>FP16输入/输出、FP32累加；原21案例、五形状、五轮全部保留。显式FP32归约的两种直接/编译FP16强库均通过21/21。默认FP16两库只通过19/21，失败仍可查看。</p><table><tr><th>设备</th><th>全部合格库整体速度比</th><th>原95% CI</th><th>原转换基线速度比</th></tr>{native_rows}</table><p class="notice">两卡均未通过原CI下限&gt;1.05门。不能把原约4倍转换基线收益写成强库胜利；不删首轮、不继续补跑逼出正结果。</p><h3>五形状中位数 · candidate / baseline时延</h3><table><tr><th>形状</th><th>GPU1速度比</th><th>GPU1时延</th><th>GPU4速度比</th><th>GPU4时延</th></tr>{shapes}</table><p>{report('native')} · {report('default_native')}</p></section>
<section id="graph"><h2>真实引文图：完整计算约1.59×</h2><p>ogbn-arxiv · 169343节点 · 1166243边。每轮相对五个数值合格库的最快值；各案例30组GPU与主机配对。两张4090独立复测。</p><p><button data-select="arxiv-full" aria-pressed="true">全图</button> <button data-select="arxiv-first-2048" aria-pressed="false">2048节点窗口</button> <button data-select="boundary-fixture" aria-pressed="false">边界案例</button></p>{graph_panels}<p class="notice">作者Codex；输入条件化、固定随机权重的两层FP32计算未训练。此结果不代表分类精度、整模型训练收益或原生AGH生成。</p><p>{report('graph')} · <a download href="sources/citation_graph_split_csr.py">图计算源码</a></p></section>
<section id="engineering"><h2>AGH具体改动：把可靠反馈送入下一次执行</h2><p>G031：首次可信DEV数值错误立即进入既有、有界的修复焦点。原始反馈保持；拒绝无来源诊断；正确候选仍单独保存；压缩/恢复后的状态有明确上限。</p><p>沿革还包括原生剩余预算、运行中有界续查、重复写入反馈、输出截断后的有界续接和可信搜索记忆。源码作者为外层Codex。</p><pre>G031 commit: {data['engineering']['source_commit']}
3633源码 / 175原运行文件 / 1587实际合成HTTP正文
24原生请求 / 4候选 / 4096输出 / 32768上下文保持</pre><p class="notice">工程与合成HTTP证明机制；没有G031正式搜索收益证明，accepted仍为0。</p><p>{report('sdk')} · <a download href="sources/G031-source.patch">G014→G031补丁</a></p></section>
<section id="measurement"><h2>全任务对照保留负结果</h2><p>最近完整V50：G014/G030每臂20任务，全部40位置保留。整体比 {pair['speedup']:.6f}，95%CI {ci(pair['ci'])}；未晋级。</p><p>后续V51尝试：父20终态、13最终评测、7环境错误、6正确；GPU3持续忙导致复验等待，已正常停止。G031子未启动，配对比为空，已预留40位置不退还。</p><p>{report('pair')} · {report('attempt')}</p></section>
<section id="failures"><h2>正常、边界与失败都有可执行证据</h2><p>正常与边界：原21个GEMM检查及真实图三案例。失败：两次原生生成失败、默认FP16库数值不合格、忙卡拒绝与最新对照环境中止。</p><details><summary>查看默认FP16库为什么不进入胜利主张</summary><p>f12/f14未过原数值门，19/21记录保留。显式FP32归约比较另行声明，仍沿用原参考与容差。</p></details><details><summary>查看忙卡如何处理</summary><p>每个GPU作业前 fresh util=0 / free≥3072MiB；不结束外部进程。旧无期限启动等待已发现，新的有限等待通过7个CPU检查；正式下一控制器接入仍待验证。</p></details><details><summary>历史请求与未知消耗</summary><p>47754请求 / 34309历史unknown / 41原生SQL均完整留档；unknown不是确认账单。本次六独立GPU复测另账2773.401651 wall-span秒；失败不退款。</p></details></section>
<section id="reproduction"><h2>按原输入、源码与样本独立核查</h2><p>默认入口生成CPU计划；只有显式 --execute 才启动GPU测量。Python3.11 / PyTorch2.5.1+cu121 / Triton3.1，实测硬件RTX4090。</p><pre>python harness-evolution/competition/reproduce_native_small_m_fp32_reduction.py --out /tmp/new-native-plan
python harness-evolution/competition/reproduce_native_small_m_fp32_reduction.py --out /tmp/new-native-run --gpu 0 --execute
python harness-evolution/competition/reproduce_citation_graph.py --archive datasets/arxiv.zip --out /tmp/new-graph-plan --case all
python harness-evolution/competition/reproduce_citation_graph.py --archive datasets/arxiv.zip --out /tmp/new-graph-run --gpu 0 --case all --execute</pre><p>各次复制保留原SQL、逐轮样本和最终结果。演示无需GPU或模型凭据；新AGH任务需使用者通过环境提供自己的凭据。</p></section>
<section id="downloads"><h2>证据与交付范围</h2><p><a download href="data.json">本页冻结数据</a> · <a download href="portable-evidence-map.json">原字节与包内路径映射</a> · <a download href="review-manifest.json">本页完整哈希清单</a></p><ul>{links}</ul><h3>限制与外部依赖</h3><ul>{limits}</ul></section><footer>{data['release_id']} · 全部主张来自闭合记录 · 不以等待RSI或比赛名次延长研发</footer>
<script>document.querySelectorAll('[data-select]').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('[data-case]').forEach(panel=>panel.hidden=panel.dataset.case!==button.dataset.select);document.querySelectorAll('[data-select]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));}}));</script></html>'''


def build(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise ValueError('fresh review output required')
    docs = load_reports(repo)
    data = make_snapshot(docs)
    out.mkdir(parents=True, mode=0o700)
    mappings = []
    for key, (name, digest) in REPORTS.items():
        copy_bound({'path': str(repo / 'harness-evolution/codex-rsi/reports' / name), 'sha256': digest}, out, 'evidence/' + name, mappings)
    copy_bound(docs['origin']['exported_kernel'], out, 'sources/K12-kernel.py', mappings)
    patch = docs['sdk']['source_patch']
    patch_path = Path(patch['path'])
    if not patch_path.is_absolute():
        patch_path = repo / 'harness-evolution/codex-rsi' / patch_path
    copy_bound({'path': str(patch_path), 'sha256': patch['sha256']}, out,
               'sources/G031-source.patch', mappings)
    for name in ('citation_graph_split_csr.py', 'benchmark_citation_graph_split_csr.py'):
        path = repo / 'harness-evolution/competition/application' / name
        copy_bound({'path': str(path), 'sha256': sha(path)}, out, 'sources/' + name, mappings)
    for ref in docs['origin']['source_bindings'][:3]:
        copy_bound(ref, out, 'raw/native-original/' + Path(ref['path']).name, mappings)
    for key in ('native', 'default_native', 'graph'):
        for replica in docs[key]['replicas']:
            for label in ('execution', 'result', 'local_SQL'):
                if label in replica:
                    copy_bound(replica[label], out,
                        f'raw/{key}/gpu{replica["physical_gpu"]}/{label}' + ('.sqlite' if label == 'local_SQL' else '.json'), mappings)
    write_json(out / 'data.json', data)
    write_json(out / 'portable-evidence-map.json', {'copies': mappings, 'original_bytes_unchanged': True,
        'original_private_paths_are_provenance_only': True, 'new_model_GPU_calls': 0})
    (out / 'index.html').write_text(render(data))
    files = [{'path': p.relative_to(out).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(out.rglob('*')) if p.is_file()]
    write_json(out / 'review-manifest.json', {'release_id': data['release_id'], 'created_at': datetime.now(timezone.utc).isoformat(),
        'files': files, 'data_sha256': sha(out / 'data.json'), 'builder_sha256': sha(__file__),
        'status': 'ASSEMBLED_BROWSER_VERIFICATION_PENDING', 'source_commit': 'fd7e800',
        'RSI_proven': False, 'new_model_GPU_calls': 0})
    return {'out': str(out), 'files': len(files) + 1, 'data_sha256': sha(out / 'data.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.out)))
