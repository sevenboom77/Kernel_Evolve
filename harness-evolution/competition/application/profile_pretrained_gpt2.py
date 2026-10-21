"""Accounted diagnostic profiling of the unchanged local GPT-2 library forward."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
C = Path('/data/lizhijun/agh-evo/codex-rsi-hackathon-v11-20261005')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
import pretrained_gpt2 as model_app

POLICY_SHA = 'b7bcffb8a7825d04becc24596048ecdc4311a4935ce20e7374e504523418f35b'
CARRY_SHA = '68b590910ce31317ce907d95956ce71e168efc3b765428778695de9be382b4ca'
BACKENDS = ('ncu', 'nsys', 'pytorch')

def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': model_app.sha(path)}

def bound(r):
    if ref(r['path']) != r:
        raise ValueError('profiling source binding changed')
    return json.loads(Path(r['path']).read_text())

def put(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    Path(path).chmod(0o600)
    return ref(path)

def make_plan(model_dir, backend):
    if backend not in BACKENDS:
        raise ValueError('declared profiling backend required')
    return {'status': 'GPU_PENDING', 'kind': 'diagnostic-pretrained-library-profiling',
            'model': model_app.model_receipt(model_dir), 'shapes': [list(x) for x in model_app.SHAPES],
            'backend': backend, 'warmup': 20, 'profiled_forwards_per_shape': 5,
            'max_wall_seconds': 900, 'exact_logits_before_and_after_required': True,
            'workload': 'identical FP16 SDPA library full forward including logits; no cache',
            'performance_measured': False, 'candidate_measured': False,
            'model_API_calls': 0, 'RSI_proven': False,
            'source': [ref(__file__), ref(HERE / 'pretrained_gpt2.py'),
                       ref(HERE / 'benchmark_pretrained_gpt2.py'), ref(HERE / 'benchmark_decoder_v3.py')]}

def validate_scope(policy, backend, queue, gpu):
    queue = Path(queue).resolve()
    expected = C / f'application/post-v26-r1/profiler-{backend}/queue.sqlite'
    if (backend not in BACKENDS or type(gpu) is not int or gpu not in [0, 1]
            or queue != expected or str(queue) not in policy['gpu_queue_paths']
            or len(policy['gpu_queue_paths']) != 163 or not policy['application_only']
            or policy['reliability_revision'] != 27 or policy['gpu_snapshot_max_age_ms'] != 2000
            or policy['prior_post_v26_closed_phase']['sha256'] != CARRY_SHA):
        raise ValueError('exact registered current profiling application scope required')
    return queue

def worker(args):
    """Nested tool process; its outer owner holds the original lock/job/observer."""
    import sqlite3
    import torch
    from benchmark_pretrained_gpt2 import parameter_sha, tensor_sha, verify_device_binding
    plan = bound({'path': str(args.worker_plan), 'sha256': args.worker_plan_sha256})
    assert plan == make_plan(args.model_dir, args.backend)
    policy = bound({'path': str(C / 'policy-post-v26-application-r1.json'), 'sha256': POLICY_SHA})
    validate_scope(policy, args.backend, args.accounting_queue, args.gpu)
    with sqlite3.connect(Path(args.accounting_queue).as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute("SELECT lease_owner FROM jobs WHERE status='running'").fetchall()
    assert rows == [(f'application:{args.outer_pid}',)]
    assert Path(f'/proc/{args.outer_pid}').exists() and os.getpgid(args.outer_pid) == os.getpgrp()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    probe = torch.empty(1, device='cuda'); torch.cuda.synchronize()
    device = verify_device_binding(args.gpu)
    model = model_app.load_model(args.model_dir).cuda().eval()
    tokens = model_app.input_ids(args.model_dir)
    params = parameter_sha(model)
    cases = []
    with torch.inference_mode():
        for label, cpu_ids in tokens.items():
            ids = cpu_ids.cuda()
            for _ in range(20): model(ids, use_cache=False).logits
            torch.cuda.synchronize()
            reference = model(ids, use_cache=False).logits
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push('gpt2_profile')
            try:
                if args.backend == 'pytorch':
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                            profile_memory=True, with_stack=False) as profiler:
                        for _ in range(5): actual = model(ids, use_cache=False).logits
                        torch.cuda.synchronize()
                    trace = args.out.parent / (label + '-pytorch-trace.json')
                    profiler.export_chrome_trace(str(trace))
                    events = [{'name': e.key, 'calls': e.count,
                               'self_CPU_us': e.self_cpu_time_total,
                               'self_device_us': e.self_device_time_total,
                               'device_us': e.device_time_total}
                              for e in profiler.key_averages()]
                    table = sorted(events, key=lambda x: x['self_device_us'], reverse=True)
                else:
                    for _ in range(5): actual = model(ids, use_cache=False).logits
                    torch.cuda.synchronize(); table = None; trace = None
            finally:
                torch.cuda.nvtx.range_pop()
            assert actual.shape == reference.shape and actual.dtype == reference.dtype
            assert torch.equal(actual, reference), 'profiling changed original library logits'
            cases.append({'label': label, 'input_sha256': tensor_sha(cpu_ids),
                          'logits_sha256': tensor_sha(actual), 'exact_logits_preserved': True,
                          'events': table, 'trace': ref(trace) if trace else None})
    assert parameter_sha(model) == params and model_app.model_receipt(args.model_dir) == plan['model']
    put(args.out, {'status': 'ACTUAL_UNCHANGED_LIBRARY_PROFILED', 'plan': ref(args.worker_plan),
                   'cases': cases, 'physical_device': device, 'parameters_sha256': params,
                   'all_original_four_shapes': True, 'performance_measured': False,
                   'source_author': 'Codex outer loop', 'model_API_calls': 0, 'RSI_proven': False})

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--backend', choices=BACKENDS, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--accounting-queue', type=Path)
    p.add_argument('--budget-argv-json', type=Path)
    p.add_argument('--budget-argv-sha256')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker-plan', type=Path)
    p.add_argument('--worker-plan-sha256')
    p.add_argument('--outer-pid', type=int)
    args = p.parse_args(argv)
    if args.out.exists(): p.error('fresh output required; preserve previous attempts')
    if args.worker_plan:
        worker(args); return 0
    plan = make_plan(args.model_dir, args.backend)
    if not args.execute:
        put(args.out, plan); return 0
    from benchmark_decoder_v3 import (ApplicationJob, BudgetChecker, NumericGpuLock,
                                      clean_snapshot, wall_limit)
    policy = bound({'path': str(C / 'policy-post-v26-application-r1.json'), 'sha256': POLICY_SHA})
    validate_scope(policy, args.backend, args.accounting_queue, args.gpu)
    checker = BudgetChecker(args.budget_argv_json, args.budget_argv_sha256)
    checker.check()
    plan_ref = put(args.out.parent / 'profile-plan.json', plan)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker-plan', plan_ref['path'],
               '--worker-plan-sha256', plan_ref['sha256'], '--outer-pid', str(os.getpid()),
               '--model-dir', str(args.model_dir), '--backend', args.backend, '--gpu', str(args.gpu),
               '--accounting-queue', str(args.accounting_queue), '--out', str(args.out.parent / 'worker-result.json')]
    if args.backend == 'ncu':
        command = ['/usr/local/cuda/bin/ncu', '--target-processes', 'all', '--launch-count', '1',
                   '--metrics', 'sm__throughput.avg.pct_of_peak_sustained_elapsed', '--csv', *command]
    elif args.backend == 'nsys':
        command = ['/usr/local/cuda/bin/nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
                   '--cpuctxsw=none', '--output=' + str(args.out.parent / 'timeline'), *command]
    report = {'status': 'PROFILE_ATTEMPT_FAILED', 'plan': plan_ref, 'argv': command,
              'model_API_calls': 0, 'performance_measured': False, 'RSI_proven': False}
    try:
        with wall_limit(900), NumericGpuLock(ROOT, args.gpu):
            report['clean_before'] = clean_snapshot(args.gpu)
            checker.check()
            protocol_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            with ApplicationJob(args.accounting_queue, 'profile-' + args.backend + '-post-v26-r1',
                                protocol_sha, ref(__file__)['sha256'], args.gpu, 900) as job:
                job.result['report_path'] = str(args.out)
                with (args.out.parent / 'tool.log').open('xb') as f:
                    result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT,
                                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu)}, timeout=850)
                report.update(tool_exit_code=result.returncode, tool_log=ref(args.out.parent / 'tool.log'))
                checker.check()
                if result.returncode:
                    raise RuntimeError('actual profiler tool refused or failed; original log retained')
                worker_report = args.out.parent / 'worker-result.json'
                value = bound(ref(worker_report))
                assert value['all_original_four_shapes'] and all(x['exact_logits_preserved'] for x in value['cases'])
                report.update(status='ACTUAL_CURRENT_LIBRARY_PROFILE_CLOSED', worker_result=ref(worker_report))
            report['clean_after'] = clean_snapshot(args.gpu)
        put(args.out, report); return 0
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:2000])
        put(args.out, report); raise

if __name__ == '__main__':
    raise SystemExit(main())
