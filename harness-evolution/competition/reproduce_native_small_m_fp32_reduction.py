"""Reproduce frozen MiMo K12 with separately declared FP32-reduction FP16 libraries.

The default creates a CPU plan. GPU execution requires --execute, a fresh output,
and per-job admission inside the same physical-card locks used by kernel-lab.
No AGH search, production queue, credential or network is used here.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent / 'application'
INPUTS = APP / 'native-small-m-reproduction-inputs-v1.json'
INPUTS_SHA256 = 'd0880ce56d8e2f60de00084a048d3d43be3980d14f1e36c1050bcc6b123a5a3e'
DEFAULT_FP16_BASELINES = ['torch_mm_fp16', 'compiled_torch_mm_fp16']
STRONG_BASELINES_REQUIRED = ['torch_mm_fp16_fp32_reduction', 'compiled_torch_mm_fp16_fp32_reduction']
ADDITIONAL_BASELINES = DEFAULT_FP16_BASELINES + STRONG_BASELINES_REQUIRED
ORIGINAL_BASELINES = ['torch_eager', 'torch_compile', 'specialized']
SCOPE = 'INDEPENDENT_LOCAL_NATIVE_SMALL_M_REPRODUCTION'


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_new(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return reference(path)


def frozen_bindings():
    if reference(INPUTS)['sha256'] != INPUTS_SHA256:
        raise ValueError('frozen input manifest changed')
    return {key: {'path': str(ROOT / value['relative_path']), 'sha256': value['sha256']}
            for key, value in json.loads(INPUTS.read_text()).items()}


def check_bindings(bindings):
    for key, value in bindings.items():
        if reference(value['path'])['sha256'] != value['sha256']:
            raise ValueError(f'frozen input changed: {key}')


def literal_cases(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'frozen input does not declare {name}')


def make_plan(bindings):
    check_bindings(bindings)
    protocol = json.loads(Path(bindings['protocol']['path']).read_text())
    spec = json.loads(Path(bindings['tasks/K12/spec.json']['path']).read_text())
    case_source = bindings['tasks/K12/cases.py']['path']
    correctness = literal_cases(case_source, 'FINAL_CORRECTNESS')
    timing = literal_cases(case_source, 'DEV_TIMING') + literal_cases(case_source, 'RESERVED_TIMING')
    if protocol.get('timing_shapes') or protocol.get('baseline_qualification_cases'):
        raise ValueError('frozen input has an unregistered case override')
    return {'status': 'GPU_PENDING', 'scope': SCOPE, 'task_id': 'K12',
            'bindings': bindings, 'original_protocol': protocol,
            'operator_source_sha256': bindings['operator_source']['sha256'],
            'original_protocol_sha256': bindings['protocol']['sha256'],
            'correctness_cases': correctness, 'correctness_case_count': len(correctness),
            'baseline_qualification_case_count': len(correctness),
            'baseline_qualification_seed': protocol['seeds']['final'],
            'original_baseline_qualification_case_count': spec['dev_correctness_cases'],
            'timing_cases': timing, 'timing': protocol['timing'],
            'verification_rounds': protocol['final_verification_rounds'],
            'bootstrap': {'unit': 'verification_round', 'samples': 2000, 'seed': 12345},
            'tolerance': {name: spec[name] for name in ('rtol', 'atol')},
            'original_baselines': ORIGINAL_BASELINES.copy(),
            'additional_baselines': ADDITIONAL_BASELINES.copy(),
            'strong_baselines_required': STRONG_BASELINES_REQUIRED.copy(),
            'retained_default_FP16_baselines': DEFAULT_FP16_BASELINES.copy(),
            'strong_baseline_precision': {'allow_fp16_reduced_precision_reduction': False,
                'constructor_control_in_fresh_original_child': True, 'FP16_inputs_cast_to_FP32': False},
            'original_score_or_protocol_modified': False,
            'performance_measured': False, 'RSI_proven': False}


def fresh_output(path):
    path = Path(path)
    resolved = path.resolve()
    if (path.exists() or path.is_symlink() or
            resolved.is_relative_to(Path('/data/lizhijun/agh-evo'))):
        raise ValueError('fresh independent output outside production experiments required')
    return resolved


def extended_baselines(original):
    return original + b'''

# Independent comparison layer; the original source above is byte-preserved.
_original_build = build
NAMES = (*NAMES, "torch_mm_fp16", "compiled_torch_mm_fp16")

class _DirectFP16(torch.nn.Module):
    def forward(self, a, b):
        return (torch.mm(a, b),)

def build(name, static):
    if name == "torch_mm_fp16":
        return _DirectFP16()
    if name == "compiled_torch_mm_fp16":
        return torch.compile(_DirectFP16(), fullgraph=True, dynamic=False)
    return _original_build(name, static)

# Explicit FP32 intermediate reduction, retaining the default FP16 controls.
_default_fp16_build = build
NAMES = (*NAMES, "torch_mm_fp16_fp32_reduction", "compiled_torch_mm_fp16_fp32_reduction")

class _FP32Reduction(_DirectFP16):
    def __init__(self):
        super().__init__()
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

def build(name, static):
    if name == "torch_mm_fp16_fp32_reduction":
        return _FP32Reduction()
    if name == "compiled_torch_mm_fp16_fp32_reduction":
        return torch.compile(_FP32Reduction(), fullgraph=True, dynamic=False)
    return _default_fp16_build(name, static)
'''


def prepare_scope(path, plan):
    check_bindings(plan['bindings'])
    root = fresh_output(path)
    root.mkdir(parents=True, mode=0o700)
    for key, value in plan['bindings'].items():
        if key.startswith(('tasks/', 'kernel_lab/')):
            target = root / key
        elif key == 'operator_source':
            target = root / 'candidate' / 'kernel.py'
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = Path(value['path']).read_bytes()
        with target.open('xb') as stream:
            stream.write(extended_baselines(raw) if key == 'tasks/K12/baselines.py' else raw)
    baseline = root / 'tasks/K12/baselines.py'
    return {'root': str(root), 'candidate': reference(root / 'candidate/kernel.py'),
            'extended_baselines': reference(baseline),
            'original_baselines_prefix_sha256': plan['bindings']['tasks/K12/baselines.py']['sha256']}


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_original_evaluator():
    bindings = frozen_bindings()
    check_bindings(bindings)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Relative imports retain the exact original helpers. A private evaluator
    # module keeps the independent TaskPackage derivation out of production.
    return load_file('kernel_lab._independent_native_small_m_evaluator',
                     bindings['kernel_lab/evaluator.py']['path'])


def qualification_evaluator(engine):
    original_package = engine.TaskPackage

    class FinalQualificationPackage(original_package):
        def dev_correctness_cases(self):
            if self.task_id == 'K12':
                return self.final_correctness_cases()
            return super().dev_correctness_cases()

    engine.TaskPackage = FinalQualificationPackage

    class FinalQualificationEvaluator(engine.Evaluator):
        def ensure_oracle(self, task_id, phase, cases):
            if task_id == 'K12' and phase == 'dev':
                phase = 'final'
            return super().ensure_oracle(task_id, phase, cases)

    return FinalQualificationEvaluator


def idle_evaluator(engine, base, lock_root):
    bindings = frozen_bindings()
    policy = load_file('independent_native_idle_policy', bindings['idle_policy']['path'])
    admission = load_file('independent_native_idle_admission', bindings['idle_admission']['path'])
    original_lock = engine.GpuLock
    engine.GpuLock = lambda _private_root, physical_key: original_lock(lock_root, physical_key)

    class IdleEvaluator(base):
        def serve(self, *args, **kwargs):
            original_finish = self.q.finish

            def owned_finish(job, status, result):
                if not self.q.heartbeat(job['job_id'], self.owner, lease_s=900):
                    raise RuntimeError('independent GPU job lease lost before finish')
                return original_finish(job, status, result)

            self.q.finish = owned_finish
            try:
                return super().serve(*args, **kwargs)
            finally:
                self.q.finish = original_finish

        def _gpu_snapshot(self):
            snapshot = super()._gpu_snapshot
            result = snapshot()
            if getattr(self, '_admission_pending', False):
                self._admission_pending = False
                result = admission.wait_for_idle(initial=result, snapshot=snapshot,
                    cancelled=lambda: self._shutdown,
                    heartbeat=lambda: self.q.heartbeat(self._job['job_id'], self.owner, lease_s=900),
                    record=lambda **fields: self.q._emit('gpu_job_idle_admission_wait', run_id=self.run_id,
                        job_id=self._job['job_id'], gpu=self.gpu, **fields))
                self._before = result
            return result

        def _dispatch(self, job):
            started = time.monotonic()
            self.q._emit('independent_gpu_span_start', run_id=self.run_id, job_id=job['job_id'],
                gpu_uuid=self._before['uuid'], owner=self.owner, pid=os.getpid())
            try:
                status, result = super()._dispatch(job)
                result['seconds'] = time.monotonic() - started
                return status, result
            finally:
                self._span_seconds = time.monotonic() - started
                self.q._emit('independent_gpu_span_end', run_id=self.run_id, job_id=job['job_id'],
                    gpu_uuid=self._before['uuid'], owner=self.owner,
                    seconds=self._span_seconds, pid=os.getpid())

        def _execute(self, job):
            self._job, self._admission_pending = job, True
            self._before, self._span_seconds = None, 0.0
            try:
                return super()._execute(job)
            except Exception as exc:
                return 'crashed', {'status': 'crashed',
                    'error': f'{type(exc).__name__}: {exc}', 'seconds': self._span_seconds,
                    'gpu_state_before': self._before, 'gpu_state_after': super()._gpu_snapshot()}
            finally:
                self._job, self._admission_pending = None, False

    return IdleEvaluator, policy


def summarize(engine, evaluator, plan):
    db = evaluator.q.conn
    jobs = [dict(row) for row in db.execute('SELECT * FROM jobs ORDER BY created')]
    results = [dict(row) for row in db.execute('SELECT * FROM results ORDER BY created')]
    baseline_rows = [dict(row) for row in db.execute('SELECT * FROM baselines ORDER BY label')]
    qualification = {row['label']: json.loads(row['qualification_json'] or '{}')
                     for row in baseline_rows if row['label'] != '__strongest__'}
    verification = [row for row in results if row['job_id'] in
                    {job['job_id'] for job in jobs if job['kind'] == 'verify'}]
    payload = json.loads(verification[0]['payload_json']) if len(verification) == 1 else {}
    rounds = [{} for _ in range(plan['verification_rounds'])]
    host_rounds = [{} for _ in rounds]
    for row in db.execute("SELECT * FROM measurements WHERE phase='final' ORDER BY id"):
        measurement = json.loads(row['payload_json'])
        index = measurement['verify_round']
        rounds[index].setdefault(row['label'], {})[row['case_id']] = row['latency_us']
        if row['host_us'] is not None:
            host_rounds[index].setdefault(row['label'], {})[row['case_id']] = row['host_us']
    cases = [case['case_id'] for case in plan['timing_cases']]
    comparisons = {}
    for name, labels in (('original_baselines', ORIGINAL_BASELINES),
                         ('all_qualified_libraries', ORIGINAL_BASELINES + ADDITIONAL_BASELINES)):
        subset = [{key: values for key, values in row.items()
                   if key == 'candidate' or key in labels} for row in rounds]
        gaps = engine.verify_round_gaps(subset, cases, plan['verification_rounds'])
        comparisons[name] = {'complete': not gaps, 'incomplete': gaps,
            'statistics': engine.verification_statistics(subset) if not gaps else None}
    strong_coverage = all(qualification.get(label, {}).get('status') == 'ok' and
        all(all(case in row.get(label, {}) for case in cases) for row in rounds)
        for label in STRONG_BASELINES_REQUIRED)
    events = [json.loads(row['payload_json']) for row in db.execute(
        "SELECT payload_json FROM events WHERE kind LIKE 'independent_gpu_span_%' ORDER BY id")]
    starts = [row for row in events if row['kind'] == 'independent_gpu_span_start']
    ends = [row for row in events if row['kind'] == 'independent_gpu_span_end']
    closure = (len(jobs) == len(results) == 2 and
               not any(job['status'] in ('queued', 'running') for job in jobs) and
               {row['job_id'] for row in starts} == {row['job_id'] for row in ends} and
               len(starts) == len(ends) == 2)
    raw_contaminated = any(json.loads(row['payload_json']).get('gpu_state_before', {}).get(
        'user_gpu_admission', {}).get('original_process_presence_contaminated', False)
        for row in results if json.loads(row['payload_json']).get('gpu_state_before'))
    measured = bool(comparisons['all_qualified_libraries']['complete'] and
                    payload.get('statistics', {}).get('timing_backend') == 'cuda_events')
    return {'scope': SCOPE, 'task_id': 'K12', 'jobs': jobs, 'results': results,
        'baseline_qualification': qualification, 'verification': payload,
        'comparisons': comparisons, 'strong_FP16_library_coverage_complete': strong_coverage,
        'strong_library_gain_verified': bool(closure and measured and strong_coverage
                                             and payload.get('status') == 'complete'
                                             and not payload.get('contaminated', True)),
        'performance_measured': measured, 'owned_resource_closure_complete': closure,
        'original_process_presence_contaminated': raw_contaminated,
        'gpu_spans': events, 'local_gpu_seconds': sum(row['seconds'] for row in ends),
        'model_requests': 0, 'production_ledger_or_queue_writes': 0,
        'foreign_processes_signaled': 0, 'RSI_proven': False}


def execute(out, plan, gpu, lock_root):
    scope = prepare_scope(out / 'scope', plan)
    engine = load_original_evaluator()
    E, policy = idle_evaluator(engine, qualification_evaluator(engine), lock_root)
    protocol = dict(plan['original_protocol'])
    run_id = 'independent-native-small-M-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    protocol.update(run_id=run_id, keep_alive=False, gpu_admission_policy=policy.POLICY)
    evaluator = E(run_id, str(gpu), scope['root'], protocol=protocol)
    engine.write_json(evaluator.paths.protocol, protocol)
    evaluator.q.create_run(run_id, 'independent-native-K12', reference(evaluator.paths.protocol)['sha256'])
    evaluator.q.enqueue(run_id, 'K12', 'baseline', priority=1, max_attempts=1)
    evaluator.q.enqueue(run_id, 'K12', 'verify', candidate_id='original-MiMo-K12',
        source_hash=scope['candidate']['sha256'], snapshot_dir=str(Path(scope['root']) / 'candidate'),
        phase='final', priority=2, max_attempts=1)
    original_contamination = engine.gpuinfo.contaminated

    def declared_contamination(before, after):
        decision = policy.assess(before, after)
        decision['original_process_presence_contaminated'] = original_contamination(before, after)
        before['user_gpu_admission'] = dict(decision)
        after['user_gpu_admission'] = dict(decision)
        return not decision['eligible']

    engine.gpuinfo.contaminated = declared_contamination
    code, failure = 0, None
    try:
        evaluator.serve(max_jobs=2)
    except Exception as exc:
        code, failure = 2, {'error_type': type(exc).__name__, 'diagnostic': str(exc)}
    finally:
        engine.gpuinfo.contaminated = original_contamination
    result = summarize(engine, evaluator, plan)
    queue_path = evaluator.paths.db
    evaluator.q.close()
    check_bindings(plan['bindings'])
    result.update(prepared_scope=scope, failure=failure, environment={
        'python': sys.version, 'torch': importlib.metadata.version('torch'),
        'triton': importlib.metadata.version('triton'), 'CUDA_runtime': engine.torch.version.cuda})
    result_ref = write_new(out / 'result.json', result)
    if not result['owned_resource_closure_complete']:
        code = 2
    write_new(out / 'execution.json', {'scope': SCOPE, 'exit_code': code,
        'result': result_ref, 'queue': reference(queue_path), 'entrypoint': reference(__file__),
        'GPU_execution_closed': result['owned_resource_closure_complete'],
        'performance_measured': result['performance_measured'],
        'strong_library_gain_verified': result['strong_library_gain_verified'],
        'model_requests': 0, 'production_ledger_or_queue_writes': 0, 'RSI_proven': False})
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(8), default=0)
    parser.add_argument('--lock-root', type=Path, default=ROOT,
                        help='project root whose .locks/ is shared by all GPU jobs')
    args = parser.parse_args(argv)
    out = fresh_output(args.out)
    plan = make_plan(frozen_bindings())
    out.mkdir(parents=True, mode=0o700)
    write_new(out / 'reproduction-plan.json', {
        'status': 'EXECUTE_REQUESTED' if args.execute else 'CPU_PLAN_ONLY',
        'scope': SCOPE, 'created': datetime.now(timezone.utc).isoformat(),
        'benchmark_plan': plan, 'entrypoint': reference(__file__),
        'gpu': args.gpu, 'shared_lock_root': str(args.lock_root.resolve()),
        'model_requests': 0, 'production_ledger_or_queue_writes': 0,
        'performance_measured': False, 'RSI_proven': False})
    if args.execute:
        return execute(out, plan, args.gpu, args.lock_root.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
