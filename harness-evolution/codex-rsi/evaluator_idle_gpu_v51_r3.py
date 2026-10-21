"""Per-job idle admission with unchanged V43 measurement contamination semantics."""
from __future__ import annotations

import math

POLICY = 'user-premeasurement-utilization-zero-v43'


def assess(before, after):
    original = any(s.get('available') and s.get('foreign_processes', 0) > 0 for s in (before, after))
    prior_pids = before.get('foreign_pids', [])
    later_pids = after.get('foreign_pids', [])
    known_pids = all(type(pid) is int and pid > 0 for pid in [*prior_pids, *later_pids])
    new_pids = sorted(set(later_pids) - set(prior_pids))
    cpu = before.get('device') == after.get('device') == 'cpu'
    util = before.get('util')
    idle = type(util) in (int, float) and math.isfinite(util) and util == 0
    available = before.get('available') is True and after.get('available') is True
    same_device = before.get('uuid') == after.get('uuid') and isinstance(before.get('uuid'), str)
    no_unknown = before.get('unknown_processes', 0) == after.get('unknown_processes', 0) == 0
    eligible = cpu or (available and same_device and idle and known_pids and no_unknown and not new_pids)
    return {'policy': POLICY, 'eligible': bool(eligible),
            'original_process_presence_contaminated': bool(original),
            'new_foreign_pids': new_pids, 'premeasurement_utilization': util,
            'after_utilization_attributed_to_foreign': False,
            'exclusive_device_confirmed': bool(eligible and not original)}


def main():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kernel_lab import cli, gpuinfo
    from kernel_lab.evaluator import Evaluator
    from idle_gpu_job_admission_v51 import wait_for_idle

    original_contamination = gpuinfo.contaminated
    original_evaluator = cli.Evaluator

    def declared_contamination(before, after):
        decision = assess(before, after)
        decision['original_process_presence_contaminated'] = original_contamination(before, after)
        before['user_gpu_admission'] = dict(decision)
        after['user_gpu_admission'] = dict(decision)
        return not decision['eligible']

    class IdleUtilizationEvaluator(Evaluator):
        def serve(self, *args, **kwargs):
            original_finish = self.q.finish

            def owned_finish(job, status, result):
                if not self.q.heartbeat(job['job_id'], self.owner, lease_s=900):
                    self.events.emit('gpu_job_lease_lost_before_finish',
                        level='error', job_id=job['job_id'], gpu=self.gpu)
                    raise SystemExit('GPU job lease ownership lost; original queue row preserved')
                return original_finish(job, status, result)

            self.q.finish = owned_finish
            try:
                return super().serve(*args, **kwargs)
            finally:
                self.q.finish = original_finish

        def _execute(self, job):
            if self.protocol.get('gpu_admission_policy') != POLICY:
                raise ValueError('explicit frozen user idle-GPU protocol required')
            self._idle_policy_before = True
            self._admission_job = job
            try:
                return super()._execute(job)
            finally:
                self._idle_policy_before = False
                self._admission_job = None

        def _gpu_snapshot(self):
            base_snapshot = super()._gpu_snapshot
            result = base_snapshot()
            if getattr(self, '_idle_policy_before', False):
                self._idle_policy_before = False
                if self.is_cuda:
                    job_id = self._admission_job['job_id']
                    result = wait_for_idle(initial=result, snapshot=base_snapshot,
                        cancelled=lambda: self._shutdown,
                        heartbeat=lambda: self.q.heartbeat(job_id, self.owner, lease_s=900),
                        record=lambda **fields: self.events.emit('gpu_job_idle_admission_wait',
                            job_id=job_id, gpu=self.gpu, **fields))
            return result

    gpuinfo.contaminated = declared_contamination
    cli.Evaluator = IdleUtilizationEvaluator
    try:
        return cli.main()
    finally:
        gpuinfo.contaminated = original_contamination
        cli.Evaluator = original_evaluator


if __name__ == '__main__':
    raise SystemExit(main())
