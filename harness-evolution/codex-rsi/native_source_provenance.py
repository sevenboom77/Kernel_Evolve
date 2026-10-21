"""Bind original trusted model writes to native submissions and frozen GPU results."""
import hashlib
import json
from pathlib import Path


def verify_native_source(*, events, source, jobs, results, episode, operator, run_id, task_id):
    """Audit existing closed evidence; neither generate source nor rerun a GPU job."""
    def require(condition):
        if not condition:
            raise ValueError('native_provenance_incomplete_or_changed')

    def trusted(event, event_type, origin):
        return (event.get('type') == event_type and event.get('origin') == origin
                and event.get('trust') == 'trusted' and event.get('lane') == 'main')

    require(hashlib.sha256(source).hexdigest() == operator['source_file_sha256'])
    require(episode['task'] == task_id)
    slots = episode['model_slots']
    require(slots and all(x['model'] == 'mimo-v2.6-pro' and x['thinking'] == 'off'
                         and x['contextWindow'] == 32768 for x in slots.values()))
    selected = [x for x in jobs if x['kind'] == 'candidate' and x['candidate_id'] == operator['candidate_id']]
    verifies = [x for x in jobs if x['job_id'] == operator['verify_job_id']]
    require(len(selected) == len(verifies) == 1)
    job, verify = selected[0], verifies[0]
    for row in (job, verify):
        require(row['run_id'] == run_id and row['task_id'] == task_id and row['status'] == 'complete'
                and row['candidate_id'] == operator['candidate_id']
                and row['source_hash'] == operator['source_package_sha256'])
    require(verify['kind'] == 'verify' and verify['snapshot_dir'] == job['snapshot_dir'])
    receipts = [x for x in events if trusted(x, 'tool/result', 'tool:kernel_submit')
                and x['data'].get('structured', {}).get('job_id') == job['job_id']]
    require(len(receipts) == 1)
    submitted = receipts[0]
    receipt = submitted['data']['structured']
    require(submitted['data'].get('isError') is False and receipt['ok'] is True
            and receipt['candidate_id'] == job['candidate_id'] and receipt['source_hash'] == job['source_hash']
            and receipt['snapshot'] == job['snapshot_dir'] and receipt['workspace'] == episode['workspace']
            and receipt['phase'] == 'dev')
    calls = [x for x in events if trusted(x, 'tool/call', 'model')
             and x['data'].get('name') == 'kernel_submit'
             and x['data']['toolUseId'] == submitted['data']['toolUseId']]
    require(len(calls) == 1)
    call = calls[0]
    args = call['data']['args']
    require(args['task_id'] == task_id and call['seq'] < submitted['seq'])
    kernel_path = Path(episode['workspace']) / args['candidate_dir'] / 'kernel.py'
    require(kernel_path.is_relative_to(episode['workspace']))
    writes = [x for x in events if trusted(x, 'tool/call', 'model')
              and x['data'].get('name') == 'write' and x['seq'] < call['seq']
              and x['data']['args'].get('path') == str(kernel_path)
              and x['data']['args'].get('content', '').encode() == source]
    require(len(writes) == 1)
    write = writes[0]
    acknowledgments = [x for x in events if trusted(x, 'tool/result', 'tool:write')
                       and x['data'].get('toolUseId') == write['data']['toolUseId']
                       and x['data'].get('isError') is False
                       and write['seq'] < x['seq'] < call['seq']]
    require(len(acknowledgments) == 1)
    prior = [x for x in jobs if x['kind'] == 'candidate' and x['created'] < job['created']]
    require(prior and args.get('parent_hash') == prior[-1]['source_hash'])
    raw_payload = results[verify['job_id']]
    require(hashlib.sha256(raw_payload.encode()).hexdigest() == operator['verification_payload_sha256'])
    payload = json.loads(raw_payload)
    correct, stats = payload['correctness'], payload['statistics']
    require(payload['status'] == 'complete' and payload['contaminated'] is False
            and correct['ok'] is True and correct['passed'] == correct['total'] == 21
            and len(correct['per_case']) == 21 and all(x['ok'] is True for x in correct['per_case'])
            and stats['rounds_used'] == operator['independent_verification_rounds'] == 5
            and stats['task_speedup_mean'] == operator['verified_speedup']
            and stats['task_speedup_ci_low'] == operator['verified_ci_low']
            and stats['task_speedup_ci_high'] == operator['verified_ci_high'])
    return {'task_id': task_id, 'run_id': run_id, 'model': slots['primary']['model'],
            'thinking': slots['primary']['thinking'], 'candidate_id': job['candidate_id'],
            'candidate_job_id': job['job_id'], 'verify_job_id': verify['job_id'],
            'model_write_seq': write['seq'], 'write_result_seq': acknowledgments[0]['seq'],
            'submit_call_seq': call['seq'], 'submit_result_seq': submitted['seq'],
            'source_file_sha256': operator['source_file_sha256'],
            'source_package_sha256': job['source_hash'],
            'correctness_cases': correct['total'], 'verification_rounds': stats['rounds_used'],
            'verified_speedup': stats['task_speedup_mean'],
            'verified_CI': [stats['task_speedup_ci_low'], stats['task_speedup_ci_high']],
            'prior_candidate_statuses': [x['status'] for x in prior],
            'original_episode_budget_exhausted_retained': episode['terminal_status'] == 'budget_exhausted',
            'new_model_GPU_calls': 0, 'harness_causal_gain_or_RSI_proven': False}
