"""Fresh per-job GPU admission, called inside the existing physical GPU lock."""
from __future__ import annotations

import math
import time


def wait_for_idle(*, initial, snapshot, cancelled, heartbeat, record):
    result = initial
    device = None
    started = time.monotonic()
    renewed = None
    attempt = 0
    while True:
        if cancelled():
            raise InterruptedError('GPU job admission cancelled before dispatch')
        now = time.monotonic()
        if renewed is None or now - renewed >= 30:
            if not heartbeat():
                raise RuntimeError('GPU job lease ownership lost before dispatch')
            renewed = now
        util, used, total = (result.get(key) for key in
                             ('util', 'mem_used_mib', 'mem_total_mib'))
        pids = result.get('foreign_pids')
        uuid = result.get('uuid')
        valid = (
            result.get('available') is True and isinstance(uuid, str) and bool(uuid)
            and all(type(value) in (int, float) and math.isfinite(value)
                    for value in (util, used, total))
            and 0 <= util <= 100 and 0 <= used <= total and total > 0
            and isinstance(pids, list)
            and all(type(pid) is int and pid > 0 for pid in pids)
            and type(result.get('unknown_processes')) is int
            and result['unknown_processes'] == 0
            and type(result.get('foreign_processes')) is int
            and result['foreign_processes'] == len(pids)
            and (device is None or uuid == device)
        )
        if not valid:
            raise ValueError('GPU job admission sample is unavailable or unverifiable')
        device = uuid
        attempt += 1
        if util == 0 and total - used >= 3072:
            if cancelled():
                raise InterruptedError('GPU job admission cancelled before dispatch')
            if not heartbeat():
                raise RuntimeError('GPU job lease ownership lost before dispatch')
            return result
        if attempt == 1 or now - started >= 30:
            record(attempt=attempt, utilization=util, free_mib=total - used,
                   waiting_seconds=now - started)
            started = now
        time.sleep(.25)
        result = snapshot()
