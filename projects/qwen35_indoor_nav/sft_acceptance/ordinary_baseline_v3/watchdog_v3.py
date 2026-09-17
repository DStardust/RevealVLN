"""Watchdog for the supervised v3 formal training: relaunch the lease if dead.

Only observes process/file state and relaunches my own lease driver. Never
signals foreign processes. Bounded: <=10 lease relaunches, <=30 h wall.
"""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
RUN = HERE / 'formal' / 'run_0001'
LEASE_DIR = HERE / 'formal' / 'lease_3gpu'
LOG = HERE / 'formal' / 'watchdog_v3.log'
RUNBOOK = HERE / 'RUNBOOK_FORMAL_3GPU.json'
TOOLS_PY = '/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
MAX_RELAUNCH = 10
MAX_WALL = 30 * 3600


def log(event):
    with LOG.open('a') as stream:
        stream.write(json.dumps(dict(unix=time.time(), **event)) + '\n')


def lease_alive():
    out = subprocess.run(['pgrep', '-f', 'lease_run.py RUNBOOK_FORMAL_3GPU.json'],
                         capture_output=True, text=True)
    return out.returncode == 0


def training_state():
    """completed | budget_stopped (terminal) | incomplete."""
    results = [RUN / 'RESULT.json'] + sorted(RUN.glob('RESULT_CONTINUATION_*.json'))
    for path in reversed(results):
        if path.is_file():
            data = json.loads(path.read_text())
            status = data.get('status')
            if status == 'EPOCHS_COMPLETED':
                return 'completed'
            if status == 'STOPPED':
                if any(str(s).startswith('BUDGET:') for s in data.get('stop', [])):
                    return 'budget_stopped'
                return 'incomplete'
    return 'incomplete'


def main():
    started = time.monotonic()
    relaunches = 0
    log(dict(event='watchdog_started'))
    while time.monotonic() - started < MAX_WALL and relaunches <= MAX_RELAUNCH:
        state = training_state()
        if state == 'completed':
            log(dict(event='training_completed'))
            break
        if state == 'budget_stopped':
            log(dict(event='training_budget_stopped_no_relaunch'))
            break
        if lease_alive():
            time.sleep(120)
            continue
        # lease dead: is a supervised attempt still running standalone?
        alive = subprocess.run(['pgrep', '-f', 'supervise_v3.py'], capture_output=True)
        if alive.returncode == 0:
            time.sleep(120)
            continue
        relaunches += 1
        # each relaunch gets its own lease output dir (lease_run requires a fresh one)
        runbook = json.loads(RUNBOOK.read_text())
        runbook['out'] = 'formal/lease_3gpu_r%d' % relaunches
        variant = HERE / ('RUNBOOK_FORMAL_3GPU_R%d.json' % relaunches)  # lease_run requires runbook in HERE
        variant.write_text(json.dumps(runbook, indent=2))
        log(dict(event='lease_relaunch', n=relaunches, runbook=variant.name))
        subprocess.Popen([TOOLS_PY, '-I', '-B', str(HERE / 'lease_run.py'), str(variant)],
                         cwd=str(HERE),
                         stdout=(HERE / 'formal' / ('lease_driver_r%d.log' % relaunches)).open('wb'),
                         stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(300)
    log(dict(event='watchdog_exit', relaunches=relaunches,
             wall_seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
