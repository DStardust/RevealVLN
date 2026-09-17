"""One-shot incident capture and exact-identity shutdown of the verified old run."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'ordinary_baseline_v3'
RUN = OLD / 'formal/run_0001'


def proc(pid):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, ppid=int(fields[1]), start=int(fields[19]), state=fields[0],
                    argv=(root / 'cmdline').read_bytes().split(b'\0')[:-1],
                    cwd=str((root / 'cwd').resolve()))
    except (FileNotFoundError, ProcessLookupError):
        return None


def serial(p):
    return {**p, 'argv': [x.decode() for x in p['argv']]}


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def signal_exact(p, signum):
    live = proc(p['pid'])
    if live is None or live['state'] in ('Z', 'X'):
        return False
    assert all(live[k] == p[k] for k in ('start', 'argv', 'cwd')), 'PID_IDENTITY_CHANGED'
    os.kill(p['pid'], signum)
    with (HERE / 'STOP_SIGNALS.jsonl').open('a') as f:
        f.write(json.dumps(dict(unix=time.time(), pid=p['pid'], start=p['start'], signal=int(signum))) + '\n')
        f.flush()
        os.fsync(f.fileno())
    return True


def main():
    # IDs are observations, never sufficient authority: every full identity and
    # ancestry is verified immediately before the bounded action.
    watchdog, lease, supervisor, launcher = [proc(p) for p in (1009475, 1009479, 1009543, 1009544)]
    assert all((watchdog, lease, supervisor, launcher)), 'OLD_TOPOLOGY_CHANGED'
    assert str(OLD / 'watchdog_v3.py').encode() in watchdog['argv']
    assert str(OLD / 'lease_run.py').encode() in lease['argv'] and lease['ppid'] == watchdog['pid']
    assert str(OLD / 'supervise_v3.py').encode() in supervisor['argv'] and supervisor['ppid'] == lease['pid']
    assert str(OLD / 'train.py').encode() in launcher['argv'] and launcher['ppid'] == supervisor['pid']
    processes = {int(p.name): proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    children = {launcher['pid']}
    while True:
        expanded = children | {pid for pid, p in processes.items() if p and p['ppid'] in children}
        if expanded == children:
            break
        children = expanded
    own = [processes[pid] for pid in sorted(children)]
    compile_worker = OLD.parents[1] / '.envs/q35n_qwen_g2_v1/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py'
    assert all(str(OLD / 'train.py').encode() in p['argv'] or
               str(compile_worker).encode() in p['argv'] for p in own), 'UNKNOWN_CHILD'
    assert all(p['cwd'] == str(OLD) for p in [watchdog, lease, supervisor, *own]), 'CWD_CHANGED'
    progress = json.loads((RUN / 'PROGRESS.json').read_text())
    assert progress['cursor']['updates'] == 30499 and time.time() - progress['unix'] > 600, 'NOT_STALLED'
    ckpt = RUN / 'checkpoint_000030400.pt'
    receipt = json.loads(Path(str(ckpt) + '.json').read_text())
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert digest == receipt['sha256'] == '49e59020ef5fa4b9b870ec524816d1a9b4217380dc3ae6ed15ef13344562d73e'
    save('INCIDENT_BEFORE.json', dict(unix=time.time(), progress=progress, checkpoint=str(ckpt),
         receipt=receipt, uncheckpointed_reported_updates=99,
         root_processes=[serial(p) for p in (watchdog, lease, supervisor)],
         train_tree=[serial(p) for p in own],
         gpu=subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                      '--format=csv,noheader'], text=True),
         old_lease_before=json.loads((OLD / 'formal/lease_3gpu_r1/LEASE_BEFORE.json').read_text()),
         cause='LOCAL_WALL_CLOCK_COLLECTIVE_DIVERGENCE_CODE_CONFIRMED_RUNTIME_STACK_UNOBSERVED'))
    # Disable re-launch first. Lease SIGTERM invokes its registered finally,
    # terminates the supervisor/torchrun, and restores the three exact holders.
    signal_exact(watchdog, signal.SIGTERM)
    signal_exact(lease, signal.SIGTERM)
    deadline = time.monotonic() + 22
    while time.monotonic() < deadline:
        if all(not proc(p['pid']) or proc(p['pid'])['state'] in ('Z', 'X') for p in own):
            break
        time.sleep(.5)
    # torchrun workers have separate sessions; clean only pre-captured own PIDs
    # if the hung collective prevented graceful termination.
    for p in own:
        signal_exact(p, signal.SIGKILL)
    save('STOP_REQUEST_RESULT.json', dict(unix=time.time(), old_relauncher_stopped=True,
         training_tree_signal_complete=True, lease_restoration_pending=True,
         scientific_pass=False, historical_failure_preserved=True))
    print('Old relauncher and exact stalled training tree stopped; lease restoration in progress.', flush=True)


if __name__ == '__main__':
    main()
