"""Executed after the submitting terminal exits; no model or GPU work."""
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
time.sleep(8)
job=HERE/'standalone_jobs'/'v15-detachment-test-20260920'
spec=json.loads((job/'JOB.json').read_text())
current=Path('/proc/self/cgroup').read_text()
assert '/system.slice/'+spec['unit'] in current
assert current!=spec['launcher_cgroup']
assert not Path('/proc',str(spec['launcher_pid'])).exists(),'SUBMITTER_STILL_ALIVE'
assert os.getuid()==spec['uid'] and os.getgid()==spec['gid']
with (HERE/'STANDALONE_CPU_TEST_RESULT.json').open('x') as out:
    json.dump(dict(passed=True,submitter_exited=True,separate_system_service_cgroup=True,
        same_uid_gid=True,pid=os.getpid(),cgroup=current,GPU_used=False,
        scope='Real detached command completed after submitting terminal command exited; no Codex API dependency.'),out,indent=2)
print('Standalone process survived submitter exit',flush=True)
