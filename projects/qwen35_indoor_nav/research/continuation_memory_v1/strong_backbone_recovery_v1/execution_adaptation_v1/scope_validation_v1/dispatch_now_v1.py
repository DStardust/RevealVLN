"""User-authorized scheduling amendment; retain the frozen scope experiment."""
import argparse
import importlib.util
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('registered_scope_pipeline',HERE/'pipeline.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
u=pipeline.u


def main(run,resume):
    for path,sha in u.read(run/'START_NOW_SOURCE_LOCK.json')['files'].items():
        if u.sha(path)!=sha:raise ValueError('START_NOW_AMENDMENT_CHANGED')
    auth=u.read(run/'START_NOW_AUTHORIZATION.json')
    if auth['user_instruction']!='不用等旧评测，直接开始':raise ValueError('MISSING_SCHEDULING_AUTHORIZATION')
    protocol=u.read(run/'PROTOCOL.json');previous=u.read(Path(protocol['wait_for_run'])/'STATUS.json')
    if previous['status'] not in ('INTERRUPTED','COMPLETE') or previous.get('workers'):
        raise ValueError('PREVIOUS_WORKERS_NOT_RELEASED')
    u.write(run/f'START_NOW_{time.time_ns()}.json',dict(
        authorization_sha256=u.sha(run/'START_NOW_AUTHORIZATION.json'),
        amendment_source_lock_sha256=u.sha(run/'START_NOW_SOURCE_LOCK.json'),
        previous_status=previous['status'],previous_sealed_groups=previous['sealed_groups'],
        scheduling_only=True,weights_inputs_actions_and_registered_denominator_unchanged=True))
    pipeline.main(run,resume)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();main(a.run.resolve(),a.resume)
