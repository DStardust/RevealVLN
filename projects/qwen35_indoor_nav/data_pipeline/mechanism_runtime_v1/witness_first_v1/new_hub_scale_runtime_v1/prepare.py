"""CPU-only source closure and exact job preparation; never approves or launches."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('scale_scout_prepare_common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

def code_dependencies():
    result=c.dependencies()
    # A draft holder is not usable: its complete final checksum manifest is mandatory.
    holder=c.WF/'special_scale_holder_v1';manifest=holder/'SHA256SUMS'
    c.require(manifest.is_file(),'HOLDER_NOT_SEALED')
    for line in manifest.read_text().splitlines():
        h,name=line.split(maxsplit=1);path=c.scoped(holder/name.lstrip('*'))
        c.require(c.sha(path)==h,'HOLDER_SEAL_CHANGED');result[path]=h
    result[manifest]=c.sha(manifest)
    transport=c.load('new_hub_scale_holder_dependency',holder/'transport.py')
    c.require(callable(getattr(transport,'make_ops',None)),'SEALED_RESTORE_ENV_API_REQUIRED')
    for path in transport.dependency_paths():result[Path(path)]=c.sha(path)
    for path in sorted(HERE.glob('*.py')):result[path]=c.sha(path)
    for path in [HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',c.WF/'multi_program_bank_v1/prepare.py',c.WF/'multi_program_bank_v1/core.py',
                 c.WF/'multi_program_bank_v2/prepare.py',c.WF/'language_realization_v1/verbalizer.py']:
        result[path]=c.sha(path)
    return result

def prepare(authorization_path):
    authorization_path=c.scoped(authorization_path);auth=c.authorization(authorization_path)
    dependencies=code_dependencies();outputs=[]
    for ident in auth['job_ids']:
        job=c.job_root(HERE/'jobs'/ident);c.require(not job.exists(),'NO_JOB_OVERWRITE_OR_RETRY')
        item=c.item_for(job);source_lock=c.verify_lock(item['source_lock']);lock=dict(source_lock)
        for path,h in dependencies.items():lock[str(path)]=h
        for path in [c.QUEUE,Path(item['source_lock']),Path(item['prepared_config']),authorization_path,Path(auth['initial_holder_identity_path'])]:lock[str(path)]=c.sha(path)
        index=auth['job_ids'].index(ident);previous=HERE/'jobs'/auth['job_ids'][index-1] if index else None
        launch=dict(authorization_path=str(authorization_path),initial_holder_identity_path=auth['initial_holder_identity_path'],
            previous_job=str(previous) if previous else None,
            previous_source_lock_sha256=c.sha(previous/'SOURCE_LOCK.json') if previous else None)
        if previous:lock[str(previous/'SOURCE_LOCK.json')]=c.sha(previous/'SOURCE_LOCK.json')
        job.mkdir(parents=True,exist_ok=False)
        c.save(job/'PREPARED_CONFIG.json',c.prepared_config(item));c.save(job/'LAUNCH_CONFIG.json',launch)
        # Actual runtime subprocess imports the node, never an unregistered worker.
        source='import importlib.util\nfrom pathlib import Path\n'
        source+='p=Path('+repr(str(HERE/'runtime.py'))+')\ns=importlib.util.spec_from_file_location("scale_scout_job",p)\nm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
        for name,call in [('worker.py','worker_main'),('run.py','run_main')]:
            with (job/name).open('x') as f:f.write(source+'m.'+call+'(Path('+repr(str(job))+'))\n')
        for name in ['PREPARED_CONFIG.json','LAUNCH_CONFIG.json','worker.py','run.py']:lock[str(job/name)]=c.sha(job/name)
        c.require(len(lock)<=2048 and sum(Path(p).stat().st_size for p in lock)<=32*1024**3,'COMPLETE_SOURCE_CAP')
        c.save(job/'SOURCE_LOCK.json',lock)
        outputs.append(dict(id=ident,gpu=7,command=[str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-B',str(job/'run.py')],
            max_seconds=5100,transport_upper_seconds=3300,audit_seconds=1800,main_approval_required=True,
            audit_command=[str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-B',str(HERE/'audit.py'),'--job',str(job)],
            completion=[dict(path=str(job/'run_v1/SUPERVISOR_RESULT.json'),equals=dict(returncode=0,error=None,cleanup_complete=True)),
                dict(path=str(job/'run_v1/LEASE_RESULT.json'),equals=dict(execute_returned=True,error=None,holder_restored=True)),
                dict(path=str(job/'run_v1/RESTORATION.json'),equals=dict(restored=True,remain_on_exit_restored=True))],
            input_hashes={str(job/'run.py'):c.sha(job/'run.py'),str(job/'SOURCE_LOCK.json'):c.sha(job/'SOURCE_LOCK.json'),str(HERE/'runtime.py'):c.sha(HERE/'runtime.py'),str(HERE/'audit.py'):c.sha(HERE/'audit.py')}))
    c.save(HERE/'jobs/JOBS.json',dict(jobs=outputs,lane_wall_seconds=43200,automatic_gpu_launch=False,source_queue_sha256=c.sha(c.QUEUE)))
    return outputs

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--authorization',required=True);a=p.parse_args();prepare(a.authorization)
