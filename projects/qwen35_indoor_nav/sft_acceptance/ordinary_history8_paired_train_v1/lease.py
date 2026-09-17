"""Reuse proven holder restoration; pin current holder identity and supervise exact child."""
import hashlib
from pathlib import Path
SOURCE=Path(__file__).resolve().parent.parent/'ordinary_sync_recovery_v1/lease_run.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='06343e1da28f19df10311759e992dab59554bc4b3824621dca937d4692b22a89'
_code=SOURCE.read_text();_entry="if __name__ == '__main__':\n    main()"
assert _code.count(_entry)==1
exec(compile(_code.replace(_entry,''),str(SOURCE)+':read-only-holder-transport','exec'),globals())
import importlib.util
import xml.etree.ElementTree as ET
_spec=importlib.util.spec_from_file_location('paired_lease_owner',HERE.parent/'ordinary_prefix_history8_v1/owned_process_r1.py')
owner=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(owner)
_original_identity=holder_identity
def holder_identity(holder):
    actual=_original_identity(holder)
    require(actual['pid']==holder['expected_pid'] and actual['starttime_ticks']==holder['expected_start'],'HOLDER_REPLACED_SINCE_PREFLIGHT')
    return actual
def gpu_snapshot(uuid):
    root=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x','-i',uuid],text=True,timeout=15))
    gpu=root.find('gpu');require(gpu.findtext('uuid')==uuid,'GPU_UUID')
    processes={}
    for row in gpu.findall('processes/process_info'):
        pid=int(row.findtext('pid'));processes[pid]=processes.get(pid,0)+float(row.findtext('used_memory').split()[0])
    return dict(uuid=uuid,processes=processes,memory_mib=float(gpu.findtext('fb_memory_usage/used').split()[0]),graphics_contexts_included=True)
def run_step(step,out,deadline):
    require(any(str(HERE) in str(x) for x in step['argv']),'STEP_SCOPE')
    env=dict(os.environ);env.update(step.get('env_extra',{}))
    guard=None;child=None;error=None
    with (out/('step_'+step['name']+'.log')).open('x') as log:
        try:
            child=subprocess.Popen(step['argv'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            guard=owner.OwnedChild(child,step['argv'],ROOT)
            save(out/('STEP_'+step['name']+'_PROCESS.json'),guard.identity)
            while guard.check()!='exited':
                require(time.monotonic()<deadline,'STEP_DEADLINE')
                time.sleep(2)
            require(child.returncode==0,'STEP_EXIT:'+str(child.returncode))
        except BaseException as exc:error=repr(exc);raise
        finally:
            cleanup=guard.cleanup(timeout=90) if guard else dict(exit_code=child.poll() if child else None,identity_registered=False)
            save(out/('STEP_'+step['name']+'_RESULT.json'),dict(error=error,cleanup=cleanup))
    return dict(name=step['name'],exit=0)
if __name__=='__main__':
    approval=json.loads((HERE/'MAIN_AGENT_APPROVAL.json').read_text())
    require(sha256(HERE/'RUNBOOK.json')==approval['runbook_sha256'],'RUNBOOK_CHANGED')
    require(sha256(HERE/'PROTOCOL.json')==approval['protocol_sha256'],'PROTOCOL_CHANGED')
    main()

