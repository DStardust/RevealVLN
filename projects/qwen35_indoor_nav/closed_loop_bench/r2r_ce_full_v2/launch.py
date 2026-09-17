"""One bounded inference process group. Never signals training or other GPU owners."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
OUT=HERE/'run_001'


def identity(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rindex(')')+2:].split()
        return dict(pid=pid,state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),
                    start_ticks=int(fields[19]),rss_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
    except (FileNotFoundError,ProcessLookupError):return None


def group_members(pgid):
    result=[]
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            item=identity(int(path.name))
            if item and item['pgid']==pgid and item['sid']==pgid:result.append(item)
    return result


def gpu_snapshot():
    blob=subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15)
    result=[]
    for index,gpu in enumerate(ET.fromstring(blob).findall('gpu')):
        contexts=[]
        for item in gpu.findall('processes/process_info'):
            contexts.append(dict(pid=int(item.findtext('pid')),kind=item.findtext('type'),
                                 name=item.findtext('process_name'),used_memory=item.findtext('used_memory')))
        result.append(dict(index=index,uuid=gpu.findtext('uuid'),
            memory_mib=int(gpu.findtext('fb_memory_usage/used').split()[0]),contexts=contexts))
    return result


def training_snapshot():
    status=json.loads((c.TRAIN/'formal/STATUS.json').read_text())
    return dict(status=status,processes=[identity(pid) for pid in (1151310,1151311,1151312,1151313)])


def terminate_owned(proc,initial):
    members=group_members(proc.pid)
    if not members:return dict(signaled=False,remaining=[])
    current=identity(proc.pid)
    if current and current['start_ticks']!=initial['start_ticks']:
        raise RuntimeError('OWN_PID_IDENTITY_CHANGED_REFUSE_SIGNAL')
    # Session was created by this Popen; other existing tasks cannot join it.
    os.killpg(proc.pid,signal.SIGTERM)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        proc.poll();live=[x for x in group_members(proc.pid) if x['state']!='Z']
        if not live:break
        time.sleep(.2)
    live=[x for x in group_members(proc.pid) if x['state']!='Z']
    if live:os.killpg(proc.pid,signal.SIGKILL)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
    return dict(signaled=True,remaining=[x for x in group_members(proc.pid) if x['state']!='Z'])


def main():
    c.verify_lock();p=json.loads((HERE/'PROTOCOL.json').read_text())
    assert json.loads((HERE/'CPU_TEST_RESULT.json').read_text())['passed']
    lease=(HERE/'gpu1_eval.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=gpu_snapshot();gpu=before[p['gpu']]
    assert gpu['uuid']==p['gpu_uuid'] and not gpu['contexts'] and gpu['memory_mib']<128,'GPU_NOT_EMPTY'
    training_before=training_snapshot()
    assert all(x for x in training_before['processes']),'TRAINING_IDENTITY_NOT_PRESENT'
    OUT.mkdir(exist_ok=False)
    for name in ('lanes','cache','cache/hf','cache/xdg','cache/torch','cache/triton','cache/cuda','cache/numba','cache/mpl','tmp'):
        (OUT/name).mkdir(exist_ok=False)
    for lane in range(p['lanes']):
        folder=OUT/'lanes'/f'lane_{lane:02d}';folder.mkdir();(folder/'frames').mkdir()
    c.write(OUT/'PREFLIGHT.json',dict(unix=time.time(),gpus=before,training=training_before,
        source_lock_sha256=c.sha(HERE/'SOURCE_LOCK.json'),borrowed_holders=[],external_signals_allowed=False),True)
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'):env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=str(p['gpu']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key,relative in dict(HF_HOME='cache/hf',XDG_CACHE_HOME='cache/xdg',TORCH_HOME='cache/torch',
        TRITON_CACHE_DIR='cache/triton',CUDA_CACHE_PATH='cache/cuda',NUMBA_CACHE_DIR='cache/numba',
        MPLCONFIGDIR='cache/mpl',TMPDIR='tmp',TMP='tmp',TEMP='tmp').items():env[key]=str(OUT/relative)
    began=time.monotonic();proc=None;initial=None;reason=None;peak_gpu=0;peak_rss=0;peak_output=0
    def interrupted(signum,frame):raise RuntimeError('LAUNCHER_SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    log=(OUT/'worker.log').open('x')
    try:
        proc=subprocess.Popen([str(c.LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(HERE/'evaluate.py')],
            env=env,cwd=c.ROOT,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        initial=identity(proc.pid);assert initial and initial['pgid']==proc.pid and initial['sid']==proc.pid
        c.write(OUT/'PROCESS.json',dict(initial,launcher_pid=os.getpid(),unix=time.time()),True)
        while proc.poll() is None:
            members=group_members(proc.pid);pids={x['pid'] for x in members}
            snapshots=gpu_snapshot();selected=snapshots[p['gpu']]
            rss=sum(x['rss_bytes'] for x in members)
            size=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file())
            elapsed=time.monotonic()-began
            peak_gpu=max(peak_gpu,selected['memory_mib']);peak_rss=max(peak_rss,rss);peak_output=max(peak_output,size)
            progress_path=OUT/'PROGRESS.json'
            progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
            foreign=[x for x in selected['contexts'] if x['pid'] not in pids]
            misplaced=[dict(x,gpu=g['index']) for g in snapshots if g['index']!=p['gpu']
                       for x in g['contexts'] if x['pid'] in pids]
            if foreign:reason='FOREIGN_GPU_CONTEXT_APPEARED'
            elif misplaced:reason='OWN_GPU_MAPPING_MISMATCH'
            elif elapsed>p['wall_seconds']:reason='WALL_BUDGET'
            elif selected['memory_mib']>p['gpu_memory_gib']*1024:reason='GPU_MEMORY_BUDGET'
            elif rss>p['cpu_memory_gib']*1024**3:reason='CPU_MEMORY_BUDGET'
            elif size>p['output_gib']*1024**3:reason='OUTPUT_BUDGET'
            elif progress.get('total_actions',0)>p['episode_count']*p['max_steps']:reason='ACTION_BUDGET'
            row=dict(unix=time.time(),elapsed_seconds=elapsed,gpu=selected,rss_bytes=rss,output_bytes=size,
                own_pids=sorted(pids),misplaced=misplaced,foreign=foreign,stop_reason=reason)
            c.append(OUT/'RESOURCE.jsonl',row)
            if reason:break
            time.sleep(5)
    except BaseException as exc:
        reason=reason or 'LAUNCHER_ERROR'
        c.write(OUT/'LAUNCH_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        cleanup=terminate_owned(proc,initial) if proc is not None and initial else dict(signaled=False,remaining=[])
        log.close();after=gpu_snapshot();training_after=training_snapshot()
        unchanged=all(a is not None and b is not None and a['start_ticks']==b['start_ticks']
                      for a,b in zip(training_before['processes'],training_after['processes']))
        live_unchanged=all(b is None or a is not None and a['start_ticks']==b['start_ticks']
                           for a,b in zip(training_before['processes'],training_after['processes']))
        resource_reasons={'FOREIGN_GPU_CONTEXT_APPEARED','WALL_BUDGET','GPU_MEMORY_BUDGET','CPU_MEMORY_BUDGET','OUTPUT_BUDGET','ACTION_BUDGET'}
        result=dict(status='COMPLETE' if reason is None and proc and proc.returncode==0 else 'RESOURCE_CENSORED' if reason in resource_reasons else 'SERVICE_FAILED',
            reason=reason,returncode=proc.returncode if proc else None,wall_seconds=time.monotonic()-began,
            peak_gpu_mib=peak_gpu,peak_rss_bytes=peak_rss,peak_output_bytes=peak_output,cleanup=cleanup,
            gpus_after=after,training_after=training_after,training_processes_unchanged=unchanged,
            borrowed_holders=[],foreign_processes_signaled=[],optimizer_updates=0,
            surviving_training_process_identities_unchanged=live_unchanged)
        c.write(OUT/'LAUNCH_RESULT.json',result,True)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        lease.close()
    if result['status']!='COMPLETE':raise SystemExit(1)
    try:
        with (OUT/'aggregate.log').open('x') as report_log:
            report=subprocess.run([str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'aggregate.py')],
                cwd=c.ROOT,stdout=report_log,stderr=subprocess.STDOUT,timeout=1800)
    except BaseException as exc:
        c.write(OUT/'AUDIT_FAILURE.json',dict(error=repr(exc),metrics_not_admitted=True),True)
        raise
    if report.returncode:
        c.write(OUT/'AUDIT_FAILURE.json',dict(returncode=report.returncode,metrics_not_admitted=True),True)
        raise SystemExit(report.returncode)


if __name__=='__main__':main()

