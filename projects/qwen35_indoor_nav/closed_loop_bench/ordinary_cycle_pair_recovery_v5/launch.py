"""Bounded shared-GPU sessions; retries resume complete pairs, never single arms."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c
owned = c.load('v5_owned_processes', c.V3/'launch.py')


def snapshot():
    doc = ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15))
    rows = []
    for index,gpu in enumerate(doc.findall('gpu')):
        processes = []
        for proc in gpu.findall('processes/process_info'):
            raw = proc.findtext('used_memory')
            processes.append(dict(pid=int(proc.findtext('pid')),kind=proc.findtext('type'),
                name=proc.findtext('process_name'),memory_mib=int(raw.split()[0]) if raw and raw.split()[0].isdigit() else None))
        rows.append(dict(index=index,uuid=gpu.findtext('uuid'),name=gpu.findtext('product_name'),driver=doc.findtext('driver_version'),
            used_mib=int(gpu.findtext('fb_memory_usage/used').split()[0]),
            free_mib=int(gpu.findtext('fb_memory_usage/free').split()[0]),processes=processes))
    return rows


def environment(session, gpu):
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'): env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key,relative in dict(HF_HOME='cache/hf',XDG_CACHE_HOME='cache/xdg',TORCH_HOME='cache/torch',
        TRITON_CACHE_DIR='cache/triton',CUDA_CACHE_PATH='cache/cuda',NUMBA_CACHE_DIR='cache/numba',
        MPLCONFIGDIR='cache/mpl',TMPDIR='tmp',TMP='tmp',TEMP='tmp').items():
        (session/relative).mkdir(parents=True,exist_ok=True)
        env[key] = str(session/relative)
    return env


def spent_seconds():
    return sum(c.read(path)['gpu_seconds'] for path in (HERE/'sessions').glob('session_*/LAUNCH_RESULT.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=int,choices=[5,20,100],default=100)
    parser.add_argument('--gpu',type=int,default=1)
    args = parser.parse_args()
    p = c.read(HERE/'PROTOCOL.json')
    (HERE/'sessions').mkdir(exist_ok=True)
    lock = (HERE/'gpu_inference.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Verify only dependencies used here, not another historical full audit.
    for path,expected in c.read(HERE/'SOURCE_LOCK.json')['existing_sources'].items():
        assert c.sha(path) == expected, 'FROZEN_SOURCE_CHANGED:'+path
    assert c.sha(HERE/'cycle_policy.py') == c.sha(c.V3/'cycle_policy.py')
    assert c.read(HERE/'CPU_TEST_RESULT.json')['passed']
    retries = sum(int(c.read(x).get('infrastructure_retry',False)) for x in (HERE/'sessions').glob('session_*/LAUNCH_RESULT.json'))
    while True:
        if retries > p['max_infrastructure_retries']:
            break
        found = c.committed_pairs()
        missing = [r for r in range(args.target) if r not in found]
        if not missing: break
        for path in (HERE/'sessions').glob('session_*/FAILURE.json'):
            if c.read(path)['status'] == 'CORRECTNESS_ERROR' and not (path.parent/'CORRECTION.json').exists():
                print('UNRESOLVED_CORRECTNESS: '+str(path),flush=True)
                return
        remaining = p['total_gpu_seconds'] - spent_seconds()
        if remaining < 180:
            c.write(HERE/'RESOURCE_BLOCK.json',dict(reason='ENGINEERING_8_GPU_HOURS_REACHED',completed=len(found)))
            break
        gpus = snapshot()
        gpu = gpus[args.gpu]
        if gpu['free_mib'] < p['start_free_mib']:
            c.write(HERE/'RESOURCE_BLOCK.json',dict(reason='INSUFFICIENT_CURRENT_FREE_MEMORY',required_free_mib=p['start_free_mib'],gpu=gpu,
                shared_usage_authorized=True,exclusive_window_required=False,completed=len(found)))
            break
        number = len(list((HERE/'sessions').glob('session_*')))+1
        session = HERE/'sessions'/f'session_{number:03d}'
        session.mkdir()
        for name in ('pairs','frames','source'): (session/name).mkdir()
        sources = {}
        for path in list(HERE.glob('*.py'))+[HERE/'PROTOCOL.json',HERE/'PAIR_ORDER.json']:
            shutil.copy2(path,session/'source'/path.name)
            sources[path.name] = c.sha(path)
        config = dict(p,gpu=gpu['index'],gpu_uuid=gpu['uuid'],scheduled_ranks=missing,source_hashes=sources)
        c.write(session/'CONFIG.json',config,True)
        c.write(session/'PREFLIGHT.json',dict(gpus=gpus,selected=gpu,shared_usage_authorized_by='User V5',
            exclusive_lease_claimed=False,prior_gpu_seconds=spent_seconds(),remaining_gpu_seconds=remaining,
            infrastructure_retries_used=retries,planned_pair_ranks=missing),True)
        began = time.monotonic()
        deadline = began+min(p['wall_seconds'],remaining)
        proc = None
        initial = None
        reason = None
        peak_rss = 0
        peak_own = 0
        concurrent_samples = 0
        stop_sent = None
        log = (session/'worker.log').open('x')
        try:
            proc = subprocess.Popen([str(c.LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(HERE/'evaluate.py'),str(session)],
                env=environment(session,gpu),cwd=c.ROOT,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            initial = owned.identity(proc.pid)
            assert initial and initial['sid'] == initial['pgid'] == proc.pid
            c.write(session/'PROCESS.json',initial,True)
            while proc.poll() is None:
                members = owned.group_members(proc.pid)
                pids = {x['pid'] for x in members}
                gpus_now = snapshot()
                selected = next(g for g in gpus_now if g['uuid']==gpu['uuid'])
                foreign = [x for x in selected['processes'] if x['pid'] not in pids]
                own_mib = sum(x['memory_mib'] or 0 for x in selected['processes'] if x['pid'] in pids)
                misplaced = [dict(x,gpu=g['uuid']) for g in gpus_now if g['uuid'] != gpu['uuid'] for x in g['processes'] if x['pid'] in pids]
                rss = sum(x['rss_bytes'] for x in members)
                peak_rss=max(peak_rss,rss);peak_own=max(peak_own,own_mib)
                concurrent_samples += bool(foreign)
                progress = c.read(session/'PROGRESS.json') if (session/'PROGRESS.json').exists() else {}
                size,misses = owned.tree.tree_size(HERE)
                if misplaced: reason='OWN_GPU_MAPPING_ERROR'
                elif selected['free_mib'] < p['minimum_free_mib']: reason='GPU_MEMORY_PRESSURE'
                elif rss > p['cpu_memory_gib']*1024**3: reason='RSS_BUDGET'
                elif size > p['output_gib']*1024**3: reason='OUTPUT_BUDGET'
                elif time.monotonic() > deadline-120: reason='SESSION_TIME_BOUNDARY'
                elif progress and time.time()-progress['unix'] > p['stall_seconds']: reason='NO_PROGRESS'
                c.append(session/'RESOURCE.jsonl',dict(unix=time.time(),elapsed_seconds=time.monotonic()-began,
                    gpu=selected,foreign=foreign,own_memory_mib=own_mib,rss_bytes=rss,output_bytes=size,
                    resource_competition=bool(foreign),stop_reason=reason,misplaced=misplaced,transient_stat_misses=misses))
                if reason and stop_sent is None:
                    current=owned.identity(proc.pid)
                    assert current and current['start_ticks']==initial['start_ticks']
                    proc.send_signal(signal.SIGTERM)
                    stop_sent=time.monotonic()
                if stop_sent is not None and time.monotonic()-stop_sent>90: break
                time.sleep(5)
        except BaseException as exc:
            reason=reason or 'LAUNCH_SERVICE_ERROR'
            c.write(session/'LAUNCH_FAILURE.json',dict(error=repr(exc)),True)
        finally:
            cleanup = owned.terminate_owned(proc,initial) if proc and initial else dict(signaled=False,remaining=[])
            log.close()
            failure = c.read(session/'FAILURE.json') if (session/'FAILURE.json').exists() else None
            infra = bool(reason or proc is None or proc.returncode != 0) and not (failure and failure['status']=='CORRECTNESS_ERROR')
            record = dict(status='CORRECTNESS_ERROR' if failure and failure['status']=='CORRECTNESS_ERROR' else 'INTERRUPTED' if infra else 'COMPLETE',
                reason=reason,returncode=proc.returncode if proc else None,gpu_seconds=time.monotonic()-began,
                gpu_uuid=gpu['uuid'],peak_own_memory_mib=peak_own,peak_group_rss_bytes=peak_rss,competition_samples=concurrent_samples,
                cleanup=cleanup,foreign_processes_signaled=[],infrastructure_retry=infra)
            c.write(session/'LAUNCH_RESULT.json',record,True)
            c.load('v5_final_aggregate',HERE/'aggregate.py').main()
            print(record,flush=True)
        if record['status']=='CORRECTNESS_ERROR': break
        if infra:
            retries += 1
            if retries > p['max_infrastructure_retries']: break
        # A new session is allowed only for still-missing whole pairs, never for score selection.
    lock.close()


if __name__=='__main__': main()
