"""GPU1 empty-only bounded lease, no holder signals, exact owned-child cleanup."""
import fcntl,importlib.util,json,os,subprocess,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def gpu():
    root=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True));g=next(x for x in root.findall('gpu') if x.findtext('uuid')=='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8')
    return dict(processes=[int(x.findtext('pid')) for x in g.findall('processes/process_info')],memory_mib=float(g.findtext('fb_memory_usage/used').split()[0]))
def identity(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,start=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],argv=p.joinpath('cmdline').read_bytes().split(b'\0')[:-1],cwd=str(p.joinpath('cwd').resolve()))
    except FileNotFoundError:return None
def main():
    lock=(HERE/'extract.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'LAUNCH_RESULT.json').exists();c.verify();assert c.read(HERE/'CPU_TEST_RESULT.json')['status']=='PASS'
    pre=gpu();assert pre['processes']==[] and pre['memory_mib']<1024
    c.write(HERE/'PREFLIGHT.json',dict(unix=time.time(),gpu=pre,resource='empty GPU1 only',source_lock_sha256=c.sha(HERE/'SOURCE_LOCK.json')))
    env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='1',TMPDIR=str(c.LINE/'.tc12'),TORCHINDUCTOR_CACHE_DIR=str(HERE/'inductor_cache'),TRITON_CACHE_DIR=str(HERE/'triton_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    (c.LINE/'.t12').mkdir(exist_ok=True)
    proc=None;ident=None;began=time.monotonic();reason=None
    try:
        with (HERE/'extract.log').open('x') as log:
            proc=subprocess.Popen([str(c.LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(HERE/'extract.py')],cwd=c.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            ident=identity(proc.pid);assert ident and ident['cwd']==str(c.ROOT)
            c.write(HERE/'PROCESS.json',dict(pid=proc.pid,start=ident['start'],argv=[x.decode() for x in ident['argv']],cwd=ident['cwd'],unix=time.time()))
            while proc.poll() is None:
                assert identity(proc.pid)==ident,'OWNED_IDENTITY_CHANGED'
                info=gpu();assert set(info['processes'])<= {proc.pid},'FOREIGN_GPU1_CONTEXT'
                status=(Path('/proc')/str(proc.pid)/'status').read_text();rss=next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0)
                size=sum(x.stat().st_size for x in HERE.rglob('*') if x.is_file())
                c.write(HERE/'RESOURCE.json',dict(unix=time.time(),pid=proc.pid,wall_seconds=time.monotonic()-began,gpu=info,rss_bytes=rss,output_bytes=size),False)
                assert time.monotonic()-began<=300 and info['memory_mib']<=25*1024 and rss<=16*1024**3 and size<=1024**3,'RESOURCE_BUDGET'
                time.sleep(5)
            assert proc.returncode==0,'EXTRACTION_EXIT_'+str(proc.returncode)
    except BaseException as exc:
        reason=repr(exc);c.write(HERE/'LAUNCH_FAILURE.json',dict(error=reason,traceback=traceback.format_exc(),unix=time.time()))
        raise
    finally:
        if proc and proc.poll() is None:
            assert identity(proc.pid)==ident;proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                assert identity(proc.pid)==ident;proc.kill();proc.wait(timeout=10)
        after=gpu();c.write(HERE/'LAUNCH_RESULT.json',dict(status='COMPLETE' if reason is None else 'FAILED',unix=time.time(),reason=reason,wall_seconds=time.monotonic()-began,gpu_after=after,owned_process_cleaned=not proc or proc.poll() is not None,other_processes_signaled=[],holders_released=[]))
if __name__=='__main__':main()
