"""Finish verified CPU postprocessing and bounded publication; never start a GPU."""
import sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import read,write,sha,immutable,HERE,c,verify_lock

def main(run):
    verify_lock(read(run/'SOURCE_LOCK.json'))
    post=read(HERE/'POSTPROCESS_SOURCE_LOCK.json')
    for name,expected in post['files'].items():
        if sha(HERE/name)!=expected:raise ValueError('POSTPROCESS_SOURCE_CHANGED')
    if sha(run/'RESULT.json')!=post['result_sha256']:raise ValueError('PRIMARY_RESULT_CHANGED')
    calibration=read(run/'CALIBRATION.json')
    if calibration['groups']!=128 or calibration['source_sha256']!=sha(HERE/'calibration_r2.py'):raise ValueError('CALIBRATION_INCOMPLETE')
    result=read(run/'RESULT.json')
    if result['complete']!=1152 or result['groups']!=128:raise ValueError('EVALUATION_INCOMPLETE')
    if not (run/'CALIBRATION_FAILURE.json').exists():immutable(run/'CALIBRATION_FAILURE.json',read(run/'STATUS.json'))
    immutable(run/'CPU_FINISH_RESULT.json',dict(status='VALID_COMPLETE_WITH_CPU_POSTPROCESS_REPAIR',
        result_sha256=sha(run/'RESULT.json'),calibration_sha256=sha(run/'CALIBRATION.json'),new_training_updates=0,new_rollouts=0))
    budget=sum(row['gpu_hours'] for row in c.records(run/'RESOURCES.jsonl'))
    paths=run/'attempts';folder=paths/f'publish_r2_{len(list(paths.glob("publish_r2_*")))+1:03d}';folder.mkdir()
    started=time.monotonic();cfg=read(run/'PROTOCOL.json')
    with (folder/'stdout.log').open('x') as log:
        proc=subprocess.Popen([cfg['standalone_python'],'-I','-B',str(HERE/'publish_r2.py'),str(run)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            write(run/'STATUS.json',dict(status='RUNNING',stage='publish',unix=time.time(),gpu_hours=budget,
                complete_rollouts=1152,models_complete=9,postprocess_complete=True,workers=[]))
            time.sleep(5)
    success=proc.returncode==0
    write(folder/'EXIT.json',dict(returncode=proc.returncode,wall_seconds=time.monotonic()-started,gpu_hours=0),True)
    write(run/'STATUS.json',dict(status='COMPLETE' if success else 'COMPLETE_UPLOAD_FAILED',stage='complete',
        unix=time.time(),gpu_hours=budget,complete_rollouts=1152,models_complete=9,postprocess_complete=True,
        upload_verified=success,upload_log=str(folder/'stdout.log'),workers=[]))

if __name__=='__main__':main(Path(sys.argv[1]))
