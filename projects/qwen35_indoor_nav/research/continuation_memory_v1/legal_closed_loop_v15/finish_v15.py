"""Standalone CPU completion of the finished frozen V15 experiment."""
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
LINE=HERE.parents[2]
STD=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
TORCH=LINE/'.envs/q35n_qwen_g2_v1/bin/python3'
sys.path.insert(0,str(HERE))
import review as r


def main():
    for name,interpreter,args,output in [
        ('resume_cpu',STD,['test_review.py'],'REVIEW_CPU_TEST_RESULT.json'),
        ('review',STD,['review.py'],None),
        ('content',TORCH,['audit_continuation_content.py',str(HERE/'continuation_run_004')],'continuation_run_004_CONTENT_REVIEW.json'),
        ('resources',STD,['resource_review.py'],None),
        ('finalize',STD,['finalize.py'],'REPORT_ZH.md'),
        ('package',STD,['package_evidence.py'],'EVIDENCE_MANIFEST.json')]:
        if output and (HERE/output).exists():continue
        command=[str(interpreter),'-I','-B']+(['-S'] if interpreter==STD else [])+[str(HERE/args[0]),*args[1:]]
        print('START',name,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        print('COMPLETE',name,flush=True)
    assert r.c.read(HERE/'RESULT.json')['complete_continuations']==216
    assert r.c.read(HERE/'EVIDENCE_MANIFEST.json')['readback_verified']
    r.c.write(HERE/'FINISH_JOB_RESULT.json',dict(status='COMPLETE',models=18,continuations=216,
        new_GPU_work=False,remaining=0),True)


if __name__=='__main__':main()
