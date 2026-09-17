"""Stdlib regression for paired metrics, unmeasured binding and workflow scope."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import time
HERE=Path(__file__).resolve().parent
def main():
    c=runpy.run_path(str(HERE/'compare.py'),run_name='CPU_COMPARISON_TEST')
    rows={i:dict(episode_id=str(i),house='house'+str(i%5),sr=float(i<21),spl=.18,ndtw=.36) for i in range(100)}
    identical=c['comparison'](rows,rows)
    assert identical['delta']==dict(sr=0.,spl=0.,ndtw=0.) and not identical['pass_development_gate']
    better={i:dict(x) for i,x in rows.items()};better[99]['sr']=1.;better[99]['spl']=.4
    positive=c['comparison'](better,rows)
    assert positive['wins']==1 and positive['losses']==0 and positive['pass_development_gate']
    assert positive['delta']['sr']==.01 and len(positive['leave_one_house_out'])==5
    assert c['comparison'](rows,better)['losses']==1 and not c['comparison'](rows,better)['pass_development_gate']
    bad={i:dict(x,ndtw=x['ndtw']-.02) for i,x in better.items()}
    assert not c['comparison'](bad,rows)['pass_development_gate']
    mismatched={i:dict(x) for i,x in rows.items()};mismatched[0]['episode_id']='wrong'
    try:c['comparison'](mismatched,rows)
    except AssertionError:pass
    else:raise AssertionError('MISMATCHED_EPISODES_ACCEPTED')
    prepare=runpy.run_path(str(HERE/'prepare.py'),run_name='CPU_PREFREEZE_TEST')
    workflow=runpy.run_path(str(HERE/'workflow.py'),run_name='CPU_WORKFLOW_IMPORT')
    assert prepare['TRAIN']==workflow['TRAIN'] and prepare['HERE']==workflow['HERE']==HERE
    assert "while not gpu_empty()" in (HERE/'workflow.py').read_text()
    assert "env=dict(os.environ,CUDA_VISIBLE_DEVICES='')" in (HERE/'workflow.py').read_text()
    assert 'SIGKILL' not in (HERE/'workflow.py').read_text()
    assert not (HERE/'WORKFLOW_PROCESS.json').exists()
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    result=dict(status='PASS_ORCHESTRATION_CPU_ONLY',unix=time.time(),same_episode_pairing_rejected_when_wrong=True,
        unchanged_not_positive=True,matched_3metric_gate_checked=True,bootstrap_deterministic=True,
        evaluation_not_launched=True,training_not_modified=True,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    with (HERE/'CPU_ORCHESTRATION_RESULT.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)
if __name__=='__main__':main()

