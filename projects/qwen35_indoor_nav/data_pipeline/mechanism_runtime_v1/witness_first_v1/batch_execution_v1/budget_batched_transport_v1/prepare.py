"""CPU-only prospective batch07 configuration using original prepare function."""
import importlib.util
import json
from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('clock_prepare_transport',HERE/'transport.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
SNAPSHOT=t.WF/'multi_program_bank_v2/language_ready_v1'

def selection(snapshot):
    draft=json.loads((snapshot/'CONFIG_DRAFT.json').read_text())
    nxt=json.loads((snapshot/'NEXT12.json').read_text())
    assert draft['runtime_allowed'] is False and nxt['runtime_allowed'] is False
    assert len(nxt['candidate_ids'])==12 and len(set(nxt['candidate_ids']))==12
    mapping={r['candidate_id']:(i,r) for i,r in enumerate(draft['candidates'])}
    assert len(mapping)==len(draft['candidates'])
    picked=[];houses=set();positions=[]
    for pos,key in enumerate(nxt['candidate_ids']):
        i,row=mapping[key]
        if row['house_id'] in houses:continue
        assert row['split']=='FIT' and len(row['configuration']['u_position'])==3
        houses.add(row['house_id']);picked.append(i);positions.append(pos)
        if len(picked)==3:break
    assert len(picked)==3
    return picked,positions

def adapted_prepare_source(source):
    old="cfg['split_sha256']=sha(split)"
    new=old+"\n    cfg.update(budget_transport='clock_batching_fresh_only_v1',fresh_only=True,resume_allowed=False,comparison='unpaired_actual_yield_and_wall_only',cohort_membership=None,next12_positions=[0,2,4])"
    assert source.count(old)==1;source=source.replace(old,new)
    old='for p in code:lock[str(p.resolve())]=sha(p)'
    new="code += extra_code\n    "+old
    assert source.count(old)==1;return source.replace(old,new)

def wrapper(entry):
    return "import importlib.util\nfrom pathlib import Path\nHERE=Path(__file__).resolve().parent\ns=importlib.util.spec_from_file_location('batch07_clock_transport',HERE.parent/'budget_batched_transport_v1/transport.py')\nt=importlib.util.module_from_spec(s);s.loader.exec_module(t)\nif __name__=='__main__':t."+entry+"(HERE)\n"

def prepare():
    snapshot=SNAPSHOT;indices,positions=selection(snapshot)
    assert indices==[0,34,60] and positions==[0,2,4],'FIXED_PREACTION_SELECTION'
    t.base('prepare.py')
    assert t.sha(t.CLOCK)==t.CLOCK_SHA
    source=(t.BE/'prepare.py').read_text()
    module=types.ModuleType('clock_private_original_prepare');module.__file__=str(t.BE/'prepare.py')
    exec(compile(adapted_prepare_source(source),str(HERE/'prepare.py'),'exec'),module.__dict__)
    module.extra_code=[*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',snapshot/'NEXT12.json',snapshot/'SOURCE_LOCK.json',
        t.CLOCK,t.CLOCK.parent/'SHA256SUMS',t.CLOCK.parent/'SPEC_ZH.md']
    # All selected source/asset hashes are checked by original preparation before
    # it creates run_v1; original constructors are instantiated without render.
    batch=t.BE/'batch_07';batch.mkdir(exist_ok=False)
    for name,entry in [('worker.py','worker_main'),('run.py','run_main')]:
        with (batch/name).open('x') as f:f.write(wrapper(entry))
    module.prepare(snapshot,'batch_07',indices,2,'winding_v1')
    cfg=t.check_inputs(batch,worker=True)
    print(json.dumps({'cpu_only':True,'gpu_started':False,'next12_positions':positions,
                      'indices':indices,'houses':[r['house_id'] for r in cfg['candidates']]}))

if __name__=='__main__':prepare()
