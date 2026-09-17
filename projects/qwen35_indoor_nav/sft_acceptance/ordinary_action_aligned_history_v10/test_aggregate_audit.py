"""Execute the actual new aggregate statements; altered selected hashes must fail."""
import ast,json,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
CASE=LINE/'closed_loop_bench/ordinary_action_aligned_history_dev_v10'
source=runpy.run_path(str(CASE/'reuse.py'))['source']('aggregate.py')
tree=ast.parse(source);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
loop=next(x for x in ast.walk(fn) if isinstance(x,ast.For) and isinstance(x.target,ast.Tuple) and [t.id for t in x.target.elts]==['policy','step'])
start=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='t')
end=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='append')
block=compile(ast.fix_missing_locations(ast.Module(body=loop.body[start:end+1],type_ignores=[])),'ACTUAL_AGGREGATE_RGB_AUDIT','exec')
assert len(loop.body[start:end+1])==6
passed=0
for t in (0,1,7,8,9,50,499):
    history=[f'hash{x}' for x in range(t+1)];selected=[0] if t==0 else [max(0,t-8),t]
    valid=dict(frame_stride=8,input_frame_indices=selected,input_rgb_sha256=[history[j] for j in selected])
    for bad in (None,'rgb','index','stride'):
        policy=dict(valid)
        if bad=='rgb':policy['input_rgb_sha256']=['bad']*len(selected)
        if bad=='index':policy['input_frame_indices']=[500]
        if bad=='stride':policy['frame_stride']=1
        ns=dict(st=t+1,index=3,policy=policy,step=dict(rgb_sha256='next'),observed={3:list(history)})
        try:exec(block,ns)
        except AssertionError:assert bad is not None
        else:assert bad is None and ns['observed'][3]==history+['next']
        passed+=1
result=dict(status='PASS',unix=time.time(),actual_aggregate_cases=passed,positive_and_tampered_hash_index_stride_checked=True,gpu_actions=0)
with (HERE/'AGGREGATE_CPU_TEST_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
