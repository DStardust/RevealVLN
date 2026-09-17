import ast
import copy
import importlib.util
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('causal_loader',OUT/'loader.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
tests=0
for p in OUT.glob('*.py'):ast.parse(p.read_text());tests+=1
p={'instruction':'Go to the bedroom.','rgb_sequence':['frame0','frame1','frame2','frame3']}
s={'actions':['turn_left','move_forward','turn_right','STOP'],'scene_id':'secret','position':[9,8,7]}
for t in range(4):
    ex=m.decision_example(p,s,t)
    assert set(ex['policy_input'])=={'instruction','rgb_history','executed_actions'}
    assert len(ex['policy_input']['rgb_history'])==t+1
    assert len(ex['policy_input']['executed_actions'])==t
    assert ex['target_action']==s['actions'][t]
    altered=copy.deepcopy(p);altered['rgb_sequence'][t+1:]=['SECRET_FUTURE']*(3-t)
    ss=copy.deepcopy(s);ss['scene_id']='different';ss['position']=[0,0,0]
    for k in range(t,3):ss['actions'][k]='turn_right'
    assert m.decision_example(altered,ss,t)['policy_input']==ex['policy_input']
    tests+=5
for t in (-1,4):
    try:m.decision_example(p,s,t)
    except ValueError:tests+=1
    else:raise AssertionError('invalid time accepted')
for actions in (['STOP']*4,['foo']*3+['STOP'],['move_forward']*4):
    try:m.decision_example(p,{'actions':actions},0)
    except ValueError:tests+=1
    else:raise AssertionError('invalid action accepted')
report={'pass':True,'checks':tests,'synthetic_loader_and_syntax_only':True,'model_tensor_or_gradient_test':False}
with (OUT/'TEST_RESULT.json').open('x') as f:json.dump(report,f,indent=2)
print(report)
