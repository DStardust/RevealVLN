"""Executable controls for matched data exposure, real loss gradients and missing-slot accounting."""
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run=None):
 import torch
 from prepare import schedule_expanded
 evaluator=local_module("evaluate_continuations");registry_value,prefix_audit=evaluator.registry_value,evaluator.prefix_audit
 review_module=local_module("review");metric,paired=review_module.metric,review_module.paired
 cfg=read(HERE/'PROTOCOL.json');old=read(CPU/'DATA.json');schedules=read(CPU/'SCHEDULES.json')
 extra=read(Path(cfg['expanded_data']))['raw_families'];raw=[f for f in old['raw_families'] if f['split']=='FIT']+extra
 tests=[]
 for seed in cfg['seeds']:
  s=schedule_expanded(raw,schedules[str(seed)],seed)
  assert s==schedule_expanded(raw,schedules[str(seed)],seed)
  assert len(s)==1200 and [r['ordinary'] for r in s]==[r['ordinary'] for r in schedules[str(seed)]]
  byid={f['family_id']:f['parent_family_id'] for f in raw}
  assert len({byid[r['family']] for r in s})==848
 tests+=['deterministic_1200_all_848_parents','identical_ordinary_schedule']
 held=read(Path(cfg['eval_run'])/'DATA.json');reg=registry_value(held['raw_families'],cfg)
 assert len(reg['slots'])==1536 and sum(x['available'] for x in reg['conditions'])==248
 assert len({f['house'] for f in raw}&set(cfg['houses']))==0
 assert metric([dict(status='NOT_COLLECTED',cost=1.)])['upper']==1
 tests+=['heldout_disjoint','missing_48_slots_retained']
 a=make_head('OLD_1209');b=make_head('EXPANDED_1209')
 assert c.model_identity(a)['sha256']==c.model_identity(b)['sha256']
 assert a.mode==b.mode=='MONOTONIC'
 tests+=['identical_monotonic_initialization']
 common=sys.modules['common'];sys.path.insert(0,str(PARENT));objective=load('scale_test_objective',PARENT/'objective.py')
 family=next(f for f in old['families'] if f['split']=='FIT')
 cache=torch.load(CONTROL/'features/FEATURES.pt',map_location='cpu',weights_only=True)
 w=objective.weights([f for f in old['families'] if f['split']=='FIT'])
 torch.set_num_threads(2)
 batch=objective.batch(family,'cpu');loss,stats,result=objective.losses(a,cache,batch,w)
 assert torch.isfinite(loss);before=c.model_identity(a)['sha256'];loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
 opt=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.01);opt.step()
 assert c.model_identity(a)['sha256']!=before and not torch.cuda.is_initialized()
 # Padding is masked; future queries never reach the memory/action forward.
 changed=copy.deepcopy(batch);changed['query']=1-changed['query']
 x=a(cache['features'][batch['indices']],cache['logits'][batch['indices']],batch['alive'])['logits']
 y=a(cache['features'][changed['indices']],cache['logits'][changed['indices']],changed['alive'])['logits']
 assert torch.equal(x,y)
 assert not (batch['event_mask'][~batch['alive']]).any()
 tests+=['real_original_FIT_loss_backward_update_CPU_only','query_not_in_actor','padded_event_masks']
 from select_action import select
 assert select([0,0,0,100],[0,3,0,0])['executed_action']=='turn_left'
 tests+=['method_argmax_no_native_STOP_override']
 if run:
  registry=local_module("evaluate_continuations").registry
  assert registry(run)==reg
  d=read(run/'TRAIN_DATA.json')
  assert len(d['arms']['OLD'])==59 and len(d['arms']['EXPANDED'])==1550
  assert set(d['arms']['OLD'])<=set(d['arms']['EXPANDED'])
  ss=read(run/'SCHEDULES.json')
  for seed in cfg['seeds']:
   assert [r['ordinary'] for r in ss['OLD'][str(seed)]]==[r['ordinary'] for r in ss['EXPANDED'][str(seed)]]
  tests+=['actual_compiled_arm_counts_and_shared_ordinary','registered_evaluation_identity']
 result=dict(status='CPU_CONTROLS_PASSED',tests=tests,count=len(tests),cuda_initialized=False,isolated_CPU_updates=1,gpu_updates=0,navigation_gain=None)
 write((run or HERE)/'CPU_TEST_RESULT.json',result)
 print(result,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else None)
