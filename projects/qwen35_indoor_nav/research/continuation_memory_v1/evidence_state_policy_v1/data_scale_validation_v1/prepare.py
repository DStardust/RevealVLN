"""Compile frozen physical data into causal shards; preserve the original teacher/loss semantics."""
import copy,random,sys,time
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from evaluator_v16 import legacy,state_sequence,query_contexts,compose

def schedule_expanded(raw,old,seed):
 groups=defaultdict(list)
 for f in raw:groups[f['parent_family_id']].append(f['family_id'])
 for v in groups.values():v.sort()
 rng=random.Random(seed);visits=Counter();parents=sorted(groups);out=[]
 while len(out)<len(old):
  order=parents.copy();rng.shuffle(order)
  for parent in order:
   if len(out)==len(old):break
   fid=groups[parent][visits[parent]%len(groups[parent])];visits[parent]+=1
   out.append(dict(family=fid,ordinary=old[len(out)]['ordinary']))
 return out

def main(run):
 cfg=config(run)
 if (run/'PREPARED.json').exists():
  for p,h in read(run/'PREPARED.json')['files'].items():
   if sha(run/p)!=h:raise ValueError('PREPARED_CHANGED:'+p)
  return
 if sha(Path(cfg['expanded_data']))!=cfg['expanded_sha256']:raise ValueError('EXPANSION_CHANGED')
 (run/'families').mkdir(exist_ok=True)
 old=read(Path(cfg['old_data']));extra=read(Path(cfg['expanded_data']))['raw_families'];natural=read(PARENT.parent/'natural_transfer_v9/DATA.json')
 oldfit=[f for f in old['families'] if f['split']=='FIT'];oldraw=[f for f in old['raw_families'] if f['split']=='FIT']
 olddev={f['house'] for f in old['raw_families'] if f['split']!='FIT'}
 ordinary_check={r['row']['scene_group'] for r in natural['records'] if r['partition']!='fit'}
 forbid=set(cfg['houses'])|olddev|ordinary_check
 if {f['house'] for f in extra+oldraw}&forbid:raise ValueError('FIT_HELDOUT_HOUSE_LEAK')
 if any(f['split']!='FIT' or f.get('training_admission')!='CONTROLLED_SEE2_ARRAY_CERTIFIED_FIT' for f in extra):raise ValueError('EXPANSION_ADMISSION')
 ids=[f['family_id'] for f in oldraw+extra]
 if len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_FAMILY_ID')
 if len({f['parent_family_id'] for f in extra})!=816 or len(extra)!=1491:raise ValueError('REGISTERED_DATA_COUNTS')
 data=dict(features=copy.deepcopy(old['features']),contents=copy.deepcopy(old['contents']))
 lookup={x['key']:i for i,x in enumerate(data['features'])};files={};arrays={};counts=Counter()
 from functools import lru_cache
 @lru_cache(maxsize=64)
 def trace_at(path,expected):
  if sha(LINE/path)!=expected:raise ValueError('TRACE_CHANGED')
  return read(LINE/path)
 def save_family(f):
  path=run/'families'/(f['family_id']+'.json');immutable(path,f)
  files[f['family_id']]=dict(path=str(path.relative_to(run)),sha256=sha(path),parent=f['parent_family_id'],house=f['house'],stratum=f['stratum'])
 for f in oldfit:save_family(f)
 certs={k:load('scale_cert_'+k,V16/n/'certify.py').certify for k,n in [('terminal_present','identifiable_data_v1'),('terminal_absent','b2_state_coverage_v1')]}
 # Load the existing array auditor with its own shared module isolated.
 local_shared=sys.modules['shared'];sys.modules.pop('shared')
 quality=load('scale_admission_audit',PARENT/'event_data_scale20_v2/quality.py')
 sys.modules['shared']=local_shared;sys.path.insert(0,str(HERE))
 def feature(trace,t,instruction):
  item=dict(instruction=instruction,rgb_refs=['sha256:'+o['rgb_hash'] for o in trace['observations'][max(0,t-1):t+1]],executed=[dict(F='move_forward',L='turn_left',R='turn_right')[a] for a in trace['actions'][max(0,t-8):t]])
  key=digest(item)
  if key in lookup:
   if data['features'][lookup[key]]['split']!='FIT':raise ValueError('CROSS_SPLIT_CAUSAL_INPUT')
  else:lookup[key]=len(data['features']);data['features'].append(dict(key=key,split='FIT',**item))
  return lookup[key]
 for j,f in enumerate(extra):
  audit_path=LINE/f['audit']['path']
  if sha(audit_path)!=f['audit']['sha256']:raise ValueError('SOURCE_ADMISSION_CHANGED')
  a=quality.audit_family(f,certs[f['stratum']],arrays)
  compiler=legacy.Compiler(**f['compiler']);rows=[]
  for h in f['histories']:
   cutoff=len(f['histories'][h])
   for task in ('task_A','task_T'):
    instruction=f['compiler']['tasks']['task_A']['instruction'] if task=='task_A' else f['terminal_instruction']
    if f['stratum']=='terminal_present':teacher='direct_stop' if h.startswith('seen') or task=='task_T' else 'acquire_anchor'
    else:teacher=min((q for q in f['suffixes'] if f['certificate']['labels'][h+'__'+q+'__'+task]['label']=='PASS'),key=lambda q:(len(f['suffixes'][q]),q))
    if f['certificate']['labels'][h+'__'+teacher+'__'+task]['label']!='PASS':raise ValueError('TEACHER_NOT_REAL_PASS')
    for q in f['suffixes']:
     ref=f['traces'][h+'__'+q];trace=trace_at(ref['path'],ref['sha256']);n=len(trace['actions'])
     states=state_sequence(compiler,trace['observations'],task);queries=query_contexts(compiler,trace,task)
     label=f['certificate']['labels'][h+'__'+q+'__'+task]['label']
     if label not in ('PASS','FAIL'):raise ValueError('UNKNOWN_TRAINING_LABEL')
     y=int(label=='PASS')
     if any(int(compose(states[t],queries[t]))!=y for t in range(n)):raise ValueError('QUERY_STATE_DISAGREEMENT')
     atoms=compiler.atoms(trace['observations']);events=[[int(bool(e[k])) for k in ('anchor','terminal')] for e in atoms]
     masks=[[int(e[k] is not None and (k!='anchor' or task!='task_T')) for k in ('anchor','terminal')] for e in atoms]
     rows.append(dict(family_id=f['family_id'],house=f['house'],split='FIT',history=h,task=task,continuation=q,features=[feature(trace,t,instruction) for t in range(n)],cutoff=cutoff,targets=['FLRS'.index(x) for x in trace['actions']],action_masks=[int(t>=cutoff and q==teacher) for t in range(n)],state_targets=states[:n],state_masks=[1]*n,query_contexts=queries,y=[y]*n,query_masks=[1]*n,source_trace=ref,teacher_selection='shortest registered real PASS continuation; no model scores',event_targets=events[:n],event_masks=masks[:n]))
     counts['causal_steps_with_reuse']+=n;counts['action_labels']+=sum(rows[-1]['action_masks'])
  save_family(dict(family_id=f['family_id'],parent_family_id=f['parent_family_id'],house=f['house'],split='FIT',stratum=f['stratum'],certificate=f['certificate'],mechanism_cases=[],sequences=rows))
  # RGB evidence is indexed by causal pixels, while semantics stays only in labels/audits.
  for p,h in a['arrays'].items():
   if p.endswith('.rgb.npy'):
    ref='sha256:'+Path(p).name.split('.')[0]
    data['contents'].setdefault(ref,dict(line_relative_path=p,file_sha256=h))
  if j%10==0:write(run/'PREPARE_PROGRESS.json',dict(complete=j+1,total=len(extra),features=len(data['features']),arrays_verified=len(arrays),unix=time.time()))
 # Ordinary FIT can overlap expanded FIT, never any heldout house. Shared for both arms.
 ordinary_fit={r['row']['scene_group'] for r in natural['records'] if r['partition']=='fit'}
 if ordinary_fit&(set(cfg['houses'])|olddev):raise ValueError('ORDINARY_HELDOUT_LEAK')
 schedules=read(CPU/'SCHEDULES.json');both=dict(OLD=schedules,EXPANDED={str(s):schedule_expanded(oldraw+extra,schedules[str(s)],s) for s in cfg['seeds']})
 for arm in both:
  allowed={f['family_id'] for f in (oldraw if arm=='OLD' else oldraw+extra)}
  for seed,rows in both[arm].items():
   if len(rows)!=cfg['steps'] or any(r['family'] not in allowed or any(natural['records'][i]['partition']!='fit' for i in r['ordinary']) for r in rows):raise ValueError('SCHEDULE_LEAK')
 immutable(run/'TRAIN_DATA.json',dict(**data,family_files=files,arms=dict(OLD=[f['family_id'] for f in oldfit],EXPANDED=ids)))
 immutable(run/'SCHEDULES.json',both)
 # Original FIT statistics, identical loss weights across both pools.
 import importlib.util
 base_common=load('scale_train_common',PARENT/'common.py');sys.modules['common']=base_common
 obj=load('scale_train_objective',PARENT/'objective.py');immutable(run/'SHARED_FIT_WEIGHTS.json',obj.weights(oldfit))
 immutable(run/'WARMUP_DATA.json',old)
 heldout=read(Path(cfg['eval_run'])/'DATA.json');immutable(run/'DATA.json',heldout)
 audit=dict(old_parents=32,old_variants=59,new_parents=816,new_variants=1491,expanded_parents=len({f['parent_family_id'] for f in oldraw+extra}),expanded_variants=len(ids),new_houses=sorted({f['house'] for f in extra}),heldout=cfg['houses'],heldout_exposed=True,arrays_verified=len(arrays),features=len(data['features']),counts=counts,ordinary_FIT_overlap_expansion=sorted(ordinary_fit&{f['house'] for f in extra}),new_physical_executions=0,mechanism_diagnosis='not recomputed; this experiment measures data scale, not an architecture or memory mechanism claim')
 registry_value=local_module("evaluate_continuations").registry_value
 immutable(run/'EVALUATION_REGISTRY.json',registry_value(heldout['raw_families'],cfg))
 audit.update(data_sha256=sha(run/'DATA.json'),registry_sha256=sha(run/'EVALUATION_REGISTRY.json'))
 immutable(run/'DATA_AUDIT.json',audit)
 exposure={}
 for arm,ss in both.items():
  exposure[arm]={s:dict(unique_variants=len({r['family'] for r in rows}),unique_parents=len({files[r['family']]['parent'] for r in rows}),variant_occurrences=dict(Counter(r['family'] for r in rows))) for s,rows in ss.items()}
 immutable(run/'EXPOSURE_SCHEDULE.json',exposure)
 names=['TRAIN_DATA.json','SCHEDULES.json','SHARED_FIT_WEIGHTS.json','WARMUP_DATA.json','DATA.json','DATA_AUDIT.json','EVALUATION_REGISTRY.json','EXPOSURE_SCHEDULE.json']
 immutable(run/'PREPARED.json',dict(files={n:sha(run/n) for n in names},family_files=len(files),feature_windows=len(data['features'])))
 write(run/'PREPARE_PROGRESS.json',dict(complete=len(extra),total=len(extra),features=len(data['features']),done=True))
if __name__=='__main__':main(Path(sys.argv[1]))
