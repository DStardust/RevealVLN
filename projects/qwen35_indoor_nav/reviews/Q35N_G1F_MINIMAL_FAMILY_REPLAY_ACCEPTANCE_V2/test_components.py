"""CPU-only compiler checks; synthetic cases are not family evidence."""
import json
from pathlib import Path
import random
import sys

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
from engine import compress,inverse,transform,task_eval,expected,make_query,digest
import numpy as np

def main():
    assert not (OUT/'COMPONENT_TESTS.json').exists()
    rng=random.Random(1109); checks=[]
    for i in range(200):
        a=[rng.choice('FLR') for _ in range(rng.randrange(1,60))]
        raw,comp=inverse(a)
        t=transform(a+comp)
        assert np.linalg.norm(t['xz'])<1e-10 and t['yaw_bin']==0
        x,y=transform(raw),transform(comp)
        assert np.linalg.norm(np.array(x['xz'])-y['xz'])<1e-10 and x['yaw_bin']==y['yaw_bin']
        assert raw.count('F')==comp.count('F')==a.count('F')
    checks.append({'test':'inverse_compression_SE2_and_translation_count','cases':200,'pass':True,'synthetic':True})
    def tr(events,complete=True):
        return {'actions':['L']*(len(events)-1)+['S'],'complete':complete,
                'observations':[{'step':i,'evidence_complete':True,'rgb_hash':digest(str(i)),
                                 'see2':{k:([1] if k in ev else []) for k in ('D','K','B','L')}} for i,ev in enumerate(events)]}
    tests=[(tr(['','D','B']),'g_D_v2','pass'),(tr(['','','B']),'g_D_v2','fail'),
           (tr(['','B','D']),'g_D_v2','fail'),(tr(['','','DB']),'g_D_v2','fail'),
           (tr(['','D','B'],False),'g_D_v2','unknown'),(tr(['','K','B']),'g_K_v2','pass')]
    for t,g,y in tests:assert task_eval(t,g)==y
    checks.append({'test':'ordered_program_including_strict_simultaneity_and_unknown','cases':len(tests),'pass':True,'synthetic':True})
    q=make_query(tr(['','D','B']))
    projection={k:q[k] for k in ('query_schema_version','coordinate_frame','sequence')}
    changed=dict(q,action_trace_ref='sha256:'+'f'*64,rgb_content_refs=['sha256:'+'a'*64])
    assert digest(projection)==digest({k:changed[k] for k in projection})
    assert 'pass' not in json.dumps(projection) and 'history_id' not in json.dumps(projection)
    checks.append({'test':'query_integrity_ID_fields_excluded_from_semantic_projection','cases':1,'pass':True,'synthetic':True})
    previews=[json.loads(x) for x in (OUT/'WITNESS_DISCOVERY.jsonl').read_text().splitlines()]
    passed=[x for x in previews if x['status']=='WITNESS_PREVIEW_PASS']
    for x in passed:
        nums=x['pixel_counts']; assert any(nums[i-1]>=256 and nums[i]>=256 for i in range(1,len(nums)))
    checks.append({'test':'real_witness_selected_pixel_counts_recomputed','cases':len(passed),'pass':True,'synthetic':False,
                   'scope':'stored preview measurements; not causal RGB recognizability or family proof'})
    (OUT/'COMPONENT_TESTS.json').write_text(json.dumps({'tests':checks,'all_pass':True,'scientific_pass':False},indent=2)+'\n')
    print(f'{sum(x["cases"] for x in checks)} component cases passed; no family/model claim.')

if __name__=='__main__':main()
