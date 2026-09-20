"""Native CPU initial-state proposals for the frozen new-house family list."""
import copy
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LINE, RESEARCH, c
import numpy as np
import habitat_sim
import magnum as mn
from habitat_sim.agent.controls.default_controls import _rotate_local,_move_along


def main():
    graph=habitat_sim.SceneGraph();node=graph.get_root_node().create_child()
    def canonical(q):
        values=(q.scalar,*q.vector)
        return tuple(-x for x in values) if next((x for x in values if x),1)<0 else values
    def components(actions,rotation):
        node.reset_transformation();node.rotation=mn.Quaternion(mn.Vector3(rotation[1:]),rotation[0])
        shifts=[];rotations=[]
        for a in actions:
            node.translation=mn.Vector3(0,0,0)
            if a in ('L','R'):_rotate_local(node,15 if a=='L' else -15,1)
            else:_move_along(node,-.25,2)
            shifts.append(list(node.translation));rotations.append(canonical(node.rotation))
        return np.array(shifts,dtype=np.float32),tuple(rotations[-2:])
    def translate(position,shifts):
        result=np.array(position,dtype=np.float32,copy=True)
        for delta in shifts:result+=delta
        return result
    families=[];audit=[]
    for source in c.read(RESEARCH/'NEW_HOUSE_ASSEMBLY.json')['families']:
        family=copy.deepcopy(source);origin=family['candidate']['position'];histories=family['candidate']['histories']
        found=False;rotations_matched=0;attempts=[]
        for delta in [0]+[s*i*1e-5 for i in range(1,501) for s in (1,-1)]:
            rotation=[math.cos(delta/2),0,math.sin(delta/2),0]
            vectors={};tails={}
            for name,actions in histories.items():vectors[name],tails[name]=components(actions,rotation)
            if len(set(tails.values()))!=1:continue
            rotations_matched+=1;starts={h:dict(position=list(origin),rotation=rotation) for h in histories};dimensions=[]
            for axis in (0,2):
                unit=abs(float(np.spacing(np.float32(origin[axis]))))
                offsets=np.array([0]+[s*i for i in range(1,1025) for s in (1,-1)],dtype=np.float32)
                positions=np.repeat(np.array(origin,dtype=np.float32)[None],len(offsets),axis=0)
                positions[:,axis]+=offsets*unit
                tables={}
                for name in histories:
                    output=translate(positions,vectors[name])[:,axis];table={}
                    for value,start in zip(output.tolist(),positions[:,axis].tolist()):table.setdefault(value,start)
                    tables[name]=table
                shared=set.intersection(*(set(t) for t in tables.values()))
                dimensions.append(dict(axis=axis,common_outputs=len(shared)))
                if not shared:break
                target=min(shared,key=lambda v:(max(abs(t[v]-origin[axis]) for t in tables.values()),abs(v-origin[axis]),v))
                for h in histories:starts[h]['position'][axis]=tables[h][target]
            attempts.append(dict(angle_delta_rad=delta,dimensions=dimensions))
            if len(dimensions)==2 and all(d['common_outputs'] for d in dimensions):
                endpoints={h:translate(starts[h]['position'],vectors[h]).tolist() for h in histories}
                assert all(p==endpoints['H_A'] for p in endpoints.values())
                # Separate native execution checks the proposed vectorized sum.
                for h,actions in histories.items():
                    node.reset_transformation();node.translation=mn.Vector3(starts[h]['position'])
                    node.rotation=mn.Quaternion(mn.Vector3(rotation[1:]),rotation[0])
                    for a in actions:
                        if a in ('L','R'):_rotate_local(node,15 if a=='L' else -15,1)
                        else:_move_along(node,-.25,2)
                    assert list(node.translation)==endpoints[h]
                family['initial_states']=starts
                fillers=family['balancing_filler_spans']
                family['balancing_filler_spans']={**fillers,**{h+'_N':v for h,v in fillers.items()}}
                families.append(family);found=True;break
            if rotations_matched>=16:break
        row=dict(family_id=family['family_id'],house=family['house'],partition=family['partition'],
            kinematic_proposal=found,rotation_matches=rotations_matched,attempts=attempts,physical_replay_required=True)
        audit.append(row);print({k:v for k,v in row.items() if k!='attempts'},flush=True)
    c.write(RESEARCH/'NEW_HOUSE_STARTS.json',dict(families=families,audit=audit,
        source_sha256=c.sha(Path(__file__)),physical_admission=False,training_admission=False),True)


if __name__=='__main__':main()
