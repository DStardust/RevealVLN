"""Bounded native-kinematic proposals for new, symmetric neutral controls."""
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
    house=sys.argv[1] if len(sys.argv)>1 else '2n8kARJN3HM'
    sync_forward=int(sys.argv[2]) if len(sys.argv)>2 else 0
    assert sync_forward in (0,2)
    graph=habitat_sim.SceneGraph();node=graph.get_root_node().create_child()
    def components(actions,rotation):
        node.reset_transformation();node.rotation=mn.Quaternion(mn.Vector3(rotation[1:]),rotation[0])
        shifts=[];rotations=[]
        for action in actions:
            node.translation=mn.Vector3(0,0,0)
            if action in ('L','R'):_rotate_local(node,15 if action=='L' else -15,1)
            else:_move_along(node,-.25,2)
            shifts.append(list(node.translation))
            q=node.rotation;rotations.append((q.scalar,*q.vector))
        return np.array(shifts,dtype=np.float32),rotations
    def translate(position,shifts):
        result=np.array(position,dtype=np.float32,copy=True)
        for delta in shifts:result+=delta
        return result
    config=c.read(RESEARCH/'RAW_PROTOCOL_R3.json')
    family=copy.deepcopy(next(f for f in config['families'] if f['house']==house))
    origin=family['candidate']['position'];old=family['candidate']['histories']
    reference=c.read(HERE/'raw_run_003'/family['family_id']/'RAW_CERTIFICATE.json')
    for name,actions in old.items():
        shifts,rot=components(actions,[1,0,0,0])
        assert translate(origin,shifts).tolist()==reference['prefixes'][name]['pose']['position']
        assert list(rot[-1])==reference['prefixes'][name]['pose']['rotation']
    rows=[];attempts=[]
    # One fixed set of eight moving shuttles; no scores or model in discovery.
    for distance in (1,2,3,4):
        for turn,back in [('L','R'),('R','L')]:
            loop=['F']*distance+[turn]*12+['F']*distance+[back]*12
            histories={}
            for name in ('H_A','H_B'):
                histories[name]=old[name]+['F']*sync_forward+['L','R']*4
                histories[name+'_N']=old[name]+loop+['F']*sync_forward+['L','R']*4
            proposal=None;tested=0;rotation_matches=0
            for delta in [0]+[s*i*1e-5 for i in range(1,501) for s in (1,-1)]:
                rotation=[math.cos(delta/2),0,math.sin(delta/2),0]
                vectors={};tails={}
                for name,actions in histories.items():
                    vectors[name],rot=components(actions,rotation);tails[name]=tuple(rot[-2:])
                tested+=1
                if len(set(tails.values()))!=1:continue
                rotation_matches+=1
                starts={name:dict(position=list(origin),rotation=rotation) for name in histories}
                valid=True
                for axis in (0,2):
                    unit=abs(float(np.spacing(np.float32(origin[axis]))))
                    offsets=np.array([0]+[s*i for i in range(1,1025) for s in (1,-1)],dtype=np.float32)
                    positions=np.repeat(np.array(origin,dtype=np.float32)[None],len(offsets),axis=0)
                    positions[:,axis]+=offsets*unit
                    tables={}
                    for name in histories:
                        out=translate(positions,vectors[name])[:,axis];table={}
                        for value,start in zip(out.tolist(),positions[:,axis].tolist()):table.setdefault(value,start)
                        tables[name]=table
                    shared=set.intersection(*(set(t) for t in tables.values()))
                    if not shared:valid=False;break
                    target=min(shared,key=lambda v:(max(abs(t[v]-origin[axis]) for t in tables.values()),abs(v-origin[axis]),v))
                    for name in histories:starts[name]['position'][axis]=tables[name][target]
                if valid:
                    # Verify composed initial positions with the actual native
                    # SceneNode operators, independently of vectorized addition.
                    endings={}
                    for name,actions in histories.items():
                        start=starts[name];node.reset_transformation()
                        node.translation=mn.Vector3(start['position']);node.rotation=mn.Quaternion(mn.Vector3(rotation[1:]),rotation[0])
                        last=[]
                        for action in actions:
                            if action in ('L','R'):_rotate_local(node,15 if action=='L' else -15,1)
                            else:_move_along(node,-.25,2)
                            q=node.rotation;last.append((tuple(node.translation),(q.scalar,*q.vector)))
                        assert list(node.translation)==translate(start['position'],vectors[name]).tolist()
                        endings[name]=tuple(last[-2:])
                    assert len(set(endings.values()))==1
                    proposal=copy.deepcopy(family)
                    proposal['family_id']=family['family_id']+f'_NEUTRAL_{distance}{turn}'
                    if sync_forward:proposal['family_id']+=f'_SYNC{sync_forward}'
                    proposal['candidate']['histories']=histories
                    if sync_forward:
                        inverse=['L']*12+['F']*sync_forward+['R']*12
                        proposal['candidate']['continuations']={k:inverse+v for k,v in family['candidate']['continuations'].items()}
                    assert max(map(len,histories.values()))+max(map(len,proposal['candidate']['continuations'].values()))<=500
                    proposal['initial_states']=starts
                    proposal['neutral_span']={name+'_N':[len(old[name]),len(old[name])+len(loop)] for name in ('H_A','H_B')}
                    proposal['source_family_id']=family['family_id'];proposal['moving_neutral_proposal']=loop
                    category,room=family['compiler']['roles']['terminal']
                    proposal['task_terminal_only']=dict(terminal='terminal',instruction=f'Get two consecutive clear views of the {category} in the {room}, then stop. No earlier object sighting is required.')
                    break
                if rotation_matches>=16:break
            attempts.append(dict(distance=distance,turn=turn,angles_tested=tested,rotation_matches=rotation_matches,kinematic_proposal=proposal is not None))
            if proposal:rows.append(proposal)
            print(attempts[-1],flush=True)
    filename='NEUTRAL_PROPOSALS.json' if house=='2n8kARJN3HM' else f'NEUTRAL_PROPOSALS_{house}.json'
    if sync_forward:filename=filename.replace('.json',f'_SYNC{sync_forward}.json')
    c.write(RESEARCH/filename,dict(families=rows,attempts=attempts,
        semantic_or_physical_admission=False,source_sha256=c.sha(Path(__file__)),
        native_vectorized_translation_verified=True,max_rotations_matching_per_shuttle=16,
        max_angles_each_shuttle=1001,common_actual_forward_steps=sync_forward,policy_or_model_scores_used=False),True)


if __name__=='__main__':main()
