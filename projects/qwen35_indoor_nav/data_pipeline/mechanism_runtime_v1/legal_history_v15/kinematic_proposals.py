"""Initial-position proposals, checked by native SceneNode controls on CPU.

No navigation mesh, sensors or labels are simulated here. The output is only
input to complete physical replay; it cannot grant training admission.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LINE, RESEARCH, c
import numpy as np
import habitat_sim
import magnum as mn
from habitat_sim.agent.controls.default_controls import _rotate_local,_move_along


def main():
    rotation_index=int(sys.argv[1]) if len(sys.argv)>1 else 0
    graph=habitat_sim.SceneGraph();node=graph.get_root_node().create_child()
    def final(actions,position,rotation):
        node.reset_transformation();node.translation=mn.Vector3(position)
        node.rotation=mn.Quaternion(mn.Vector3(rotation[1:]),rotation[0])
        for action in actions:
            if action in ('L','R'):_rotate_local(node,15 if action=='L' else -15,1)
            else:_move_along(node,-.25,2)
        q=node.rotation
        return dict(position=list(node.translation),rotation=[q.scalar,*q.vector])
    rotations={x['family_id']:x for x in c.read(RESEARCH/'ROTATION_PROPOSALS.json')['rows']}
    rows=[]
    for family in c.read(RESEARCH/'RAW_PROTOCOL_R3.json')['families']:
        if rotation_index and family['family_id']!='WF_MULTI_e0239f96cb5b52ffde971cd1':continue
        origin=family['candidate']['position'];histories=family['candidate']['histories']
        old=c.read(HERE/'raw_run_003'/family['family_id']/'RAW_CERTIFICATE.json')
        for name,actions in histories.items():
            measured=final(actions,origin,[1,0,0,0])
            assert all(measured[k]==old['prefixes'][name]['pose'][k] for k in measured)
        rotation=rotations[family['family_id']]['candidates'][rotation_index]['initial_rotation']
        starts={name:dict(position=list(origin),rotation=rotation) for name in histories}
        dimensions=[]
        for axis in (0,2):
            tables={}
            unit=abs(float(np.spacing(np.float32(origin[axis]))))
            for name,actions in histories.items():
                table={}
                for offset in [0]+[s*i for i in range(1,1025) for s in (1,-1)]:
                    position=list(origin);position[axis]=float(np.float32(origin[axis]+offset*unit))
                    output=final(actions,position,rotation)['position'][axis]
                    table.setdefault(output,position[axis])
                tables[name]=table
            shared=set.intersection(*(set(t) for t in tables.values()))
            dimensions.append(dict(axis=axis,origin_ulp=unit,common_outputs=len(shared)))
            if not shared:break
            target=min(shared,key=lambda v:(max(abs(t[v]-origin[axis]) for t in tables.values()),abs(v-origin[axis]),v))
            for name in histories:starts[name]['position'][axis]=tables[name][target]
        endpoints={name:final(histories[name],**starts[name]) for name in histories}
        matched=all(e==endpoints['H_A'] for e in endpoints.values())
        rows.append(dict(family_id=family['family_id'],source_reference_matched=True,
            initial_states=starts,dimensions=dimensions,predicted_endpoints=endpoints,
            kinematic_matched=matched,physical_replay_required=True,training_admission=False))
        print(rows[-1],flush=True)
    filename='KINEMATIC_PROPOSALS.json' if rotation_index==0 else f'KINEMATIC_PROPOSALS_R{rotation_index}.json'
    c.write(RESEARCH/filename,dict(rows=rows,rotation_candidate_index=rotation_index,grid_offsets_each_axis=2049,
        source_sha256=c.sha(Path(__file__)),model_loads=0,gpu_hours=0,training_admission=False),True)


if __name__=='__main__':main()
