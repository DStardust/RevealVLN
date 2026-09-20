"""CPU proposals using the installed Habitat SceneNode turn implementation.

These are rotation calculations, never certified trajectories or observations.
Real full-history replay remains necessary for every proposed initial state.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LINE, RESEARCH, c
import habitat_sim
import magnum as mn
from habitat_sim.agent.controls.default_controls import _rotate_local


def main():
    config=c.read(RESEARCH/'RAW_PROTOCOL_R3.json')
    graph=habitat_sim.SceneGraph();node=graph.get_root_node().create_child()
    def final(history,angle):
        node.reset_transformation()
        node.rotation=mn.Quaternion(mn.Vector3(0,math.sin(angle/2),0),math.cos(angle/2))
        for action in history:
            if action in ('L','R'):_rotate_local(node,15 if action=='L' else -15,1)
        q=node.rotation
        return (q.scalar,*q.vector)
    rows=[]
    for family in config['families']:
        reference=c.read(HERE/'raw_run_003'/family['family_id']/'RAW_CERTIFICATE.json')
        history=family['candidate']['histories']
        original_angle=family['candidate']['yaw_bin']*math.pi/12
        for name,actions in history.items():
            assert list(final(actions,original_angle))==reference['prefixes'][name]['pose']['rotation'],name
        # Fixed bounded grid independent of labels/model scores. Try smaller
        # initial perturbations first, and require one shared angle for all rows.
        found=[]
        attempts=0
        for delta in [0]+[s*i*1e-5 for i in range(1,501) for s in (1,-1)]:
            angle=original_angle+delta
            rotations={name:final(actions,angle) for name,actions in history.items()}
            attempts+=1
            if len(set(rotations.values()))==1:
                found.append(dict(initial_rotation=[math.cos(angle/2),0,math.sin(angle/2),0],
                    angle_delta_rad=delta,final_rotation=list(next(iter(rotations.values())))))
                if len(found)==4:break
        rows.append(dict(family_id=family['family_id'],actual_raw_reference_matched=True,
            candidates=found,angles_tested=attempts))
    c.write(RESEARCH/'ROTATION_PROPOSALS.json',dict(rows=rows,scope='CPU native rotation proposals only; no physical or semantic admission',
        source_sha256=c.sha(Path(__file__)),model_loads=0,gpu_hours=0),True)
    print(rows)


if __name__=='__main__':main()
