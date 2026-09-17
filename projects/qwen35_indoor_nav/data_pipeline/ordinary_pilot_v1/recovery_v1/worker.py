import importlib.util
import json
from pathlib import Path
import time

OUT=Path(__file__).resolve().parent
ORIGINAL=OUT.parent
spec=importlib.util.spec_from_file_location('original_compiler',ORIGINAL/'worker.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
core.OUT=OUT


def main():
    jobs=json.loads((ORIGINAL/'JOBS.json').read_text())
    previous=[json.loads(s) for s in (ORIGINAL/'LEDGER.jsonl').read_text().splitlines()]
    assert len(previous)==16 and [r['job_id'] for r in previous]==[j['job_id'] for j in jobs[:16]]
    assert not (OUT/'LEDGER.jsonl').exists()
    counts={'primitive_actions':0,'observations':0,'resets':0,'logical_stops':0,'collisions':0,'simulator_constructions':0}
    results=[];sim=None;scene=None;started=time.time()
    try:
        for j in jobs[16:]:
            if scene!=j['scene_id']:
                if sim:sim.close();sim=None
                sim=core.simulator(j['scene_id']);scene=j['scene_id'];counts['simulator_constructions']+=1
            results.append(core.run_job(sim,j,counts))
            (OUT/'COUNTS_LIVE.json').write_text(json.dumps(counts,indent=2))
    finally:
        if sim:sim.close()
        core.save(OUT/'ACTUAL_COUNTS.json',dict(counts,wall_seconds=time.time()-started,completed_candidates=len(results)))
    combined=previous+results
    passed=[r for r in combined if r['status']=='CERTIFIED']
    per_scene={s:sum(r['status']=='CERTIFIED' for r in combined if r['scene_id']==s) for s in sorted({j['scene_id'] for j in jobs})}
    core.save(ORIGINAL/'GENERATION_RESULT.json',{'attempted_unique_candidates':len(combined),'completed':len(combined),
        'interrupted_candidate_retried':1,'certified_routes':len(passed),'certified_per_scene':per_scene,
        'instruction_records':sum(r['instruction_records'] for r in passed),
        'unique_route_decisions':sum(r['decisions'] for r in passed),
        'generation_yield_pass':len(passed)>=80 and min(per_scene.values())>=10,
        'integrity_acceptance':'PENDING_AUDIT','scientific_pass':False,'navigation_gain':None})


if __name__=='__main__':main()
