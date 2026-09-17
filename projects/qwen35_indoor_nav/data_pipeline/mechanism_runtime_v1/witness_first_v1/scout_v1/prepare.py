"""CPU manifest/source lock only, runtime_allowed=false; no GPU imports/operations."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import RUNTIME,ROOT,sha,save

def main():
    source=RUNTIME/'feedback_generation_v1/run_v1/EXECUTION_CONFIG.json'
    original=json.loads(source.read_text())
    assert [c['house_id'] for c in original['candidates']]==['17DRP5sb8fy','1LXtFkjw3qL','1pXnuDYAj8r']
    cfg=dict(node='Q35N_WITNESS_FIRST_CROSS_HOUSE_SCOUT_V1',runtime_allowed=False,training_allowed=False,
        gpu_device=1,gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',
        candidates=original['candidates'],source_configuration_sha256=sha(source),
        budget=dict(total_actions=60000,total_seconds=4200,discovery_actions=20000,discovery_seconds=1200,
                    certification_actions=1,certification_seconds=1),
        supervision_wall_seconds=4500,content_budget_bytes=6*1024**3,total_disk_budget_bytes=8*1024**3,
        ram_budget_bytes=8*1024**3,max_hubs_per_house=2,min_hub_distance_m=1,
        yaw_bin=0,max_groups_per_hub=12,max_targets_per_group=2,max_target_distance_m=8,
        max_outbound_actions=140,max_compact_loop_actions=504,max_programs_per_hub=32,
        physical_family_certification=False,scientific_pass=False,executable=False)
    assert all(c['split']=='FIT' for c in cfg['candidates'])
    save(HERE/'PREPARED_CONFIG.json',cfg)
    paths=list(HERE.glob('*.py'))+[HERE/'README.md',HERE/'PREPARED_CONFIG.json',source]
    paths += [RUNTIME/'core_bridge.py',RUNTIME/'runtime_journal.py',RUNTIME/'guard.py',RUNTIME/'habitat_backend.py',RUNTIME/'SHA256SUMS',
        RUNTIME/'feedback_generation_v1/feedback.py',RUNTIME/'feedback_generation_v1/store.py',RUNTIME/'compact_loop_v2/run.py',
        RUNTIME.parent/'mechanism_factory_v2/compiler.py',RUNTIME.parent/'mechanism_factory_v2/factory.py',
        RUNTIME.parent/'mechanism_factory_v2/planning.py',RUNTIME.parent/'mechanism_factory_v2/SHA256SUMS',
        RUNTIME.parent/'mechanism_scale_v1/planner.py',HERE.parent/'compatibility_cpu/audit.py',
        HERE.parent/'bank_cpu/bank.py',HERE.parent/'bank_cpu/SHA256SUMS',
        HERE.parent/'budget_and_trace/adapters.py',HERE.parent/'budget_and_trace/SHA256SUMS',
        RUNTIME/'feedback_diagnosis_v1/budget_optimization_cpu/budget.py',
        RUNTIME/'feedback_diagnosis_v1/budget_optimization_cpu/SHA256SUMS']
    lock={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for candidate in cfg['candidates']:
        for name,expected in candidate['assets'].items():
            path=Path(name);assert sha(path)==expected,name
            lock[str(path.relative_to(ROOT))]=expected
    save(HERE/'INPUT_LOCK.json',lock)
    print(json.dumps(dict(prepared=True,executable=False,houses=3,locked_inputs=len(lock),gpu_operations=0)))

if __name__=='__main__':main()
