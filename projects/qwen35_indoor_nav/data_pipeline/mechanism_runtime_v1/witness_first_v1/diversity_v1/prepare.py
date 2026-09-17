"""Deterministic, house-balanced selection and fresh-run input closure."""
from collections import Counter, defaultdict
import copy
import json

from common import HERE, LINE, OUT, RUNTIME, WF, load, read, save, sha
from language import aliases, realize


def main():
    snapshot_path = LINE/'reports/opencode_navbench_fusion_20260911/LIVE_READONLY_SNAPSHOT.json'
    snapshot = read(snapshot_path)['special']
    sources = snapshot['export_manifests']
    counts = Counter(x['house'] for x in sources)
    available = defaultdict(list)
    receipts = {}
    for item in snapshot['audit_receipts']:
        assert sha(item['path']) == item['sha256']
        document = read(item['path'])
        for row in document.get('observation', {}).get('attempts', []):
            if row.get('quality_pass') and row.get('source_and_phase_binding_verified'):
                receipts[row['attempt_id']] = item
    for source in sources:
        path = __import__('pathlib').Path(source['path'])
        assert sha(path) == source['sha256']
        manifest = read(path)
        candidate = manifest['candidate']
        size = len(candidate['histories']['H_A'])
        continuation = max(map(len, candidate['continuations'].values()))
        if size <= 440 and continuation <= 112 and source['attempt_id'] in receipts:
            available[source['house']].append((size, continuation, source['attempt_id'], source, manifest))
    selected = [min(available[h], key=lambda x: x[:3]) for h in sorted(available, key=lambda h: (counts[h], h))[:6]]
    assert len(selected) == 6
    OUT.mkdir(exist_ok=False)
    locked = {str(snapshot_path): sha(snapshot_path)}
    candidates = []
    all_aliases = []
    split_sha = None
    for index, (_, _, aid, source, manifest) in enumerate(selected):
        mp = __import__('pathlib').Path(source['path'])
        run = mp.parents[3]
        source_config = read(run/'EXECUTION_CONFIG.json')
        original = next(r for r in source_config['candidates'] if r['candidate_id'] == aid)
        tasks = realize(original['roles'], original['tasks'], index % 5)
        new_id = 'DV1_' + aid.removeprefix('WF_MULTI_')
        candidate = dict(candidate_id=new_id, source_family_id=aid, house_id=original['house_id'],
                         scene_glb=original['scene_glb'], assets=original['assets'], roles=original['roles'],
                         tasks=tasks, expected_eligible=original['expected_eligible'],
                         source_candidate=manifest['candidate'], split='FIT',
                         configuration=copy.deepcopy(original['configuration']),
                         configurations=[dict(u_position=manifest['candidate']['position'],
                             yaw_bin=(manifest['candidate']['yaw_bin']+offset)%24,
                             public_tail='FFFFFFFF') for offset in (0,6,12,18)],
                         source_manifest=str(mp), source_manifest_sha256=source['sha256'],
                         source_quality_receipt=receipts[aid], language_variant=index % 5,
                         training_admission=False)
        candidates.append(candidate)
        all_aliases.append(dict(source_family_id=aid, new_family_id=new_id,
                                variants=aliases(original['roles'], original['tasks'], new_id)))
        for path in [mp, mp.parent.parent/'CERTIFICATE.json', mp.parent.parent/'FROZEN_CANDIDATE.json',
                     run/'EXECUTION_CONFIG.json', run/'INPUT_LOCK.json', __import__('pathlib').Path(receipts[aid]['path'])]:
            locked[str(path)] = sha(path)
        locked.update(original['assets'])
        old_lock = read(run/'INPUT_LOCK.json')
        split_sha = source_config['split_sha256'] if split_sha is None else split_sha
        assert source_config['split_sha256'] == split_sha
        split_paths = [p for p, value in old_lock.items() if value == split_sha]
        assert len(split_paths) == 1 and candidate['house_id'] in read(split_paths[0])['FIT']
        locked[split_paths[0]] = split_sha
    # Seal actual imported dependencies; old whole-batch locks remain immutable
    # provenance, not a claim that unrelated old RGB files are read by this run.
    dependency_dirs = [RUNTIME, RUNTIME.parent/'mechanism_factory_v2', WF/'budget_and_trace',
                       WF/'budget_clock_batching_cpu_v1', WF/'special_scale_transport_v1',
                       WF/'quality_cpu', WF/'quality_cpu/batch_acceptance_v1',
                       WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity',
                       WF/'language_realization_v1']
    paths = [*HERE.glob('*.py'), HERE/'SPEC_ZH.md',
             RUNTIME/'feedback_generation_v1/store.py',
             RUNTIME/'feedback_diagnosis_v1/budget_optimization_cpu/budget.py',
             RUNTIME/'feedback_diagnosis_v1/budget_optimization_cpu/SHA256SUMS',
             RUNTIME/'compact_loop_v2/run.py', RUNTIME.parent/'mechanism_scale_v1/planner.py']
    for folder in dependency_dirs:
        paths.extend(folder.glob('*.py'))
        if (folder/'SHA256SUMS').is_file(): paths.append(folder/'SHA256SUMS')
    transport_common = load('diversity_transport_source_closure', WF/'special_scale_transport_v1/common.py')
    paths.extend(transport_common.SOURCES)
    for path in paths: locked[str(path)] = sha(path)
    cfg = dict(node='Q35N_SPECIAL_DIVERSITY_V1', runtime_allowed=True, executable=True,
               training_allowed=False, gpu_device=1, gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',
               control_type='completed_subgoal_revisit_placement_not_event_free_detour',
               candidates=candidates, split_sha256=split_sha, language_version='q35n.controlled_language.v2',
               budget=dict(total_actions=60000,total_seconds=3600,discovery_actions=15000,
                           discovery_seconds=1000,certification_actions=20000,certification_seconds=1500),
               supervision_wall_seconds=3900, scientific_pass=False, auto_retry=False,
               source_selection='house_count_then_short_history_then_continuation_then_id',
               source_closed_physical_replay_is_not_reused_as_new_certificate=True,
               public_tail_revision='existing_F8_real_motion_branch',
               source_house_counts=dict(counts))
    save(OUT/'EXECUTION_CONFIG.json',cfg)
    save(HERE/'LANGUAGE_ALIASES_PROPOSAL.json',all_aliases)
    locked[str(OUT/'EXECUTION_CONFIG.json')] = sha(OUT/'EXECUTION_CONFIG.json')
    locked[str(HERE/'LANGUAGE_ALIASES_PROPOSAL.json')] = sha(HERE/'LANGUAGE_ALIASES_PROPOSAL.json')
    assert len(locked) <= 1024
    assert sum(__import__('pathlib').Path(p).stat().st_size for p in locked) <= 32*1024**3
    save(OUT/'INPUT_LOCK.json',locked)
    print(json.dumps(dict(candidates=[dict(id=r['candidate_id'],house=r['house_id'],
                     original_history=len(r['source_candidate']['histories']['H_A']),
                     instruction=r['tasks']['task_A']['instruction']) for r in candidates],
                     locked_files=len(locked),approval_created=False),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
