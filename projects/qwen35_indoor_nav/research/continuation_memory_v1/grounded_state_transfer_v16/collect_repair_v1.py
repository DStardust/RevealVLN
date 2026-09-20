"""Versioned physical repair collector for the rqfALeAoiTq TEST shortfall.

This module deliberately imports the frozen V16 collector for all simulator,
SEE2, collision, replay, and label semantics.  The only changed discovery
path is that every pre-declared neutral yaw is tried for a candidate before
the candidate is rejected.  If the old candidate list remains insufficient,
the same deterministic pathfinder stream supplies additional hubs.  No model
outputs or scores are read.
"""
import itertools
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v16_common import *
from collect import ContentStore, Rejected, Runner, instructions, terminal_instruction
from evaluator_v16 import legacy, evaluate


REPAIR_VERSION = 'v16_rqf_full_neutral_yaw_search_v1'
REPAIR_HOUSE = 'rqfALeAoiTq'
REPAIR_EXTRA_HUBS = 24
REPAIR_MAX_PROPOSALS = 256


def _save_trace(path, runner):
    if not path.exists():
        runner.save(path)


def _attempt_yaw(backend, proposal, house, cfg, compiler, targets, folder, yaw):
    """Try one frozen yaw and return a certified family or a rejection."""
    yaw_folder = folder / ('yaw_%02d' % yaw)
    yaw_folder.mkdir(parents=True, exist_ok=True)
    runner = Runner(backend, compiler, folder, cfg['history_cap'])
    current_path = yaw_folder / 'INITIAL_VIEW_PROBE.json'
    try:
        runner.start(proposal['start'], yaw)
        for action in ('L', 'R'):
            runner.move(action)
        runner.trace['complete'] = True
        runner.save(current_path)
        events = compiler.atoms(runner.trace['observations'])
        if any(e['anchor_A'] or e['anchor_B'] for e in events):
            raise Rejected('NO_ANCHOR_NEUTRAL_START_VIEW')

        histories = {}
        recovery_records = {}
        for history_id, role, recovery in (
            ('H_A', 'anchor_A', False), ('H_B', 'anchor_B', False),
            ('H_A_R', 'anchor_A', True), ('H_B_R', 'anchor_B', True),
        ):
            current_path = yaw_folder / (history_id + '_DISCOVERY.json')
            runner.cap = cfg['history_cap']
            runner.start(proposal['start'], yaw)
            runner.witness(role, targets[role])
            if recovery:
                perturbation_start = len(runner.trace['actions'])
                before = runner.trace['observations'][-1]['pose']['position']
                for action in cfg['perturbation']:
                    runner.move(action)
                after = runner.trace['observations'][-1]['pose']['position']
                if math.dist(before, after) < .2:
                    raise Rejected('PERTURBATION_NO_DISPLACEMENT')
                recovery_records[history_id] = dict(
                    perturbation_start=perturbation_start,
                    actions=cfg['perturbation'], before=before, after=after,
                    meaning='wrong heading plus actual displaced step before resuming registered return target',
                    teacher_recovery_end=None,
                )
            angle = yaw * math.pi / 12
            runner.navigate(
                proposal['start'],
                [proposal['start'][0] - math.sin(angle), proposal['start'][1],
                 proposal['start'][2] - math.cos(angle)],
            )
            # Keep the original V16 public tail and its fixed length.
            for action in ('L', 'R') * 5:
                runner.move(action)
            events = compiler.atoms(runner.trace['observations'])
            other = 'anchor_B' if role == 'anchor_A' else 'anchor_A'
            if not any(e[role] for e in events) or any(e[other] for e in events):
                raise Rejected('HISTORY_EVENT_STATE_NOT_ISOLATED')
            runner.trace['complete'] = True
            runner.save(current_path)
            histories[history_id] = list(runner.trace['actions'])
            if recovery:
                recovery_records[history_id]['teacher_recovery_end'] = len(histories[history_id])

        runner.cap = cfg['full_budget']
        traces = {}
        for history_id, history in histories.items():
            reference = read(yaw_folder / (history_id + '_DISCOVERY.json'))
            for continuation, chain in (
                ('C0', ['terminal']), ('C_A', ['anchor_A', 'terminal']),
                ('C_B', ['anchor_B', 'terminal']),
            ):
                current_path = yaw_folder / (history_id + '__' + continuation + '.json')
                runner.start(proposal['start'], yaw)
                for action in history:
                    runner.move(action)
                for actual, expected in zip(runner.trace['observations'], reference['observations']):
                    if actual['rgb_hash'] != expected['rgb_hash'] or actual['semantic_hash'] != expected['semantic_hash']:
                        raise ValueError('HISTORY_RAW_REPLAY_MISMATCH')
                cutoff = len(history)
                for role in chain:
                    runner.witness(role, targets[role])
                runner.move('S')
                runner.trace['complete'] = True
                runner.trace['cutoff'] = cutoff
                runner.save(current_path)
                labels = {task: evaluate(compiler, runner.trace, task, cutoff)
                          for task in ('task_A', 'task_B', 'task_T')}
                if any(result['safe_v16_label'] == 'UNKNOWN' for result in labels.values()):
                    raise Rejected('UNKNOWN_CROSS_LABEL')
                traces[history_id + '__' + continuation] = dict(
                    path=str(current_path.relative_to(LINE)), sha256=sha(current_path), labels=labels,
                )

        for history_id in histories:
            own_task = 'task_A' if history_id.startswith('H_A') else 'task_B'
            other_task = 'task_B' if own_task == 'task_A' else 'task_A'
            if traces[history_id + '__C0']['labels'][own_task]['safe_v16_label'] != 'PASS':
                raise Rejected('MISSING_REQUIRED_HISTORY_SENSITIVE_LABELS')
            if traces[history_id + '__C0']['labels'][other_task]['safe_v16_label'] != 'FAIL':
                raise Rejected('MISSING_REQUIRED_HISTORY_SENSITIVE_LABELS')
            for task in ('task_A', 'task_B', 'task_T'):
                if not any(traces[history_id + '__' + continuation]['labels'][task]['safe_v16_label'] == 'PASS'
                           for continuation in ('C0', 'C_A', 'C_B')):
                    raise Rejected('NO_REAL_PASS_TEACHER')

        family = dict(
            family_id=proposal['id'], house=house['house'], split=house['split'], scene=house['scene'],
            roles={name: house['roles'][idx]['spec'] for name, idx in
                   zip(('anchor_A', 'anchor_B', 'terminal'), proposal['role_indices'])},
            compiler=dict(roles={k: [v['mpcat40'], v['room']] for k, v in
                                 {name: house['roles'][idx]['spec'] for name, idx in
                                  zip(('anchor_A', 'anchor_B', 'terminal'), proposal['role_indices'])}.items()},
                          tasks=instructions({name: house['roles'][idx]['spec'] for name, idx in
                                              zip(('anchor_A', 'anchor_B', 'terminal'), proposal['role_indices'])}, house['split']),
                          eligible={name: house['roles'][idx]['eligible'] for name, idx in
                                    zip(('anchor_A', 'anchor_B', 'terminal'), proposal['role_indices'])}),
            terminal_instruction=terminal_instruction(
                {name: house['roles'][idx]['spec'] for name, idx in
                 zip(('anchor_A', 'anchor_B', 'terminal'), proposal['role_indices'])}, house['split']),
            histories=histories, initial_position=proposal['start'], initial_yaw=yaw,
            traces=traces, recovery=recovery_records, proposal=proposal,
            shared_cross_history_window_claim=False,
            training_admission='V16_PHYSICAL_AND_LABEL_CERTIFIED_PENDING_SPLIT_AUDIT',
            repair=dict(version=REPAIR_VERSION, tested_yaws=list(cfg['neutral_start_yaw_candidates']),
                        selected_yaw=yaw, no_model_scores_read=True),
        )
        write(folder / 'FAMILY.json', family, True)
        return family, None
    except BaseException as exc:
        _save_trace(current_path, runner)
        return None, exc


def collect_family_repair(backend, proposal, house, run, cfg, store):
    folder = run / 'proposals' / proposal['id']
    folder.mkdir(parents=True, exist_ok=False)
    role_names = ('anchor_A', 'anchor_B', 'terminal')
    role_indices = proposal['role_indices']
    roles = {name: house['roles'][idx]['spec'] for name, idx in zip(role_names, role_indices)}
    eligible = {name: backend.eligible['role_' + str(idx)] for name, idx in zip(role_names, role_indices)}
    compiler = legacy.Compiler(
        roles={key: [value['mpcat40'], value['room']] for key, value in roles.items()},
        tasks=instructions(roles, house['split']), eligible=eligible,
    )
    targets = dict(zip(role_names, proposal['targets']))
    yaw_results = []
    for yaw in cfg['neutral_start_yaw_candidates']:
        family, error = _attempt_yaw(backend, proposal, house, cfg, compiler, targets, folder, yaw)
        row = dict(yaw=yaw, status='CERTIFIED' if family else 'REJECTED',
                   error=None if family is not None else repr(error), unix=time.time())
        append(folder / 'REPAIR_YAW_ATTEMPTS.jsonl', row)
        yaw_results.append(row)
        if family is not None:
            family['content_root'] = str(store.root.relative_to(LINE))
            write(folder / 'FAMILY.json', family, True)
            write(folder / 'REPAIR_YAW_SEARCH.json', dict(version=REPAIR_VERSION, attempts=yaw_results,
                                                            selected_yaw=yaw), True)
            return family
    write(folder / 'REPAIR_YAW_SEARCH.json', dict(version=REPAIR_VERSION, attempts=yaw_results,
                                                   selected_yaw=None), True)
    write(folder / 'FAILURE.json', dict(error="Rejected('REPAIR_NEUTRAL_YAW_EXHAUSTED')",
                                         physical_attempt=True, repair_version=REPAIR_VERSION,
                                         yaw_attempts=yaw_results), True)
    return None


def _extra_proposals(backend, house, cfg, original_count):
    """Generate a deterministic continuation of the frozen hub stream."""
    roles = {name: house['roles'][int(name[5:])]['spec'] for name in backend.eligible}
    feedback = load('v16_repair_feedback', LINE / 'data_pipeline/mechanism_runtime_v1/feedback_generation_v1/feedback.py')
    backend.sim.pathfinder.seed(1209)
    starts = [backend.sim.pathfinder.get_random_navigable_point().tolist()
              for _ in range(cfg['hubs_per_house'] + REPAIR_EXTRA_HUBS)]
    proposals = []
    for hub, start in enumerate(starts[cfg['hubs_per_house']:], start=cfg['hubs_per_house']):
        if not all(math.isfinite(x) for x in start):
            continue
        targets = []
        for name in roles:
            idx = int(name[5:])
            target = feedback.candidate_targets(backend, start, name, limit=cfg['targets_per_role'])
            if target:
                targets.append((target[0]['distance'], idx, target))
        targets.sort(key=lambda row: (row[0], row[1]))
        targets = targets[:cfg['roles_per_hub']]
        for triple in itertools.combinations(targets, 3):
            indices = [row[1] for row in triple]
            if len({house['roles'][idx]['signature'][1] for idx in indices}) < 3:
                continue
            for terminal in range(3):
                order = [idx for idx in range(3) if idx != terminal] + [terminal]
                proposal = dict(
                    start=start, hub=hub,
                    role_indices=[indices[idx] for idx in order],
                    targets=[triple[idx][2] for idx in order],
                )
                proposal['id'] = 'V16R1_' + digest(dict(version=REPAIR_VERSION, proposal=proposal))[:20]
                proposal['repair_variant'] = REPAIR_VERSION
                proposal['base_proposal_id'] = None
                proposals.append(proposal)
    ordered = []
    for hub in range(cfg['hubs_per_house'], cfg['hubs_per_house'] + REPAIR_EXTRA_HUBS):
        group = sorted((p for p in proposals if p['hub'] == hub),
                       key=lambda p: (p['targets'][2][0]['distance'],
                                      sum(t[0]['distance'] for t in p['targets']), p['id']))
        ordered.extend(group)
    return ordered


def _repair_proposals(backend, house, cfg, source):
    prior_path = source / ('PROPOSALS_' + house['house'] + '.json')
    prior = read(prior_path)
    proposals = []
    for original in prior:
        proposal = dict(original)
        proposal['id'] = 'V16R1_' + digest(dict(version=REPAIR_VERSION, base=original))[:20]
        proposal['repair_variant'] = REPAIR_VERSION
        proposal['base_proposal_id'] = original['id']
        proposal['source_proposal_path'] = str((prior_path).relative_to(LINE))
        proposals.append(proposal)
    proposals.extend(_extra_proposals(backend, house, cfg, len(proposals)))
    return proposals[:REPAIR_MAX_PROPOSALS]


def main(run, splits=('TEST',)):
    cfg = read(run / 'PROTOCOL.json')
    manifest = read(HERE / 'DATA_MANIFEST.json')
    source = Path(read(run / 'REUSED_PHYSICAL_ASSETS.json')['source'])
    runtime = LINE / 'data_pipeline/mechanism_runtime_v1'
    sys.path.insert(0, str(runtime))
    backend_module = load('v16_repair_backend', runtime / 'habitat_backend.py')
    store = ContentStore(run / 'content', cfg['artifact_gib'] * 2 ** 30)
    began = time.monotonic(); families = []; shortfalls = []
    for house in manifest['houses']:
        if house['split'] not in splits:
            continue
        if house['house'] != REPAIR_HOUSE:
            record_path = run / ('HOUSE_' + house['house'] + '.json')
            if not record_path.exists():
                raise ValueError('REPAIR_REUSE_RECORD_MISSING:' + house['house'])
            record = read(record_path); families.extend(record['families'])
            if not record.get('complete', False):
                shortfalls.append(dict(house=house['house'], actual=len(record['families']), target=house['target_families']))
            continue
        housefile = run / ('HOUSE_' + house['house'] + '.json')
        if housefile.exists() and read(housefile).get('complete', False):
            families.extend(read(housefile)['families']); continue
        roles = {'role_' + str(i): row['spec'] for i, row in enumerate(house['roles'])}
        planner = load('v16_repair_planner', LINE / 'data_pipeline/mechanism_scale_v1/planner.py')
        objects = planner.parse_house(Path(house['scene']).with_suffix('.house').read_text())
        reserved = [obj for idx, obj in objects.items() if idx in backend_module.RESERVED_MASKS]
        excluded = {key: spec for key, spec in roles.items()
                    if any(backend_module.role_matches(obj, spec) for obj in reserved)}
        immutable(run / ('RESERVED_ROLE_EXCLUSIONS_' + house['house'] + '.json'),
                  dict(excluded=excluded, rule='exclude whole role signature when a reserved mask would satisfy it; never reinterpret reserved semantic pixels'))
        roles = {key: value for key, value in roles.items() if key not in excluded}
        backend = backend_module.HabitatBackend(
            house['scene'], cfg['gpu'], roles, store,
            dict(runtime_allowed=True, scene_glb=house['scene'], gpu_device=cfg['gpu']),
        )
        prior = run / ('PRIOR_HOUSE_' + house['house'] + '.json')
        accepted = list(read(prior)['families']) if prior.exists() else []
        attempts = sum(1 for row in c.records(run / 'COLLECTION_ATTEMPTS.jsonl')
                       if row.get('house') == house['house'] and row.get('status') == 'START') \
            if (run / 'COLLECTION_ATTEMPTS.jsonl').exists() else 0
        try:
            expected = {key: house['roles'][int(key[5:])]['eligible'] for key in roles}
            if backend.eligible != expected:
                raise ValueError('SEMANTIC_ASSET_IDENTITY')
            proposals_path = run / ('PROPOSALS_' + house['house'] + '.json')
            if proposals_path.exists():
                proposals = read(proposals_path)
            else:
                proposals = _repair_proposals(backend, house, cfg['collection'], source)
                immutable(proposals_path, proposals)
            for proposal in proposals:
                if len(accepted) >= house['target_families']:
                    break
                path = run / 'proposals' / proposal['id']
                if (path / 'FAMILY.json').exists():
                    family = read(path / 'FAMILY.json')
                elif (path / 'FAILURE.json').exists():
                    continue
                else:
                    attempts += 1
                    append(run / 'COLLECTION_ATTEMPTS.jsonl', dict(
                        status='START', house=house['house'], split=house['split'],
                        proposal=proposal['id'], base_proposal_id=proposal.get('base_proposal_id'),
                        repair_version=REPAIR_VERSION, unix=time.time(),
                    ))
                    family = collect_family_repair(backend, proposal, house, run, cfg['collection'], store)
                    append(run / 'COLLECTION_ATTEMPTS.jsonl', dict(
                        status='CERTIFIED' if family else 'REJECTED', proposal=proposal['id'],
                        repair_version=REPAIR_VERSION, unix=time.time(),
                    ))
                if family:
                    accepted.append(family)
                write(run / 'COLLECTION_PROGRESS.json', dict(
                    house=house['house'], split=house['split'], attempts=attempts,
                    accepted=len(accepted), target=house['target_families'],
                    seconds=time.monotonic() - began, repair_version=REPAIR_VERSION,
                ))
                print(house['house'], attempts, len(accepted), flush=True)
            write(housefile, dict(house=house['house'], families=accepted, attempts=attempts,
                                  target=house['target_families'], complete=len(accepted) == house['target_families'],
                                  repair_version=REPAIR_VERSION, source_run=str(source)), True)
            families.extend(accepted)
            if len(accepted) < house['target_families']:
                shortfalls.append(dict(house=house['house'], actual=len(accepted), target=house['target_families']))
        finally:
            backend.close()
    if shortfalls:
        write(run / 'DATA_SHORTFALL.json', dict(houses=shortfalls, accepted_families=len(families),
                                                repair_version=REPAIR_VERSION), True)
        raise RuntimeError('REGISTERED_HOUSE_FAMILY_SHORTFALL: ' + str(shortfalls))
    write(run / 'REPAIR_COLLECTION_COMPLETE.json', dict(version=REPAIR_VERSION, house=REPAIR_HOUSE,
                                                        accepted_families=len(families), completed_unix=time.time()), True)
    return families


if __name__ == '__main__':
    main(Path(sys.argv[1]), tuple(sys.argv[2:]) or ('TEST',))
