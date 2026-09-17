"""One read-only CPU audit of existing assets/logs; no model or simulator imports."""
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
BASE = '9d22783e646ef86232ba3a3a59d7bb9ce8a8c0a7'
V3 = LINE / 'closed_loop_bench/ordinary_cycle_pair_gpu1_v3'
V4 = LINE / 'closed_loop_bench/ordinary_cycle_pair_recovery_v4'
RUNTIME = LINE / 'data_pipeline/mechanism_runtime_v1'
BUNDLE = RUNTIME / 'witness_first_v1/short_revisit_v3/run_v1/bundles/WF_SHORT_REVISIT_V3_004'
READS = {}


def blob(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('READ_OUTSIDE_PROJECT')
    data = path.read_bytes()
    READS[str(path.relative_to(ROOT))] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return data


def read(path):
    return json.loads(blob(path))


def lines(path):
    return [json.loads(x) for x in blob(path).decode().splitlines() if x]


def write(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load(name, path):
    blob(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budget_audit():
    rows = []
    for name in ('ordinary_cycle_recovery_v1', 'ordinary_cycle_pair_v2', 'ordinary_cycle_pair_gpu1_v3'):
        folder = V3.parent / name / 'run_001'
        launch, progress = read(folder / 'LAUNCH_RESULT.json'), read(folder / 'PROGRESS.json')
        batches = lines(folder / 'INFERENCE_BATCHES.jsonl')
        infer_actions = sum(r['size'] for r in batches)
        infer_seconds = sum(r['seconds'] for r in batches)
        rows.append(dict(case=name, source_status=launch['status'],
            actions=progress['total_actions'], completed=progress['completed'],
            recorded_launcher_seconds=launch['wall_seconds'],
            recorded_evaluation_seconds=progress['wall_seconds'],
            evaluation_seconds_per_action=progress['wall_seconds']/progress['total_actions'],
            preprocessing_and_inference_seconds_per_action=infer_seconds/infer_actions,
            unaccounted_startup_cleanup_and_snapshot_lag_seconds=launch['wall_seconds']-progress['wall_seconds']))
    paired = rows[1:]
    low, high = min(r['evaluation_seconds_per_action'] for r in paired), max(r['evaluation_seconds_per_action'] for r in paired)
    overhead = max(r['unaccounted_startup_cleanup_and_snapshot_lag_seconds'] for r in paired)
    scenarios = []
    for name, decisions in [('two_historical_native_action_counts_for_sensitivity_only', 2*19122),
                            ('historical_native_plus_B_at_500_each', 19122+50000),
                            ('full_decision_cap', 100000)]:
        scenarios.append(dict(name=name, assumed_decisions=decisions,
            seconds_interval=[overhead+low*decisions, overhead+high*decisions]))
    out = dict(method='Extrapolation from existing censored logs only, not a new timing experiment',
        records=rows, scenarios=scenarios, budget_seconds=4100,
        timing_verdict='AT_RISK_NOT_DEMONSTRATED_FEASIBLE',
        guaranteed_finish=False, obvious_impossibility_proven=False,
        new_tensor_and_full_parameter_fingerprint_overhead_seconds=None,
        caution='Episode/house/instruction lengths, scene loads, interrupted sampling and new read-only instrumentation are not controlled by this extrapolation; historical A action count is not the new A result',
        additional_navigation_starts=0)
    write(V4/'codex_return_20260917/BUDGET_FEASIBILITY.json', out)
    return out


def inventory():
    root = RUNTIME / 'witness_first_v1'
    rows = []
    skip = {'content','frames','traces','prefixes','full_policy','cache','__pycache__','deps','.git'}
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('.') and not (Path(parent)/d).is_symlink()]
        if 'MANIFEST.json' not in files:
            continue
        p = Path(parent)/'MANIFEST.json'
        m = read(p)
        if m.get('schema_version') != 'q35n.family_export.v4':
            continue
        rows.append(dict(path=str(p.relative_to(LINE)), family_id=m['family_id'],
            house_id=m['group']['house_id'], split=m['split'],
            canonical_traces=m['canonical_traces'], cells=m['cells'],
            training_admission=m['training_admission'],
            control_type=m.get('provenance',{}).get('control_type'),
            candidate_sha256=hashlib.sha256(json.dumps(m['candidate'],sort_keys=True).encode()).hexdigest()))
    return dict(scope='Metadata inventory of witness_first_v1 family exports, not a certification census',
        exports=len(rows), declared_houses=sorted({r['house_id'] for r in rows}),
        unique_candidate_hashes=len({r['candidate_sha256'] for r in rows}),
        summed_cells_not_independent_trajectories=sum(r['cells'] for r in rows),
        declared_training_admissions=sum(r['training_admission'] for r in rows),
        independent_physical_family_count=None, rows=rows)


def asset_audit():
    module = load('return_compiler', LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    loader_module = load('return_loader', RUNTIME/'loader.py')
    export = BUNDLE/'export_v4'
    manifest = read(export/'MANIFEST.json')
    compiler = module.Compiler(**manifest['compiler_config'])
    loader = loader_module.FamilyLoader(export, compiler)
    recomputed = loader.validate_supervision_contract()
    # Record files read inside the existing loader, without claiming a new replay.
    for line in blob(export/'SHA256SUMS').decode().splitlines():
        _, relative = line.split('  ',1)
        blob(export/relative)
    content = read(export/'CONTENT_INDEX.json')
    for item in content.values():
        data = blob(LINE/item['line_relative_path'])
        if hashlib.sha256(data).hexdigest() != item['file_sha256']:
            raise ValueError('CONTENT_HASH_MISMATCH')
        loader_module.npy_pixels(data,item['raw_pixel_sha256'],item['kind'])
    certificate = read(BUNDLE/'CERTIFICATE.json')
    result = read(BUNDLE/'result.json')
    run_result = read(BUNDLE.parents[1]/'result.json')
    matrix, history_events, normalization = [], {}, []
    short_keys = set()
    for cell in loader.cells:
        trace = read(export/cell['trace_path'])
        cutoff = cell['prefix_cutoff']
        matrix.append({k:cell[k] for k in ('history_id','task_id','continuation_id','y')})
        if cell['continuation_id']=='C0':
            task = compiler.tasks[cell['task_id']]
            events = compiler.atoms(trace['observations'][:cutoff+1])
            matches = [t for t,event in enumerate(events) if event[task['anchor']]]
            history_events[cell['history_id']+'/'+cell['task_id']] = dict(
                last_anchor_step=matches[-1] if matches else None,
                steps_since_last_anchor=cutoff-matches[-1] if matches else None,
                full_prefix_observations=cutoff+1,
                evidence_rgb_sha256=[trace['observations'][t]['rgb_hash'] for t in (matches[-1]-1,matches[-1])] if matches else [],
                state_at_cutoff=compiler.m2(trace,cell['task_id'])[cutoff])
        if cell['task_id']==next(iter(compiler.tasks)):
            short_keys.add(json.dumps([[o['rgb_hash'] for o in trace['observations'][cutoff-1:cutoff+1]],trace['actions'][cutoff-8:cutoff]]))
            normalization += [dict(history=cell['history_id'],continuation=cell['continuation_id'],
                 step=e['step'], raw_rgb_sha=e['raw_record']['rgb_hash'],
                 canonical_rgb_sha=e['canonical_record']['rgb_hash'],
                 corrections=e['corrections'], protocol=e['protocol']) for e in trace.get('normalization_events',[])]
    first_id = next(iter(loader.prefix_index))
    prefix = loader.prefix_records(first_id)
    payload = loader.policy_payload(prefix[-1])
    out = dict(status='DATA_ASSET_GAP', candidate_family=manifest['family_id'],
        export_path=str(export.relative_to(LINE)), manifest_sha256=hashlib.sha256(blob(export/'MANIFEST.json')).hexdigest(),
        measured_scope='CPU existing export/label/state/content validation, no new physical replay',
        source_recorded_physical_certified=result['physical_certified'],
        source_training_admission=manifest['training_admission'],
        task_revision=manifest['compiler_config']['task_revision'],
        task_programs=manifest['compiler_config']['tasks'],
        control_semantics=manifest['provenance']['control_type'],
        physical_house=manifest['group']['house_id'],
        histories=3,continuations=3,tasks=2,canonical_traces=9,
        recorded_certificate_replays=certificate['replays'],recorded_certificate_evaluations=certificate['evaluations'],
        recorded_entire_run_counts=run_result['counts'], labels_recomputed=recomputed,
        content_arrays_sha_and_raw_bytes_verified=len(content), current_short_window_variants=len(short_keys),
        policy_payload_keys=sorted(payload), decoded_rgb_sizes=[len(x) for x in payload['rgb']],
        prefix_observations=len(prefix), history_event_distances=history_events,
        matrix=matrix, normalization_events=normalization,
        new_physical_trajectories=0, new_labels_generated=0,
        gaps=['SEE2 visual witness is not geodesic room visit',
              'Completed-subgoal revisit control is not an event-free irrelevant spatial detour',
              'No history-irrelevant goal-only task in the chosen two-task family',
              'This one inspected house is already exposed; no independently certified new house/language split in this audit',
              'Numerical normalization is recorded; raw/canonical boundary must be evaluated under any new shared-state contract',
              '248-step prefix is not evidence that differentiable memory unroll fits the future budget',
              'No qualified research backbone, memory implementation or measured cross-step gradients',
              'Typed query indices depend on per-compiler vocabulary; cannot train directly on unbound local category IDs',
              'Global family grouping and all-arm shared action-pool manifest are not prepared for this research protocol'])
    write(HERE/'DATA_ASSET_AUDIT.json',out)
    write(HERE/'ASSET_INVENTORY.json',inventory())
    return out


def stop_audit():
    folder = V3.parent/'ordinary_expanded_dev_after_single_v1/run_001'
    repeat = read(LINE/'reviews/Q35N_RECOVERY_20260917/REPEATED_INPUTS.json')
    repeated = {r['index']:r['repeated_input_decisions'] for r in repeat['episodes']}
    rows=[]
    for lane in sorted((folder/'lanes').glob('lane_*')):
        policies=collections.defaultdict(list)
        for r in lines(lane/'POLICY_STEPS.jsonl'):policies[r['index']].append(r)
        for path in sorted(lane.glob('episode_*.json')):
            e=read(path);near=[i+1 for i,d in enumerate(e['distances'][:-1]) if d<3.]
            category=('success' if e['success'] else
                'left_goal_then_stopped_far' if e['oracle_success'] and e['stopped'] else
                'entered_goal_budget_no_legal_stop' if e['oracle_success'] else
                'far_stop_never_entered' if e['stopped'] else 'budget_never_entered')
            margins=[]
            for p in policies[e['index']]:
                if p['step'] in near:
                    margins.append(p['logits'][3]-max(p['logits'][:3]))
            rows.append(dict(index=e['index'],episode_id=e['episode_id'],category=category,
                steps=e['steps'],repeated_decisions=repeated[e['index']],
                in_goal_before_action_decisions=len(near),
                in_goal_stop_margin_median=statistics.median(margins) if margins else None,
                terminal_stop_margin=policies[e['index']][-1]['logits'][3]-max(policies[e['index']][-1]['logits'][:3])))
    if len(rows)!=100:raise ValueError('EXPECTED_HISTORICAL_100')
    write(HERE/'STOP_FAILURE_AUDIT.json',dict(scope='Read-only historical best4k INTERNAL_DEV100; not new A/B',
        episodes=100,category_counts=dict(collections.Counter(r['category'] for r in rows)),
        failed_entered_with_repetition=sum(r['repeated_decisions']>0 and r['category'] in ('left_goal_then_stopped_far','entered_goal_budget_no_legal_stop') for r in rows),
        stop_margin_definition='native STOP logit minus max of three motion logits at pre-action geodesic distance <3m; audit-only distance',rows=sorted(rows,key=lambda r:r['index'])))


def main():
    started=time.monotonic()
    required=['CURRENT_STATUS.json','MAINLINE_FREEZE_V3.md','reviews/Q35N_P1_PAPER_CORE_ADJUDICATION_V2/REPORT_ZH.md',
        'reviews/Q35N_RECOVERY_20260917/FINAL_REVIEW.json','reviews/Q35N_RECOVERY_20260917/REPEATED_INPUTS.json',
        'reviews/Q35N_RECOVERY_20260917/NUMERIC_TRANSPORT_DIAGNOSIS.json',
        'closed_loop_bench/ordinary_cycle_recovery_v1/review.py','closed_loop_bench/r2r_ce_tiny_v1/common.py',
        'closed_loop_bench/r2r_ce_tiny_v1/metrics.py','sft_acceptance/ordinary_sync_recovery_v1/model.py',
        'sft_acceptance/ordinary_sync_recovery_v1/data.py','sft_acceptance/ordinary_baseline_v2/data.py',
        'deployment/ordinary_v1/MODEL_CARD.json','deployment/ordinary_v1/DEPLOYMENT_ACCEPTANCE.json','deployment/ordinary_v1/predict.py',
        'sft_acceptance/v1/policy.py','data_pipeline/mechanism_runtime_v1/exporter.py',
        'data_pipeline/mechanism_runtime_v1/witness_first_v1/ordered_visit_v5_cpu/visit.py']
    for name in ('cycle_policy.py','evaluate.py','common.py','executor.py','aggregate.py','review.py','launch.py','PROTOCOL.json','SOURCE_LOCK.json'):
        required.append(str((V3/name).relative_to(LINE)))
    for relative in required:blob(LINE/relative)
    for p in (ROOT/'AGENTS.md',LINE/'AGENTS.md',ROOT/'research/EXECUTION_RULES.md'):blob(p)
    mismatches=[]
    for path,expected in read(V3/'SOURCE_LOCK.json')['files'].items():
        digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=expected:mismatches.append(path)
    write(HERE/'WORKSPACE_AUDIT.json',dict(base_commit=BASE,
        actual_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        actual_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        diff_from_base=subprocess.check_output(['git','diff','--name-status',BASE],cwd=ROOT,text=True).splitlines(),
        source_lock_entries=198,source_lock_mismatches=mismatches,
        v4_preexists_and_closed=True,old_v4_result_sha256=hashlib.sha256(blob(V4/'RESULT.json')).hexdigest()))
    if mismatches:raise ValueError('PREFLIGHT_BLOCKED: SOURCE_IDENTITY_MISMATCH')
    budget=budget_audit()
    scheduler={c:shutil.which(c) for c in ('scontrol','squeue','salloc','qstat','bjobs')}
    allocation={k:v for k,v in os.environ.items() if k.startswith(('SLURM_','PBS_','LSB_'))}
    write(V4/'codex_return_20260917/PREFLIGHT.json',dict(status='RESOURCE_NOT_RESERVED',
        scheduler_commands=scheduler,scheduler_environment=allocation,coordinated_window_receipt=None,
        existing_v4_result='../RESULT.json',formal_starts=0,completed_pairs=0,required_pairs=100,
        evaluation_valid=False,benefit='UNKNOWN',adopted=False,
        budget_timing_verdict=budget['timing_verdict'],runtime_integration_ready=False,
        polling_or_device_switch_attempts=0,old_results_overwritten=False,
        source_mismatch_or_proven_budget_infeasibility_status='PREFLIGHT_BLOCKED'))
    asset=asset_audit();stop_audit()
    write(HERE/'READ_MANIFEST.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        required_paths=required,reads=READS,unavailable_required_paths=[],
        wall_seconds=time.monotonic()-started,new_gpu_starts=0,new_simulation_actions=0,
        note='Listed reads are existing bytes/content, not proof of executing models or replaying physics; source seal streaming reads are counted separately'))
    print(json.dumps(dict(source_mismatches=mismatches,engineering='RESOURCE_NOT_RESERVED',
        asset_status=asset['status'],labels=asset['labels_recomputed'],
        content_arrays=asset['content_arrays_sha_and_raw_bytes_verified'],wall_seconds=time.monotonic()-started)))


if __name__=='__main__':main()
