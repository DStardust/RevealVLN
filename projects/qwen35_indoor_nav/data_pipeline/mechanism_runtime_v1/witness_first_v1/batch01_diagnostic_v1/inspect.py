"""Read-only batch01 diagnostic snapshots, never formal quality certification."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
WF=HERE.parent
RUNTIME=WF.parent
RUN=WF/'batch_execution_v1/batch_01/run_v1'
sys.path.insert(0,str(RUNTIME))
from core_bridge import Compiler,compiler,FamilyFactory,factory,digest
spec=importlib.util.spec_from_file_location('diagnostic_winding_plan',WF/'winding_balance_v1/method.py')
method=importlib.util.module_from_spec(spec);spec.loader.exec_module(method)


def stable_json(path):
    before=path.stat();raw=path.read_bytes();after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('ACTIVE_FILE_CHANGED')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def inspect(output):
    output=output.resolve()
    if output.parent!=HERE or not output.name.startswith('snapshot') or output.exists():raise ValueError('NEW_LOCAL_SNAPSHOT_REQUIRED')
    cfg,cfg_sha=stable_json(RUN/'EXECUTION_CONFIG.json');bundles=[];source_hashes={str(RUN/'EXECUTION_CONFIG.json'):cfg_sha}
    active_unread=[]
    for row in cfg['candidates']:
        comp=Compiler({k:(v['mpcat40'],v['room']) for k,v in row['roles'].items()},row['tasks'],row['expected_eligible'])
        p=method.plan(row['components']['a'],row['components']['b'],row['components']['i'],**row['balance'])
        tail=list(row['configuration']['public_tail'])
        c0=row['components']['terminal']+['S']
        cs={'C0':c0,'C_A':row['components']['a']+c0,'C_B':row['components']['b']+c0}
        folder=RUN/'bundles'/row['candidate_id'];traces=[]
        if folder.is_dir():
            for path in sorted((folder/'traces').glob('*.json')):
                try:trace,h=stable_json(path)
                except (json.JSONDecodeError,ValueError,FileNotFoundError):
                    active_unread.append(str(path));continue
                source_hashes[str(path)]=h
                complete=compiler.complete(trace)
                actions=trace['actions'];phases=[]
                if not actions:phases.append('reset_probe')
                for direction in p['required_neutral_spin_directions']:
                    if actions==[direction]*24:phases.append('neutral_spin_'+direction)
                for name,a in row['components'].items():
                    if actions==a:phases.append('component_'+name)
                for history,a in p['histories'].items():
                    if actions==a:phases.append('history_'+history)
                    if actions==a+tail:phases.append('public_tail_'+history)
                entry={'trace_file':str(path.relative_to(RUN)),'file_sha256':h,'complete':complete,
                       'motion_actions':sum(a in ('F','L','R') for a in actions),
                       'F_L_R_S_counts':{a:actions.count(a) for a in ('F','L','R','S')},
                       'collisions':trace['collisions'],'matched_phases':phases,'matrix_check':None}
                if complete:
                    body={k:v for k,v in trace.items() if k!='trace_hash'}
                    entry['trace_self_hash_pass']=trace.get('trace_hash')==digest(body)
                    atoms=comp.atoms(trace['observations'])
                    entry['first_role_events']={role:next((t for t,event in enumerate(atoms) if event[role]),None) for role in row['roles']}
                    initial=trace['observations'][0]['pose'];last=trace['observations'][-1]['pose']
                    entry['endpoint_pose_deltas']={'agent':factory.pose_distance(initial,last),
                        **{sensor:factory.pose_distance(initial['sensors'][sensor],last['sensors'][sensor]) for sensor in ('rgb','semantic')}}
                    for history,a in p['histories'].items():
                        for continuation,c in cs.items():
                            if actions==a+tail+c:
                                cutoff=len(a)+len(tail)
                                ys={task:comp.evaluate(trace,task) for task in row['tasks']}
                                expected={task:FamilyFactory.expected(i,history,continuation) for i,task in enumerate(('task_A','task_B'))}
                                entry['matrix_check']={'history':history,'continuation':continuation,'seed':trace['seed'],
                                    'observed':ys,'expected':expected,'label_match':ys==expected,
                                    'query_sequence_length':len(comp.query_from_trace(comp.slice_continuation(trace,cutoff))['sequence'])}
                traces.append(entry)
        result=None
        if (folder/'result.json').is_file():
            try:result,h=stable_json(folder/'result.json');source_hashes[str(folder/'result.json')]=h
            except (ValueError,json.JSONDecodeError,FileNotFoundError):active_unread.append(str(folder/'result.json'))
        language_risks=[]
        for task,specification in row['tasks'].items():
            if ' in the tv ' in specification['instruction'].lower() or ' in the tv,' in specification['instruction'].lower():
                language_risks.append({'task':task,'type':'ROOM_CODE_TV_NOT_RENDERED_AS_TV_ROOM','formal_quality_decision_deferred':True})
        bundles.append({'candidate_id':row['candidate_id'],'house_id':row['house_id'],
            'configured_history_F_L_R':p['action_counts'],'configured_counts_match_expected':p['action_counts']==row['expected_history_action_counts'],
            'configured_history_actions':p['history_actions'],'required_neutral_spins':p['required_neutral_spin_directions'],
            'trace_count_read':len(traces),'traces':traces,'terminal_result':result,'language_risks':language_risks,
            'frozen_candidate_present':(folder/'FROZEN_CANDIDATE.json').is_file(),
            'physical_certificate_file_present':(folder/'CERTIFICATE.json').is_file()})
    progress=None
    if (RUN/'PROGRESS.json').is_file():
        try:progress,h=stable_json(RUN/'PROGRESS.json');source_hashes[str(RUN/'PROGRESS.json')]=h
        except (ValueError,json.JSONDecodeError,FileNotFoundError):active_unread.append(str(RUN/'PROGRESS.json'))
    output.mkdir()
    result={'scope':'READ_ONLY_DIAGNOSTIC_NOT_FORMAL_QUALITY','batch':'batch_01','bundles':bundles,
            'progress':progress,'process_record_present':(RUN/'PROCESS.json').is_file(),
            'gpu_admission_record_present':(RUN/'GPU_BEFORE.json').is_file(),
            'batch_terminal_result_present':(RUN/'result.json').is_file(),
            'partially_written_or_changing_files_not_judged':active_unread,
            'source_hashes_at_read':source_hashes,'runtime_or_process_modified':False,'scientific_pass':False}
    with (output/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'process_record_present':result['process_record_present'],'progress':progress,
        'bundles':[{'candidate_id':x['candidate_id'],'traces':x['trace_count_read'],'result':x['terminal_result']} for x in bundles]}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    inspect(parser.parse_args().output)
