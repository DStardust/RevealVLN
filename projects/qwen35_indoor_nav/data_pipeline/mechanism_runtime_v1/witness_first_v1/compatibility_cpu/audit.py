"""Witness-first CPU compatibility; partial evidence, never a family certificate."""
import collections
import importlib.util
import itertools
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
ROOT=RUNTIME.parents[3]
sys.path.insert(0,str(RUNTIME))
from core_bridge import Compiler,compiler,digest,factory

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

planner=load('witness_allowed_roles',RUNTIME.parent/'mechanism_scale_v1/planner.py')

def stable_json(path):
    before=path.stat();raw=path.read_bytes();after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'ACTIVE_FILE_CHANGED'
    return json.loads(raw),digest(raw)

def save(name,value):
    content=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n';p=HERE/name
    if p.exists():assert p.read_text()==content,('NEW_SNAPSHOT_VERSION_REQUIRED',name)
    else:
        with p.open('x') as handle:handle.write(content)

def role_groups(objects):
    groups=collections.defaultdict(list)
    for key,obj in objects.items():
        idx=int(key);raw=' '.join(obj['raw'].lower().replace('#',' ').split())
        if idx in planner.RESERVED or obj['room'] not in planner.INDOOR_ROOMS or raw not in planner.RAW.get(obj['mpcat40'],set()):continue
        signature=(obj['mpcat40'],obj['room'],raw)
        groups[signature].append(idx)
    return {sig:sorted(ids) for sig,ids in sorted(groups.items())}

def instance_events(observations):
    """Same-instance, consecutive-frame threshold; exact compiler validation."""
    result=[]
    for t,obs in enumerate(observations):
        assert compiler._known_observation(obs,t),'UNKNOWN_OBSERVATION_CANNOT_BE_NEGATIVE'
        if t==0:result.append({});continue
        previous=observations[t-1]
        ids=sorted(int(k) for k,v in obs['pixels'].items() if v>=256 and compiler.pixel(previous,int(k))>=256)
        result.append({idx:dict(step=t,frames=[t-1,t],pixels=[compiler.pixel(previous,idx),compiler.pixel(obs,idx)]) for idx in ids})
    return result

def closed_pose(first,last):
    if set(first.get('sensors',{}))!={'rgb','semantic'} or set(last.get('sensors',{}))!={'rgb','semantic'}:return False
    return all(max(factory.pose_distance(a,b))<=1e-5 for a,b in [(first,last)]+[(first['sensors'][k],last['sensors'][k]) for k in ('rgb','semantic')])

def trace_summary(trace,objects,path,sha,house):
    assert compiler.complete(trace),'TRACE_NOT_COMPLETE'
    body={k:v for k,v in trace.items() if k!='trace_hash'}
    assert digest(body)==trace['trace_hash'],'TRACE_SELF_HASH'
    events=instance_events(trace['observations']);groups=role_groups(objects)
    first={};instance_first={}
    for ev in events:
        for idx,evidence in ev.items():instance_first.setdefault(str(idx),evidence)
    for signature,ids in groups.items():
        witnessed=[(instance_first[str(idx)]['step'],idx) for idx in ids if str(idx) in instance_first]
        if witnessed:first[json.dumps(signature,separators=(',',':'))]=min(witnessed)
    pose=trace['observations'][0]['pose']
    return dict(path=str(path.relative_to(ROOT)),sha256=sha,trace_hash=trace['trace_hash'],house_id=house,
        actions=len(trace['actions']),start_pose=pose,start_key=digest(dict(house=house,pose=pose)),
        initial_position=trace['initial_position'],initial_yaw_bin=trace['initial_yaw_bin'],
        all_instance_first_witnesses=instance_first,role_first=first,
        closed_loop=bool(trace['actions']) and closed_pose(pose,trace['observations'][-1]['pose']),
        public_tail_probe=trace['actions']==list('LRLRLRLR'))

def witness(rows,required,forbidden=(),closed=False):
    choices=[]
    for row in rows:
        if required not in row['role_first']:continue
        cutoff=row['role_first'][required][0]
        if closed:
            if not row['closed_loop'] or row['actions']>504:continue
            if any(role in row['role_first'] for role in forbidden):continue
            cutoff=row['actions']
        elif any(role in row['role_first'] and row['role_first'][role][0]<=cutoff for role in forbidden):continue
        choices.append(dict(path=row['path'],sha256=row['sha256'],prefix_cutoff=cutoff,
            required_role=required,required_instance=row['role_first'][required][1],
            required_event_step=row['role_first'][required][0],forbidden_roles=list(forbidden),
            full_closed_loop=closed,source_trace_actions=row['actions']))
    return min(choices,key=lambda w:(w['prefix_cutoff'],w['path'])) if choices else None

def semantic_distinct(signatures):return len({tuple(json.loads(s)[:2]) for s in signatures})==len(signatures)

def compatibility(rows):
    by_start=collections.defaultdict(list)
    for row in rows:by_start[row['start_key']].append(row)
    pairs=[];programs=[]
    for key,traces in sorted(by_start.items()):
        roles=sorted({role for tr in traces for role in tr['role_first']})
        for a,b in itertools.combinations(roles,2):
            if not semantic_distinct((a,b)):continue
            wa=witness(traces,a,[b]);wb=witness(traces,b,[a])
            if not wa or not wb:continue
            neutral=[tr for tr in traces if tr['public_tail_probe'] and a not in tr['role_first'] and b not in tr['role_first']]
            loops=[witness(traces,a,[b],closed=True),witness(traces,b,[a],closed=True)]
            pair=dict(start_key=key,house_id=traces[0]['house_id'],start_pose=traces[0]['start_pose'],
                anchor_A=a,anchor_B=b,witness_A=wa,witness_B=wb,
                neutral_public_tail_witnesses=[tr['path'] for tr in neutral],
                closed_loop_A=loops[0],closed_loop_B=loops[1],family_certified=False)
            pairs.append(pair)
            if not neutral:continue
            for terminal in roles:
                if not semantic_distinct((a,b,terminal)):continue
                nt=[tr for tr in neutral if terminal not in tr['role_first']]
                if not nt:continue
                wt=witness(traces,terminal,[a,b])
                if not wt:continue
                for irrelevant in roles:
                    if not semantic_distinct((a,b,terminal,irrelevant)):continue
                    wi=witness(traces,irrelevant,[b,terminal])
                    if not wi:continue
                    rolespec={name:planner.role_spec(json.loads(sig)) for name,sig in
                              [('anchor_A',a),('anchor_B',b),('terminal',terminal),('irrelevant',irrelevant)]}
                    programs.append(dict(start_key=key,house_id=traces[0]['house_id'],
                        initial_position=traces[0]['initial_position'],initial_yaw_bin=traces[0]['initial_yaw_bin'],
                        roles=rolespec,tasks=planner.task_spec(rolespec),witness_A=wa,witness_B=wb,
                        terminal_only_prefix=wt,irrelevant_prefix=wi,neutral_public_tail=nt[0]['path'],
                        closed_loop_A=loops[0],closed_loop_B=loops[1],
                        closed_loop_irrelevant=witness(traces,irrelevant,[b,terminal],closed=True),
                        evidence_tier='CAUSAL_PREFIX_ROLE_COMPATIBILITY_NOT_PHYSICAL_FAMILY',
                        physical_family_certified=False,training_admission=False,
                        unproven=['three_history_shared_join','padding_neutrality','composed_H_A_I',
                            'three_legal_continuations','18_cell_matrix','three_seed_replay','shortcut_audit']))
    programs.sort(key=lambda p:(-sum(p[k] is not None for k in ('closed_loop_A','closed_loop_B','closed_loop_irrelevant')),
        sum(p[k]['prefix_cutoff'] for k in ('witness_A','witness_B','terminal_only_prefix','irrelevant_prefix')),
        p['house_id'],p['start_key'],json.dumps(p['roles'],sort_keys=True)))
    return pairs,programs

def snapshot():
    path=HERE/'SNAPSHOT.json'
    if path.exists():return json.loads(path.read_text())
    rows=[];excluded=[];metadata={}
    for runrel in ('feedback_generation_v1/run_v1','compact_loop_v2/recovery_v1/run_v1'):
        run=RUNTIME/runrel
        cfg,config_hash=stable_json(run/'EXECUTION_CONFIG.json')
        metadata[str((run/'EXECUTION_CONFIG.json').relative_to(ROOT))]=config_hash
        for candidate in cfg['candidates']:
            assert candidate['split']=='FIT','HELDOUT_FORBIDDEN'
            folder=run/'bundles'/candidate['candidate_id']
            inv=folder/'SEMANTIC_INVENTORY.json'
            if not inv.exists():continue
            _,inv_hash=stable_json(inv);metadata[str(inv.relative_to(ROOT))]=inv_hash
            for tracefile in sorted((folder/'traces').glob('*.json')):
                try:
                    trace,trace_hash=stable_json(tracefile)
                    if not compiler.complete(trace):
                        excluded.append(dict(path=str(tracefile.relative_to(ROOT)),reason='INCOMPLETE_TRACE_NOT_NEGATIVE'));continue
                    assert digest({k:v for k,v in trace.items() if k!='trace_hash'})==trace['trace_hash']
                except (json.JSONDecodeError,AssertionError) as error:
                    excluded.append(dict(path=str(tracefile.relative_to(ROOT)),reason=repr(error)));continue
                rows.append(dict(path=str(tracefile.relative_to(ROOT)),sha256=trace_hash,
                    inventory=str(inv.relative_to(ROOT)),house_id=candidate['house_id'],source_run=runrel))
    data=dict(status='FIXED_READ_ONLY_SNAPSHOT_NOT_V2_TERMINAL',traces=rows,metadata_hashes=metadata,
              excluded=excluded,heldout_read=False,all_future_files_excluded=True)
    save('SNAPSHOT.json',data);return data

def main():
    snap=snapshot();summaries=[];inventories={};catalog={}
    for name,expected in snap['metadata_hashes'].items():
        _,actual=stable_json(ROOT/name);assert actual==expected,name
    for row in snap['traces']:
        trace,actual=stable_json(ROOT/row['path']);assert actual==row['sha256'],row['path']
        if row['inventory'] not in inventories:inventories[row['inventory']]=stable_json(ROOT/row['inventory'])[0]['objects']
        objects=inventories[row['inventory']]
        summaries.append(trace_summary(trace,objects,ROOT/row['path'],actual,row['house_id']))
        catalog[row['house_id']]={json.dumps(sig,separators=(',',':')):ids for sig,ids in role_groups(objects).items()}
    pairs,programs=compatibility(summaries)
    save('TRACE_WITNESSES.json',summaries);save('ROLE_CATALOG.json',catalog)
    save('COMPATIBLE_PAIRS.json',pairs);save('PROGRAM_CANDIDATES.json',programs[:100])
    result=dict(status='CPU_WITNESS_COMPATIBILITY_READY',threshold='same_instance_256px_in_two_consecutive_known_frames',
        frozen_complete_traces=len(summaries),snapshot_excluded=len(snap['excluded']),
        houses=sorted({tr['house_id'] for tr in summaries}),compatible_same_start_pairs=len(pairs),
        pairs_with_neutral_public_tail=sum(bool(p['neutral_public_tail_witnesses']) for p in pairs),
        pairs_with_both_actual_closed_loops=sum(bool(p['closed_loop_A']) and bool(p['closed_loop_B']) for p in pairs),
        four_role_prefix_compatible_programs=len(programs),exported_top_programs=min(100,len(programs)),
        programs_with_three_existing_closed_loops=sum(all(p[k] for k in ('closed_loop_A','closed_loop_B','closed_loop_irrelevant')) for p in programs),
        new_physical_families=0,training_admitted_families=0,scientific_pass=False,gpu_operations=0,
        active_v2_terminal_read=False,old_files_modified=False)
    save('result.json',result)
    save('SOURCE_CODE_HASHES.json',{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in
        [RUNTIME.parent/'mechanism_factory_v2/compiler.py',RUNTIME.parent/'mechanism_factory_v2/factory.py',
         RUNTIME.parent/'mechanism_scale_v1/planner.py',HERE/'audit.py',HERE/'test_audit.py']})
    print(json.dumps(result,indent=2))
    if programs:print(json.dumps(programs[0],indent=2))

if __name__=='__main__':main()
