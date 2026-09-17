"""Distinct semantic program selection; stored observations are not new replays."""
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent
WF=HERE.parent
RUNTIME=WF.parent
ROOT=next(p for p in HERE.parents if p.name=='vla')
sys.path.insert(0,str(WF/'winding_balance_v1'))
from method import solve,closed,plan,CONTROL_TYPE
sys.path.insert(0,str(RUNTIME))
from core_bridge import Compiler,compiler,digest
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
bank=load('multi_program_frozen_bank',WF/'bank_cpu/bank.py')
loader=load('multi_program_pixel_reader',RUNTIME/'loader.py')
acceptance=load('multi_program_committed_journal',WF/'quality_cpu/batch_acceptance_v1/acceptance.py')

def stable(path,max_bytes=512*1024**2):
    path=Path(path);assert path.resolve()==path and path.is_relative_to(ROOT)
    before=path.stat();assert before.st_size<=max_bytes
    raw=path.read_bytes();after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'SOURCE_CHANGED_WHILE_READ'
    return raw,hashlib.sha256(raw).hexdigest()
def canonical_program(proposal):
    """Raw aliases, A/B swap, I identity, seed, padding and names create no group."""
    def semantic(name):
        r=proposal['roles'][name];return (r['mpcat40'],r['room'])
    anchors=tuple(sorted([semantic('anchor_A'),semantic('anchor_B')]))
    return (anchors,semantic('terminal'))
def semantic_id(proposal):
    return digest({'house':proposal['house_id'],'physical_hub_position':proposal['hub_pose']['position'],
                   'ordered_visual_event_then_stop_program_pair':canonical_program(proposal)})
def rank(candidate):
    return (candidate['history_actions_before_public_tail'],candidate['max_continuation_actions'],
            candidate['source_proposal_id'],candidate['balance']['k'],candidate['balance']['j'])
def select_programs(candidates,cap=24):
    if type(cap) is not int or not 1<=cap<=24:raise ValueError('PER_HUB_CAP')
    groups=collections.defaultdict(list)
    for row in candidates:groups[row['canonical_program_id']].append(row)
    chosen=[];decisions=[]
    for key,rows in sorted(groups.items()):
        rows.sort(key=rank);chosen.append(rows[0])
        for row in rows[1:]:decisions.append({'candidate_id':row['candidate_id'],'reason':'SAME_SEMANTIC_PROGRAM_NONMINIMAL_REPRESENTATIVE','representative':rows[0]['candidate_id']})
    chosen.sort(key=lambda r:(rank(r),r['canonical_program_id']))
    for row in chosen[cap:]:decisions.append({'candidate_id':row['candidate_id'],'reason':'PER_HUB_24_CAP'})
    return chosen[:cap],decisions
def spin_verdict(comp,required,witnesses):
    report={};rejected=False
    for direction in required:
        traces=[w for w in witnesses if w['direction']==direction]
        if not traces:report[direction]={'status':'UNTESTED_NO_DIRECTION_WITNESS'};continue
        rows=[]
        for witness in traces:
            trace=witness['trace'];assert trace['actions']==[direction]*24 and compiler.complete(trace)
            assert closed(trace['observations'][0]['pose'],trace['observations'][-1]['pose'])
            events=comp.atoms(trace['observations'])
            forbidden=[{'step':i,'anchor_A':e['anchor_A'],'anchor_B':e['anchor_B']} for i,e in enumerate(events) if e['anchor_A'] or e['anchor_B']]
            rows.append({'trace_ref':witness['trace_ref'],'trace_sha256':witness['sha256'],
                         'forbidden_events':forbidden,'neutral':not forbidden})
        bad=any(not x['neutral'] for x in rows);rejected|=bad
        report[direction]={'status':'REJECTED_OBSERVED_ANCHOR_EVENT' if bad else 'STORED_DIRECTION_WITNESS_NEUTRAL_NOT_FAMILY_PASS','witnesses':rows}
    return not rejected,report
def verify_semantic_pixels(trace,content_root,lock,cache):
    for obs in trace['observations']:
        key=obs['semantic_hash'];path=content_root/(key+'.semantic.npy')
        if str(path) not in cache:
            raw,h=stable(path,1024**2);lock[str(path)]=h
            pixels=loader.npy_pixels(raw,key,'semantic')
            cache[str(path)]={str(k):v for k,v in collections.Counter(x[0] for x in struct.iter_unpack('<I',pixels)).items()}
        assert {str(k):v for k,v in obs['pixels'].items() if v}==cache[str(path)],'SEMANTIC_PIXELS_NOT_RECOMPUTED'
def read_committed_prefix(run,snapshot_dir,lock):
    """Snapshot a verified HEAD-sized immutable journal prefix, even if appending."""
    raw_cfg,h=stable(run/'EXECUTION_CONFIG.json');lock[str(run/'EXECUTION_CONFIG.json')]=h;cfg=json.loads(raw_cfg)
    raw_lock,h=stable(run/'INPUT_LOCK.json');lock[str(run/'INPUT_LOCK.json')]=h;source_lock=json.loads(raw_lock)
    assert source_lock[str(run/'EXECUTION_CONFIG.json')]==lock[str(run/'EXECUTION_CONFIG.json')]
    # HEAD is atomically replaced while active. Pin a single inode through its
    # open descriptor: requiring the pathname inode to remain current would
    # incorrectly reject a valid older committed prefix during a normal append.
    with (run/'journal/HEAD.json').open('rb') as stream:
        first=os.fstat(stream.fileno());assert first.st_size<=65536
        head_raw=stream.read(65537);last=os.fstat(stream.fileno())
        assert (first.st_ino,first.st_size,first.st_mtime_ns)==(last.st_ino,last.st_size,last.st_mtime_ns)
        assert len(head_raw)==first.st_size
    head=json.loads(head_raw)
    assert head['byte_length']<=512*1024**2
    with (run/'journal/events.jsonl').open('rb') as f:raw=f.read(head['byte_length'])
    assert len(raw)==head['byte_length']
    records=acceptance.parse_journal(raw,head,cfg)
    snapshot_dir.mkdir(parents=True,exist_ok=False)
    for name,blob in [('HEAD.json',head_raw),('events.jsonl',raw)]:
        path=snapshot_dir/name
        with path.open('xb') as f:f.write(blob)
        lock[str(path)]=hashlib.sha256(blob).hexdigest()
    return cfg,records
