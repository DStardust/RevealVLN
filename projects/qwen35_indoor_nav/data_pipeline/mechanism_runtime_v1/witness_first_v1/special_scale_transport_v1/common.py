"""Frozen-source utilities for explicit new special-production transports."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
WF=HERE.parent
BE=WF/'batch_execution_v1'
RUNTIME=WF.parent
LINE=RUNTIME.parents[1]
ROOT=LINE.parents[1]
BUDGET=dict(total_actions=60000,total_seconds=3600,discovery_actions=15000,
            discovery_seconds=1000,certification_actions=20000,certification_seconds=1500)
AMENDMENT=dict(version='special_active_conservative_accounting_v1',
    inconsistent_sample_total_must_be_less_than_mib=4096,
    external_process_max_mib=768,external_sum_max_mib=2048,
    active_own_present_only=True,raw_xml_preserved=True,
    idle_restore_original_guard=True,original_guard_pass_not_claimed=True)
SOURCES={
    BE/'shared.py':'3b717bdf07a53ce06191a2445a20e86982073c17b43412b30b1332c8e01a53b0',
    BE/'prepare.py':'14b0288ccdc027df9556ef3ff9cb1b478c3fd0c9158e6587a1a6432d8278d718',
    BE/'readiness_v1.py':'6ad5b369845917a57963ce2fb4718bf2580b8d87a408c8f467cb70ddc61b7600',
    BE/'budget_batched_transport_v1/transport.py':'386e05e4ae16a1ed062846120fd5352ed4819594a8b4da4462a8645a2e9b1a2c',
    BE/'gpu5_transport_v2/transport.py':'a48d8f1c3454b84804531a051b209db6ce419995493d103078a4a2cb92749a7f',
    RUNTIME/'compact_loop_v2/run.py':'9aabce69787bc717f6283b7679fef3c2f811018f0456741dc51516ebe6e986ca',
    LINE/'data_pipeline/ordinary_fullscale_source_v1/auto_generation_v4/telemetry.py':'c0488a18a30477ec5323b409f0b4a31d38b668677be8fe923195fb682d6f3976',
    WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/gate.py':'9aa097b918611537a9d0f2517528a6e257ec6504e860cc53fe28af8c9119fb83'}

def require(ok,reason):
    if not ok:raise ValueError(reason)
def scoped(path):
    p=Path(path).absolute()
    require(p==p.resolve() and p.is_relative_to(LINE),'EXACT_LINE_PATH_REQUIRED')
    return p
def sha(path):
    p=Path(path).absolute();require(p==p.resolve() and p.is_relative_to(ROOT),'HASH_SCOPE')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(scoped(path).read_text())
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def save(path,value):
    with scoped(path).open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False));f.flush();os.fsync(f.fileno())
def append(path,value):
    with scoped(path).open('a') as f:
        f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
def load(name,path):
    p=scoped(path)
    if p in SOURCES:require(sha(p)==SOURCES[p],'FROZEN_SOURCE_CHANGED:'+str(p))
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
def check_sources():
    for p,h in SOURCES.items():require(sha(p)==h,'FROZEN_SOURCE_CHANGED:'+str(p))
def exact(source,old,new):
    require(source.count(old)==1 and old!=new,'EXACT_SOURCE_SUBSTITUTION')
    changed=source.replace(old,new);require(changed.count(new)==1 and changed.replace(new,old)==source,'SOURCE_REVERSE_CHECK')
    return changed
def validate_authorization(document,queue_path):
    require(document.get('approved') is True and document.get('training_allowed') is False,'EXPLICIT_DATA_ONLY_APPROVAL')
    require(document.get('version')=='special_scale_transport_v1','AUTH_VERSION')
    parent=scoped(document['main_scope_authorization_path'])
    require(sha(parent)==document['main_scope_authorization_sha256'],'MAIN_SCOPE_AUTH_HASH')
    broad=read(parent)
    require(broad.get('approved') is True and broad.get('training_allowed') is False and broad.get('project_root')==str(ROOT),'MAIN_SCOPE_APPROVAL')
    require(broad.get('special_lane_wall_seconds')==43200 and broad.get('no_automatic_retry_of_failed_or_partial') is True,'MAIN_SCOPE_LIMITS')
    require(broad.get('special_per_batch')==dict(supervisor_seconds=3900,factory_seconds=3600,action_cap=60000,disk_gib=7,audit_seconds=1200),'MAIN_BATCH_LIMITS')
    require(broad.get('unknown_or_external_process_signals_allowed') is False and broad.get('gpu0_operations_allowed') is False,'MAIN_SIGNAL_SCOPE')
    require(broad.get('gpu_accounting_amendment')==dict(version='active_conservative_max_v1',raw_first=True,
        derived_decision_separate=True,strict_original_idle_restore=True,preserve_original_quality_thresholds=True,
        resource_amendment_indicated_in_audit=True,active_only=True,conservative_total_must_be_less_than_mib=4096,
        external_per_process_mib=768,external_total_mib=2048),'MAIN_AMENDMENT_SCOPE')
    gpu=document.get('gpu_device');require(type(gpu) is int and gpu in (1,2,3,4,5,7),'GPU_SCOPE')
    require(isinstance(document.get('gpu_uuid'),str) and re.fullmatch(r'GPU-[a-f0-9-]{36}',document['gpu_uuid']),'GPU_UUID_REQUIRED')
    require(document.get('mode') in ('nonexclusive','holder_chain'),'LEASE_MODE')
    require(document.get('budget')==BUDGET and document.get('supervision_wall_seconds')==3900,'ORIGINAL_BUDGET_REQUIRED')
    require(document.get('accounting_amendment')==AMENDMENT,'EXPLICIT_ACCOUNTING_AMENDMENT')
    require(document.get('automatic_retry') is False and document.get('replace_failed_candidate') is False,'NO_RETRY_OR_REPLACEMENT')
    require(document.get('lane_wall_seconds')==43200,'FROZEN_LANE_BUDGET')
    path=scoped(queue_path);require(document.get('queue_path')==str(path) and document.get('queue_sha256')==sha(path),'QUEUE_AUTH_BINDING')
    queue=read(path);items=[r for r in queue['batches'] if r['gpu']==gpu]
    require(gpu in broad['special_gpus'] and len(queue['batches'])<=broad['max_special_jobs'],'MAIN_GPU_JOB_SCOPE')
    require(sum(len(r['candidate_ids']) for r in queue['batches'])<=broad['max_special_candidates'],'MAIN_CANDIDATE_SCOPE')
    require(items and document.get('batch_ids')==[r['id'] for r in items],'ORDERED_LANE_BINDING')
    require(len({r['id'] for r in queue['batches']})==len(queue['batches']),'DUPLICATE_BATCH_ID')
    for r in items:
        require(re.fullmatch(r'batch_[0-9]+',r['id']) and len(r['candidate_ids'])==3,'THREE_CANDIDATE_BATCH_REQUIRED')
    if document['mode']=='holder_chain':
        require(gpu in (3,4,5,7),'BORROW_SCOPE')
        require(document.get('restore_holder_on_success_failure_or_interruption') is True,'RESTORATION_REQUIRED')
        require(document.get('chain_identity_from_immediate_previous_restoration_only') is True,'EXACT_RESTORATION_CHAIN_REQUIRED')
        p=scoped(document['initial_holder_identity_path'])
        require(sha(p)==document['initial_holder_identity_sha256'],'INITIAL_IDENTITY_HASH')
    return items

def check_config(cfg):
    require(cfg.get('runtime_transport_version')=='special_scale_transport_v1','TRANSPORT_VERSION')
    require(cfg.get('accounting_amendment')==AMENDMENT,'AMENDMENT_CONFIG')
    require(cfg.get('budget')==BUDGET and cfg.get('supervision_wall_seconds')==3900,'UNCHANGED_BUDGET')
    require(cfg.get('budget_transport')=='clock_batching_fresh_only_v1' and cfg.get('fresh_only') is True and cfg.get('resume_allowed') is False,'FRESH_LEDGER_ONLY')
    require(cfg.get('runtime_allowed') is True and cfg.get('executable') is True and cfg.get('training_allowed') is False,'DATA_ONLY_CONFIG')
    require(cfg.get('cohort_membership') is None and cfg.get('auto_retry') is False,'NO_OLD_COHORT_OR_RETRY')
    require(cfg.get('factory_variant')=='winding_v1','UNCHANGED_FACTORY')
