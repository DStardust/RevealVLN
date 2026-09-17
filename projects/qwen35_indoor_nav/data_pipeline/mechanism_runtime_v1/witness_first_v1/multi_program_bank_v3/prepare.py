"""Read-only V2 bank algorithm adapter for the next closed three-house source."""
import hashlib
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
WF=HERE.parent
PREVIOUS=WF/'multi_program_bank_v2/prepare.py'
PREVIOUS_SHA='61e2e9dc840c11f5fb1f1dfdec46579f9e2f78324f2010a02871f4a5257e5b02'

def patches():
    return [
        ("SCOUT=WF/'bulk_source_v1/shard_00/run_v1'","SCOUT=WF/'bulk_source_v1/next_shard_runtime_v1/run_v1'"),
        ("HOUSES=['5q7pvUzZiYa','759xd9YjKW5','7y3sRwLe3Va']","HOUSES=['8WUmhLawc2A','D7N2EKCX4Sj','E9uDoFAP3SH']"),
        ("# shard_00 runtime binds its parent's SOURCE_LOCK, not a run INPUT_LOCK.","# next-shard runtime copies its parent SOURCE_LOCK into run INPUT_LOCK."),
        ("manifest=source.parents[1]/'SOURCE_LOCK.json'","manifest=source.parent/'SOURCE_LOCK.json'"),
        ("lock=read(manifest);assert len(lock)<=2048","lock=read(manifest);assert len(lock)<=2048\n    assert read(source/'INPUT_LOCK.json')==lock,'EXACT_RUNTIME_INPUT_MANIFEST'"),
        ("approval=source.parents[1]/'MAIN_AGENT_APPROVAL_SHARD_00.json'","approval=source.parent/'MAIN_AGENT_APPROVAL.json'"),
        ("{'approved':True,'shard':0,'gpu':1,'source_lock_sha256':sha(manifest)}","{'approved':True,'shard':1,'gpu':2,'source_lock_sha256':sha(manifest)}"),
        ("for p in (manifest,approval,source/'EXECUTION_CONFIG.json'):","for p in (manifest,approval,source/'EXECUTION_CONFIG.json',source/'INPUT_LOCK.json'):"),
        ("for p in (manifest,approval):lock[str(p)]=sha(p)","for p in (manifest,approval,source/'INPUT_LOCK.json'):lock[str(p)]=sha(p)"),
        ("sha(SCOUT.parents[1]/'SOURCE_LOCK.json')","sha(SCOUT.parent/'SOURCE_LOCK.json')"),
        ("Q35N_MULTI_PROGRAM_BANK_V2_CLOSED_NEW_HOUSES","Q35N_MULTI_PROGRAM_BANK_V3_CLOSED_NEW_HOUSES"),
        ("Q35N_MULTI_PROGRAM_BANK_V2_FINITE_LANGUAGE","Q35N_MULTI_PROGRAM_BANK_V3_FINITE_LANGUAGE"),
        ("deps=[source,OLD/'core.py'","deps=[Path("+repr(str(PREVIOUS))+"),Path("+repr(str(PREVIOUS.parent/'SHA256SUMS'))+"),source,OLD/'core.py'"),
    ]
def adapt(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PREVIOUS_SHA,'FROZEN_V2_PREPARE_CHANGED'
    value=source
    for old,new in patches():
        assert value.count(old)==1,(old,value.count(old));value=value.replace(old,new)
    reverse=value
    for old,new in reversed(patches()):reverse=reverse.replace(new,old)
    assert reverse==source,'NONREVERSIBLE_SOURCE_ONLY_TRANSPORT'
    return value
def load():
    module=types.ModuleType('multi_program_bank_v3_private_v2');module.__file__=str(HERE/'prepare.py')
    exec(compile(adapt(PREVIOUS.read_text()),str(PREVIOUS)+'::closed_next_source_v3','exec'),module.__dict__)
    return module
def main():load().main()
if __name__=='__main__':main()
