"""Versioned CPU audit transport: source manifest count 1024 -> 2048 only.

Original acceptance and cohort gate files remain byte-for-byte unchanged.
No simulator/model/GPU calls; imports only the frozen CPU evidence stack.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
ACCEPTANCE=HERE.parent/'acceptance.py'
GATE_V2=HERE.parent/'gate_v2/gate.py'
ACCEPTANCE_SHA='a2cae535b132b652c289d753bcf11dc2f55413bb942fd731528198a8ab4fa7d8'
GATE_SHA='31a7271baeff1244a1ef825c0e3c64bd206077ae84ab121f547187b987840975'
OLD_EXPR="len(lock)<=1024 and sum(path_safe(p).stat().st_size for p in lock)<=32*1024**3"
NEW_EXPR="len(lock)<=2048 and sum(path_safe(p).stat().st_size for p in lock)<=32*1024**3"

def exact_source(source):
    if hashlib.sha256(source.encode()).hexdigest()!=ACCEPTANCE_SHA:raise ValueError('FROZEN_ACCEPTANCE_CHANGED')
    if source.count(OLD_EXPR)!=1 or source.count(NEW_EXPR)!=0:raise ValueError('EXACT_SINGLE_CAP_SUBSTITUTION')
    adapted=source.replace(OLD_EXPR,NEW_EXPR)
    if adapted.replace(NEW_EXPR,OLD_EXPR)!=source:raise ValueError('NONREVERSIBLE_SOURCE_TRANSFORM')
    return adapted

def load_stack():
    original=ACCEPTANCE.read_text();adapted=exact_source(original)
    raw=GATE_V2.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=GATE_SHA:raise ValueError('FROZEN_COHORT_GATE_CHANGED')
    # An isolated instance: never monkeypatch a module used by another audit.
    spec=importlib.util.spec_from_file_location('capacity_v3_private_original_cohort',GATE_V2)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    old=types.ModuleType('capacity_v3_private_acceptance')
    old.__file__=str(ACCEPTANCE)
    exec(compile(adapted,str(ACCEPTANCE)+'::manifest_capacity_v3','exec'),old.__dict__)
    gate.old=old;gate.require=old.require
    # Relocate only the new audit output boundary; original acceptance parent
    # remains its source root, and accepts this strictly-contained new folder.
    gate.HERE=HERE
    return gate,original,adapted

gate,ORIGINAL_SOURCE,ADAPTED_SOURCE=load_stack()
evaluate=gate.evaluate
validate_cohort=gate.validate_cohort

def audit_all(batch_root,output,recovery_index=None,cohort_manifest=None):
    result=gate.audit_all(batch_root,output,recovery_index,cohort_manifest)
    gate.old.save(Path(output)/'MANIFEST_CAPACITY_ADAPTER.json',{
        'version':'MANIFEST_CAPACITY_V3','original_acceptance_sha256':ACCEPTANCE_SHA,
        'adapted_acceptance_sha256':hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest(),
        'original_cohort_gate_sha256':GATE_SHA,'source_manifest_count_cap':2048,
        'source_total_bytes_cap':32*1024**3,'single_file_bytes_cap':8*1024**3,
        'source_transform_exact_reversible':True,'other_validation_source_unchanged':True,
        'cohort_thresholds_unchanged':True,'gpu_operations':0,'scientific_pass':False})
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--batch-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--recovery-index');p.add_argument('--cohort-manifest',required=True)
    a=p.parse_args();print(json.dumps(audit_all(a.batch_root,a.output,a.recovery_index,a.cohort_manifest),indent=2))
