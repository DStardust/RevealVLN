"""CPU-only; requires new explicit authorization and exact holder identity."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v5_prepare_private_transport',HERE/'transport.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
p=t.load('v5_private_v2_prepare',t.v4.v3.checked('prepare.py'))
original_adapter=p.adapted_prepare_source

def adapted_prepare_source(source):
    value=original_adapter(source)
    old='cfg.update(GPU5_TRANSPORT_METADATA)'
    lines=[old,"cfg['identity_binding_version']='gpu5_transport_v3'",
        "cfg['input_manifest_capacity_version']='gpu5_transport_v4_2048'",
        "cfg['runtime_transport_version']='gpu5_transport_v5'",
        'cfg[\'sampling_amendment\']='+repr(t.AMENDMENT),
        "cfg.update(thresholds_unchanged=True,supervision_sampling_policy_changed=True,cohort_member=False,cohort_replacement=False,engineering_retry_count=1)",
        "cfg['engineering_retry_of']="+repr(str(t.base.BE/'batch_06r1/run_v1'))]
    assert value.count(old)==1;return value.replace(old,'\n    '.join(lines))

def extended_prepare_function(source):
    start=source.index('def prepare(snapshot,name,indices,authorization,holder_identity):')
    value=source[start:source.index("\nif __name__=='__main__':",start)]
    old="t.BE/'BATCH02_MAIN_RECOVERY_RECEIPT_V1.json',t.BE/'gpu5_transport_v1/SHA256SUMS']"
    new=old+'\n    module.GPU5_TRANSPORT_LOCK_PATHS += t.dependency_paths()'
    assert value.count(old)==1
    return value.replace(old,new)

p.t=t;p.HERE=HERE
p.LAUNCHER=p.LAUNCHER.replace('gpu5_transport_v2/transport.py','gpu5_transport_v5/transport.py')
p.adapted_prepare_source=adapted_prepare_source
exec(compile(extended_prepare_function(t.v4.v3.checked('prepare.py').read_text()),str(HERE/'prepare.py'),'exec'),p.__dict__)

def prepare(snapshot,name,indices,authorization,holder_identity):
    assert name=='batch_06r2' and indices==[9,10,11]
    assert Path(snapshot).resolve()==t.base.WF/'multi_program_bank_v1/language_ready_v2'
    # Old snapshot is unchanged. Fresh INPUT_LOCK adds this exact full adapter
    # chain and wrapper before the first GPU action; no old lock is amended.
    return p.prepare(snapshot,name,indices,authorization,holder_identity)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--indices',type=int,nargs='+',required=True);parser.add_argument('--authorization',required=True)
    parser.add_argument('--holder-identity',required=True);a=parser.parse_args()
    prepare(a.snapshot,a.name,a.indices,a.authorization,a.holder_identity)
