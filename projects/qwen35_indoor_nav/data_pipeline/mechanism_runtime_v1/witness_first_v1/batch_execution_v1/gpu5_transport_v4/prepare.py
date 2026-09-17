"""Prospective V4 preparation. Source, sampling and physical budgets unchanged."""
import importlib.util
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport as t
spec=importlib.util.spec_from_file_location('gpu5_v4_v3_prepare',t.checked_v3('prepare.py'))
p3=importlib.util.module_from_spec(spec);spec.loader.exec_module(p3)
previous_adapter=p3.p.adapted_prepare_source
def adapted(source):
    value=previous_adapter(source)
    old="cfg['identity_binding_version']='gpu5_transport_v3'"
    assert value.count(old)==1
    return value.replace(old,old+"\n    cfg['input_manifest_capacity_version']='gpu5_transport_v4_2048'")
p3.p.t=t
p3.p.HERE=HERE
p3.p.LAUNCHER=p3.p.LAUNCHER.replace('gpu5_transport_v3/transport.py','gpu5_transport_v4/transport.py')
p3.p.adapted_prepare_source=adapted
def prepare(snapshot,name,indices,authorization,holder_identity):
    lock=t.read(Path(snapshot)/'SOURCE_LOCK.json')
    for path in map(t.checked_v3,t.V3_HASHES):
        assert lock.get(str(path))==t.sha(path),'V3_SNAPSHOT_CLOSURE_REQUIRED'
    return p3.prepare(snapshot,name,indices,authorization,holder_identity)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--indices',type=int,nargs='+',required=True)
    parser.add_argument('--authorization',required=True);parser.add_argument('--holder-identity',required=True)
    args=parser.parse_args();prepare(args.snapshot,args.name,args.indices,args.authorization,args.holder_identity)
