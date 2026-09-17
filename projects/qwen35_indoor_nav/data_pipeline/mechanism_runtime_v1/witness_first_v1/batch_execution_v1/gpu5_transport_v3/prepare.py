"""CPU-only V2 preparation, with separately frozen exact holder identity binding."""
import importlib.util
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport as t
spec = importlib.util.spec_from_file_location('gpu5_v3_exact_prepare',t.checked('prepare.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
original_adapter = p.adapted_prepare_source
def adapted(source):
    value = original_adapter(source)
    old = 'cfg.update(GPU5_TRANSPORT_METADATA)'
    assert value.count(old) == 1
    return value.replace(old, old+"\n    cfg['identity_binding_version']='gpu5_transport_v3'")
p.t = t
p.HERE = HERE
p.LAUNCHER = p.LAUNCHER.replace('gpu5_transport_v2/transport.py','gpu5_transport_v3/transport.py')
p.adapted_prepare_source = adapted

def prepare(snapshot,name,indices,authorization,holder_identity):
    # The original preparation also locks all V3 .py files. Add immutable V2
    # implementation closure through the prospective snapshot rather than
    # modifying any already sealed input file.
    source_lock = t.read(Path(snapshot)/'SOURCE_LOCK.json')
    for path in [t.checked('transport.py'),t.checked('prepare.py')]:
        t.require(source_lock.get(str(path)) == t.sha(path),'SNAPSHOT_V2_TRANSPORT_CLOSURE_REQUIRED')
    return p.prepare(snapshot,name,indices,authorization,holder_identity)

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--indices',type=int,nargs='+',required=True)
    parser.add_argument('--authorization',required=True);parser.add_argument('--holder-identity',required=True)
    args=parser.parse_args()
    prepare(args.snapshot,args.name,args.indices,args.authorization,args.holder_identity)
