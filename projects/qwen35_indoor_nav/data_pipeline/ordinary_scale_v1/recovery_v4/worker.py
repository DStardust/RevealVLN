"""Single explicit AST GPU transport 2 -> 5; same frozen jobs and auditor."""
import ast
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def transport(source):
    tree=ast.parse(source)
    changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            target=node.targets[0]
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='cfg' and target.attr=='gpu_device_id':
                assert isinstance(node.value,ast.Constant) and node.value.value==2,'UNEXPECTED_OLD_DEVICE'
                node.value=ast.Constant(value=5);changed.append(node.lineno)
    assert len(changed)==1,('TRANSPORT_ASSIGNMENT_COUNT',changed)
    return ast.fix_missing_locations(tree),changed

def main():
    load('v4_preflight',HERE/'preflight.py').verify()
    source=BASE/'worker.py'
    tree,changed=transport(source.read_text())
    module=types.ModuleType('ordinary_gpu5_transport_v4')
    module.__file__=str(source)
    exec(compile(tree,str(source),'exec'),module.__dict__)
    module.audit=load('unchanged_strict_quarantine',BASE/'recovery_v1/audit.py')
    module.audit.HERE=HERE
    def atomic(name,obj):
        target=HERE/name;pending=target.with_suffix(target.suffix+'.pending')
        with pending.open('w') as handle:json.dump(obj,handle,indent=2,allow_nan=False)
        pending.replace(target)
    module.atomic=atomic
    with (HERE/'TRANSPORT_APPLIED.json').open('x') as handle:
        json.dump(dict(source=str(source),old_gpu=2,new_gpu=5,changed_lines=changed,assignments_changed=1),handle,indent=2)
    module.main()

if __name__=='__main__':main()
