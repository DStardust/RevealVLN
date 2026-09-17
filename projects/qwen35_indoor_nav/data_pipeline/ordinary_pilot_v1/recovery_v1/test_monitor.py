"""Bounded synthetic regression for atomic-file disappearance, no GPU calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('recovery_runner',OUT/'run.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class Item:
    def __init__(self,value):self.value=value
    def is_file(self):return True
    def stat(self):
        if isinstance(self.value,Exception):raise self.value
        return SimpleNamespace(st_size=self.value)


class Tree:
    def __init__(self,values):self.values=values
    def rglob(self,pattern):return (Item(v) for v in self.values)


cases=0
for n in range(1,51):
    m.PARENT=Tree([n,FileNotFoundError('atomic promotion'),2*n])
    assert m.disk_size()==3*n;cases+=1
m.PARENT=Tree([PermissionError('not a permitted race')])
try:m.disk_size()
except PermissionError:cases+=1
else:raise AssertionError('Unexpected errors must not be hidden')
with (OUT/'MONITOR_REGRESSION_RESULT.json').open('x') as f:
    json.dump({'pass':True,'synthetic_cases':cases,'gpu_called':False},f,indent=2)
print(f'{cases} monitor regression tests passed')
