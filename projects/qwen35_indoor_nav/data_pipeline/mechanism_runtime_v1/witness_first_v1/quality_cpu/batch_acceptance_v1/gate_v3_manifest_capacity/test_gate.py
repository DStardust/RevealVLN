"""Differential tests are CPU fixtures, never physical evidence or a cohort PASS."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import types
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('capacity_v3_test',HERE/'gate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def batch(n):
    return {'run_root':'batch'+str(n),'bindings_verified':True,'all_frozen_candidates_attempted_and_terminal':True,
        'attempts':[{'house_id':'house'+str(n),'hub_position':[2.*i,0.,0.],'quality_pass':True,
        'source_and_phase_binding_verified':True,'control_type':'completed_subgoal_revisit_placement_not_event_free_detour'} for i in range(3)]}

def source_cap_run(source,count,size=1,corrupt=False):
    # Execute the actual inspect_run function through the entire SHA loop, then
    # stop at the journal boundary. Nothing after this is claimed tested here.
    f=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='inspect_run')
    class P:
        def __init__(self,name):self.name=name
        def __truediv__(self,name):return P(self.name+'/'+name)
        def __str__(self):return self.name
        def is_relative_to(self,other):return True
        def stat(self):return types.SimpleNamespace(st_size=size,st_mtime=0)
        def read_bytes(self):return b''
    config='run/EXECUTION_CONFIG.json'
    lock={config:'good',**{str(i):'good' for i in range(count-1)}}
    if corrupt:lock[str(count-2)]='bad'
    values={'EXECUTION_CONFIG.json':{},'INPUT_LOCK.json':lock,'PROCESS.json':{'started_unix':1}}
    calls=[]
    def sha(p):calls.append(str(p));return 'good'
    def require(ok,why):
        if not ok:raise ValueError(why)
    def stop(*args):raise RuntimeError('REACHED_JOURNAL_AFTER_FULL_SOURCE_CHECK')
    namespace={'path_safe':lambda p:p if isinstance(p,P) else P(p),'require':require,'LINE':None,
        'read':lambda p:values.get(p.name.rsplit('/',1)[-1],{}),'sha':sha,'parse_journal':stop}
    exec(compile(ast.Module(body=[f],type_ignores=[]),'<CPU_INSPECT_RUN_FIXTURE>','exec'),namespace)
    try:namespace['inspect_run'](P('run'))
    except (ValueError,RuntimeError) as e:return str(e),calls
    raise AssertionError('UNEXPECTED_FULL_FIXTURE_PASS')

class Tests(unittest.TestCase):
    def test_reverse_exact_whole_source(self):
        self.assertEqual(m.ADAPTED_SOURCE.replace(m.NEW_EXPR,m.OLD_EXPR),m.ORIGINAL_SOURCE)
        self.assertEqual(m.ADAPTED_SOURCE.count(m.NEW_EXPR),1)
    def test_original_disk_hashes_unchanged(self):
        self.assertEqual(hashlib.sha256(m.ACCEPTANCE.read_bytes()).hexdigest(),m.ACCEPTANCE_SHA)
        self.assertEqual(hashlib.sha256(m.GATE_V2.read_bytes()).hexdigest(),m.GATE_SHA)
    def test_changed_original_refused(self):
        with self.assertRaises(ValueError):m.exact_source(m.ORIGINAL_SOURCE+'\n')
    def test_below_old_cap_identical(self):
        for count in (1,12,1024):self.assertEqual(source_cap_run(m.ORIGINAL_SOURCE,count),source_cap_run(m.ADAPTED_SOURCE,count))
    def test_new_range_only_expanded(self):
        for count in (1025,1026,2048):
            self.assertEqual(source_cap_run(m.ORIGINAL_SOURCE,count)[0],'SOURCE_READ_CAP')
            why,calls=source_cap_run(m.ADAPTED_SOURCE,count)
            self.assertEqual(why,'REACHED_JOURNAL_AFTER_FULL_SOURCE_CHECK');self.assertEqual(len(calls),count+1)
    def test_2049_refused(self):self.assertEqual(source_cap_run(m.ADAPTED_SOURCE,2049)[0],'SOURCE_READ_CAP')
    def test_32gib_still_exact(self):
        self.assertEqual(source_cap_run(m.ADAPTED_SOURCE,2048,16*1024**2)[0],'REACHED_JOURNAL_AFTER_FULL_SOURCE_CHECK')
        self.assertEqual(source_cap_run(m.ADAPTED_SOURCE,2048,16*1024**2+1)[0],'SOURCE_READ_CAP')
    def test_sha_corruption_refused_inside_new_range(self):
        self.assertTrue(source_cap_run(m.ADAPTED_SOURCE,1026,corrupt=True)[0].startswith('LOCKED_SOURCE_CHANGED:'))
    def test_original_sha_corruption_identical(self):
        self.assertEqual(source_cap_run(m.ORIGINAL_SOURCE,12,corrupt=True),source_cap_run(m.ADAPTED_SOURCE,12,corrupt=True))
    def test_output_scope_new_child(self):
        self.assertEqual(m.gate.HERE,HERE);self.assertTrue(HERE.is_relative_to(m.gate.old.HERE))
    def test_cohort_functions_bytecode_unchanged(self):
        original=types.ModuleType('untouched_gate_comparison');original.__file__=str(m.GATE_V2)
        exec(compile(m.GATE_V2.read_text(),str(m.GATE_V2),'exec'),original.__dict__)
        for name in ('evaluate','validate_cohort','observe_run','audit_all'):
            self.assertEqual(getattr(original,name).__code__.co_code,getattr(m.gate,name).__code__.co_code)
        rows=[batch(i) for i in range(3)]
        self.assertEqual(original.evaluate(rows),m.evaluate(rows))
        rows[0]['bindings_verified']=False;self.assertEqual(original.evaluate(rows),m.evaluate(rows))
    def test_failure_and_hub_requirements_still_enforced(self):
        rows=[batch(i) for i in range(3)];rows[0]['attempts'][1]['hub_position']=[0.,0.,0.]
        self.assertFalse(m.evaluate(rows)['preliminary_cross_house_production_pass'])
        self.assertFalse(m.evaluate([batch(0)])['preliminary_cross_house_production_pass'])

if __name__=='__main__':unittest.main()
