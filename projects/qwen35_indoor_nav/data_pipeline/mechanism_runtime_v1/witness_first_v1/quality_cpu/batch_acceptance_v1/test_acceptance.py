"""CPU-only rule/chain/resource tests; no invented qualified physical data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
if __package__:
    from . import acceptance as a
else:
    spec=importlib.util.spec_from_file_location('batch_acceptance_cpu',Path(__file__).resolve().with_name('acceptance.py'))
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def frozen_journal(events):
    prev='0'*64;lines=[]
    for i,(kind,payload) in enumerate(events):
        body={'seq':i,'prev':prev,'kind':kind,'payload':payload};prev=hashlib.sha256(a.canonical(body)).hexdigest()
        lines.append(a.canonical(dict(body,hash=prev)))
    raw=b'\n'.join(lines)+b'\n'
    head={'count':len(lines),'byte_length':len(raw),'last_hash':prev,'config_hash':hashlib.sha256(a.canonical(events[0][1])).hexdigest()}
    return raw,head

def batches():
    rows=[{'house_id':'h'+str(i//2),'hub_position':[float(i%2),0.,0.],'quality_pass':True,
           'source_and_phase_binding_verified':True,'control_type':'completed_subgoal_revisit_placement_not_event_free_detour'} for i in range(6)]
    return [{'bindings_verified':True,'attempts':rows[:3]},{'bindings_verified':True,'attempts':rows[3:]}]

class BatchTests(unittest.TestCase):
    def test_six_hubs_and_three_houses_preliminary_only(self):
        report=a.gate_from_verified_batches(batches())
        self.assertTrue(report['preliminary_cross_house_production_pass'])
        self.assertFalse(report['statistical_stability_pass']);self.assertFalse(report['model_generalization_pass'])
    def test_exactly_one_meter_distinct_below_one_not_distinct(self):
        rows=batches()[0]['attempts'][:2]
        self.assertEqual(len(a.independent_hubs(rows)),2)
        rows[1]['hub_position'][0]=.999999
        self.assertEqual(len(a.independent_hubs(rows)),1)
    def test_role_or_seed_renaming_not_new_hub(self):
        b=batches();b[0]['attempts'][1]['hub_position']=[0.,0.,0.]
        self.assertFalse(a.gate_from_verified_batches(b)['preliminary_cross_house_production_pass'])
    def test_connected_overlap_is_conservative(self):
        rows=[{'house_id':'h','hub_position':[x,0.,0.]} for x in (0.,.8,1.6)]
        self.assertEqual(len(a.independent_hubs(rows)),1)
    def test_each_batch_requires_three_attempted_hubs(self):
        b=batches();b[0]['attempts'].pop()
        self.assertIn('BATCH_NEEDS_THREE_ATTEMPTED_DISTINCT_HUBS',a.gate_from_verified_batches(b)['errors'])
    def test_unverified_or_unknown_reports_cannot_pass(self):
        b=batches();b[0]['attempts'][0]['source_and_phase_binding_verified']=False
        self.assertFalse(a.gate_from_verified_batches(b)['preliminary_cross_house_production_pass'])
        b=batches();b[1]['bindings_verified']=False
        self.assertFalse(a.gate_from_verified_batches(b)['preliminary_cross_house_production_pass'])
    def test_control_types_retained(self):
        b=batches();b[0]['attempts'][0]['control_type']='event_free'
        self.assertEqual(len(a.gate_from_verified_batches(b)['control_type_counts']),2)
    def test_invalid_hub_rejected(self):
        with self.assertRaises(ValueError):a.independent_hubs([{'house_id':'h','hub_position':[float('nan'),0,0]}])
    def test_actual_certification_phase_selected_not_discovery(self):
        cfg={'fixture':True};candidate={'candidate_hash':'fixture_not_physical'}
        events=[('__config__',cfg),('budget',{'active':['x','discovery']}),
                ('trace_saved',{'bundle':'x','index':0,'complete':True}),
                ('freeze',{'x':{'candidate':candidate}}),('budget',{'active':['x','certification']}),
                ('action_completed',{'bundle':'x','value':{'action':'F'}}),
                ('trace_saved',{'bundle':'x','index':1,'complete':True}),('budget',{'active':None})]
        raw,head=frozen_journal(events);records=a.parse_journal(raw,head,cfg);phase=a.phase_evidence(records,'x')
        self.assertEqual([r['index'] for r in phase['certification_trace_records']],[1])
        self.assertLess(phase['frozen_seq'],phase['certification_started_seq'])
    def test_config_and_chain_tamper_rejected(self):
        raw,head=frozen_journal([('__config__',{'x':1})])
        with self.assertRaises(ValueError):a.parse_journal(raw,head,{'x':2})
        with self.assertRaises(ValueError):a.parse_journal(raw.replace(b'"x":1',b'"x":2'),head,{'x':2})
    def test_truncated_tail_rejected(self):
        raw,head=frozen_journal([('__config__',{'x':1})])
        with self.assertRaises(ValueError):a.parse_journal(raw[:-1],head,{'x':1})
    def test_graphics_renderer_resource_and_cleanup(self):
        sample={'uuid':'gpu','memory_mib':1500,'processes':{'5':{'mib':1000,'type':'G'},'8':{'mib':246,'type':'C'}}}
        self.assertFalse(a.gpu_snapshot(sample,5,'gpu')['own_renderer_absent'])
        sample['processes'].pop('5');sample['memory_mib']=300
        self.assertTrue(a.gpu_snapshot(sample,5,'gpu')['own_renderer_absent'])
    def test_external_or_unknown_memory_not_accepted(self):
        sample={'uuid':'gpu','memory_mib':1500,'processes':{'8':{'mib':769,'type':'G'}}}
        with self.assertRaises(ValueError):a.gpu_snapshot(sample,5,'gpu')
        sample['processes']['8']={'mib':100,'type':None}
        with self.assertRaises(ValueError):a.gpu_snapshot(sample,5,'gpu')
    def test_builder_missing_closed_run_fails_without_false_pass(self):
        with tempfile.TemporaryDirectory(prefix='batch_cpu_',dir=a.HERE) as tmp:
            tmp=Path(tmp);report=a.build_family_evidence(tmp/'missing_run','x',tmp/'review')
            self.assertFalse(report['quality_pass']);self.assertTrue(report['errors'])
            self.assertTrue((tmp/'review/FAMILY_REPORT.json').is_file())

if __name__=='__main__':unittest.main()
