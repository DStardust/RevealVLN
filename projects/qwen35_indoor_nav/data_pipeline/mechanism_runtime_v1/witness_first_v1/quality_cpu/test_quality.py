"""Synthetic rule tests and old sealed interface readback, never physical gain."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
if __package__:
    from . import quality as q
else:
    spec=importlib.util.spec_from_file_location('quality_cpu_test',Path(__file__).resolve().with_name('quality.py'))
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

def fixture():
    histories={'H_A':list('FLRF'),'H_B':list('RFFL'),'H_A_I':list('LFRF')}
    cells=[]
    for h in histories:
        for t in ('task_A','task_B'):
            for c in ('C0','C_A','C_B'):
                y=(h!='H_B' or c=='C_A') if t=='task_A' else (h=='H_B' or c=='C_B')
                cells.append(dict(history_id=h,task_id=t,continuation_id=c,y=int(y)))
    return histories,cells

def batch_fixture():
    families={str(i):{'quality_pass':True,'evidence_verified':True,'house_id':'house'+str(i//2),
                      'hub_position':[float((i%2)*2),0.,0.],
                      'control_type':'completed_subgoal_revisit_placement_not_event_free_detour'} for i in range(6)}
    batches=[]
    for ids in (['0','1','2'],['3','4','5']):
        batches.append({'freeze_unix':1,'sealed_input_verified':True,'frozen_attempt_ids':ids,
                        'attempts':[{'id':i,'started_unix':2} for i in ids]})
    return batches,families

class QualityTests(unittest.TestCase):
    def test_matrix_all_axes_and_counts(self):
        result=q.matrix_checks(*fixture());self.assertTrue(result['history_reversals']);self.assertTrue(result['task_differences'])
    def test_count_shortcut_rejected(self):
        h,c=fixture();h['H_A'].append('F')
        with self.assertRaisesRegex(ValueError,'ACTION_COUNT'):q.matrix_checks(h,c)
    def test_unknown_label_not_negative(self):
        h,c=fixture();c[0]['y']=None
        with self.assertRaises(ValueError):q.matrix_checks(h,c)
    def test_no_history_reversal_rejected(self):
        h,c=fixture()
        for row in c:row['y']=int(row['task_id']=='task_A')
        with self.assertRaisesRegex(ValueError,'NO_HISTORY'):q.matrix_checks(h,c)
    def test_a_control_invariance_required(self):
        h,c=fixture();c[-1]['y']=1-c[-1]['y']
        with self.assertRaisesRegex(ValueError,'A_CONTROL'):q.matrix_checks(h,c)
    def test_no_task_difference_rejected(self):
        h,c=fixture()
        for row in c:row['y']=int(row['history_id']!='H_B')
        with self.assertRaisesRegex(ValueError,'NO_TASK'):q.matrix_checks(h,c)
    def test_missing_external_seal_never_pass(self):
        report=q.audit_family(q.HERE,{})
        self.assertFalse(report['quality_pass']);self.assertEqual(report['grade'],'MISSING_OR_REJECTED_EVIDENCE')
    def test_six_hubs_three_houses_is_only_draft(self):
        result=q.evaluate_batch_draft(*batch_fixture())
        self.assertTrue(result['draft_threshold_met']);self.assertFalse(result['engineering_pass']);self.assertFalse(result['scientific_pass'])
    def test_role_or_seed_relabel_same_hub_does_not_count(self):
        b,f=batch_fixture();f['1']['hub_position']=f['0']['hub_position'][:]
        result=q.evaluate_batch_draft(b,f)
        self.assertFalse(result['draft_threshold_met']);self.assertEqual(result['distinct_physical_hubs'],5)
    def test_nearby_numeric_changes_do_not_create_new_hub(self):
        b,f=batch_fixture();f['1']['hub_position']=[.000001,0,0]
        self.assertFalse(q.evaluate_batch_draft(b,f)['draft_threshold_met'])
    def test_late_freeze_or_missing_attempt_rejected(self):
        b,f=batch_fixture();b[0]['freeze_unix']=3
        self.assertFalse(q.evaluate_batch_draft(b,f)['draft_threshold_met'])
        b,f=batch_fixture();b[0]['attempts'].pop()
        self.assertFalse(q.evaluate_batch_draft(b,f)['draft_threshold_met'])
    def test_unknown_quality_is_not_accepted(self):
        b,f=batch_fixture();f['0']['quality_pass']=None
        self.assertFalse(q.evaluate_batch_draft(b,f)['draft_threshold_met'])
    def test_control_types_separately_counted(self):
        b,f=batch_fixture();f['1']['control_type']='spatial_detour'
        result=q.evaluate_batch_draft(b,f)
        self.assertEqual(len(result['control_type_counts']),2);self.assertFalse(result['mixed_control_types_pooled_as_same_mechanism'])
    def test_actual_original_loader_compiler_old_interface_only(self):
        root=q.RUNTIME/'smoke_v1/export_v4';manifest=json.loads((root/'MANIFEST.json').read_text())
        cfg=manifest['compiler_config'];compiler=q.Compiler(cfg['roles'],cfg['tasks'],cfg['eligible'],task_revision=cfg['task_revision'])
        loader=q.FamilyLoader(root,compiler);result=loader.validate_supervision_contract()
        self.assertEqual(result['labels_recomputed'],18);self.assertEqual(manifest['split'],'interface_only')
        self.assertFalse(q.audit_family(root,{})['quality_pass'])
        prefix=next(iter(loader.prefix_index));record=loader.prefix_records(prefix)[-1]
        self.assertNotIn('query',loader.policy_payload(record))
        bad=copy.deepcopy(record);bad['query']={'future':True}
        with self.assertRaises(ValueError):loader.policy_payload(bad)
        bad=copy.deepcopy(record);bad['observations'][-1]['step']+=1
        with self.assertRaises(ValueError):loader.policy_payload(bad)

if __name__=='__main__':unittest.main()
