"""CPU helper/negative acceptance tests; synthetic fixtures are not real families."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tested_recovery',HERE/'recovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def journal_rows(payloads,cfg):
    previous='0'*64;rows=[]
    for seq,(kind,payload) in enumerate([('__config__',cfg)]+payloads):
        body={'seq':seq,'kind':kind,'payload':payload,'prev':previous}
        h=hashlib.sha256(m.old.canonical(body)).hexdigest();rows.append(dict(body,hash=h));previous=h
    raw=b''.join(m.old.canonical(r)+b'\n' for r in rows)
    head={'version':1,'count':len(rows),'last_hash':previous,'byte_length':len(raw),
          'config_hash':hashlib.sha256(m.old.canonical(cfg)).hexdigest()}
    return raw,head,rows

def family_records():
    rows=[]
    def append(kind,value):rows.append({'seq':len(rows),'kind':kind,'payload':copy.deepcopy(value)})
    limits=dict(total_actions=10,total_seconds=10,discovery_actions=5,discovery_seconds=5,
                certification_actions=5,certification_seconds=5)
    ledger=m.sem.bridge.BudgetLedger(limits,clock=lambda:1.0,persist=lambda s:append('budget',s))
    candidate={'dummy':'CPU_interface_only'}
    ledger.start_bundle('x','discovery');ledger.reserve_action();append('action_completed',{'bundle':'x'})
    append('freeze',{'x':{'candidate':candidate,'status':'frozen'}})
    ledger.finish_phase();ledger.start_bundle('x','certification');ledger.reserve_action();append('action_completed',{'bundle':'x'})
    for i in range(27):append('trace_saved',{'bundle':'x','index':i,'complete':True,'sha256':'0'*64})
    append('freeze',{'x':{'candidate':candidate,'status':'certified'}});ledger.finish_phase()
    return rows,candidate

def npy(raw):
    header=repr({'descr':'|u1','fortran_order':False,'shape':(224,224,3)}).encode('ascii')
    header+=b' '*((-10-len(header)-1)%64)+b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+raw

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cpu_test_',dir=HERE);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_prefix_ignores_only_explicit_uncommitted_tail(self):
        raw,head,rows=journal_rows([('a',{})],{})
        got,report=m.committed_prefix(raw+b'{partial',head,{})
        self.assertEqual(got,rows);self.assertEqual(report['uncommitted_tail_bytes'],8)
        self.assertFalse(report['tail_used_for_labels_or_budget'])
    def test_bad_prefix_hash_rejected(self):
        raw,head,_=journal_rows([('a',{})],{})
        with self.assertRaises(ValueError):m.committed_prefix(raw.replace(b'"a"',b'"b"'),head,{})
    def test_truncated_head_range_rejected(self):
        raw,head,_=journal_rows([],{})
        with self.assertRaises(ValueError):m.committed_prefix(raw[:-1],head,{})
    def test_config_mismatch_rejected(self):
        raw,head,_=journal_rows([],{})
        with self.assertRaises(ValueError):m.committed_prefix(raw,head,{'changed':True})
    def test_completed_budget_and_prior_reservations(self):
        rows,candidate=family_records();phase,evidence=m.family_completion(rows,'x',candidate)
        self.assertEqual(len(phase['certification_trace_records']),27)
        self.assertEqual(evidence['confirmed_actions_by_phase'],dict(discovery=1,certification=1))
    def test_incomplete_family_not_accepted(self):
        rows,candidate=family_records()
        with self.assertRaises(ValueError):m.family_completion(rows[:-1],'x',candidate)
    def test_twenty_six_replays_rejected(self):
        rows,candidate=family_records();rows=[r for r in rows if not(r['kind']=='trace_saved' and r['payload']['index']==26)]
        for i,r in enumerate(rows):r['seq']=i
        with self.assertRaises(ValueError):m.family_completion(rows,'x',candidate)
    def test_action_without_reservation_rejected(self):
        rows,candidate=family_records();action=next(i for i,r in enumerate(rows) if r['kind']=='action_completed')
        rows[action]['payload']['bundle']='foreign'
        with self.assertRaises(ValueError):m.family_completion(rows,'x',candidate)
    def test_reader_does_not_modify_bytes(self):
        p=self.root/'data';p.write_bytes(b'content');before=p.stat().st_mtime_ns
        reader=m.Reader();self.assertEqual(reader.read(p),b'content');reader.verify_again()
        self.assertEqual(p.stat().st_mtime_ns,before)
    def test_reader_rejects_between_read_mutation(self):
        p=self.root/'data';p.write_bytes(b'a');reader=m.Reader();reader.read(p);p.write_bytes(b'b')
        with self.assertRaises(ValueError):reader.verify_again()
    def test_symlink_rejected(self):
        p=self.root/'data';p.write_bytes(b'a');link=self.root/'link';link.symlink_to(p)
        with self.assertRaises(ValueError):m.Reader().read(link)
    def make_store(self):
        store=self.root/'content';store.mkdir();raw=b'\x05'*(224*224*3)
        (store/(hashlib.sha256(raw).hexdigest()+'.rgb.npy')).write_bytes(npy(raw));return store
    def test_readonly_inventory_valid(self):
        value=m.scan_store(self.make_store(),m.Reader())
        self.assertTrue(value['content_inventory_pass']);self.assertTrue(value['not_original_store_close'])
        self.assertFalse(value['original_poisoned_state_known'])
    def test_partial_counts_and_rejects(self):
        store=self.make_store();(store/'x.partial').write_bytes(b'12345')
        value=m.scan_store(store,m.Reader());self.assertFalse(value['content_inventory_pass'])
        self.assertEqual(value['actual_bytes_all_entries'],sum(p.stat().st_size for p in store.iterdir()))
    def test_capacity_rejects(self):
        self.assertFalse(m.scan_store(self.make_store(),m.Reader(),limit=1)['content_inventory_pass'])
    def test_wrong_pixel_hash_rejected(self):
        store=self.make_store();p=next(store.iterdir());p.write_bytes(p.read_bytes()[:-1]+b'\x06')
        self.assertFalse(m.scan_store(store,m.Reader())['content_inventory_pass'])
    def test_fake_incomplete_source_never_passes(self):
        run=self.root/'missing_run';run.mkdir();out=self.root/'audit'
        result=m.audit_recovery(run,'x',out)
        self.assertFalse(result['recovery_content_pass']);self.assertFalse(result['quality_pass'])
        self.assertFalse(result['training_admission']);self.assertFalse(result['original_batch_pass'])

if __name__=='__main__':unittest.main()
