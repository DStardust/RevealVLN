import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tested_required_scope',HERE/'family_scope_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def inventory(digest='1'*64):
    name='.'+digest+'.semantic.npy.'+'a'*32+'.partial'
    return {'actual_bytes_all_entries':100,'limit_bytes':200,'files':[{'name':name,'bytes':100,'sha256':'2'*64}],
            'problems':[{'name':name,'error':'FAILURE_PARTIAL_OR_FOREIGN_ENTRY'}]}

class Tests(unittest.TestCase):
    def test_allowed_unrelated_partial_remains_in_inventory(self):
        inv=inventory();out=m.unrelated_partials(inv,{'3'*64+'.semantic'},[b'{}'])
        self.assertEqual(out[0]['bytes'],100);self.assertTrue(out[0]['retained_in_original_store'])
        self.assertEqual(len(inv['problems']),1)
    def test_partial_target_in_index_rejected(self):
        with self.assertRaises(ValueError):m.unrelated_partials(inventory(),{'1'*64+'.semantic'},[b'{}'])
    def test_partial_target_in_history_or_query_rejected(self):
        with self.assertRaises(ValueError):m.unrelated_partials(inventory(),set(),[b'{"query": "'+b'1'*64+b'"}'])
    def test_partial_full_name_in_graph_rejected(self):
        inv=inventory()
        with self.assertRaises(ValueError):m.unrelated_partials(inv,set(),[inv['files'][0]['name'].encode()])
    def test_failure_log_not_exception(self):
        inv=inventory();inv['problems'][0]['name']='FAILURES.jsonl'
        with self.assertRaises(ValueError):m.unrelated_partials(inv,set(),[])
    def test_unknown_or_corrupt_content_not_exception(self):
        inv=inventory();inv['problems'][0]['error']='CONTENT_HASH'
        with self.assertRaises(ValueError):m.unrelated_partials(inv,set(),[])
    def test_uninventoried_partial_rejected(self):
        inv=inventory();inv['files']=[]
        with self.assertRaises(ValueError):m.unrelated_partials(inv,set(),[])
    def test_full_store_bytes_cap_unchanged(self):
        inv=inventory();inv['actual_bytes_all_entries']=201
        with self.assertRaises(ValueError):m.unrelated_partials(inv,set(),[])
    def test_ref_parser_includes_normalization_and_policy(self):
        doc={'normalization_events':[{'raw_record':{'semantic_hash':'1'*64}}],
             'policy':[{'rgb_ref':'sha256:'+'2'*64}]}
        self.assertEqual(m.pixel_refs(doc),{'1'*64+'.semantic','2'*64+'.rgb'})
    def test_malformed_ref_rejected(self):
        with self.assertRaises(ValueError):m.pixel_refs({'rgb_ref':'guessed'})

if __name__=='__main__':unittest.main()
