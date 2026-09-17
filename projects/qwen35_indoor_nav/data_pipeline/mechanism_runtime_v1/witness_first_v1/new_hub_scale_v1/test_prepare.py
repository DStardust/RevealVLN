import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load():
    s=importlib.util.spec_from_file_location('new_hub_scale_cpu',HERE/'prepare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=load()
class HubScaleTests(unittest.TestCase):
    def test_no_gpu_import(self):
        with patch('subprocess.run',side_effect=AssertionError('NO_PROCESS')),patch('subprocess.check_output',side_effect=AssertionError('NO_GPU')):load()
    def test_exact_four_budget(self):
        with self.assertRaises(AssertionError):p.select([],[],5)
    def test_determinism(self):
        positions=[[x,0,z] for x in range(5) for z in range(5)]
        self.assertEqual(p.select(positions,[]),p.select(positions,[]))
    def test_existing_hubs_excluded(self):
        selected,ledger=p.select([[0,0,0],[.9,0,0],[1,0,0],[4,0,0],[8,0,0],[12,0,0]],[[0,0,0]])
        self.assertEqual(len(selected),4);self.assertTrue(all(math.dist(r['position'],[0,0,0])>=1 for r in selected))
    def test_new_selected_hubs_separated(self):
        selected,_=p.select([[x/4,0,0] for x in range(50)],[])
        self.assertEqual(len(selected),4)
        self.assertTrue(all(math.dist(a['position'],b['position'])>=1 for i,a in enumerate(selected) for b in selected[:i]))
    def test_yaw_duplicates_not_new_hubs(self):
        selected,ledger=p.select([[0,0,0]]*50,[]);self.assertEqual(len(selected),1)
        self.assertEqual(sum(x['status']=='DUPLICATE_OFFICIAL_POSITION' for x in ledger),49)
    def test_three_dimensional_separation(self):
        selected,_=p.select([[0,y,0] for y in (0,3,6,9)],[]);self.assertEqual(len(selected),4)
    def test_nonfinite_rejected(self):
        with self.assertRaises(AssertionError):p.select([[float('nan'),0,0]],[])
    def test_live_reachability_never_claimed(self):
        selected,_=p.select([[x,0,0] for x in range(8)],[])
        self.assertTrue(all(r['status']=='PROSPECTIVE_OFFICIAL_POSITION_LIVE_NAVMESH_AND_WITNESS_UNTESTED' for r in selected))
    def test_real_frozen_fit_house_scope(self):
        houses={r['house_id'] for r in p.read(p.MANIFEST)};fit=set(p.read(p.SPLIT)['FIT'])
        self.assertEqual(len(houses),43);self.assertTrue(houses<=fit)
    def test_closed_source_houses_scope(self):
        for root in p.CLOSED:self.assertEqual(p.read(root/'result.json')['status'],'SCOUT_CLOSED')
    def test_unselected_positions_accounted(self):
        positions=[[x,0,0] for x in range(10)];selected,ledger=p.select(positions,[])
        self.assertEqual(len(ledger),10);self.assertEqual(len(selected),4)
if __name__=='__main__':unittest.main()
