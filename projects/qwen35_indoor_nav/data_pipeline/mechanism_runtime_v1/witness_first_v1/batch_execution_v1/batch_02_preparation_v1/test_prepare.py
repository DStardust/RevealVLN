import copy
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import prepare as p


class ProspectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.draft=json.loads((p.SOURCE/'CONFIG_DRAFT.json').read_text())

    def test_selection_fixed_and_inert(self):
        rows=p.select(self.draft)
        self.assertEqual([(r['house_id'],r['hub_index']) for r in rows],list(p.EXPECTED))
        self.assertTrue(all(not r['executable'] for r in rows))

    def test_source_untouched(self):
        before=copy.deepcopy(self.draft)
        rows=p.select(self.draft);p.surface_revision(rows[0])
        self.assertEqual(before,self.draft)

    def test_below_1m_same_house_rejected(self):
        a=dict(house_id='a',configuration={'u_position':[0,0,0]})
        b=dict(house_id='a',configuration={'u_position':[.99,0,0]})
        with self.assertRaises(AssertionError):p.independent(a,b)
        b['configuration']['u_position']=[1,0,0]
        self.assertEqual(p.independent(a,b)['distance_m'],1)

    def test_different_houses_no_coordinate_distance(self):
        a=dict(house_id='a',configuration={'u_position':[0,0,0]})
        b=dict(house_id='b',configuration={'u_position':[0,0,0]})
        self.assertIsNone(p.independent(a,b)['distance_m'])

    def test_exact_recipe_invariants(self):
        for old in p.select(self.draft):
            new,revision=p.surface_revision(old)
            for key in ['roles','components','expected_eligible','balance','configuration','assets']:
                self.assertEqual(new[key],old[key])
            self.assertEqual(p.lang.structure(new['tasks']),p.lang.structure(old['tasks']))
            self.assertFalse(revision['executable'])

    def test_three_real_constructors_without_backend(self):
        for old in p.select(self.draft):
            new,_=p.surface_revision(old)
            result=p.validate_constructor(old,new)
            self.assertTrue(result['cpu_constructor_pass'])
            self.assertEqual(result['backend_calls'],0)


if __name__=='__main__':unittest.main()
