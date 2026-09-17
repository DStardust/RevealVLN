import copy
import json
from pathlib import Path
import unittest
import planner as p


class PlannerTests(unittest.TestCase):
    def fixture(self):
        def obj(i,cat,room,x):
            raw = {'tv_monitor':'tv'}.get(cat,cat)
            return {'object_index':i,'raw':raw,'mpcat40':cat,'room':room,'level':0,
                    'region_index':i,'house_frame_center':[x,0,0]}
        return {i:obj(i,c,r,x) for i,c,r,x in [
            (1,'bed','bedroom',0),(2,'sink','bathroom',3),
            (3,'sofa','living room',6),(4,'chair','living room',7),
            (5,'table','dining room',9)]}

    def test_parse(self):
        text='ASCII 1.1\nR 0 1 0 0 d 0 0 0\nC 0 1 dining#chair 3 chair\nO 1 0 0 1 2 3\n'
        row=p.parse_house(text)[1]
        self.assertEqual((row['raw'],row['room'],row['level']),('dining chair','dining room',1))

    def test_version_denied(self):
        with self.assertRaises(ValueError): p.parse_house('ASCII 1.0')

    def test_duplicate_denied(self):
        with self.assertRaises(ValueError):
            p.parse_house('ASCII 1.1\nR 0 0 0 0 b\nR 0 0 0 0 b\n')

    def test_dangling_denied(self):
        with self.assertRaises(ValueError):p.parse_house('ASCII 1.1\nO 1 0 0 0 0 0\n')

    def test_nonfinite_denied(self):
        with self.assertRaises(ValueError):p.parse_house('ASCII 1.1\nO 1 -1 -1 nan 0 0\n')

    def test_flexible_not_old_fixed_combination(self):
        proposals,stats=p.propose('toy',self.fixture())
        self.assertGreater(len(proposals),0)
        self.assertTrue(all(r['quality_tier']=='SEMANTIC_CANDIDATE' for r in proposals))
        self.assertTrue(all(r['visibility_pass'] is None for r in proposals))

    def test_determinism_input_order(self):
        x=self.fixture()
        self.assertEqual(p.propose('toy',x),p.propose('toy',dict(reversed(list(x.items())))))

    def test_candidate_dedup(self):
        rows,_=p.propose('toy',self.fixture())
        self.assertEqual(len(rows),len({r['candidate_id'] for r in rows}))

    def test_four_roles_distinct(self):
        rows,_=p.propose('toy',self.fixture())
        for r in rows:
            self.assertEqual(len({p.canonical(v) for v in r['roles'].values()}),4)

    def test_bound(self):
        rows,stats=p.propose('toy',self.fixture(),2)
        self.assertEqual(len(rows),2)
        self.assertGreater(stats['trimmed_by_predeclared_limit'],0)

    def test_insufficient_roles(self):
        rows,_=p.propose('toy',{1:self.fixture()[1]})
        self.assertEqual(rows,[])

    def test_reserved_instance(self):
        obj=self.fixture()[1]
        self.assertEqual(p.semantic_groups({0:obj}),{})

    def test_unrecognized_raw_not_guessed(self):
        obj=self.fixture()[1];obj['raw']='bed-like unknown'
        self.assertEqual(p.semantic_groups({1:obj}),{})

    def test_unknown_room_not_guessed(self):
        obj=self.fixture()[1];obj['room']=None
        self.assertEqual(p.semantic_groups({1:obj}),{})

    def test_quality_cannot_promote(self):
        with self.assertRaises(ValueError):p.validate_quality({'quality_tier':'MECHANISM_TRAIN_READY','checks':{}})

    def test_candidate_cannot_train(self):
        with self.assertRaises(ValueError):p.validate_quality({'quality_tier':'SEMANTIC_CANDIDATE','training_admission':True})

    def test_cpu_parser_matches_five_saved_runtime_inventories(self):
        houses=['17DRP5sb8fy','1LXtFkjw3qL','1pXnuDYAj8r','29hnd4uzFmX','5LpN3gDmAk7']
        for index,house in enumerate(houses):
            source=p.scoped(p.ROOT/'third_party/ETP-R1/data/scene_datasets/mp3d'/house/(house+'.house'))
            parsed=p.parse_house(source.read_text())
            inventory=p.LINE/'data_pipeline/mechanism_runtime_v1/p0_execution_v1/run/bundles'/('MP5_%02d'%index)/'SEMANTIC_INVENTORY.json'
            observed=json.loads(inventory.read_text())['objects']
            self.assertGreater(len(observed),0)
            for idx,obj in observed.items():
                self.assertEqual({k:parsed[int(idx)][k] for k in ('raw','mpcat40','room')},
                                 {k:obj[k] for k in ('raw','mpcat40','room')},(house,idx))


if __name__=='__main__':unittest.main()
