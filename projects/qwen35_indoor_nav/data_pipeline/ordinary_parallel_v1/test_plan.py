import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('parallel_plan',HERE/'plan.py')
plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)

def job(key,house='house',source='R2R',aliases=3):
    return dict(physical_source_route_sha256=key,scene_id=house,source=source,
                instruction_alias_episodes=[dict(episode_id=i) for i in range(aliases)])

class PlanTests(unittest.TestCase):
    def test_excludes_all_prior_candidates(self):
        selected,stats=plan.select_sources({'R2R':[job('old'),job('new')],'RxR':[]},{'old'},500)
        self.assertEqual([j['physical_source_route_sha256'] for j in selected],['new'])
        self.assertEqual(stats['R2R']['shortfall_routes'],499)
    def test_cross_source_route_is_not_duplicated(self):
        selected,_=plan.select_sources({'R2R':[job('same')],'RxR':[job('same',source='RxR')]},set(),500)
        self.assertEqual(len(selected),1)
    def test_partition_whole_house_and_aliases(self):
        jobs=[job(f'{h}-{i}',h,aliases=3) for h in 'abcdefg' for i in range(3)]
        shards,mapping=plan.partition(jobs)
        self.assertEqual(sum(map(len,shards)),len(jobs))
        self.assertEqual(sum(len(j['instruction_alias_episodes']) for s in shards for j in s),63)
        self.assertEqual(len({j['physical_source_route_sha256'] for s in shards for j in s}),21)
        for sid,shard in enumerate(shards):
            self.assertTrue(all(mapping[j['scene_id']]==sid for j in shard))
    def test_input_order_does_not_change_assignment(self):
        jobs=[job(f'{h}-{i}',h) for h in 'abcde' for i in range(4)]
        self.assertEqual(plan.partition(jobs),plan.partition(list(reversed(jobs))))
    def test_source_selection_is_deterministic(self):
        jobs=[job(str(i),str(i%3)) for i in range(21)]
        a=plan.select_sources({'R2R':jobs,'RxR':[]},set(),10)
        b=plan.select_sources({'R2R':list(reversed(jobs)),'RxR':[]},set(),10)
        self.assertEqual(a,b)
    def test_empty_pools_and_shards(self):
        jobs,stats=plan.select_sources({'R2R':[],'RxR':[]},set())
        self.assertEqual(jobs,[])
        self.assertEqual(stats['RxR']['shortfall_routes'],500)
        self.assertEqual(plan.partition(jobs),([[],[],[],[]],{}))

if __name__=='__main__':unittest.main()
