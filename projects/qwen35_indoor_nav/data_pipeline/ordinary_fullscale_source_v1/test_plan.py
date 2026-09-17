import copy
import json
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import plan


def job(number, house='h'):
    e=dict(episode_id=number, scene_id=f'mp3d/{house}/{house}.glb', start_position=[number,0,0],
        start_rotation=[0,0,0,1], reference_path=[[number,0,0],[number+2,0,0]], goals=[dict(position=[number+2,0,0])],
        instruction=dict(instruction_text='Walk forward.',language='en'))
    return dict(job_id=str(number),source='R2R',scene_id=house,split='FIT',physical_source_route_sha256=plan.key(e),
        episode=e,instruction_alias_episodes=[e])


class Tests(unittest.TestCase):
    def test_reference_length(self):
        self.assertEqual(plan.features(job(0)), dict(reference_polyline_m=2,reference_waypoints=2,aliases=1))
    def test_no_alias_loss_partition(self):
        jobs=[job(i,str(i//3)) for i in range(20)]
        shards=plan.partition(jobs,7)
        self.assertTrue(all(len(s)<=7 for s in shards))
        plan.validate(jobs,shards,set(),{j['scene_id'] for j in jobs})
        self.assertEqual(plan.partition(list(reversed(jobs)),7),shards)
    def test_whole_house_when_fits(self):
        jobs=[job(i,str(i//3)) for i in range(12)]
        shards=plan.partition(jobs,7)
        for house in {j['scene_id'] for j in jobs}:
            self.assertEqual(sum(any(j['scene_id']==house for j in s) for s in shards),1)
    def test_large_house_route_indivisible(self):
        shards=plan.partition([job(i) for i in range(21)],5)
        self.assertEqual(sorted(map(len,shards)),[1,5,5,5,5])
    def test_duplicate_exclusion_fails(self):
        j=job(0)
        with self.assertRaises(AssertionError):plan.validate([j,j],[[j,j]],set(),{'h'})
        with self.assertRaises(AssertionError):plan.validate([j],[[j]],{j['physical_source_route_sha256']},{'h'})
    def test_wrong_language_and_alias_route_fail(self):
        j=job(0); j['instruction_alias_episodes']=[job(1)['episode']]
        with self.assertRaises(AssertionError):plan.validate([j],[[j]],set(),{'h'})
        j=job(0);j['episode']['instruction']['language']='hi-IN'
        with self.assertRaises(AssertionError):plan.validate([j],[[j]],set(),{'h'})
    def test_heldout_fails(self):
        j=job(0)
        with self.assertRaises(AssertionError):plan.validate([j],[[j]],set(),{'other'})
    def test_summary_and_empty(self):
        self.assertEqual(plan.summary([])['n'],0)
        self.assertEqual(plan.summary([1,2,3,4])['median'],2.5)
    def test_actual_freeze(self):
        if not (HERE/'PLAN.json').exists():self.skipTest('preparation not generated')
        p=plan.read(HERE/'PLAN.json');jobs=plan.read(HERE/'JOBS.json')
        excluded=set(plan.read(HERE/'EXCLUSION_MANIFEST.json')['physical_route_keys'])
        shards=[plan.read(HERE/s['jobs_path']) for s in p['shards']]
        plan.validate(jobs,shards,excluded,plan.read(HERE/'SPLIT_FREEZE.json')['FIT'])
        self.assertEqual(len(excluded),2100)
        self.assertFalse(p['executable'])
        for path,h in plan.read(HERE/'MANIFEST_HASHES.json').items():self.assertEqual(plan.sha(HERE/path),h)
    def test_forecast_not_generated(self):
        if not (HERE/'FORECAST.json').exists():self.skipTest('preparation not generated')
        p=plan.read(HERE/'FORECAST.json')
        self.assertEqual(p['generated_action_records_this_node'],0)
        self.assertFalse(p['scientific_pass'])
        for variant in p['forecasts'].values():
            for values in variant.values():
                for scenario in values['scenarios'].values():
                    self.assertLessEqual(scenario['yield_adjusted_instruction_conditioned_decisions'],scenario['all_pass_instruction_conditioned_decisions'])


if __name__=='__main__':unittest.main()
