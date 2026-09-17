"""Compare against the actual official SPL body, plus all FIT zero-offset data."""
import collections
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
s=importlib.util.spec_from_file_location('metric_v2_test',HERE/'counterfactual_v2.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


class MetricTests(unittest.TestCase):
    def test_actual_official_class(self):
        rng=np.random.RandomState(1309)
        positions=rng.normal(size=(200,3)).astype(np.float32)
        state=NS(position=positions[0]);sim=NS(get_agent_state=lambda:state)
        values={'distance_to_goal':NS(get_metric=lambda:4.),'success':NS(get_metric=lambda:1.)}
        task=NS(measurements=NS(measures=values,check_measure_dependencies=lambda *x:None))
        m=c.official.official_classes()['SPL'](sim=sim,config=NS())
        m.reset_metric(episode=NS(),task=task)
        for position in positions[1:]:state.position=position;m.update_metric(episode=NS(),task=task)
        self.assertEqual(float(m.get_metric()),float(4./max(4.,c.official_traveled(positions.tolist()))))
    def test_all_64_zero_bias_same_tolerance(self):
        case=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
        p=json.loads((case/'PROTOCOL.json').read_text());gt=json.load(gzip.open(p['gt_path']))
        count=0;maximum=collections.defaultdict(float)
        for lane in sorted((case/'run_001/lanes').glob('lane_*')):
            ds=collections.defaultdict(list)
            for line in (lane/'POLICY_STEPS.jsonl').read_text().splitlines():
                row=json.loads(line);ds[row['index']].append(row)
            for path in lane.glob('episode_*.json'):
                e=json.loads(path.read_text());r=c.prefix(e,ds[e['index']],gt[str(e['episode_id'])]['locations'],0)
                for key in ('success','spl','ndtw','steps','navigation_error_m','path_length_m'):
                    diff=abs(r[key]-e[key]);maximum[key]=max(maximum[key],diff)
                    self.assertLessEqual(diff,1e-9,(e['index'],key,diff))
                count+=1
        self.assertEqual(count,64)
        print(json.dumps(dict(zero_bias_episodes=count,max_abs_error=maximum)))
    def test_policy_unchanged(self):
        self.assertIs(c.choose,c.old.choose)
        self.assertEqual(c.GRID,c.old.GRID)


if __name__=='__main__':unittest.main()
