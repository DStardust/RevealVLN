import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cycle_policy',HERE/'cycle_policy.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class CycleTests(unittest.TestCase):
    def test_attempt_all_movements_and_preserve_stop(self):
        guard=m.CycleRecovery()
        args=('go',[b'rgb'],[],[4.,3.,2.,-1.])
        self.assertEqual([guard.choose(*args)[0] for _ in range(3)],list(m.ACTIONS[:3]))
        self.assertEqual(guard.choose('go',[b'rgb'],[],[0.,1.,2.,3.])[0],'STOP')

    def test_complete_input_and_episode_isolation(self):
        guard=m.CycleRecovery()
        self.assertEqual(guard.choose('go',[b'rgb'],[],[4,3,2,1])[0],'move_forward')
        for instruction,images,executed in [('new',[b'rgb'],[]),('go',[b'new'],[]),('go',[b'rgb'],['turn_left'])]:
            self.assertFalse(guard.choose(instruction,images,executed,[4,3,2,1])[1]['cycle_override'])
        guard.reset()
        self.assertFalse(guard.choose('go',[b'rgb'],[],[4,3,2,1])[1]['cycle_override'])

    def test_actual_original_100_episode_traces(self):
        # Shadow evaluation of the first possible intervention; after that the
        # original recorded trajectory is no longer a counterfactual rollout.
        run=HERE.parent/'ordinary_expanded_dev_after_single_v1/run_001'
        def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
        affected=[];successes=0;episodes=0
        for lane in sorted((run/'lanes').glob('lane_*')):
            initial={x['index']:x for x in read(lane/'INTERFACE.jsonl')}
            states={(x['index'],x['step']):x for x in read(lane/'STEPS_PRIVILEGED.jsonl')}
            policies={}
            for row in read(lane/'POLICY_STEPS.jsonl'):policies.setdefault(row['index'],[]).append(row)
            for index,rows in policies.items():
                guard=m.CycleRecovery();images=[initial[index]['rgb_sha256'].encode()];executed=[];override=None
                # Hashes stand in for raw RGB equality here; GPU runtime uses bytes.
                for row in rows:
                    action,detail=guard.choose(str(index),images[-2:],executed[-8:],row['logits'])
                    self.assertEqual(detail['native_action'],row['action'])
                    if detail['cycle_override']:
                        override=row['step'];break
                    images.append(states[index,row['step']]['rgb_sha256'].encode());executed.append(action)
                result=json.loads((lane/('episode_%02d.json'%index)).read_text())
                episodes+=1;successes+=int(result['success'])
                if result['success']:self.assertIsNone(override)
                if override:affected.append(index)
        self.assertEqual((episodes,successes,len(affected)),(100,21,23))


if __name__=='__main__':unittest.main()
