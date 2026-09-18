import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('stop_alignment',Path(__file__).resolve().parent/'analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data_spec=importlib.util.spec_from_file_location('stop_labels',Path(__file__).resolve().parent/'prepare_fit.py')
data=importlib.util.module_from_spec(data_spec);data_spec.loader.exec_module(data)
split_spec=importlib.util.spec_from_file_location('split_stop_inputs',Path(__file__).resolve().parent/'split_inputs.py')
split=importlib.util.module_from_spec(split_spec);split_spec.loader.exec_module(split)


class Tests(unittest.TestCase):
    def test_terminal_arrival_is_not_a_prior_decision(self):
        self.assertEqual(module.decision_labels(dict(steps=2,distances=[5.,4.,2.])),[0,0])
        self.assertEqual(module.decision_labels(dict(steps=2,distances=[5.,2.,2.])),[0,1])

    def test_strict_radius_and_auc_ties(self):
        self.assertEqual(module.decision_labels(dict(steps=2,distances=[3.,2.999,0.])),[0,1])
        self.assertEqual(module.auc([(1,1,1),(1,0,1)]),.5)
        self.assertEqual(module.auc([(2,1,1),(1,0,1)]),1.)
        self.assertIsNone(module.auc([(1,0,1)]))

    def test_unknown_distance_and_policy_firewall(self):
        self.assertEqual(data.stop_target(float('inf')),dict(can_stop=None,loss_mask=0))
        self.assertEqual(data.stop_target(3.),dict(can_stop=0,loss_mask=1))
        row=dict(policy=dict(instruction='stop by the door',rgb_paths=['a.png'],executed_actions=[]),audit_only=dict(goal=[1,2,3]))
        self.assertNotIn('goal',data.policy_payload(row))
        row['policy']['goal']=[1,2,3]
        with self.assertRaisesRegex(ValueError,'PRIVILEGED_POLICY_FIELD'):data.policy_payload(row)

    def test_separate_labels_and_alignment(self):
        policy=dict(sample_id='a',partition='head_fit',policy=dict(instruction='stop',rgb_paths=['a.png'],executed_actions=[]))
        label=dict(sample_id='a',supervision=dict(can_stop=1,loss_mask=1),audit_only=dict(goal=[1,2,3]))
        inputs,target=list(split.aligned_rows([policy],[label]))[0]
        self.assertEqual(set(inputs),{'instruction','rgb_paths','executed_actions'})
        self.assertEqual(target,dict(can_stop=1,loss_mask=1))
        with self.assertRaisesRegex(ValueError,'LABEL_ALIGNMENT'):list(split.aligned_rows([policy],[]))
        label['sample_id']='b'
        with self.assertRaisesRegex(ValueError,'LABEL_ALIGNMENT'):list(split.aligned_rows([policy],[label]))


if __name__=='__main__':
    unittest.main()
