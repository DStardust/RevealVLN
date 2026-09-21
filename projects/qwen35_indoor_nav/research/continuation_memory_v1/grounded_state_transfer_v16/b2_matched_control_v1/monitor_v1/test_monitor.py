import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import Reader


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = self.root / 'evaluate/session_001'
        self.job = self.root / 'job'
        conditions = [dict(endpoint='main'),dict(endpoint='control')]
        slots = [dict(rank=i*9+j,condition=i,model=f'{a}_{s}',arm=a,seed=s)
                 for i in range(2) for j,(s,a) in enumerate((s,a) for s in (1209,1210,1211) for a in ('B1','Terminal','B2'))]
        self.registry = dict(conditions=conditions,slots=slots,models=[s['model'] for s in slots[:9]])
        write(self.root/'EVALUATION_REGISTRY.json',self.registry)
        write(self.root/'STATUS.json',dict(status='RUNNING',stage='evaluate_continuations',elapsed=0))
        write(self.root/'PROTOCOL.json',dict(gpu_session_hours=24,arms=['B1','B2','Terminal'],devices=[dict(gpu=i) for i in range(8)],steps_per_arm=1200))
        files = {}
        for slot in slots[:9]:
            name=f"rollouts/{slot['rank']:04d}/TASK_RESULT.json"
            write(self.session/name,dict(rank=slot['rank'],model=slot['model'],condition=conditions[0],
                safe_v16_label='PASS' if slot['arm']=='B2' else 'FAIL',total_decisions=30,collisions=0,stopped=True,budget_exhausted=False))
            files[name] = hashlib.sha256((self.session/name).read_bytes()).hexdigest()
        self.group = self.session/'GROUP_000.json'
        write(self.group,dict(condition=0,ranks=list(range(9)),files=files,audits=[dict(input_prefix_matched=True,
            action_prefix_matched=True,argmax_flip_count=0,logits_bitwise_equal=True,max_logit_delta=0)]*6))

    def seal(self):
        write(self.session/'STATE_SEAL_000.json',dict(base_unchanged=True,heads_unchanged=True,completed_ranks=list(range(9))))

    def test_training_progress_without_global_training_result(self):
        write(self.root/'TRAIN_PROGRESS_0.json',dict(tag='B1_1209',step=55,seconds=2))
        write(self.root/'train/B2_1209/RESULT.json',dict(updates=1200))
        write(self.root/'STATUS.json',dict(status='RUNNING',stage='review',workers=[dict(device=None)]))
        value=Reader(self.root,self.job).snapshot()
        self.assertEqual(value['models_complete'],1)
        self.assertEqual(value['training_steps'],1255)
        self.assertEqual(value['training_planned_steps'],10800)
        self.assertEqual(value['workers'],[])

    def test_unsealed_group_never_counts(self):
        value=Reader(self.root,self.job).snapshot()
        self.assertEqual(value['complete'],0)
        self.assertEqual(value['planned'],18)
        self.assertTrue(all(m['partial_rate'] is None for m in value['metrics']))

    def test_sealed_partial_keeps_planned_unknown_denominator(self):
        self.seal();value=Reader(self.root,self.job).snapshot()
        self.assertEqual(value['complete'],9)
        ours=next(m for m in value['metrics'] if m['arm']=='B2' and m['endpoint']=='main')
        self.assertEqual((ours['passed'],ours['complete'],ours['planned']),(3,3,3))
        control=next(m for m in value['metrics'] if m['endpoint']=='control')
        self.assertEqual((control['complete'],control['not_run']),(0,3))
        self.assertEqual((control['identification_lower'],control['identification_upper']),(0,1))
        pair=next(p for p in value['paired'] if p['endpoint']=='main' and p['baseline']=='Terminal')
        self.assertEqual((pair['win'],pair['loss'],pair['tie']),(3,0,0))

    def test_corrupt_new_group_is_visible_and_excluded(self):
        self.seal()
        (self.session/'rollouts/0000/TASK_RESULT.json').write_text('{}')
        value=Reader(self.root,self.job).snapshot()
        self.assertEqual(value['complete'],0)
        self.assertIn('SHA mismatch',value['errors'][0])

    def test_changed_model_seal_is_not_admitted(self):
        self.seal();q=self.session/'STATE_SEAL_000.json'
        record=json.loads(q.read_text());record['base_unchanged']=False;write(q,record)
        value=Reader(self.root,self.job).snapshot()
        self.assertEqual(value['complete'],0)
        self.assertIn('Model state changed',value['errors'][0])


if __name__=='__main__':unittest.main()
