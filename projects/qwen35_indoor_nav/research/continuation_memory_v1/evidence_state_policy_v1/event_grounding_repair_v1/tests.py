"""Real loss equivalence, FIT isolation, sampling, update and checkpoint checks."""
from pathlib import Path
import copy
import random
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import *
from event_loss import Pool, ROLES, original_mass, predictions
from probe_review import summarize
from train_one import objective, save, restore


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = read(CPU/'DATA.json')
        cls.records = [r for r in read(PARENT/'recovery_localization_v1/runs/diagnosis_001/EVENT_INDEX.json')['records'] if r['split'] == 'FIT']
        cls.families = [f for f in cls.data['families'] if f['split'] == 'FIT']
        cls.cache = {k: v.float() for k, v in torch.load(CONTROL/'features/FEATURES.pt', map_location='cpu', weights_only=True).items()}
        cls.schedule = read(CPU/'SCHEDULES.json')['1209']

    def test_pool_mass_and_actual_labels(self):
        pool = Pool(self.records); mass = pool.mass()
        self.assertAlmostEqual(sum(mass), 1.)
        for ids in pool.cells.values():
            self.assertAlmostEqual(sum(mass[i] for i in ids), 1/len(pool.cells))
            self.assertEqual(len({mass[i] for i in ids}), 1)
        self.assertEqual(len(pool.records), 8271)

    def test_duplicate_nonfit_unknown_rejected(self):
        for invalid in ([self.records[0]]*2, [dict(self.records[0], split='DEV')], [dict(self.records[0], target=-1)]):
            with self.assertRaises(ValueError):
                Pool(invalid)

    def test_sampling_is_replayable_without_global_rng(self):
        pool = Pool(self.records); before = random.getstate()
        first = pool.sample(1209, 200, 256)
        self.assertEqual(first, pool.sample(1209, 200, 256))
        self.assertNotEqual(first, pool.sample(1209, 201, 256))
        self.assertEqual(before, random.getstate())

    def test_four_house_isolation(self):
        for house in sorted({r['house'] for r in self.records}):
            train = [r for r in self.records if r['house'] != house]
            check = [r for r in self.records if r['house'] == house]
            self.assertFalse({r['feature_index'] for r in train} & {r['feature_index'] for r in check})
            families = [f for f in self.families if f['house'] != house]
            mass, stats = original_mass(families, train, self.schedule)
            self.assertAlmostEqual(sum(mass), 1.)
            self.assertEqual(set(stats['family_exposures']), {f['family_id'] for f in families})

    def test_original_measure_equals_real_family_loss(self):
        net = make_head('ORIGINAL_1209')
        # Restrict only the schedule/families for a bounded exact numerical test.
        families = self.families[:2]
        schedule = [{'family': families[0]['family_id']}] * 2 + [{'family': families[1]['family_id']}]
        mass, stats = original_mass(families, self.records, schedule)
        with torch.no_grad():
            flat = F.binary_cross_entropy_with_logits(predictions(net, self.cache, self.records),
                      torch.tensor([r['target'] for r in self.records], dtype=torch.float32), reduction='none')
            represented = (flat * torch.tensor(mass)).sum()
            weighted = 0.
            w = objective.weights(families)
            for family, exposure in zip(families, (2/3, 1/3)):
                batch = objective.batch(family, 'cpu')
                _, _, detail = objective.losses(net, self.cache, batch, w)
                weighted = weighted + exposure * detail['event_loss']
            self.assertTrue(torch.allclose(represented, weighted, atol=1e-6, rtol=1e-6))

    def test_only_event_loss_is_replaced_and_teachers_unchanged(self):
        net = make_head('EVENT_1209'); before = digest(self.families[0])
        loss, stats, detail = objective.losses(net, self.cache, objective.batch(self.families[0], 'cpu'), objective.weights(self.families))
        replacement = Pool(self.records).sample_loss(net, self.cache, 1209, 0, 256)
        modified = loss-detail['event_loss']+replacement
        self.assertAlmostEqual(float(modified.detach()), stats['action_ce']+stats['cutoff_ce']+stats['state_bce']+stats['preservation_kl']+float(replacement.detach()), places=5)
        modified.backward()
        self.assertGreater(float(net.events[0].weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
        self.assertEqual(before, digest(self.families[0]))

    def test_optimizer_and_rng_restore_matches_uninterrupted_update(self):
        torch.manual_seed(1209); random.seed(1209); np.random.seed(1209)
        net = torch.nn.Linear(3, 2); opt = torch.optim.AdamW(net.parameters(), lr=.001)
        x = torch.randn(4, 3)
        def update():
            opt.zero_grad(); net(x).square().sum().backward(); opt.step()
        update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'STEP_0001.pt'; binding = {'test': 'actual_optimizer_rng'}
            save(path, net, opt, 1, binding)
            update(); expected = copy.deepcopy(net.state_dict())
            self.assertEqual(restore(path, net, opt, binding), 1)
            update()
            self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k, v in expected.items()))
            with self.assertRaises(ValueError):
                restore(path, net, opt, {'test': 'different'})

    def test_screen_missing_or_recall_degradation_cannot_advance(self):
        cfg = read(HERE/'PROTOCOL.json'); records = []
        for fold in range(4):
            for arm in cfg['arms']:
                metric = dict(brier=.2 if arm == 'ORIGINAL' else .1, recall=.8 if arm == 'ORIGINAL' else .6, fpr=.1)
                records.append(dict(arm=arm, fold=fold, held_house=str(fold), initial='same', metrics={r: metric for r in ROLES}))
        self.assertFalse(summarize(records, cfg['preflight_gate'])['advance_full_policy'])
        with self.assertRaises(ValueError):
            summarize(records[:-1], cfg['preflight_gate'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    write(HERE/'CPU_TEST_RESULT.json', dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        success=result.wasSuccessful(), cuda_initialized=torch.cuda.is_initialized(),
        scope='CPU loss identity, real FIT labels, gradients and optimizer/RNG restore. No model efficacy result.'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
