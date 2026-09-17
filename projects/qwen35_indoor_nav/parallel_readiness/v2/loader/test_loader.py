"""Real-data CPU tests; corruptions exist only in in-memory copies."""
import copy
import io
import random
import unittest

import loader as L


class RealLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = L.FamilyLoader()
        cls.record = copy.deepcopy(next(iter(cls.data.current.values())))
        cls.sample = next(iter(cls.data.by_id))

    def test_01_all_six_complete_prefixes(self):
        self.assertEqual(len(self.data.prefix_groups()), 6)
        total = 0
        for key in self.data.prefix_groups():
            stream = list(self.data.iter_prefix(*key))
            self.assertEqual([x['control']['decision_step'] for x in stream], list(range(207)))
            self.assertEqual([i for i, x in enumerate(stream) if x['policy']['memory_reset']], [0])
            self.assertEqual(len(stream[0]['policy']['rgb']), 1)
            self.assertEqual(len(stream[-1]['policy']['rgb']), 2)
            total += len(stream)
        self.assertEqual(total, 1242)

    def test_02_whitelist_and_raw_pixels(self):
        payload = self.data.policy_payload(self.record)
        self.assertEqual(set(payload), {'instruction', 'rgb', 'executed_actions', 'memory_reset'})
        for pixels in payload['rgb']:
            self.assertIs(type(pixels), bytes)
            self.assertEqual(len(pixels), 224 * 224 * 3)

    def test_03_shuffled_records_are_ordered(self):
        rows = copy.deepcopy(next(iter(self.data.prefixes.values())))
        random.Random(7).shuffle(rows)
        self.assertEqual(L.ordered_prefix(rows, 207), next(iter(self.data.prefixes.values())))

    def test_04_duplicate_or_missing_step_rejected(self):
        rows = copy.deepcopy(next(iter(self.data.prefixes.values())))
        rows[10] = copy.deepcopy(rows[9])
        with self.assertRaisesRegex(ValueError, 'PREFIX_GAP'):
            L.ordered_prefix(rows, 207)
        with self.assertRaisesRegex(ValueError, 'PREFIX_COUNT'):
            L.ordered_prefix(rows[:-1], 207)

    def test_05_reset_required(self):
        r = copy.deepcopy(self.record)
        r['memory_reset'] = True
        with self.assertRaisesRegex(ValueError, 'RESET'):
            L.validate_policy(r)

    def test_06_id_rename_invariant_policy(self):
        r = copy.deepcopy(self.record)
        r['sample_id'] = 'arbitrary-renamed-id'
        self.assertEqual(self.data.policy_payload(r), self.data.policy_payload(self.record))

    def test_07_query_and_label_do_not_enter_policy(self):
        before = self.data.policy_payload(self.record)
        s = copy.deepcopy(self.data.by_id[self.sample])
        others = [x for x in self.data.supervision if x['query_hash'] != s['query_hash']]
        self.assertTrue(others)
        s['query'] = copy.deepcopy(others[0]['query'])
        s['query_hash'] = others[0]['query_hash']
        s['outcome'] = 'fail'
        # Separate supervision has no argument position in policy_payload.
        self.assertEqual(self.data.policy_payload(self.record), before)
        r = copy.deepcopy(self.record)
        r['query'] = s['query']
        with self.assertRaisesRegex(ValueError, 'POLICY_FIELDS'):
            self.data.policy_payload(r)

    def test_08_return_mutation_no_cross_sample_contamination(self):
        first = self.data.policy_payload(self.record)
        expected = self.data.policy_payload(self.record)
        first['instruction'] = 'changed'
        first['rgb'] = ()
        self.assertEqual(self.data.policy_payload(self.record), expected)
        keys = self.data.prefix_groups()
        a, b = self.data.iter_prefix(*keys[0]), self.data.iter_prefix(*keys[-1])
        self.assertTrue(next(a)['policy']['memory_reset'])
        self.assertTrue(next(b)['policy']['memory_reset'])
        self.assertFalse(next(a)['policy']['memory_reset'])

    def test_09_future_observation_rejected(self):
        r = copy.deepcopy(self.record)
        r['observations'][-1]['step'] += 1
        with self.assertRaisesRegex(ValueError, 'OBS_CAUSAL'):
            L.validate_policy(r)

    def test_10_unexecuted_target_as_input_rejected(self):
        r = copy.deepcopy(self.record)
        r['executed_actions'][-1]['step'] = r['causal_cutoff_step']
        with self.assertRaisesRegex(ValueError, 'ACTION_CAUSAL'):
            L.validate_policy(r)

    def test_11_policy_semantic_pose_injection_rejected(self):
        for field in ('pose', 'semantic', 'target', 'house_id', 'm2_program_state'):
            r = copy.deepcopy(self.record)
            r[field] = 'forbidden'
            with self.assertRaisesRegex(ValueError, 'POLICY_FIELDS'):
                L.validate_policy(r)

    def test_12_pixel_hash_detects_corruption(self):
        r = self.record['observations'][0]['rgb_ref']
        blob = (self.data.root / 'content' / (r[7:] + '.rgb.npy')).read_bytes()
        self.assertEqual(len(L.rgb_from_npy(blob, r)), 150528)
        bad = blob[:-1] + bytes([blob[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, 'RGB_PIXEL_HASH'):
            L.rgb_from_npy(bad, r)
        with self.assertRaisesRegex(ValueError, 'NPY_PIXEL_LENGTH'):
            L.rgb_from_npy(blob[:-1], r)
        with self.assertRaisesRegex(ValueError, 'NPY_MAGIC'):
            L.rgb_from_npy(b'wrong-magic', r)

    def test_13_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PATH_ESCAPE'):
            L.safe_path(self.data.root, '../outside.json')

    def test_14_real_queries_typed_fixed_vocab(self):
        for s in self.data.supervision:
            q = self.data.query_input(s['sample_id'])
            self.assertEqual(q[-1], (5,))
            self.assertTrue(all(type(i) is int for row in q for i in row))
            for row in q:
                self.assertEqual(len(row), {3: 3, 4: 5, 5: 1}[row[0]])
        self.assertEqual(L.CATEGORY, {'chair': 20, 'sink': 21, 'bed': 22, 'tv_monitor': 23})

    def test_15_query_integrity_id_renaming_invariant(self):
        s = copy.deepcopy(self.data.by_id[self.sample])
        expected = L.typed_query(s['query'])
        s['sample_id'] = 'new'
        s['continuation_trace_id'] = 'renamed'
        s['query']['action_trace_ref'] = 'sha256:' + '1' * 64
        s['query']['rgb_content_refs'] = ['sha256:' + '2' * 64]
        self.assertEqual(L.typed_query(s['query']), expected)
        self.assertEqual(L.ref(L.query_projection(s['query'])), s['query_hash'])

    def test_16_query_extra_answer_rejected(self):
        q = copy.deepcopy(self.data.by_id[self.sample]['query'])
        q['answer'] = 1
        with self.assertRaisesRegex(ValueError, 'QUERY_FIELDS'):
            L.typed_query(q)

    def test_17_query_unknown_category_and_threshold_rejected(self):
        for field, value in [('object_category', 'unknown'), ('min_pixels', 255), ('consecutive_frames', 1)]:
            q = copy.deepcopy(self.data.by_id[self.sample]['query'])
            next(x for x in q['sequence'] if x['kind'] == 'observe')[field] = value
            with self.assertRaisesRegex(ValueError, 'OBSERVE_VALUE'):
                L.typed_query(q)

    def test_18_query_order_stop_repeat_rejected(self):
        q = copy.deepcopy(self.data.by_id[self.sample]['query'])
        q['sequence'][0]['order'] = 1
        with self.assertRaisesRegex(ValueError, 'QUERY_ORDER'):
            L.typed_query(q)
        q = copy.deepcopy(self.data.by_id[self.sample]['query'])
        q['sequence'][-1]['action'] = 'TURN_LEFT'
        with self.assertRaisesRegex(ValueError, 'QUERY_STOP'):
            L.typed_query(q)
        q = copy.deepcopy(self.data.by_id[self.sample]['query'])
        next(x for x in q['sequence'] if x['kind'] == 'movement')['repeat'] = 0
        with self.assertRaisesRegex(ValueError, 'MOVEMENT_VALUE'):
            L.typed_query(q)

    def test_19_bad_query_hash_rejected(self):
        s = copy.deepcopy(self.data.by_id[self.sample])
        s['query_hash'] = 'sha256:' + '0' * 64
        with self.assertRaisesRegex(ValueError, 'QUERY_HASH'):
            L.validate_supervision(s)

    def test_20_true_negative_bce_but_no_ce(self):
        fail = [s for s in self.data.supervision if s['outcome'] == 'fail']
        self.assertEqual(len(fail), 6)
        for s in fail:
            self.assertEqual(self.data.label(s['sample_id']), {'target': 0, 'bce_mask': 1})
            self.assertFalse(any(s['action_loss_mask']))
        s = copy.deepcopy(fail[0])
        s['action_loss_mask'][0] = 1
        with self.assertRaisesRegex(ValueError, 'NONPASS_ACTION_MASK'):
            L.validate_supervision(s)

    def test_21_unknown_rejected_mask_not_negative(self):
        s = copy.deepcopy(next(s for s in self.data.supervision if s['outcome'] == 'fail'))
        s['outcome'] = 'unknown'
        self.assertEqual(L.supervision_label(s), {'target': None, 'bce_mask': 0})
        s['sample_status'] = 'rejected'
        s['outcome'] = None
        self.assertEqual(L.supervision_label(s), {'target': None, 'bce_mask': 0})

    def test_22_target_cutoff_allowed_earlier_rejected(self):
        s = copy.deepcopy(self.data.by_id[self.sample])
        self.assertEqual(s['action_target_steps'][0], s['prefix_cutoff_step'])
        L.validate_supervision(s)
        s['action_target_steps'][0] -= 1
        with self.assertRaisesRegex(ValueError, 'TARGET_TIME'):
            L.validate_supervision(s)

    def test_23_grouped_sampling_order_invariant(self):
        groups = self.data.grouped_cells()
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(len(cells) for tasks in groups.values() for cells in tasks.values()), [9, 9])
        first = list(L.sample_grouped([self.data], 500, 1109))
        other = copy.copy(self.data)
        other.supervision = list(reversed(self.data.supervision))
        self.assertEqual(first, list(L.sample_grouped([other], 500, 1109)))
        self.assertEqual(len({sample for _, _, sample in first}), 18)

    def test_24_unique_ce_owners_and_bad_owner_rejected(self):
        self.data.validate_owners()
        self.assertEqual(len(self.data.dedup['owners']), 1410)
        other = copy.copy(self.data)
        other.dedup = copy.deepcopy(self.data.dedup)
        next(iter(other.dedup['owners'].values()))[1] += 1
        with self.assertRaisesRegex(ValueError, 'CE_OWNER_MISMATCH'):
            other.validate_owners()

    def test_25_all_real_action_streams_causal_no_future(self):
        ce_count = 0
        for sid, s in self.data.by_id.items():
            stream = self.data.iter_action_stream(sid)
            for t, item in enumerate(stream):
                self.assertEqual(item['control']['decision_step'], t)
                self.assertEqual(item['policy']['memory_reset'], t == 0)
                self.assertLessEqual(len(item['policy']['rgb']), 2)
                self.assertLessEqual(len(item['policy']['executed_actions']), 8)
                self.assertEqual(set(item), {'control', 'policy', 'action_supervision'})
                if t < s['prefix_cutoff_step']:
                    self.assertEqual(item['action_supervision'], {'action': None, 'ce_mask': 0})
                ce_count += item['action_supervision']['ce_mask']
        self.assertEqual(ce_count, 1410)

    def test_26_future_trace_mutation_cannot_change_current_payload(self):
        s = self.data.by_id[self.sample]
        t = s['prefix_cutoff_step']
        expected = self.data.policy_payload(self.data.decision_record(s, t))
        other = copy.copy(self.data)
        other._traces = copy.deepcopy(self.data._traces)
        tr = other._traces[(s['history_id'], s['continuation_trace_id'])]
        tr['observations'][t+1]['rgb_hash'] = '0' * 64
        tr['actions'][t+1] = 'R' if tr['actions'][t+1] != 'R' else 'L'
        self.assertEqual(other.policy_payload(other.decision_record(s, t)), expected)

    def test_27_prefix_action_consistency_rejected(self):
        rows = copy.deepcopy(next(iter(self.data.prefixes.values())))
        a = rows[10]['executed_actions'][0]
        a['action'] = 'TURN_LEFT' if a['action'] != 'TURN_LEFT' else 'TURN_RIGHT'
        with self.assertRaisesRegex(ValueError, 'ACTION_PREFIX_INCONSISTENT'):
            L.ordered_prefix(rows, 207)

    def test_28_changed_task_requires_distinct_prefix_replay(self):
        streams = [list(self.data.iter_prefix(t, 'H_T')) for t in self.data.manifest['task_ids']]
        self.assertNotEqual(streams[0][0]['policy']['instruction'], streams[1][0]['policy']['instruction'])
        self.assertTrue(all(s[0]['policy']['memory_reset'] for s in streams))
        self.assertEqual(streams[0][0]['policy']['rgb'], streams[1][0]['policy']['rgb'])


def run_tests():
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RealLoaderTests))
    return result, log.getvalue()
