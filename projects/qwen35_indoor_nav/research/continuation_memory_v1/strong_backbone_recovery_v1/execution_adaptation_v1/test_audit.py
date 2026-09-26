"""Synthetic CPU trace contracts only, not a real navigation evaluation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import audit_pair

TOKENS, EOS, HEADER = [10, 11, 12, 13], 99, [90, 91]


def trace(chunks, overrides=None, drift=0.0):
    overrides = overrides or {}
    rows = [dict(event='reset', rgb_sha256='start', instruction_sha256='instruction')]
    rgb, step = 'start', 0
    for query_index, actions in enumerate(chunks):
        start = step
        body = [TOKENS[action] for action in actions] + [EOS]
        records = []
        for offset, token in enumerate(body):
            native_token = overrides.get((query_index, offset), token)
            native_action = TOKENS.index(native_token) if native_token in TOKENS else 1
            native = [drift] * 4
            native[native_action] += 4
            method = list(native)
            if token in TOKENS:
                method[TOKENS.index(token)] += 8
            method_action = max(range(4), key=lambda a: method[a])
            records.append(dict(offset=offset, native_logits=native, method_logits=method,
                native_action=native_action, method_action=method_action, native_token=native_token,
                method_token=token, generated_token=token, query_writes=start + 1,
                query_memory_sha256='memory' + str(start), actor_feature_sha256='actor', residual_applied=True))
        evidence = {name: dict(shape=[1], dtype='float32', sha256=rgb + str(start))
                    for name in ('inputs', 'images', 'depths', 'poses', 'intrinsics')}
        evidence.update(time_ids=list(range(start + 1)), rgb_sha256=rgb)
        rows.append(dict(event='generation', environment_step=start, input=evidence, header_ids=HEADER,
            generated_ids=HEADER + body, tokens=records, parsed_actions=actions, empty_action_fallback=not actions))
        executed = list(enumerate(actions)) if actions else [(None, 0)]
        for offset, action in executed:
            before = rgb
            rgb += ':' + str(action) if action else ''
            step += 1
            rows.append(dict(event='action', step=step, executed_action=action,
                before_rgb_sha256=before, after_rgb_sha256=rgb, query_environment_step=start,
                token_offset=offset, distance=10 - step, collision=False))
            if action == 0:
                return rows
    return rows


def result(rows, success=0):
    return dict(steps=sum(row['event'] == 'action' for row in rows), success=success)


def audit(a, b, **kwargs):
    return audit_pair(a, b, result(a), result(b), action_token_ids=TOKENS, eos_token_id=EOS, **kwargs)


class AuditTests(unittest.TestCase):
    def test_all_token_identity_and_finite_drift_are_distinct(self):
        a = trace([[1, 2, 3, 1], [0]])
        exact = audit(a, copy.deepcopy(a), zero=True)
        self.assertTrue(exact['logits_bitwise_equal'])
        self.assertEqual(exact['compared_token_decisions'], 7)
        drift = audit(a, trace([[1, 2, 3, 1], [0]], drift=.01))
        self.assertTrue(drift['full_trajectory_matched'])
        self.assertFalse(drift['logits_bitwise_equal'])
        self.assertGreater(drift['max_logit_delta'], 0)

    def test_second_or_later_action_intervention_is_included(self):
        a = trace([[1, 2, 0]])
        b = trace([[1, 2, 3], [0]], overrides={(0, 2): TOKENS[0]})
        value = audit(a, b)
        self.assertEqual(value['first_generated_difference']['token_offset'], 2)
        self.assertEqual(value['compared_token_decisions'], 3)
        self.assertEqual(value['first_executed_override_step'], 2)

    def test_eos_intervention_allows_later_query_cadence_change(self):
        a = trace([[1], [0]])
        b = trace([[1, 2], [0]], overrides={(0, 1): EOS})
        value = audit(a, b)
        self.assertTrue(value['first_generated_difference']['reference_is_eos'])
        self.assertEqual(value['first_generated_difference']['token_offset'], 1)
        self.assertEqual(value['compared_queries'], 1)
        self.assertEqual(value['first_executed_override_step'], 1)

    def test_eos_fallback_is_not_four_class_argmax(self):
        a = trace([[]])
        self.assertEqual(a[1]['tokens'][0]['native_action'], 1)
        self.assertEqual(a[-1]['executed_action'], 0)
        self.assertIsNone(a[-1]['token_offset'])
        self.assertTrue(audit(a, copy.deepcopy(a))['full_trajectory_matched'])

    def test_full_vocabulary_flip_and_four_class_flip_fail(self):
        a = trace([[1, 0]])
        with self.assertRaisesRegex(ValueError, 'NATIVE_FULL_VOCABULARY_FLIP'):
            audit(a, trace([[1, 0]], overrides={(0, 0): EOS}))
        with self.assertRaisesRegex(ValueError, 'NATIVE_FOUR_CLASS_ARGMAX_FLIP'):
            audit(a, trace([[1, 0]], overrides={(0, 0): TOKENS[2]}))

    def test_input_change_at_intervention_decision_is_rejected(self):
        a = trace([[1, 0]])
        b = trace([[2, 0]], overrides={(0, 0): TOKENS[1]})
        b[1]['input']['poses']['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'PROCESSED_QUERY_PREFIX'):
            audit(a, b)

    def test_generated_action_must_be_the_one_physically_executed(self):
        a = trace([[1, 0]])
        b = copy.deepcopy(a)
        b[2]['token_offset'] = 1
        with self.assertRaisesRegex(ValueError, 'EXECUTED_TOKEN_OFFSET'):
            audit(a, b)

    def test_unknown_tokens_and_nonfinite_scores_fail_closed(self):
        a = trace([[1, 0]])
        b = copy.deepcopy(a)
        b[1]['generated_ids'][2] = 555
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_GENERATED'):
            audit(a, b)
        b = copy.deepcopy(a)
        b[1]['tokens'][1]['native_logits'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'NONFINITE'):
            audit(a, b)

    def test_no_intervention_requires_complete_terminal_identity(self):
        a = trace([[1, 0]])
        with self.assertRaisesRegex(ValueError, 'TERMINAL_DIVERGED'):
            audit_pair(a, copy.deepcopy(a), result(a), result(a, 1), action_token_ids=TOKENS, eos_token_id=EOS)
        b = trace([[2, 0]], overrides={(0, 0): TOKENS[1]})
        with self.assertRaisesRegex(ValueError, 'ZERO_RESIDUAL'):
            audit(a, b, zero=True)

    def test_stop_must_not_create_memory(self):
        a = trace([[1, 0]])
        b = copy.deepcopy(a) + [dict(event='memory_write', environment_step=2)]
        with self.assertRaisesRegex(ValueError, 'STOP_CREATED_MEMORY'):
            audit(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
