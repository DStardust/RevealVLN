"""Bounded CPU tests; fixtures stay inside this new project namespace."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ordinary_baseline_data', HERE / 'data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def png(rgb):
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    rows = b''.join(b'\x00' + rgb[i:i+672] for i in range(0, len(rgb), 672))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 224, 224, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cpu_fixture_', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pixels = bytes([12, 45, 78]) * (224 * 224)
        self.ref = hashlib.sha256(self.pixels).hexdigest() + '.png'
        (self.root / self.ref).write_bytes(png(self.pixels))
        self.policy = {'instruction': 'Go to the door.', 'rgb_sequence': [self.ref] * 12}
        self.supervision = {'actions': ['MOVE_FORWARD'] * 11 + ['STOP'], 'pose': 'privileged'}
        self.row = {'sourceRoot': str(self.root.relative_to(data.ROOT)),
                    'policy_file': 'policy.json', 'supervision_file': 'supervision.json',
                    'rgb_reference_root': '.', 'source': 'R2R', 'scene_group': 'test_house',
                    'physical_source_route_sha256': 'a' * 64, 'decisions': 12}
        self.write()

    def write(self):
        (self.root / 'policy.json').write_text(json.dumps(self.policy))
        (self.root / 'supervision.json').write_text(json.dumps(self.supervision))

    def test_metadata_is_lazy(self):
        with mock.patch.object(data, 'load_rgb', side_effect=AssertionError('pixel read')):
            record = data.OrdinaryRecord(self.row)
            steps = list(record.iter_decisions())
        self.assertEqual(len(steps), 12)
        self.assertEqual(steps[10]['policy']['executed_actions'], ['move_forward'] * 8)
        self.assertEqual(len(steps[10]['control']['rgb_refs']), 2)
        self.assertEqual(record.metadata()['action_counts']['STOP'], 1)
        self.assertFalse(record.metadata()['pixel_content_verified'])

    def test_future_and_privileged_isolation(self):
        before = data.OrdinaryRecord(self.row).decision_metadata(3)
        self.supervision['pose'] = 'changed'
        self.supervision['actions'][3:11] = ['TURN_RIGHT'] * 8
        other = 'b' * 64 + '.png'
        (self.root / other).write_bytes(b'not opened by metadata')
        self.policy['rgb_sequence'][4:] = [other] * 8
        self.write()
        after = data.OrdinaryRecord(self.row).decision_metadata(3)
        self.assertEqual(before['control'], after['control'])
        self.assertEqual(before['policy'], after['policy'])
        self.assertNotEqual(before['supervision'], after['supervision'])

    def test_terminal_stop_and_action_dictionary(self):
        for actions in (['STOP'] * 12, ['move_forward'] * 12,
                        ['move_forward'] * 11 + ['stop'], ['F'] * 11 + ['STOP']):
            with self.subTest(actions=actions):
                self.supervision['actions'] = actions
                self.write()
                with self.assertRaises(ValueError):
                    data.OrdinaryRecord(self.row)
        self.assertEqual(data.normalize_action('TURN_LEFT'), 'turn_left')

    def test_policy_extra_fields_rejected(self):
        self.policy['goal'] = 'secret'
        self.write()
        with self.assertRaisesRegex(ValueError, 'POLICY_FIELDS'):
            data.OrdinaryRecord(self.row)

    def test_path_escape_absolute_parent_and_source_root(self):
        for key, value in [('sourceRoot', '/root'), ('sourceRoot', '.'),
                           ('sourceRoot', '../other'), ('policy_file', '../policy.json'),
                           ('rgb_reference_root', '/tmp')]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    data.OrdinaryRecord(dict(self.row, **{key: value}))

    def test_symlink_escape(self):
        (self.root / 'outside').symlink_to(data.ROOT)
        with self.assertRaisesRegex(ValueError, 'PATH_ESCAPE'):
            data.OrdinaryRecord(dict(self.row, rgb_reference_root='outside'))

    def test_missing_or_wrong_metadata(self):
        for row in (dict(self.row, decisions=True), dict(self.row, decisions=11),
                    dict(self.row, physical_source_route_sha256='bad')):
            with self.assertRaises(ValueError):
                data.OrdinaryRecord(row)
        row = dict(self.row)
        del row['sourceRoot']
        with self.assertRaisesRegex(ValueError, 'ROW_FIELDS'):
            data.OrdinaryRecord(row)

    def test_instruction_and_iteration_reset(self):
        first = data.OrdinaryRecord(self.row)
        self.policy['instruction'] = 'Turn around then stop.'
        self.write()
        second = data.OrdinaryRecord(self.row)
        for record in (first, second, first):
            steps = list(record.iter_decisions())
            self.assertTrue(steps[0]['control']['memory_reset'])
            self.assertEqual(steps[0]['policy']['executed_actions'], [])
            self.assertFalse(any(s['control']['memory_reset'] for s in steps[1:]))
        self.assertNotEqual(first.instruction, second.instruction)

    def test_decoded_policy_whitelist_and_window(self):
        with mock.patch.object(data, 'load_rgb', return_value=object()) as decoder:
            decision = data.OrdinaryRecord(self.row).decision(5)
        self.assertEqual(decoder.call_count, 2)
        self.assertEqual(set(decision['policy']), {'instruction', 'images', 'executed_actions'})
        self.assertEqual(set(decision['control']), {'decision_step', 'memory_reset'})
        self.assertNotIn('target_action', decision['policy'])

    def test_time_bounds_and_return_mutation(self):
        record = data.OrdinaryRecord(self.row)
        for t in (-1, 12, True):
            with self.assertRaisesRegex(ValueError, 'DECISION_TIME'):
                record.decision_metadata(t)
        sample = record.decision_metadata(1)
        sample['policy']['executed_actions'][0] = 'STOP'
        self.assertEqual(record.decision_metadata(1)['policy']['executed_actions'], ['move_forward'])

    def test_real_png_raw_hash(self):
        try:
            import PIL
        except ImportError:
            self.skipTest('stdlib interpreter: existing PIL environment required for PNG decode')
        image = data.load_rgb(self.root / self.ref)
        self.assertEqual(image.tobytes(), self.pixels)
        self.assertNotEqual(hashlib.sha256((self.root / self.ref).read_bytes()).hexdigest(), Path(self.ref).stem)
        wrong = self.root / ('c' * 64 + '.png')
        wrong.write_bytes((self.root / self.ref).read_bytes())
        with self.assertRaisesRegex(ValueError, 'RGB_RAW_PIXEL_HASH'):
            data.load_rgb(wrong)

    def test_optional_snapshot_hashes(self):
        row = dict(self.row,
                   policy_sha256=hashlib.sha256((self.root / 'policy.json').read_bytes()).hexdigest(),
                   supervision_sha256=hashlib.sha256((self.root / 'supervision.json').read_bytes()).hexdigest())
        self.assertEqual(len(data.OrdinaryRecord(row)), 12)
        for field in ('policy_sha256', 'supervision_sha256'):
            with self.assertRaisesRegex(ValueError, 'SOURCE_JSON_HASH'):
                data.OrdinaryRecord(dict(row, **{field: '0' * 64}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
