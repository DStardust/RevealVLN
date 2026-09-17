import importlib.util
from pathlib import Path
import unittest

s = importlib.util.spec_from_file_location('snapshot_build', Path(__file__).with_name('build.py'))
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


class Gates(unittest.TestCase):
    def row(self, **kw):
        return dict(dict(sourceRoot='same', job_id='job', scene_group='fit', split='FIT',
                         physical_source_route_sha256='physical', source='HUMAN', policy_file='policy_1.json'), **kw)

    def test_complete_aliases(self):
        p=set(); a=set()
        m.accept_group([self.row(), self.row(policy_file='policy_2.json')], {'fit'}, p, a)
        self.assertEqual((len(p), len(a)), (1, 2))

    def test_duplicate_physical(self):
        with self.assertRaisesRegex(AssertionError, 'DUPLICATE_PHYSICAL'):
            m.accept_group([self.row()], {'fit'}, {'physical'}, set())

    def test_duplicate_alias(self):
        with self.assertRaisesRegex(AssertionError, 'DUPLICATE_INSTRUCTION'):
            m.accept_group([self.row(), self.row()], {'fit'}, set(), set())

    def test_held_out_house(self):
        with self.assertRaisesRegex(AssertionError, 'HOUSE_LEAKAGE'):
            m.accept_group([self.row(scene_group='dev')], {'fit'}, set(), set())

    def test_mislabelled_split(self):
        with self.assertRaisesRegex(AssertionError, 'HOUSE_LEAKAGE'):
            m.accept_group([self.row(split='INTERNAL_DEV')], {'fit'}, set(), set())

    def test_scoped(self):
        with self.assertRaisesRegex(AssertionError, 'SOURCE_OUTSIDE_LINE'): m.scoped('/etc/passwd')

    def test_original_strict_unchanged(self): self.assertEqual(m.sha(m.STRICT), m.STRICT_SHA)


if __name__ == '__main__': unittest.main()
