import ast
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch
import coordinator as c


class ParallelTests(unittest.TestCase):
    def test_whole_conditions_cover_remainder_without_duplicates(self):
        remaining=list(range(9,64));devices=[dict(gpu=i) for i in range(2,8)]
        actual=c.split_conditions(remaining,devices)
        self.assertEqual(Counter(x for values in actual.values() for x in values),Counter(remaining))
        self.assertLessEqual(max(map(len,actual.values()))-min(map(len,actual.values())),1)
        self.assertEqual(actual,c.split_conditions(reversed(remaining),devices))

    def test_frozen_registry_prefix_and_admission_are_identical(self):
        original=ast.parse((c.PILOT/'evaluate_continuations.py').read_text())
        parallel=ast.parse((c.HERE/'evaluate.py').read_text())
        for name in ('prefix_audit','registry_value','registry','admitted'):
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in parallel.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))

    def test_unrelated_process_cannot_be_released_as_holder(self):
        state=dict(manual_paused=False,leases=[],external_pids=[],device_count=6,worker_pids={'0':123})
        unrelated=dict(uid=0,argv=['python','real_training.py'],cgroup='/system.slice/training.service')
        with patch.object(c,'lease',return_value=state) as request,patch.object(c,'process',return_value=unrelated):
            with self.assertRaisesRegex(RuntimeError,'NOT_VERIFIED_PLACEHOLDER'):c.verify_placeholders()
            request.assert_called_once_with('status')

    def test_original_runtime_hash_is_still_frozen(self):
        amendment=c.read(c.HERE/'AMENDMENT.json')
        self.assertEqual(c.sha(c.PILOT/'evaluate_continuations.py'),amendment['original_runtime_sha256'])
        self.assertEqual(c.sha(c.RUN/'PROTOCOL.json'),amendment['original_protocol_sha256'])


if __name__=='__main__':unittest.main()
