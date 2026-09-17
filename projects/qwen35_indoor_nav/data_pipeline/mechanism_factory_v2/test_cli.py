import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import cli


class CliTests(unittest.TestCase):
    def test_prepare_frozen_five_and_no_runtime(self):
        with tempfile.TemporaryDirectory(dir=cli.HERE, prefix='test_cli_') as directory:
            output = Path(directory)/'new'
            result = cli.prepare(output)
            self.assertEqual(result['candidate_count'], 5)
            self.assertEqual(len({r['house_id'] for r in result['candidates']}), 5)
            self.assertIsNone(result['eligible'])
            self.assertFalse(result['executable'])
            self.assertEqual(result['new_physical_families'], 0)
            self.assertEqual(set(result['tasks']), {'task_A', 'task_B'})
            with self.assertRaises(FileExistsError):
                cli.prepare(output)

    def test_output_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'OUTPUT_SCOPE'):
            cli.prepare(cli.HERE.parent/'not_allowed')

    def test_runtime_subcommands_fail_closed(self):
        for command in ('generate', 'certify'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                cli.main([command])
            self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
