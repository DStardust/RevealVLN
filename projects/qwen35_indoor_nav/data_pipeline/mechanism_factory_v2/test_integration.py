import json
from pathlib import Path
import tempfile
import unittest

from journal import Journal
from planning import BudgetLedger
from factory import TraceRunner
from test_factory import candidate_fixture, fixture_compiler, RecordedBackend, traces_fixture

HERE = Path(__file__).resolve().parent


class IntegrationTests(unittest.TestCase):
    def test_persisted_reservations_match_completed_fixture_steps(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='test_integration_') as directory:
            root = Path(directory)/'journal'
            config = {'fixture_only': True}
            with Journal(root, config) as journal:
                ledger = BudgetLedger(clock=lambda: 0., persist=lambda state: journal.append('budget', state))
                ledger.start_bundle('fixture')
                backend = RecordedBackend(traces_fixture())
                runner = TraceRunner(backend, fixture_compiler(), ledger, journal.append)
                candidate = candidate_fixture()
                actions = candidate['histories']['H_A'][:3]
                trace = runner.run(candidate['position'], candidate['yaw_bin'], actions)
                self.assertTrue(trace['complete'])
                self.assertEqual(journal.latest('budget')['total_reserved_actions'], 3)
                self.assertEqual(journal.latest('action_completed')['confirmed_action_returns'], 3)
            with Journal(root, config, resume=True) as journal:
                ledger = BudgetLedger(state=journal.latest('budget'), clock=lambda: 0.,
                                      persist=lambda state: journal.append('budget', state))
                ledger.start_bundle('fixture')
                self.assertEqual(ledger.snapshot()['total_reserved_actions'], 3)
                ledger.reserve_action()
                self.assertEqual(journal.latest('budget')['total_reserved_actions'], 4)
                # Reservation is not a fictitious completed fourth physical action.
                self.assertEqual(journal.latest('action_completed')['confirmed_action_returns'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
