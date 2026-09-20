"""Check actual inventory on a directory timestamp change; never waive hashes."""
from common import HERE, RUNTIME, load

sealed = load('v15_sealed_store', RUNTIME/'feedback_generation_v1/store.py')
sealed.FEEDBACK_ROOT = HERE


class AuditedContentStore(sealed.TrackedContentStore):
    def __init__(self, root, max_bytes):
        super().__init__(root, max_bytes)
        self.timestamp_checks = []

    def _check_root(self):
        self._check_owner()
        sealed.ContentStore._check_root(self)
        if self.poisoned:
            raise sealed.StoreError('store is poisoned')
        if self._stamp() != self._directory_stamp:
            before = self._directory_stamp
            audit = self.full_audit()
            expected = [{'file': None, 'reason': 'directory_differs_from_last_successful_commit'}]
            if audit['problems'] != expected or not audit['accounting_pass']:
                raise sealed.StoreError(f'directory inventory changed: {audit}')
            self.timestamp_checks.append(dict(previous=before, current=self._stamp(), audit=audit))
            self._directory_stamp = self._stamp()
