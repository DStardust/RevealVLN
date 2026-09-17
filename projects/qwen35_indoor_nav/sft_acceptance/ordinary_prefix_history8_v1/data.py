"""Ordinary-only history view. Frozen source decoder and original labels."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
SOURCE = HERE.parent / 'ordinary_expanded_v1/data.py'
ADAPTER = HERE.parent / 'ordinary_baseline_v2/data.py'
assert hashlib.sha256(ADAPTER.read_bytes()).hexdigest() == '7631d0f57384d2979c8d5b1d7e35748c9f428377c805ea3292e7b0260e43b788'
base = runpy.run_path(str(SOURCE), run_name='READONLY_EXPANDED_ORDINARY_DATA')
decoder = runpy.run_path(str(ADAPTER), run_name='READONLY_ORDINARY_DECODER')
history = runpy.run_path(str(HERE / 'history.py'))
for name in ('load_rows', 'load_sample_index', 'plan_epoch_batches', 'ACTIONS', 'advance_epoch_boundary'):
    globals()[name] = base[name]

class SampleStore:
    def __init__(self, rows):
        self.rows = rows
        self.records = OrderedDict()

    def record(self, record_idx):
        if type(record_idx) is not int or not 0 <= record_idx < len(self.rows):
            raise ValueError('RECORD_INDEX')
        if record_idx not in self.records:
            row = self.rows[record_idx]
            if row['split'] != 'FIT':
                raise ValueError('NON_FIT_ROW')
            self.records[record_idx] = decoder['OrdinaryRecord'](row)
            if len(self.records) > 128:
                self.records.popitem(last=False)
        self.records.move_to_end(record_idx)
        return self.records[record_idx]

    def get(self, record_idx, t):
        from PIL import Image
        record = self.record(record_idx)
        meta = record.decision_metadata(t)
        refs = history['choose'](record._refs, t, None)
        images = [Image.new('RGB', (224, 224)) if ref is None
                  else decoder['load_rgb'](decoder['_relative'](record.rgb_root, ref)) for ref in refs]
        return dict(instruction=record.instruction, images=images,
                    executed=list(meta['policy']['executed_actions']),
                    target=ACTIONS.index(meta['supervision']['target_action']))

