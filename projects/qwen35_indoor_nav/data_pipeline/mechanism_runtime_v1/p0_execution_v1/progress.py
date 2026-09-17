"""Bounded read-only progress: no simulator imports and no writes."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = HERE/'run'
budget = None
log = out/'journal/events.jsonl'
if log.exists():
    with log.open('rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size-131072))
        lines = f.read().splitlines()
    for raw in reversed(lines):
        try:
            record = json.loads(raw)
        except (ValueError, UnicodeError):
            continue
        if record['kind'] == 'budget':
            budget = record['payload']
            break
results = []
for folder in sorted((out/'bundles').glob('*')):
    item = {'bundle': folder.name, 'saved_traces': len(list((folder/'traces').glob('*.json')))}
    result = folder/'result.json'
    if result.exists():
        r = json.loads(result.read_text())
        item.update(status=r['status'], reason=r['reason'], counters=r['counters'])
    else:
        item['status'] = 'IN_PROGRESS_OR_INTERRUPTED_NOT_A_PASS'
    results.append(item)
print(json.dumps({'active': budget['active'] if budget else None,
                  'reserved_actions': budget['total_reserved_actions'] if budget else None,
                  'elapsed_ledger_seconds': budget['last_clock']-budget['created'] if budget else None,
                  'bundles': results, 'worker_result_written': (out/'result.json').exists(),
                  'supervisor_closed': (out/'SUPERVISOR_RESULT.json').exists()}, ensure_ascii=False, indent=2))
