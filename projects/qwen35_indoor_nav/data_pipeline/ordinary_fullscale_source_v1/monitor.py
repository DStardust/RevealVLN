"""Read-only active progress snapshots. Only complete JSON/JSONL records are read."""
import collections
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan


def snapshot():
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), shards=[], lanes=[],
                  observation_scope='read_only_nonatomic_across_files_not_final_acceptance')
    for sid in range(4):
        root = plan.PARALLEL/'production'/f'shard_{sid:04d}'
        item = dict(shard=sid, progress=None, complete=None)
        for name, field in [('PROGRESS.json','progress'), ('GENERATION_COMPLETE.json','complete')]:
            if (root/name).exists():
                item[field] = plan.read(root/name)
        audits = []
        for path in sorted((root/'shards').glob('*.audit.json')):
            audit = plan.read(path)
            index = path.with_name(path.name.replace('.audit.json', '.jsonl'))
            assert audit['integrity_pass'] and plan.sha(index) == audit['index_sha256']
            audits.append(audit)
        item['published_strict_routes'] = sum(a['certified_routes'] for a in audits)
        item['published_quarantine_routes'] = sum(a['quarantined_routes'] for a in audits)
        item['published_instruction_decisions'] = sum(a['instruction_conditioned_decisions'] for a in audits)
        result['shards'].append(item)
    for gpu in (6,7):
        lane = plan.PARALLEL/'runtime_v1/lanes'/f'gpu_{gpu}'
        entries = []
        for attempt in sorted(lane.glob('attempt_*')):
            entries.append(dict(attempt=attempt.name,
                result=plan.read(attempt/'RESULT.json') if (attempt/'RESULT.json').exists() else None))
        result['lanes'].append(dict(gpu=gpu, attempts=entries))
    return result


if __name__ == '__main__':
    value = snapshot()
    name = 'monitor/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
    plan.save(name, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
