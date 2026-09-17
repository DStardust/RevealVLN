"""Read-only live monitor; rolling training metrics are NOT terminal evaluation."""
import collections
import json
from pathlib import Path
import statistics

p=Path(__file__).resolve().parent
def rows(name):
    f=p/name
    if not f.exists():return []
    return [json.loads(x) for x in f.read_bytes().splitlines(keepends=True) if x.endswith(b'\n')]

u=[x for x in rows('rank0_events.jsonl') if x['event']=='update']
steps=sorted([x for r in range(2) for x in rows(f'rank{r}_train_steps.jsonl')],key=lambda x:x['update'])
recent=steps[-320:]
durations=[b['unix']-a['unix'] for a,b in zip(u[-21:],u[-20:])] if len(u)>20 else [b['unix']-a['unix'] for a,b in zip(u,u[1:])]
print(json.dumps(dict(updates=len(u),last=u[-1] if u else None,
    recent_median_update_seconds=statistics.median(durations) if durations else None,
    rolling_decisions=len(recent),rolling_training_accuracy=sum(x['target']==x['prediction'] for x in recent)/len(recent) if recent else None,
    rolling_predictions=collections.Counter(str(x['prediction']) for x in recent),
    rolling_target_counts=collections.Counter(str(x['target']) for x in recent),
    rolling_memory_rms_max=max((x['memory_rms'] for x in recent),default=None),
    execution_closed=(p/'EXECUTION.json').exists()),ensure_ascii=False))
