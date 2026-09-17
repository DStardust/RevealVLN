"""Save one read-only training status snapshot; do not treat window metrics as eval."""
from pathlib import Path
import json
import hashlib
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'sft_acceptance/ordinary_baseline_v3/formal/run_0001/PROGRESS.json'
raw=SOURCE.read_bytes();progress=json.loads(raw)
result=dict(source=str(SOURCE),source_bytes_sha256=hashlib.sha256(raw).hexdigest(),
            read_utc=datetime.now(timezone.utc).isoformat(),progress=progress,
            frozen_training_pool_decisions=1394744,
            cumulative_compute_is_not_unique_sample_coverage=True,
            rolling_training_metrics_are_not_closed_loop_evaluation=True)
with (HERE/'TRAINING_READONLY_SNAPSHOT.json').open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps(dict(updates=progress['cursor']['updates'],
    compute_decisions=progress['cumulative_compute']['decisions'],unix=progress['unix'],
    status=progress['status']),indent=2))
