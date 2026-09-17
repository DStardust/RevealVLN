"""CPU-only receipt gate for censored full-source shards 0/1."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
FULL=HERE.parent
SOURCE=FULL.parent/'ordinary_parallel_v1/merge_receipt_v2/gate.py'
raw=SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='f513199fff559e44fc81f271d1e5b9b22a8e7091faa84afa9f1618066a821317'
source=raw.decode()
def exact(old,new):
    global source
    assert source.count(old)==1,('EXACT_GATE_COUNT',old)
    source=source.replace(old,new)
exact("PARALLEL=HERE.parent", "PARALLEL=HERE.parent.parent/'ordinary_parallel_v1'")
exact("RECOVERY=PARALLEL/'holder_recovery_v2'", "RECOVERY=PARALLEL/'holder_recovery_v3'")
exact('8e0a0e222ec3ef10dcd8cd447def24ce3e23fcfd3c8156204ab735472c25085d',
      'eebae76ebaa2a30406b6a095eb14dff3c7f8267e7029c1e88434796c6e08b1b7')
exact("assert gpu in (6,7),'THIS_MERGE_ONLY_ORIGINAL_PARALLEL_GPU67'", "assert gpu in (3,4),'THIS_GATE_ONLY_CENSORED_GPU34'")
exact("assert result['error'] is None", "assert result['error'] in {\"AssertionError('SHARD_WALL_BUDGET')\",\"AssertionError('LANE_WALL_BUDGET')\"}")
exact("PARALLEL/'runtime_v1/lanes'", "PARALLEL.parent/'ordinary_fullscale_source_v1/runtime_v2/lanes'")
exact("('RESULT.json','RESTORATION.json','LEASE_BEFORE.json','LEASE_ACTIVE.json')", "('RESULT.json','RESTORATION.json','LEASE_BEFORE.json','LEASE_ACTIVE.json',f'PROCESS_{gpu-3}.json')")
exact("node='EMPTY_ARGV_SAME_SLEEPER_NATURAL_CLOSE_RECOVERY_V2'", "node='BUDGET_CENSOR_EMPTY_ARGV_HOLDER_RECOVERY_V3'")
exec(compile(source,str(__file__),'exec'),globals())
