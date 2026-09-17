"""Exact source freezer adaptation: new V3 partial excluded, no old output writes."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
raw=(HERE.parent/'auto_generation_v3/source.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='5e91ab37bd097ea32710e616f717c7851bb21fe487e6b41e6a7931b2372596c6'
code=raw.decode()
def replace(old,new,count=1):
    global code
    assert code.count(old)==count,('SOURCE_ADAPTER',old,code.count(old),count)
    code=code.replace(old,new)
replace("PREVIOUS=FULL/'auto_generation_v2'","PREVIOUS=FULL/'auto_generation_v3'")
replace("result['error']==\"FileNotFoundError(2, 'No such file or directory')\"","result['error']==\"AssertionError('GPU_MEMORY_ACCOUNTING')\"")
replace("PREVIOUS/'PREVIOUS_FAILURES.json'","PREVIOUS/'source_v1/PREVIOUS_FAILURES.json'",2)
replace("root=PREVIOUS/'production'/f'shard_{shard:04d}'","root=FULL/'auto_generation_v3_production'/f'shard_{shard:04d}'")
replace("FULL/'auto_generation_v1/source_v1/RESCUE_JOBS.json'","FULL/'auto_generation_v3/source_v1/RESCUE_JOBS.json'")
replace("failed_supervisor_path_unknown=True","failed_supervisor_path_unknown=False,failed_contexts_proven_by_saved_traceback=True")
replace("'RESULT.json','RESTORATION.json','PROCESS_2.json','worker_2.log'","'RESULT.json','RESTORATION.json','PROCESS_2.json','worker_2.log','SUPERVISOR_EXCEPTION.json','GPU_SNAPSHOTS.jsonl'")
replace("PREVIOUS/'production/shard_0002/PRODUCER.lock'","FULL/'auto_generation_v3_production/shard_0002/PRODUCER.lock'")
replace("PREVIOUS/'production/shard_0004/PRODUCER.lock'","FULL/'auto_generation_v3_production/shard_0004/PRODUCER.lock'")
replace('len(rows)==1363','len(rows)==1362')
replace("{'2':365,'4':998}","{'2':364,'4':998}")
replace('NEVER_ATTEMPTED_AFTER_V2_SUPERVISOR_FAILURE','NEVER_ATTEMPTED_AFTER_V3_ACCOUNTING_FAILURE')
replace("FULL/'auto_generation_v3_production').relative_to(ROOT)","FULL/'auto_generation_v4_production').relative_to(ROOT)")
exec(compile(code,str(__file__),'exec'),globals())
