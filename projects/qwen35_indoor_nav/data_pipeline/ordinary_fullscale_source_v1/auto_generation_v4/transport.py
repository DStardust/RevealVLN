"""V4 sibling transport plus explicit active-only conservative telemetry amendment."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT_V3=HERE.parent/'auto_generation_v3'
assert hashlib.sha256((PARENT_V3/'INPUT_LOCK.json').read_bytes()).hexdigest()=='cf3d6eb17f17c5c5b274cd4478b8236ae9ec8e4e541745b887aa60612c2e42c7'
raw=(PARENT_V3/'transport.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()==json.loads((PARENT_V3/'INPUT_LOCK.json').read_text())[str((PARENT_V3/'transport.py').relative_to(HERE.parents[4]))]
code=raw.decode()
def replace(old,new,count=1):
    global code
    assert code.count(old)==count,('V4_TRANSPORT',old,code.count(old),count)
    code=code.replace(old,new)
replace('b97447713ba4c840d461d115ce6d388be73ba140b4301d97502da7f0a787b3ce','ca0a291592b8a008e8acee91115260b69112d0e8e628493bc6c31f007ce84dcf')
replace('auto_generation_v3_production','auto_generation_v4_production',3)
replace("PREVIOUS=HERE.parent/'auto_generation_v2'","PREVIOUS=HERE.parent/'auto_generation_v3'")
replace('((2,365),(4,998))','((2,364),(4,998))')
replace("{'2':365,'4':998}","{'2':364,'4':998}")
replace("auth['max_new_routes']==1363","auth['max_new_routes']==1362")
replace('Q35N_ORDINARY_AUTO_GENERATION_UNATTEMPTED_V3','Q35N_ORDINARY_AUTO_GENERATION_UNATTEMPTED_V4')
replace('original_safe_size_and_resource_guards_unchanged=True,',
        "disk_census_unchanged=True,quantitative_memory_limits_unchanged=True,gpu_accounting_guard_amended=True,")
exec(compile(code,str(__file__),'exec'),globals())
_parent_run_source=run_source
_parent_prepare_source=prepare_source

def run_source():
    source=_parent_run_source()
    helpers='''def active_accounting(snapshot,own,out,shard,elapsed):
    import telemetry
    def record(receipt):
        row=dict(receipt,shard=shard,elapsed_seconds=elapsed,
                 original_sample_file='GPU_SNAPSHOTS.jsonl',original_sample_elapsed=elapsed)
        with (out/'GPU_ACCOUNTING_DECISIONS.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\\n');handle.flush();os.fsync(handle.fileno())
    return telemetry.active_assessment(snapshot,own,contexts,record)


'''
    source=exact(source,'def stop_own(proc):',helpers+'def stop_own(proc):')
    old="                        with (out/'GPU_SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(dict(snapshot,shard=shard,elapsed=time.monotonic()-started))+'\\n')\n                        contexts(snapshot,proc.pid)"
    new="""                        snapshot_elapsed=time.monotonic()-started
                        with (out/'GPU_SNAPSHOTS.jsonl').open('a') as f:
                            f.write(json.dumps(dict(snapshot,shard=shard,elapsed=snapshot_elapsed),allow_nan=False)+'\\n');f.flush();os.fsync(f.fileno())
                        active_accounting(snapshot,proc.pid,out,shard,snapshot_elapsed)"""
    source=exact(source,old,new)
    return source

def prepare_source():
    source=_parent_prepare_source()
    source=exact(source,"    assert auth['preserve_both_queue_approval_failures'] is True",
        "    assert auth['preserve_both_queue_approval_failures'] is True\n    assert auth['gpu_accounting_guard_amended'] is True and auth['quantitative_memory_limits_unchanged'] is True\n    amendment=auth['gpu_accounting_amendment']\n    assert amendment['version']=='conservative_max_active_v1'\n    assert amendment['reported_total']=='max(device_memory_mib,sum(process_memory_mib))'\n    assert amendment['inconsistent_sample_total_must_be_less_than_mib']==4096\n    assert amendment['external_process_max_mib']==768 and amendment['external_sum_max_mib']==2048\n    assert amendment['raw_readings_preserved'] is True\n    assert amendment['old_guard_pass_not_claimed_for_inconsistent_samples'] is True")
    source=exact(source,'    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})',
        "    paths += [transport.PARENT_V3/'transport.py',transport.PARENT_V3/'INPUT_LOCK.json',transport.PARENT_V3/'source.py']\n    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})")
    return source
