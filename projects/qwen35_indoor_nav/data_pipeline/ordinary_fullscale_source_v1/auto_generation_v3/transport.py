"""V3: sibling data namespace; original size/resource/quality guards unchanged."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_AUTO=HERE.parent/'auto_generation_v1'
assert hashlib.sha256((SOURCE_AUTO/'INPUT_LOCK.json').read_bytes()).hexdigest()=='bda832bbf488af351737d13e7f9fbbfcb9db079d27fb9eabbf162f2d44a432cc'
raw=(SOURCE_AUTO/'transport.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()==json.loads((SOURCE_AUTO/'INPUT_LOCK.json').read_text())[str((SOURCE_AUTO/'transport.py').relative_to(HERE.parents[4]))]
exec(compile(raw,str(__file__),'exec'),globals())
RESCUE_SHA='b97447713ba4c840d461d115ce6d388be73ba140b4301d97502da7f0a787b3ce'
DATA=HERE.parent/'auto_generation_v3_production'
PREVIOUS=HERE.parent/'auto_generation_v2'
_auto_common_source=common_source
_auto_run_source=run_source
_auto_prepare_source=prepare_source

def rescue_jobs():
    import source
    rows,report=source.evidence()
    assert report==json.loads((RESCUE/'PREVIOUS_FAILURES.json').read_text()),'PREVIOUS_SOURCE_CHANGED'
    assert hashlib.sha256((RESCUE/'RESCUE_JOBS.json').read_bytes()).hexdigest()==RESCUE_SHA
    assert rows==json.loads((RESCUE/'RESCUE_JOBS.json').read_text())
    byid={j['job_id']:j for j in rows};result={}
    for shard,count in ((2,365),(4,998)):
        original=json.loads((HERE.parent/'manifests'/f'shard_{shard:04d}/JOBS.json').read_text())
        result[shard]=[j for j in original if j['job_id'] in byid]
        assert len(result[shard])==count
    assert [j for s in (2,4) for j in result[s]]==rows
    return result

def common_source():
    source=exact(_auto_common_source(),"return HERE/'production'/f'shard_{shard:04d}'",
                 "return HERE.parent/'auto_generation_v3_production'/f'shard_{shard:04d}'")
    source+='''

def claim_previous_auto_lane():
    import fcntl
    import transport
    handle=(transport.PREVIOUS/'lanes/gpu_6/PRODUCER.lock').open('rb')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    transport.rescue_jobs()
    return handle
'''
    return source

def run_source():
    source=_auto_run_source()
    source=exact(source,"    old_source_locks=c.claim_original_source_lanes()",
        "    previous_auto_lane=c.claim_previous_auto_lane()\n    old_source_locks=c.claim_original_source_lanes()")
    source=exact(source,"previous_seconds=sum(r['wall_seconds'] for r in old)",
        "previous_seconds=sum(r['wall_seconds'] for r in old)+c.read(HERE/'source_v1/PREVIOUS_FAILURES.json')['prior_wall_seconds']")
    source=exact(source,"    assert previous_seconds<lane_limit-120,'LANE_BUDGET_EXHAUSTED'",
        "    for s,value in c.read(HERE/'source_v1/PREVIOUS_FAILURES.json')['prior_worker_seconds'].items():spent[int(s)]+=value\n    assert previous_seconds<lane_limit-120,'LANE_BUDGET_EXHAUSTED'")
    source=exact(source,'    except BaseException as exc:error=repr(exc)', '''    except BaseException as exc:
        import traceback
        error=repr(exc)
        failure=dict(error=error,exception_type=type(exc).__name__,filename=getattr(exc,'filename',None),
                     filename2=getattr(exc,'filename2',None),errno=getattr(exc,'errno',None),
                     traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        try:c.save(out/'SUPERVISOR_EXCEPTION.json',failure)
        except BaseException as log_error:error+='; EXCEPTION_PERSISTENCE_FAILED:'+repr(log_error)''')
    return source

def prepare_source():
    source=_auto_prepare_source()
    source=source.replace("HERE/'production'","HERE.parent/'auto_generation_v3_production'")
    source=exact(source,"{'2':426,'4':998}","{'2':365,'4':998}")
    source=exact(source,"auth['max_new_routes']>=1424","auth['max_new_routes']==1363")
    source=exact(source,'Q35N_ORDINARY_AUTO_GENERATION_UNATTEMPTED_V1','Q35N_ORDINARY_AUTO_GENERATION_UNATTEMPTED_V3')
    source=exact(source,"    assert c.LANES==lanes and c.ROOT==transport.ROOT",'''    assert c.LANES==lanes and c.ROOT==transport.ROOT
    import fcntl
    previous_auto_lane=(transport.PREVIOUS/'lanes/gpu_6/PRODUCER.lock').open('rb')
    fcntl.flock(previous_auto_lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=c.read(transport.RESCUE/'PREVIOUS_FAILURES.json')
    assert auth['prior_preworker_wall_seconds']==previous['prior_wall_seconds']
    assert abs(auth['remaining_chain_wall_seconds']-(auth['lane_wall_seconds']['6']-previous['prior_wall_seconds']))<1e-9
    assert not transport.DATA.is_relative_to(HERE),'METADATA_DATA_MUST_BE_DISJOINT'
    assert auth['preserve_both_queue_approval_failures'] is True''')
    source=exact(source,'restore_exact_holder=True,scientific_pass=False,',
        "restore_exact_holder=True,scientific_pass=False,metadata_and_production_disjoint=True,\n        original_safe_size_and_resource_guards_unchanged=True,\n        prior_chain_wall_seconds=previous['prior_wall_seconds'],prior_worker_seconds=previous['prior_worker_seconds'],")
    source=exact(source,'    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})',
        "    paths += [transport.SOURCE_AUTO/'transport.py',transport.SOURCE_AUTO/'INPUT_LOCK.json',transport.RESCUE/'PREVIOUS_FAILURES.json']\n    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})")
    return source
