"""New parent+worker runtime after two zero-route approval failures; no bridge."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_AUTO=HERE.parent/'auto_generation_v1'
AUTO_LOCK='bda832bbf488af351737d13e7f9fbbfcb9db079d27fb9eabbf162f2d44a432cc'
assert hashlib.sha256((SOURCE_AUTO/'INPUT_LOCK.json').read_bytes()).hexdigest()==AUTO_LOCK
version_bytes=(SOURCE_AUTO/'transport.py').read_bytes()
assert hashlib.sha256(version_bytes).hexdigest()==json.loads((SOURCE_AUTO/'INPUT_LOCK.json').read_text())[str((SOURCE_AUTO/'transport.py').relative_to(HERE.parents[4]))]
version_code=version_bytes.decode()
old="\"RESCUE=HERE/'source_v1'\""
assert version_code.count(old)==1
version_code=version_code.replace(old,"\"RESCUE=HERE.parent/'auto_generation_v1/source_v1'\"")
assert version_code.count('Q35N_ORDINARY_AUTO_GENERATION_UNATTEMPTED_V1')==1
version_code=version_code.replace('Q35N_ORDINARY_AUTO_GENERATION_UNATTEMPTED_V1','Q35N_ORDINARY_AUTO_GENERATION_UNATTEMPTED_V2')
exec(compile(version_code,str(__file__),'exec'),globals())
_auto_common_source=common_source
_auto_run_source=run_source
_auto_prepare_source=prepare_source


def previous_evidence():
    project=HERE.parents[4]
    attempt=SOURCE_AUTO/'lanes/gpu_6/attempt_000'
    result=json.loads((attempt/'RESULT.json').read_text())
    assert result['error']=="AssertionError('WORKER_FAILED')"
    assert result['restoration']==json.loads((attempt/'RESTORATION.json').read_text()) and result['restoration']['restored'] is True
    assert len(result['workers'])==1 and result['workers'][0]['shard']==2 and result['workers'][0]['returncode']==1
    assert 'MAIN_APPROVAL_REQUIRED' in (attempt/'worker_2.log').read_text()
    process=json.loads((attempt/'PROCESS_2.json').read_text());assert not Path('/proc',str(process['pid'])).exists()
    roots={}
    for shard in (2,4):
        root=SOURCE_AUTO/'production'/f'shard_{shard:04d}'
        assert {p.name for p in root.iterdir()}=={'INPUT_LOCK.json','JOBS.json','SOURCE_INVENTORY.json','SPLIT_FREEZE.json','quality'},'PREVIOUS_ACTUAL_ROUTE_OR_LEDGER_PRESENT'
        assert not list((root/'quality').iterdir()),'PREVIOUS_QUALITY_NOT_EMPTY'
        roots[str(shard)]=dict(routes=0,ledger_rows=0,path=str(root.relative_to(project)))
    queue=HERE.parent.parent/'auto_production_v1'
    paths=[attempt/name for name in ('RESULT.json','RESTORATION.json','PROCESS_2.json','worker_2.log')]
    for version,ident in (('ordinary_queue_v1','ordinary_gpu6_unattempted_1424_v1'),('ordinary_queue_v2','ordinary_gpu6_unattempted_1424_approval_v2')):
        lane=queue/version/'lane_gpu_6';receipt=json.loads((lane/'RESULT.json').read_text())
        assert receipt['states']=={ident:'FAILED_NO_AUTOMATIC_RETRY'}
        assert receipt['automatic_retries']==0 and receipt['gpu_processes_signalled']==0
        paths += [lane/'RESULT.json',lane/(ident+'_production.log')]
    paths += [SOURCE_AUTO/'MAIN_AGENT_APPROVAL_GPU6.json',SOURCE_AUTO/'approval_bridge_v1/APPROVAL.json',SOURCE_AUTO/'approval_bridge_v1/launch.py']
    return dict(prior_wall_seconds=result['wall_seconds'],prior_worker_seconds={'2':result['workers'][0]['wall_seconds'],'4':0},
        zero_actual_route_evidence=roots,both_queue_failures_preserved=True,
        prior_hashes={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def common_source():
    source=_auto_common_source()
    source+='''

def claim_previous_auto_lane():
    import fcntl
    import transport
    handle=(transport.SOURCE_AUTO/'lanes/gpu_6/PRODUCER.lock').open('rb')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert transport.previous_evidence()==read(HERE/'PREVIOUS_FAILURES.json')
    return handle
'''
    return source


def run_source():
    source=_auto_run_source()
    source=exact(source,"    old_source_locks=c.claim_original_source_lanes()",
        "    previous_auto_lane=c.claim_previous_auto_lane()\n    old_source_locks=c.claim_original_source_lanes()")
    source=exact(source,"previous_seconds=sum(r['wall_seconds'] for r in old)",
        "previous_seconds=sum(r['wall_seconds'] for r in old)+c.read(HERE/'PREVIOUS_FAILURES.json')['prior_wall_seconds']")
    source=exact(source,"    assert previous_seconds<lane_limit-120,'LANE_BUDGET_EXHAUSTED'",
        "    for s,value in c.read(HERE/'PREVIOUS_FAILURES.json')['prior_worker_seconds'].items():spent[int(s)]+=value\n    assert previous_seconds<lane_limit-120,'LANE_BUDGET_EXHAUSTED'")
    return source


def prepare_source():
    source=_auto_prepare_source()
    source=exact(source,"    assert c.LANES==lanes and c.ROOT==transport.ROOT",
        "    assert c.LANES==lanes and c.ROOT==transport.ROOT\n    import fcntl\n    previous_auto_lane=(transport.SOURCE_AUTO/'lanes/gpu_6/PRODUCER.lock').open('rb')\n    fcntl.flock(previous_auto_lane,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    previous=transport.previous_evidence()\n    assert auth['prior_preworker_wall_seconds']==previous['prior_wall_seconds']\n    assert abs(auth['remaining_chain_wall_seconds']-(auth['lane_wall_seconds']['6']-previous['prior_wall_seconds']))<1e-9\n    assert auth['preserve_both_queue_approval_failures'] is True\n    c.save(HERE/'PREVIOUS_FAILURES.json',previous)")
    source=exact(source,"    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})",
        "    paths += [transport.SOURCE_AUTO/'transport.py',transport.SOURCE_AUTO/'INPUT_LOCK.json',HERE/'PREVIOUS_FAILURES.json']\n    paths += [c.ROOT/p for p in previous['prior_hashes']]\n    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})")
    return source
