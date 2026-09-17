"""Versioned holder-only recovery after exact budget censor; does not certify partial data."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'holder_recovery_v2'
EXPECTED_SOURCE_LOCK='8e0a0e222ec3ef10dcd8cd447def24ce3e23fcfd3c8156204ab735472c25085d'
PROJECT=HERE.parents[4]
ALLOWED_BUDGET_ERRORS={"AssertionError('SHARD_WALL_BUDGET')","AssertionError('LANE_WALL_BUDGET')"}


def source_file(name):
    lock_path=SOURCE/'INPUT_LOCK.json';raw=lock_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED_SOURCE_LOCK
    lock=json.loads(raw);path=SOURCE/name
    assert path.resolve(strict=True).is_relative_to(PROJECT)
    content=path.read_bytes()
    assert hashlib.sha256(content).hexdigest()==lock[str(path.relative_to(PROJECT))]
    return content.decode()


def exact(source,old,new):
    assert source.count(old)==1,('EXACT_RECOVERY_TRANSPORT_COUNT',old)
    return source.replace(old,new)


def validate_resource_workers(result,processes,assigned,gpu,process_exists=None):
    assert result['error'] in ALLOWED_BUDGET_ERRORS,'UNREGISTERED_CENSOR_ERROR'
    assert {w['shard'] for w in result['workers']}==set(assigned)
    assert len(result['workers'])==len(assigned)
    if process_exists is None:process_exists=lambda pid:Path('/proc',str(pid)).exists()
    for worker in result['workers']:
        assert type(worker['returncode']) is int and worker['returncode'] in (0,1,-15,-9),'UNKNOWN_WORKER_EXIT'
        process=processes[worker['shard']]
        assert process['gpu']==gpu and process['shard']==worker['shard']
        assert type(process['pid']) is int and process['pid']>0
        assert not process_exists(process['pid']),'OWN_WORKER_PID_STILL_PRESENT'


def recovery_source():
    source=source_file('recover.py')
    source=exact(source,"NODE='EMPTY_ARGV_SAME_SLEEPER_NATURAL_CLOSE_RECOVERY_V2'",
        "NODE='BUDGET_CENSOR_EMPTY_ARGV_HOLDER_RECOVERY_V3'")
    source=exact(source,"\n\n\ndef sha(path):", 
        "\nEXPECTED_RUNTIME_LOCKS={gpu:value for gpu,value in EXPECTED_RUNTIME_LOCKS.items() if gpu in (3,4)}\n\n\ndef sha(path):")
    source=exact(source,"assert result['error'] is None,'NOT_NATURAL_CLOSURE'",
        "assert result['error'] in ALLOWED_BUDGET_ERRORS,'UNREGISTERED_CENSOR_ERROR'")
    source=exact(source,"    assert all(w['returncode']==0 for w in result['workers']),'WORKER_NOT_SUCCESSFULLY_CLOSED'\n    for shard in c.LANES[gpu]:assert c.state(shard)['complete'],'INCOMPLETE_SHARD'",
        "    processes={s:read(attempt/f'PROCESS_{s}.json') for s in c.LANES[gpu]}\n    validate_resource_workers(result,processes,c.LANES[gpu],gpu)")
    source=exact(source,"files=[attempt/name for name in ('RESULT.json','RESTORATION.json','LEASE_BEFORE.json','LEASE_ACTIVE.json')]",
        "files=[attempt/name for name in ('RESULT.json','RESTORATION.json','LEASE_BEFORE.json','LEASE_ACTIVE.json')]\n    files += [attempt/f'PROCESS_{s}.json' for s in c.LANES[gpu]]")
    source=exact(source,"old_runtime_result_overridden=False,external_processes_stopped=0,scientific_pass=False)",
        "old_runtime_result_overridden=False,external_processes_stopped=0,scientific_pass=False,\n            data_status='UNCHANGED_RESOURCE_CENSORED_NOT_FULLY_GENERATED',full_dataset_certified=False)")
    source=exact(source,'choices=[3,4,6,7]','choices=[3,4]')
    return source


exec(compile(recovery_source(),str(__file__),'exec'),globals())
