"""Prospective EnvDrop transport: unchanged compiler/audit, explicit V4 telemetry."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PIPE=HERE.parent.parent
SOURCE=PIPE/'ordinary_parallel_v1/runtime_v1'
V4=PIPE/'ordinary_fullscale_source_v1/auto_generation_v4'
DATA=HERE.parent/'envdrop_production_v1'
MANIFEST=HERE.parent/'envdrop_source_v1'
EXPECTED={
 'common.py':'c6fa96828a5d5d0e07be467b9dcb02180ce5b73184a789748aa8b5ffb69e4840',
 'run.py':'c674250e1dff16028e933041ed63c5242427ce6ee22f617dab604e896258e5dd',
 'worker.py':'d25df35d2d27840100e1fdf1bc525a0105dd27094a2fb420c9dd69593fed1bc2',
 'merge.py':'0c16272e4430b229f199c8b5329f681129c4ad0cb948fb10fd99896db352735f',
 'safe_size.py':'957ec7851bd448685690d9065882ce208461f733731bbaaa65838cf287b4530d',
}

def original(name):
    raw=(SOURCE/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED[name],('SEALED_SOURCE_CHANGED',name)
    return raw.decode()

def exact(source,old,new,count=1):
    assert source.count(old)==count,('EXACT_ENVDROP_TRANSPORT_COUNT',old,source.count(old),count)
    return source.replace(old,new)

def common_source():
    source=original('common.py')
    source=exact(source,'ORDINARY_PARALLEL_GPU67_IDENTITIES_V1.json','ORDINARY_ENVDROP_GPU6_IDENTITY_V1.json')
    source=exact(source,'ORDINARY_PARALLEL_PRODUCTION_V1.json','ORDINARY_ENVDROP_GPU6_PRODUCTION_V1.json')
    source=exact(source,'LANES={6:(0,2),7:(1,3)}','LANES={6:tuple(range(21))}\nSELECTED_SHARDS=tuple(range(21))')
    source=exact(source,'0<=shard<4','0<=shard<21')
    source=exact(source,"return PARALLEL/'production'/f'shard_{shard:04d}'","return PARALLEL/'envdrop_production_v1'/f'shard_{shard:04d}'")
    source=exact(source,'def lane_for(shard):return 6 if shard in (0,2) else 7',
        'def lane_for(shard):\n    assert type(shard) is int and shard in SELECTED_SHARDS\n    return 6')
    return source

def run_source():
    source=original('run.py')
    source=exact(source,"        call('tmux','respawn-pane','-t',identity['pane_target'],'-c',str(c.ROOT),'sleep 24000')\n        sleeper=int(pane(identity,'#{pane_pid}'));sleeper_identity=process_identity(sleeper)",
        "        # Dead pane is retained until finally: no temporary sleeper or argv race.")
    source=exact(source,'started=time.monotonic();c.approved(gpu)',
        "started=time.monotonic();c.approved(gpu)\n    limits=c.read(HERE/'PREPARED_CONFIG.json')\n    lane_limit=limits['lane_seconds'][str(gpu)]\n    shard_limits={int(s):v for s,v in limits['shard_seconds'].items()}\n    assert limits['cleanup_margin_seconds']==120 and lane_limit==82800")
    source=exact(source,'<7080','<lane_limit-120',3)
    source=exact(source,'<3480','<shard_limits[shard]-120',2)
    source=exact(source,'<48*1024**3','<23*1024**3')
    source=exact(source,'# 1GiB early-stop margin within the independently allocated 49GiB cap.',
        '# 1GiB early-stop margin within the independently allocated 24GiB cap.')
    source=exact(source,"    old=[]\n    for p in sorted(lane.glob('attempt_*')):",
        "    assert not list(lane.glob('attempt_*')),'NEW_ENVDROP_LANE_NO_AUTOMATIC_RETRY'\n    old=[]\n    for p in sorted(lane.glob('attempt_*')):")
    source=exact(source,"        for shard in pending:\n            assert time.monotonic()-started+previous_seconds<lane_limit-120,'LANE_WALL_BUDGET'",
        "        for shard in pending:\n            assert time.monotonic()-started+previous_seconds<lane_limit-120,'LANE_WALL_BUDGET'\n            require_phase_budget(lane_limit,time.monotonic()-started+previous_seconds,shard_limits[shard],out,shard)\n            if shard!=0:require_sentinel_receipt(out)")
    source=exact(source,"                disk_guard(shard)\n            finally:",
        "                disk_guard(shard)\n                if shard==0:run_sentinel_gate(out,env)\n            finally:")
    source=exact(source,'def stop_own(proc):',HELPERS+'def stop_own(proc):')
    source=exact(source,"                        with (out/'GPU_SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(dict(snapshot,shard=shard,elapsed=time.monotonic()-started))+'\\n')\n                        contexts(snapshot,proc.pid)",
        "                        snapshot_elapsed=time.monotonic()-started\n                        with (out/'GPU_SNAPSHOTS.jsonl').open('a') as f:\n                            f.write(json.dumps(dict(snapshot,shard=shard,elapsed=snapshot_elapsed),allow_nan=False)+'\\n');f.flush();os.fsync(f.fileno())\n                        active_accounting(snapshot,proc.pid,out,shard,snapshot_elapsed)")
    source=exact(source,'    except BaseException as exc:error=repr(exc)',
        "    except BaseException as exc:\n        import traceback\n        error=repr(exc)\n        failure=dict(error=error,exception_type=type(exc).__name__,filename=getattr(exc,'filename',None),errno=getattr(exc,'errno',None),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)\n        try:c.save(out/'SUPERVISOR_EXCEPTION.json',failure)\n        except BaseException as log_error:error+='; EXCEPTION_PERSISTENCE_FAILED:'+repr(log_error)")
    return exact(source,'choices=[6,7]','choices=[6]')

HELPERS='''def active_accounting(snapshot,own,out,shard,elapsed):
    import telemetry
    def record(receipt):
        row=dict(receipt,shard=shard,elapsed_seconds=elapsed,
                 original_sample_file='GPU_SNAPSHOTS.jsonl',original_sample_elapsed=elapsed)
        with (out/'GPU_ACCOUNTING_DECISIONS.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\\n');handle.flush();os.fsync(handle.fileno())
    return telemetry.active_assessment(snapshot,own,contexts,record)


def require_phase_budget(lane_limit,elapsed,phase_limit,out,shard):
    remaining=lane_limit-elapsed
    if remaining<phase_limit+120:
        c.save(out/'TAIL_NOT_STARTED.json',dict(shard=shard,remaining_seconds=remaining,
            required_phase_and_restore_seconds=phase_limit+120,status='UNATTEMPTED_BUDGET_CENSOR'))
        raise AssertionError('INSUFFICIENT_FULL_PHASE_BUDGET_NO_START')


def require_sentinel_receipt(out):
    value=c.read(out/'SENTINEL_GATE.json')
    assert value['strict_pass'] is True and value['strict_routes']==3,'SENTINEL_GATE_REQUIRED'
    assert value['source_jobs_sha256']==c.sha(c.shard_root(0)/'JOBS.json')
    assert value['input_lock_sha256']==c.sha(HERE/'INPUT_LOCK.json')
    for path,sha in value['input_hashes'].items():assert c.sha(c.ROOT/path)==sha,'SENTINEL_EVIDENCE_CHANGED'


def run_sentinel_gate(out,env):
    with (out/'SENTINEL_GATE.log').open('x') as log:
        result=subprocess.run([str(c.ENV/'bin/python3'),'-I','-B',str(HERE/'sentinel_gate.py'),
            '--output',str(out/'SENTINEL_GATE.json')],cwd=c.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert result.returncode==0,'SENTINEL_STRICT_GATE_FAILED_NO_REPLACEMENT'
    require_sentinel_receipt(out)


'''

def merge_source():
    source=original('merge.py')
    source=exact(source,"out=c.PARALLEL/'merge'","out=HERE/'merge'")
    source=exact(source,"            assert result['restoration']['restored'],'HOLDER_NOT_RESTORED'",
        "            assert result['restoration']['restored'],'HOLDER_NOT_RESTORED'\n            assert result['error'] is None and all(w['returncode']==0 for w in result['workers']),'PRODUCTION_NOT_SUCCESSFULLY_CLOSED'")
    source=exact(source,"c.read(c.PARALLEL/'JOBS.json')","c.read(c.PARALLEL/'envdrop_source_v1/JOBS.json')")
    source=exact(source,"c.read(c.PARALLEL/'EXCLUSION_MANIFEST.json')['unique_physical_route_keys_excluded']",
        "c.read(c.PARALLEL/'envdrop_source_v1/PHYSICAL_EXCLUSION.json')['physical_route_keys']")
    source=exact(source,'len(expected_all)==1000','len(expected_all)==20000')
    source=exact(source,'for shard in range(4):','for shard in c.SELECTED_SHARDS:')
    source=exact(source,"        if not state['complete']:\n            receipt=c.read(root/'CENSOR_RECEIPT.json')\n            assert receipt==dict(approved=True,shard=shard,ledger_sha256=state['ledger_sha256'],\n                status='EXPLICIT_RESOURCE_CENSOR_NO_UNATTEMPTED_NEGATIVE_LABELS')",
        "        assert state['complete'],'FULL_BATCH_REQUIRED_NO_PARTIAL_PROMOTION'")
    source=exact(source,'<199*1024**3','<511*1024**3')
    return source
