"""Prospective V3: exact V2 transport plus stable sleeper capture, CPU prepared only."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_V2=HERE.parent/'runtime_v2'
EXPECTED_V2={
    'transport.py':'33831db012a28419ff896bf8b029e28712a7cc9b9c7c1d15c6aa610739f23a0a',
    'prepare.py':'9a01f007de65ecdbc7a8a65e6c2e061def64cbfab04851d3a171bc9b949a15ac',
    'merge_all.py':'8dc1012a2124ac78365caa7568d93a535696a8c88cfcedf8ab3bb1d148aee43a',
    'test_safety_inherited.py':'bc654603423cb5945210dd64ad4663242eda1e410243de83812260ffd1998ffc'}


def v2_source(name):
    path=SOURCE_V2/name;raw=path.read_bytes()
    assert path.resolve(strict=True).is_relative_to(HERE.parents[4])
    assert hashlib.sha256(raw).hexdigest()==EXPECTED_V2[name],('SEALED_V2_CHANGED',name)
    return raw.decode()


exec(compile(v2_source('transport.py'),str(__file__),'exec'),globals())
_v2_setup=setup
_v2_run_source=run_source


def setup():
    cfg=_v2_setup()
    if not (HERE/'SETUP.json').exists():cfg=dict(cfg,gpu_to_shards={'6':[2,4],'7':[3,5]})
    return cfg


def run_source():
    source=_v2_run_source()
    helper='''def wait_stable_sleeper(identity):
    previous=None;signature=None
    for _ in range(80):
        assert pane(identity,'#{pane_id}')==identity['pane_id'],'PANE_ID_CHANGED'
        pid=int(pane(identity,'#{pane_pid}'))
        assert pid>0 and pid!=identity['pid'],'NEW_SLEEPER_PID_REQUIRED'
        assert pane(identity,'#{pane_dead}')=='0','NEW_SLEEPER_NOT_LIVE'
        try:actual=process_identity(pid)
        except FileNotFoundError:
            time.sleep(.25);continue
        assert actual['cwd']==identity['cwd'] and actual['proc_uid']==identity['proc_uid'],'SLEEPER_CONTEXT_MISMATCH'
        current=(actual['pid'],actual['starttime_ticks'],actual['proc_uid'],actual['cwd'])
        if signature is not None:assert current==signature,'SLEEPER_PID_REUSED_OR_REPLACED'
        signature=current
        if actual['cmdline']==['']:
            previous=None;time.sleep(.25);continue
        assert actual['cmdline']==['sleep','24000'],'UNEXPECTED_NEW_SLEEPER_COMMAND'
        if previous==actual:return actual
        previous=actual;time.sleep(.25)
    raise AssertionError('SLEEPER_EXEC_STABILITY_TIMEOUT')


'''
    source=exact(source,'def restore_holder(identity,sleeper_identity,remain,out,released_pid=None):',
        helper+'def restore_holder(identity,sleeper_identity,remain,out,released_pid=None):')
    source=exact(source,"sleeper=int(pane(identity,'#{pane_pid}'));sleeper_identity=process_identity(sleeper)",
        "sleeper_identity=wait_stable_sleeper(identity);sleeper=sleeper_identity['pid']")
    source=exact(source,'started=time.monotonic();c.approved(gpu)',
        "started=time.monotonic();c.approved(gpu)\n    limits=c.read(HERE/'PREPARED_CONFIG.json')\n    lane_limit=limits['lane_seconds'][str(gpu)]\n    shard_limits={int(s):v for s,v in limits['shard_seconds'].items()}\n    assert limits['cleanup_margin_seconds']==120 and lane_limit<24000")
    assert source.count('<7080')==3,'LANE_GUARD_EXACT_COUNT'
    source=source.replace('<7080','<lane_limit-120')
    assert source.count('<3480')==2,'SHARD_GUARD_EXACT_COUNT'
    source=source.replace('<3480','<shard_limits[shard]-120')
    return source


def budget_from_authorization(auth,lanes):
    assert auth['cleanup_margin_seconds']==120
    shards=auth['shard_wall_seconds'];lane=auth['lane_wall_seconds']
    assert set(shards)=={str(s) for rows in lanes.values() for s in rows}
    assert set(lane)=={str(g) for g in lanes}
    assert all(type(v) is int and 120<v<24000 for v in list(shards.values())+list(lane.values()))
    for gpu,rows in lanes.items():
        assert sum(shards[str(s)] for s in rows)+120<=lane[str(gpu)],'LANE_MUST_INCLUDE_ALL_PHASES_AND_RESTORE'
        assert lane[str(gpu)]<=auth['max_wall_seconds_per_gpu_chain']
    return shards,lane


def prepare_source():
    source=v2_source('prepare.py')
    source=exact(source,"default='{\"3\":[0],\"4\":[1]}'","default='{\"6\":[2,4],\"7\":[3,5]}'")
    source=exact(source,"node='Q35N_ORDINARY_FULLSCALE_RUNTIME_V2'","node='Q35N_ORDINARY_FULLSCALE_RUNTIME_V3'")
    source=exact(source,"assert auth['max_wall_seconds_per_gpu_chain']>=7200",
        "shard_budget,lane_budget=transport.budget_from_authorization(auth,lanes)")
    source=exact(source,'shard_seconds=3600,lane_seconds=7200,worker_seconds_per_shard_with_cleanup_margin=3480,',
        'shard_seconds=shard_budget,lane_seconds=lane_budget,cleanup_margin_seconds=120,\n        worker_seconds_per_shard_with_cleanup_margin={s:v-120 for s,v in shard_budget.items()},')
    source=exact(source,'paths += [transport.SOURCE/name for name in transport.EXPECTED]',
        "paths += [transport.SOURCE/name for name in transport.EXPECTED]\n    paths += [transport.SOURCE_V2/name for name in transport.EXPECTED_V2]\n    paths += [HERE/'budget_preflight_v1'/name for name in ('FORECAST.json','INPUT_HASHES.json','estimate.py')]")
    return source
