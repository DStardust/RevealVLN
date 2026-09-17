"""GPU7-only dead-pane retry; preserve V3 preworker failure and all root metadata."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_V3=HERE.parent/'runtime_v3'
V3_LOCK='17de3b1d3377f1b2f4a13175d1a273c4714561a9d3a99399addbb166b4991684'
raw=(SOURCE_V3/'transport.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='148432857e063e2c6b81fbb0de609c896e167e1476213f734338e582821382fe'
exec(compile(raw,str(__file__),'exec'),globals())
_v3_setup=setup
_v3_common_source=common_source
_v3_run_source=run_source
_v3_prepare_source=prepare_source


def setup():
    cfg=_v3_setup()
    if not (HERE/'SETUP.json').exists():cfg=dict(cfg,gpu_to_shards={'7':[3,5]})
    return cfg


def inspect_preworker_handoff(c):
    assert c.LANES=={7:(3,5)}
    assert c.sha(SOURCE_V3/'INPUT_LOCK.json')==V3_LOCK
    attempt=SOURCE_V3/'lanes/gpu_7/attempt_000'
    result=c.read(attempt/'RESULT.json')
    assert result['workers']==[] and result['error']=="AssertionError('UNEXPECTED_NEW_SLEEPER_COMMAND')"
    assert result['restoration']==c.read(attempt/'RESTORATION.json')
    assert not result['restoration']['restored']
    roots={}
    for shard in (3,5):
        root=c.shard_root(shard)
        assert {p.name for p in root.iterdir()}=={'JOBS.json','SPLIT_FREEZE.json','SOURCE_INVENTORY.json','INPUT_LOCK.json','quality'},'PREWORKER_ROOT_NOT_EMPTY'
        assert not list((root/'quality').iterdir()),'PREWORKER_QUALITY_NOT_EMPTY'
        assert c.sha(root/'INPUT_LOCK.json')==V3_LOCK
        roots[str(shard)]=dict(output_root=str(root.relative_to(c.ROOT)),root_input_lock_sha256=V3_LOCK,
            jobs_sha256=c.sha(root/'JOBS.json'),route_outputs_present=False)
    files=[attempt/name for name in ('RESULT.json','RESTORATION.json','LEASE_BEFORE.json')]
    files.append(SOURCE_V3/'INPUT_LOCK.json')
    return dict(status='PREWORKER_ONLY_EXCLUSIVE_OWNERSHIP_TRANSFER',old_runtime='runtime_v3',new_runtime='runtime_v4',
        gpu=7,shards=[3,5],old_failure_preserved=True,roots=roots,
        previous_failed_wall_seconds=result['wall_seconds'],
        old_receipt_sha256={str(p.relative_to(c.ROOT)):c.sha(p) for p in files})


def common_source():
    source=_v3_common_source()
    source+='''

LEGACY_ROOT_LOCK='''+repr(V3_LOCK)+'''

def claim_legacy_lane():
    import fcntl
    legacy=HERE.parent/'runtime_v3/lanes/gpu_7'
    handle=(legacy/'PRODUCER.lock').open('rb')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    handoff=read(HERE/'PREWORKER_OWNERSHIP.json')
    for path,h in handoff['old_receipt_sha256'].items():assert sha(ROOT/path)==h,'PREWORKER_RECEIPT_CHANGED'
    assert handoff['shards']==[3,5] and LANES=={7:(3,5)}
    return handle
'''
    return source


def run_source():
    source=_v3_run_source()
    source=exact(source,"        call('tmux','respawn-pane','-t',identity['pane_target'],'-c',str(c.ROOT),'sleep 24000')\n        sleeper_identity=wait_stable_sleeper(identity);sleeper=sleeper_identity['pid']",
        '        # Keep the verified original pane dead; no temporary process is created.')
    source=exact(source,"    limits=c.read(HERE/'PREPARED_CONFIG.json')",
        "    legacy_lane_lock=c.claim_legacy_lane()\n    limits=c.read(HERE/'PREPARED_CONFIG.json')")
    source=exact(source,"previous_seconds=sum(r['wall_seconds'] for r in old)",
        "previous_seconds=sum(r['wall_seconds'] for r in old)+c.read(HERE/'PREWORKER_OWNERSHIP.json')['previous_failed_wall_seconds']")
    source=exact(source,'choices=[3,4,6,7]','choices=[7]')
    return source


def worker_entry_source():
    return exact(original('worker.py'),"assert c.read(root/'INPUT_LOCK.json')==lock,'SHARD_INPUT_LOCK_MISMATCH'",
        "assert c.sha(root/'INPUT_LOCK.json')==c.LEGACY_ROOT_LOCK,'SHARD_INPUT_LOCK_MISMATCH'")


def prepare_source():
    source=_v3_prepare_source()
    source=exact(source,"default='{\"6\":[2,4],\"7\":[3,5]}'","default='{\"7\":[3,5]}'")
    source=exact(source,"node='Q35N_ORDINARY_FULLSCALE_RUNTIME_V3'","node='Q35N_ORDINARY_FULLSCALE_GPU7_DEAD_PANE_RUNTIME_V4'")
    source=exact(source,'    assert c.LANES==lanes and c.ROOT==transport.ROOT',
        "    assert c.LANES==lanes and c.ROOT==transport.ROOT\n    assert lanes=={7:(3,5)}\n    import fcntl\n    prior_lane_lock=(transport.SOURCE_V3/'lanes/gpu_7/PRODUCER.lock').open('rb')\n    fcntl.flock(prior_lane_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    handoff=transport.inspect_preworker_handoff(c)\n    c.save(HERE/'PREWORKER_OWNERSHIP.json',handoff)")
    source=exact(source,"root=c.shard_root(shard);root.mkdir(parents=True,exist_ok=False)\n        (root/'quality').mkdir()",
        "root=c.shard_root(shard);assert root.is_dir()\n        assert c.sha(root/'INPUT_LOCK.json')==transport.V3_LOCK")
    source=exact(source,'c.save(root/name,value);outputs.append(root/name)',
        "assert c.read(root/name)==value,'PREWORKER_METADATA_CHANGED';outputs.append(root/name)")
    source=exact(source,"for shard in c.SELECTED_SHARDS:c.save(c.shard_root(shard)/'INPUT_LOCK.json',lock)",
        "for shard in c.SELECTED_SHARDS:assert c.sha(c.shard_root(shard)/'INPUT_LOCK.json')==transport.V3_LOCK")
    source=exact(source,"HERE/'budget_preflight_v1'/name", "transport.SOURCE_V3/'budget_preflight_v1'/name")
    source=exact(source,"    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})",
        "    paths += [transport.SOURCE_V3/'transport.py',transport.SOURCE_V3/'INPUT_LOCK.json',HERE/'PREWORKER_OWNERSHIP.json']\n    paths += [c.ROOT/p for p in handoff['old_receipt_sha256']]\n    paths += [c.shard_root(s)/'INPUT_LOCK.json' for s in c.SELECTED_SHARDS]\n    paths += [c.PARALLEL.parent/'ordinary_parallel_v1/preworker_holder_recovery_v3/run_v1/RESULT.json']\n    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})")
    return source
