"""Fresh shard6 on GPU3: sealed V3 transport with no temporary sleeper."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_V3=HERE.parent/'runtime_v3'
raw=(SOURCE_V3/'transport.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='148432857e063e2c6b81fbb0de609c896e167e1476213f734338e582821382fe'
exec(compile(raw,str(__file__),'exec'),globals())
_v3_setup=setup
_v3_run_source=run_source
_v3_prepare_source=prepare_source


def setup():
    cfg=_v3_setup()
    if not (HERE/'SETUP.json').exists():cfg=dict(cfg,gpu_to_shards={'3':[6]})
    assert cfg['gpu_to_shards']=={'3':[6]},'GPU3_FRESH_SHARD6_ONLY'
    return cfg


def run_source():
    source=_v3_run_source()
    source=exact(source,"        call('tmux','respawn-pane','-t',identity['pane_target'],'-c',str(c.ROOT),'sleep 24000')\n        sleeper_identity=wait_stable_sleeper(identity);sleeper=sleeper_identity['pid']",
        '        # Retain the original dead pane until verified final holder restoration.')
    return exact(source,'choices=[3,4,6,7]','choices=[3]')


def prepare_source():
    source=_v3_prepare_source()
    source=exact(source,"default='{\"6\":[2,4],\"7\":[3,5]}'","default='{\"3\":[6]}'")
    source=exact(source,"node='Q35N_ORDINARY_FULLSCALE_RUNTIME_V3'","node='Q35N_ORDINARY_FULLSCALE_FRESH_SHARD6_RUNTIME_V5'")
    source=exact(source,'    lanes=transport.selected_queues(queues)',
        "    lanes=transport.selected_queues(queues)\n    assert lanes=={3:(6,)}\n    assert not (HERE.parent/'production/shard_0006').exists(),'SHARD6_MUST_BE_NEVER_ATTEMPTED'")
    source=exact(source,"HERE/'budget_preflight_v1'/name", "transport.SOURCE_V3/'budget_preflight_v1'/name")
    source=exact(source,'    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})',
        "    paths += [transport.SOURCE_V3/'transport.py']\n    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})")
    return source
