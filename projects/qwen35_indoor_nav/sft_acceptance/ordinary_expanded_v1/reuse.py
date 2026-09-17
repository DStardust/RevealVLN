"""Hash-bound read-only source reuse; no write to frozen predecessor."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'ordinary_sync_recovery_v1'
HASHES = {
    'data.py':'b54f149d16c13a37a535cf118464b83c398ec21145fe90bb6c7396c990715c70',
    'model.py':'89c2aac37d0d0519c01f51acf8f19b42087434f42e4cf56fc76fd01cf5297389',
    'control.py':'7502c5389f93581d0596be5d25c029ccba9a6adcbb7186ae234ca87caa10f68d',
    'train_filestore.py':'b56913d7d6f109c5a29b932653b3e31ef36021bfe36ce5b02eb65c973e814c2a',
    'supervise_filestore.py':'0937bbd041f8560ae24d8755c58c5c70825c33ff5d6130aca110ed08c3f3cb95',
    'launcher.py':'631dac96bbb386ec81bcba0a0ebf2039cecd92bce6c21df39d864f278b0cb58f',
    'lease_run.py':'06343e1da28f19df10311759e992dab59554bc4b3824621dca937d4692b22a89',
}


def source(name):
    raw = (OLD/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASHES[name], 'FROZEN_PARENT_CHANGED:'+name
    text = raw.decode()
    replacements = {}
    if name == 'train_filestore.py':
        replacements = {"'Q35N_ORDINARY_SYNC_RECOVERY_V1'":"'Q35N_ORDINARY_EXPANDED_V1'",
                        "HERE.parent / 'ordinary_baseline_v3/SAMPLE_INDEX.jsonl'":"HERE / 'SAMPLE_INDEX.jsonl'",
                        'legacy_charge_includes_conservative_reserve=True':'legacy_charge_includes_conservative_reserve=False'}
    if name == 'supervise_filestore.py':
        replacements = {'range(1, 4)':'range(1, 2)', 'max_attempts=3':'max_attempts=1',
                        '127.0.0.1:18767':'127.0.0.1:18766',
                        'triton_sync_recovery_v1':'triton_ordinary_expanded_v1',
                        'inductor_sync_recovery_v1':'inductor_ordinary_expanded_v1'}
    for old, new in replacements.items():
        assert text.count(old) == 1, 'TRANSFORM_COUNT:'+old
        text = text.replace(old, new)
    return text


def execute(name, namespace):
    exec(compile(source(name), str(HERE/name)+':hash-bound-parent', 'exec'), namespace)
