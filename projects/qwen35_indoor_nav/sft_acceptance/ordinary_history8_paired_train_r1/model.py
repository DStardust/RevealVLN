"""Original Qwen2B policy with already-validated FP32 master action parameters."""
import hashlib
from pathlib import Path
SOURCE=Path(__file__).resolve().parent.parent/'ordinary_sync_recovery_v1/model.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='89c2aac37d0d0519c01f51acf8f19b42087434f42e4cf56fc76fd01cf5297389'
exec(compile(SOURCE.read_text(),str(SOURCE)+':read-only-baseline','exec'),globals())
_native_build=build_policy
def build_policy(seed=1209):
    policy=_native_build(seed)
    policy.exec_embed.float()
    policy.action_query=nn.Parameter(policy.action_query.detach().float())
    return policy


