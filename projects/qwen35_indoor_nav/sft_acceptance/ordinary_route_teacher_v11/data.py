"""Same loader and input whitelist; hash-bound route targets and data binding."""
import hashlib
from pathlib import Path
_p=Path(__file__).resolve().parent.parent/'ordinary_onpolicy_adapt_v6/data.py'
assert hashlib.sha256(_p.read_bytes()).hexdigest()=='c11a85fe227d3243ee7e2cb6811ccde64b420c30a9ac136dac2541903f34029b'
_text=_p.read_text().replace('ordinary_onpolicy_recovery_v2/run_001','ordinary_route_teacher_v11/run_001').replace('AUDITED_ONPOLICY_GOAL_TEACHER','AUDITED_ONPOLICY_ROUTE_TEACHER').replace('PROTOCOL_FILESTORE.json','DATA_BINDING.json')
exec(compile(_text,str(_p)+':route-targets-new-binding','exec'),globals())
