"""Include historical pilot recovery (one explicitly recorded legacy retry)."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
code=(HERE/'coverage.py').read_text()
def replace(old,new,count=1):
    global code
    assert code.count(old)==count,(old,code.count(old));code=code.replace(old,new)
replace("return [PIPE/'ordinary_pilot_v1',PIPE/'ordinary_scale_v1']", "return [PIPE/'ordinary_pilot_v1',PIPE/'ordinary_pilot_v1/recovery_v1',PIPE/'ordinary_scale_v1']")
replace("jobs=read(root/'JOBS.json');", "jobs=read((root.parent if root==PIPE/'ordinary_pilot_v1/recovery_v1' else root)/'JOBS.json');")
replace("key=byid[ident]['physical_source_route_sha256'];assert key not in attempts,('DUPLICATE_PHYSICAL_ATTEMPT',ident)","key=byid[ident]['physical_source_route_sha256']\n            if key in attempts:\n                assert root==PIPE/'ordinary_pilot_v1/recovery_v1' and attempts[key]==str((PIPE/'ordinary_pilot_v1').relative_to(ROOT)) and key not in terminal\n                assert read(PIPE/'ordinary_pilot_v1/GENERATION_RESULT.json')['interrupted_candidate_retried']==1")
replace("partial_routes=len(partials),", "partial_routes=sum(p['physical_route_key'] not in terminal for p in partials),historical_pilot_interruption_retried=1,")
replace("never_attempted_jobs=missing,partials=partials,", "never_attempted_jobs=missing,partials=partials,initial_audit_omitted_pilot_recovery=True,")
replace("out=HERE/'COVERAGE.json'", "out=HERE/'COVERAGE_V2.json'")
exec(compile(code,str(__file__),'exec'),globals())
