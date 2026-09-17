"""Replace exactly our old monitor; protect restored holders and running eval contexts."""
import hashlib,subprocess,json
from pathlib import Path
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'monitor_charts_targeted_v1/deploy.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='5b925111e038e90bd795d3af9b7fee774e036501ee7d0603a22c2b683e660723'
text=PARENT.read_text()
changes={
"OLD=HERE.parent/'monitor_charts_expanded_v1/server_r1.py';NEW=HERE/'server.py'":"OLD=HERE.parent/'monitor_charts_onpolicy_v6/server.py';NEW=HERE/'server.py'",
"PANE='%294';OLD_PID=2847498":"PANE='%294';OLD_PID=2847498",
"d['monitor_version']=='ordinary_targeted_v1'":"d['monitor_version']=='ordinary_onpolicy_v6r1'",
"d['navigation']['after_batch8']['result']['sr']==.17":"d['navigation']['matched']['result']['sr']==.21",
"(2099035,2099174,2099284,3996270,112240,1563749,1353421)":"protected_pids()",
}
for old,new in changes.items():
    assert text.count(old)==1,old
    text=text.replace(old,new)
def protected_pids():
    line=HERE.parents[1];train=HERE.parent/'ordinary_r2r_adapt_v5'
    holders=json.loads((train/'lease_v1/RESTORATION.json').read_text())['holders']
    pids=[3996270,112240,1563749,1353421]
    for row in holders:
        old=row['process_identity'];pid=old['pid'];proc=Path('/proc')/str(pid)
        assert int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])==old['starttime_ticks']
        assert (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()==old['command']
        pids.append(pid)
    root=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15))
    case=line/'data_pipeline/ordinary_onpolicy_recovery_v2'
    for ctx in root.findall('gpu')[1].findall('processes/process_info'):
        pid=int(ctx.findtext('pid'));argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
        assert any(str(case)+'/' in a for a in argv),'UNKNOWN_EVAL_CONTEXT'
        pids.append(pid)
    return tuple(set(pids))
exec(compile(text,__file__+':hash-bound-parent','exec'),globals())
