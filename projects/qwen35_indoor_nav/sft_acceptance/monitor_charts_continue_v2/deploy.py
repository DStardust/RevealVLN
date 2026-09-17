"""Reuse exact-monitor replacement; protect live continuation workers too."""
import hashlib
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
OLD_DEPLOY=HERE.parent/'monitor_charts_targeted_v1/deploy.py'
assert hashlib.sha256(OLD_DEPLOY.read_bytes()).hexdigest()=='5b925111e038e90bd795d3af9b7fee774e036501ee7d0603a22c2b683e660723'
text=OLD_DEPLOY.read_text()
replacements={
 "OLD=HERE.parent/'monitor_charts_expanded_v1/server_r1.py';NEW=HERE/'server.py'":"OLD=HERE.parent/'monitor_charts_targeted_v1/server.py';NEW=HERE/'server.py'",
 "PANE='%294';OLD_PID=2062178":"PANE='%294';OLD_PID=2375702",
 "d['monitor_version']=='ordinary_targeted_v1'":"d['monitor_version']=='ordinary_continue_v2'",
 "assert d['training_completion']['completed_budget'] and d['training_completion']['holder_restoration_verified']":"assert d['training_completion']['cumulative_expanded_updates']>=4000 and not d['controller_globally_enabled']",
 "d['navigation']['after_batch8']['result']['sr']==.17":"d['navigation']['matched']['result']['sr']==.21",
 "(2099035,2099174,2099284,3996270,112240,1563749,1353421)":"tuple([3996270,112240,1563749,1353421]+live_training_pids())",
}
for old,new in replacements.items():
    assert text.count(old)==1,old
    text=text.replace(old,new)


def live_training_pids():
    root=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15))
    result=[]
    for g in root.findall('gpu')[3:6]:
        contexts=g.findall('processes/process_info');assert len(contexts)==1
        pid=int(contexts[0].findtext('pid'))
        argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
        assert str(HERE.parent/'ordinary_expanded_continue_v2/train_filestore.py') in argv
        result.append(pid)
    assert len(set(result))==3
    return result


exec(compile(text,str(HERE/'deploy.py')+':frozen-parent','exec'),globals())
