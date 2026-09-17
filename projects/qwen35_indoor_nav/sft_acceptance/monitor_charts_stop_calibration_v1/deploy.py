"""Replace only the exact previous monitor, with preview and rollback checks."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD_DEPLOY=HERE.parent/'monitor_charts_targeted_v1/deploy.py'
assert hashlib.sha256(OLD_DEPLOY.read_bytes()).hexdigest()=='5b925111e038e90bd795d3af9b7fee774e036501ee7d0603a22c2b683e660723'
text=OLD_DEPLOY.read_text()
replacements={
 "OLD=HERE.parent/'monitor_charts_expanded_v1/server_r1.py';NEW=HERE/'server.py'":"OLD=HERE.parent/'monitor_charts_low_lr_v3/server.py';NEW=HERE/'server.py'",
 "PANE='%294';OLD_PID=2062178":"PANE='%294';OLD_PID=2524111",
 "d['monitor_version']=='ordinary_targeted_v1'":"d['monitor_version']=='ordinary_stop_calibration_v1'",
 "d['navigation']['after_batch8']['result']['sr']==.17":"d['navigation']['matched']['result']['sr']==.21",
 "(2099035,2099174,2099284,3996270,112240,1563749,1353421)":"(2539486,2539600,2539708,3996270,112240,1563749,1353421)",
}
for old,new in replacements.items():
    assert text.count(old)==1,old
    text=text.replace(old,new)
exec(compile(text,str(HERE/'deploy.py')+':frozen-parent','exec'),globals())
