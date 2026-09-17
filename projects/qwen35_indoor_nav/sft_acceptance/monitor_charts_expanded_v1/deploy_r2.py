"""Exact monitor-only revision for full snapshot card binding; preserve V1 receipts."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
text=(HERE/'deploy.py').read_text()
pairs={
    "OLD=HERE.parent/'monitor_charts_recovery_v1/server.py'":"OLD=HERE/'server.py'",
    'ident(1201250)':'ident(2057076)',
    "str(HERE/'server.py')":"str(HERE/'server_r1.py')",
    'HERE/"server.py"':'HERE/"server_r1.py"',
    "fd=os.pidfd_open(old['pid']);assert ident(old['pid'])==old\n    signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)":
    "assert ident(old['pid'])==old\n    os.kill(old['pid'],signal.SIGTERM)",
    "fd=os.pidfd_open(pid);assert ident(pid)==now;signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)":
    "assert ident(pid)==now;os.kill(pid,signal.SIGTERM)",
    "'preview.log'":"'preview_r2.log'",
    "'DEPLOY_BEFORE.json'":"'DEPLOY_R2_BEFORE.json'",
    "'DEPLOY_RESULT.json'":"'DEPLOY_R2_RESULT.json'",
    "'ROLLBACK.json'":"'ROLLBACK_R2.json'",
}
for old,new in pairs.items():
    assert old in text,old
    text=text.replace(old,new)
text=text.replace("assert d['monitor_version']=='ordinary_expanded_v1'","assert d['monitor_version']=='ordinary_expanded_v1' and d['snapshot_counts']['instruction_conditioned_decisions']==2650347")
exec(compile(text,str(HERE/'deploy.py')+':r2','exec'),globals())
