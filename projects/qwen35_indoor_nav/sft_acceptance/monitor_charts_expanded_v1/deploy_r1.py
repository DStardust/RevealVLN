"""Versioned pre-signal transport repair: project Python has no pidfd_open."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
assert (HERE/'DEPLOY_BEFORE.json').exists() and not (HERE/'DEPLOY_RESULT.json').exists()
with (HERE/'DEPLOY_V1_FAILURE.json').open('x') as f:
    json.dump(dict(error='AttributeError: os.pidfd_open unavailable',old_monitor_signalled=False,
                   prior_source_preserved=True,new_attempt='deploy_r1.py'),f,indent=2)
text=(HERE/'deploy.py').read_text()
pairs={
    "fd=os.pidfd_open(old['pid']);assert ident(old['pid'])==old\n    signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)":
    "assert ident(old['pid'])==old\n    os.kill(old['pid'],signal.SIGTERM)",
    "fd=os.pidfd_open(pid);assert ident(pid)==now;signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)":
    "assert ident(pid)==now;os.kill(pid,signal.SIGTERM)",
    "'preview.log'":"'preview_r1.log'",
    "'DEPLOY_BEFORE.json'":"'DEPLOY_R1_BEFORE.json'",
}
for old,new in pairs.items():
    assert text.count(old)==1
    text=text.replace(old,new)
exec(compile(text,str(HERE/'deploy.py')+':r1', 'exec'),globals())
