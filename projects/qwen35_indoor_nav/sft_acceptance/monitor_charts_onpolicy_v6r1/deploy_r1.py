"""Correct only the transform lookup key; no prior monitor was signaled."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE/'deploy.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='963bfa2d6b23cde6dddf6d881c74c2b4f715adab63d9bc3a68c50dd087269be9'
text=PARENT.read_text()
old='"PANE=\'%294\';OLD_PID=2847498":'
new='"PANE=\'%294\';OLD_PID=2062178":'
assert text.count(old)==1
text=text.replace(old,new)
exec(compile(text,__file__+':prior-transform-key-fix','exec'),globals())
