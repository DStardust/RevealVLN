"""Read-only reuse of reviewed V4 active-accounting implementation."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import transport
p=transport.V4/'telemetry.py'
lock_path=transport.V4/'INPUT_LOCK.json'
assert hashlib.sha256(lock_path.read_bytes()).hexdigest()=='8dfe1664f70ba1f38863deb0fc0b2fad3c6badd8ab44c112d35749b7358c9eb3'
expected=json.loads(lock_path.read_text())[str(p.relative_to(transport.ROOT))]
raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
exec(compile(raw,str(p),'exec'),globals())
