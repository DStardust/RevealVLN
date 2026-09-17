"""Run with the real controller AND worker Python; no mocks or GPU operations."""
from pathlib import Path
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
if __name__=='__main__':
    lock=c.approved(6)
    assert len(c.read(HERE/'MAIN_AGENT_APPROVAL_GPU6.json'))==5
    print(json.dumps(dict(interpreter=sys.executable,common_file=c.__file__,approved_call_pass=True,immutable_files=len(lock),approval=c.approval_value(6))))
