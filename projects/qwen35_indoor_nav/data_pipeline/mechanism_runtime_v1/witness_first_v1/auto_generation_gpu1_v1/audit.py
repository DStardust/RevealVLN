"""Original strong gate3 audit, new GPU1 output inside permitted scope."""
import argparse
import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('gpu1_audit_adapter',Path(__file__).resolve().parent/'adapter.py')
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
m=a.module('audit.py')
audit_main=m.audit_main
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);audit_main(p.parse_args().batch)
