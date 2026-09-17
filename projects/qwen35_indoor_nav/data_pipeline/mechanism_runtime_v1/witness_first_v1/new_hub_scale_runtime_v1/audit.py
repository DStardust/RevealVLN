"""Independent bounded CPU closed-bank command; never launches or borrows GPU."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('scale_scout_cpu_audit',HERE/'runtime.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();m.audit_main(a.job)
