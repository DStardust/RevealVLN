"""Read-only import of sealed strict audit, with new batch IDs and audit outputs."""
import argparse
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'special_scale_transport_v1/audit.py'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('sealed_manual_recovery_audit',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Original physical/semantic and active-accounting audit; no threshold edits.
    module.audit_main(Path(args.batch))

if __name__=='__main__':main()
