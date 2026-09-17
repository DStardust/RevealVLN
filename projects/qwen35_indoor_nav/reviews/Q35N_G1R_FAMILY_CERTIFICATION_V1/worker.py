import importlib.util
from pathlib import Path
import traceback

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
DISCOVERY = OUT.parent/'Q35N_G1R_TASK_INSTANCE_V3'


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def main():
    cert = module('certification', LINE/'data_pipeline/mechanism_family_v1/certify.py')
    source = module('frozen_discovery', DISCOVERY/'worker.py')
    try:
        result = cert.certify(DISCOVERY, OUT, source)
    except Exception as e:
        result = {'decision': 'FROZEN_FAMILY_CERTIFICATION_FAIL', 'error': str(e),
                  'traceback': traceback.format_exc(), 'family_certified': False, 'scientific_pass': False,
                  'candidate_reselection_allowed': False}
    cert.write(OUT/'result.json', result)
    print(result['decision'], flush=True)


if __name__ == '__main__': main()
