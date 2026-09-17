"""Acquire only pinned public code references; never execute/install their repository."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
REVISION = '76b98f233dd0fff05dfcd69435eec6740febff9d'
FILES = ('llava/mm_utils.py', 'llava/data/dataset.py', 'scripts/train/sft_8frames.sh', 'LICENSE')

def main():
    target = HERE / 'references'
    target.mkdir(exist_ok=True)
    records = []
    for name in FILES:
        url = 'https://raw.githubusercontent.com/AnjieCheng/NaVILA/' + REVISION + '/' + name
        request = urllib.request.Request(url, headers={'User-Agent': 'q35n-sr40-readonly-reference'})
        blob = urllib.request.urlopen(request, timeout=25).read()
        path = target / name.replace('/', '__')
        with path.open('xb') as stream:
            stream.write(blob)
        records.append(dict(path=str(path), source=url, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob)))
    result = dict(status='PINNED_REFERENCE_ACQUIRED', revision=REVISION, unix=time.time(),
                  files=records, upstream_code_executed=False, upstream_weights_or_data_downloaded=False)
    with (HERE / 'REFERENCE.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
if __name__ == '__main__':
    main()

