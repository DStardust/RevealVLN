"""Verify every acquired distribution against official PyPI release metadata."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
WHEELS = OUT.parents[1] / '.cache/q35n_habitat_v017_g0r/wheelhouse'

def verify(record):
    url = f'https://pypi.org/pypi/{record["name"]}/{record["version"]}/json'
    response = subprocess.run(['curl','--fail','--silent','--show-error','--connect-timeout','8',
                               '--max-time','30',url],capture_output=True,check=True)
    data = json.loads(response.stdout)
    info = next(x for x in data['urls'] if x['filename'] == record['file'])
    assert info['url'].startswith('https://files.pythonhosted.org/')
    p = WHEELS / record['file']
    actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual_hash == info['digests']['sha256'] == record['sha256']
    assert p.stat().st_size == info['size'] == record['bytes']
    return {'package':record['name'],'version':record['version'],'filename':p.name,
            'metadata_source':url,'artifact_source':info['url'],'sha256':actual_hash,
            'distribution_bytes':info['size'],'metadata_response_bytes':len(response.stdout),
            'official_hash_verified':True,'license_expression':record['license_expression'],
            'requires_python':info['requires_python'],'yanked':info['yanked']}

if __name__ == '__main__':
    target = OUT / 'OFFICIAL_WHEEL_ORIGINS.json'
    assert not target.exists()
    records = json.loads((OUT / 'WHEEL_INVENTORY.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        origins = list(pool.map(verify,records))
    target.write_text(json.dumps(origins,indent=2) + '\n')
    print(f'Official hashes verified: {len(origins)}/{len(records)}; wheel bytes: {sum(x["distribution_bytes"] for x in origins)}')
