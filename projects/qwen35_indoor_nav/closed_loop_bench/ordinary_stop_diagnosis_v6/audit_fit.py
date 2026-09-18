"""CPU recheck of exported causal windows, original labels and decoded RGB."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    folder = HERE / 'fit_sample_001'
    result = read(folder / 'RESULT.json')
    source, images, partition_routes, counts = {}, {}, {}, {}
    for record in result['records']:
        row = record['source']
        root = ROOT / row['sourceRoot']
        policy, supervision = root / row['policy_file'], root / row['supervision_file']
        assert sha(policy) == row['policy_sha256'] and sha(supervision) == row['supervision_sha256']
        source[row['record_id']] = (root, read(policy), read(supervision))
    decisions = [json.loads(line) for line in (folder / 'DECISIONS.jsonl').read_text().splitlines()]
    for row in decisions:
        audit = row['audit_only']
        root, policy, supervision = source[audit['record_id']]
        t = audit['step']
        assert row['policy']['instruction'] == policy['instruction']
        assert row['policy']['executed_actions'] == supervision['actions'][max(0, t-8):t]
        assert row['supervision']['teacher_action'] == supervision['actions'][t]
        expected = [str((root / ref).relative_to(ROOT)) for ref in policy['rgb_sequence'][max(0,t-1):t+1]]
        assert row['policy']['rgb_paths'] == expected
        assert audit['position'] == supervision['frames'][t]['position']
        for step, name in zip(range(max(0,t-1),t+1), expected):
            digest = supervision['frames'][step]['rgb_sha256']
            if name not in images:
                with Image.open(ROOT / name) as image:
                    value = np.asarray(image)
                assert value.shape == (224,224,3) and value.dtype == np.uint8
                images[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            assert images[name] == digest
        partition_routes.setdefault(row['partition'],set()).add(audit['physical_route'])
        counts[row['partition']] = counts.get(row['partition'],0) + 1
    assert partition_routes['head_fit'].isdisjoint(partition_routes['head_check'])
    hashes = {str(path.relative_to(LINE)): sha(path) for path in HERE.glob('*.py')}
    for name in ('official_distance.py','cycle_policy.py'):
        path = HERE.parent / 'ordinary_cycle_pair_gpu1_v3' / name
        hashes[str(path.relative_to(LINE))] = sha(path)
    evidence = dict(status='REAL_RGB_AND_CAUSAL_ALIGNMENT_VERIFIED', decisions=len(decisions),
                    routes=len(source), decoded_rgb_files=len(images), partition_decisions=counts,
                    original_action_labels_unchanged=True, source_sha256=hashes,
                    gpu_used=False, new_simulation=False)
    with (folder / 'AUDIT.json').open('x') as stream:
        json.dump(evidence,stream,indent=2)
    print(evidence)


if __name__ == '__main__':
    main()
