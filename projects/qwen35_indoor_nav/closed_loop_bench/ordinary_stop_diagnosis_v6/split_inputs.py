"""Separate policy-only inputs from offline labels; retain the source audit rows."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def model_inputs(row):
    policy = row['policy']
    if set(policy) != {'instruction', 'rgb_paths', 'executed_actions'}:
        raise ValueError('PRIVILEGED_POLICY_FIELD')
    return policy


def aligned_rows(policies, labels):
    if len(policies) != len(labels):
        raise ValueError('LABEL_ALIGNMENT')
    for policy, label in zip(policies, labels):
        if policy['sample_id'] != label['sample_id']:
            raise ValueError('LABEL_ALIGNMENT')
        yield model_inputs(policy), label['supervision']


def main():
    folder = HERE / 'fit_sample_001'
    source = folder / 'DECISIONS.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    policies = [{k: row[k] for k in ('sample_id', 'partition', 'policy')} for row in rows]
    labels = [{k: row[k] for k in ('sample_id', 'supervision', 'audit_only')} for row in rows]
    assert len({row['sample_id'] for row in rows}) == len(rows)
    assert len(list(aligned_rows(policies, labels))) == len(rows)
    hashes = {}
    for name, values in (('POLICY_INPUTS.jsonl', policies), ('OFFLINE_SUPERVISION.jsonl', labels)):
        path = folder / name
        with path.open('x') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(rows=len(rows), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  file_sha256=hashes, model_fields=['instruction', 'rgb_paths', 'executed_actions'],
                  ids_used_as_model_inputs=False, goal_distance_used_as_model_input=False,
                  original_audit_rows_preserved=True)
    with (folder / 'INPUT_SPLIT_RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(result)


if __name__ == '__main__':
    main()
