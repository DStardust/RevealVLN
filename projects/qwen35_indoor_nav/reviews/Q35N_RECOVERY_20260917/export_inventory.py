"""Reviewable source/result whitelist for the requested GitHub handoff."""
import collections
import hashlib
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
PRUNE = {'runtime', '.envs', 'cache', '.cache', '__pycache__', 'content', 'routes',
         'scene_datasets', 'episodes', 'node_modules', '.git', 'deps', 'quarantine_build_old_prefix'}
SOURCE = {'.py', '.md', '.sh', '.yaml', '.yml', '.toml', '.html', '.css', '.js'}
JSON_NAMES = {'RESULT.json', 'result.json', 'REVIEW.json', 'REPORT.json', 'AUDIT.json',
              'CPU_TEST_RESULT.json', 'PROTOCOL.json', 'PROTOCOL_FILESTORE.json', 'MODEL_CARD.json',
              'TOKENIZER_ADDITION.json', 'PREPARATION.json', 'PREPARATION_CORRECTION.json',
              'DATA_MIX.json', 'REPEATED_INPUTS.json', 'START.json', 'FIRST_UPDATE.json', 'INTERFACE.json',
              'EVAL_0000.json', 'EVAL_0100.json', 'EVAL_0200.json', 'EVAL_0400.json', 'INFERENCE_RESULT.json',
              'LAUNCH_RESULT.json', 'PARITY_REPORT.json', 'DEPLOYMENT_ACCEPTANCE.json', 'CURRENT_STATUS.json',
              'RUNTIME_VERSIONS.json', 'ACCEPTANCE_PLAN.json', 'NUMERIC_TRANSPORT_DIAGNOSIS.json',
              'CONTROLLED_TRANSPORT_ABORT.json', 'INFERENCE_FAILURE.json', 'AUDIT_FAILURE.json',
              'SOURCE_LOCK.json', 'CHECKPOINT_SELECTION.json', 'SELECTION_SEAL.json', 'SPLIT.json',
              'FINAL_REVIEW.json'}
SECRETS = [re.compile(pattern) for pattern in (
    rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----',
    rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{40,}',
    rb'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{32,}',
    rb'https?://[^\s/:"\x27]+:[^\s/@"\x27]{4,}@')]


def main():
    selected = []
    counts = collections.Counter()
    for folder, dirs, files in os.walk(LINE, followlinks=False):
        dirs[:] = [d for d in dirs if d not in PRUNE and not d.startswith(('.t', 'official_'))
                   and not (Path(folder) / d).is_symlink()]
        for name in files:
            path = Path(folder) / name
            wanted = path.suffix in SOURCE or name in JSON_NAMES
            wanted |= path.parent == LINE / 'authorizations' and path.suffix == '.json'
            wanted |= any(word in name.upper() for word in ('LICENSE', 'COPYING', 'NOTICE')) and path.suffix in ('', '.txt', '.md')
            if not wanted or name == 'ssh_proxy.py' or path.is_symlink() or path.stat().st_size > 2 * 1024**2:
                continue
            blob = path.read_bytes()
            if any(pattern.search(blob) for pattern in SECRETS):
                raise ValueError('Review potential credential before export: ' + str(path.relative_to(LINE)))
            selected.append(dict(path=str(path.relative_to(ROOT)), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
            counts[path.suffix] += 1
    checkpoint = LINE / 'sft_acceptance/ordinary_expanded_v1/formal/attempt_001/checkpoint_000004000.pt'
    blob = checkpoint.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == 'c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
    selected.append(dict(path=str(checkpoint.relative_to(ROOT)), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
    selected.sort(key=lambda row: row['path'])
    result = dict(scope='Q35N project source, selected small result/protocol records, retained best4k adapter checkpoint',
                  excluded='Base weights, environments, datasets, raw trajectories/frames, caches, credentials and other root work',
                  counts=dict(counts), files=selected, total_bytes=sum(row['bytes'] for row in selected))
    (HERE / 'EXPORT_MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
    (HERE / 'EXPORT_PATHS.txt').write_text('\n'.join(row['path'] for row in selected) + '\n')
    print(json.dumps(dict(files=len(selected), total_bytes=result['total_bytes'], counts=dict(counts))))


if __name__ == '__main__':
    main()
