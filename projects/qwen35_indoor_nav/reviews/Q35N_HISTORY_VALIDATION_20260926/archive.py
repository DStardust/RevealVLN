"""Snapshot sealed confirmation logs and the new registered validation, never weights."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
REPO=PROJECT.parents[1]
BASE=PROJECT/'research/continuation_memory_v1/strong_backbone_recovery_v1'
CONFIRM=BASE/'recovery_confirmation_v2/runs/confirmation_001'
NEW=BASE/'history_validation_v4/runs/validation_001'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert read(CONFIRM/'RESULT.json')['evaluation_complete']
    assert not (HERE/'SEALED_CONFIRMATION_LOGS.tar.gz').exists()
    files=set()
    rows=[]
    manifest={e['id']:e for e in read(CONFIRM/'unseen/DATA_MANIFEST.json')['episodes']}
    for session in sorted((CONFIRM/'unseen/evaluation').iterdir()):
        seal=read(session/'STATE_SEAL.json')
        assert seal['heads_unchanged'] and seal['base_before']==seal['base_after']
        files.update(session/n for n in ('STATE_SEAL.json','RUNTIME_IDENTITY.json','RESULT.json'))
        for cp in sorted(session.glob('episodes/*/COMPLETE.json')):
            g=read(cp);e=manifest[g['id']]
            assert g['runtime_identity_sha256']==sha(session/'RUNTIME_IDENTITY.json')
            files.add(cp)
            for arm,expected in g['trace_hashes'].items():
                trace=cp.parent/arm/'TRACE.jsonl'
                assert sha(trace)==expected
                files.update((trace,cp.parent/arm/'result.json'))
                v=g['outcomes'][arm]
                rows.append(dict(id=g['id'],episode_id=e['episode_id'],trajectory_id=e['trajectory_id'],house=e['house'],arm=arm,
                    success=v['success'],spl=v['spl'],steps=v['steps'],trace=str(trace.relative_to(REPO)),trace_sha256=expected))
    assert len(rows)==1400 and len({r['id'] for r in rows})==200
    with (HERE/'CONFIRMATION_ROLLOUTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with tarfile.open(HERE/'SEALED_CONFIRMATION_LOGS.tar.gz','w:gz',compresslevel=6) as t:
        for p in sorted(files):
            t.add(p,arcname=str(p.relative_to(REPO)),recursive=False)
    for name in ('RESULT.json','REPORT_ZH.md','STATUS.json','PROTOCOL.json','SOURCE_LOCK.json','CPU_TEST_RESULT.json'):
        target=HERE/'confirmation_final'/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(CONFIRM/name,target)
    for split in ('dev','unseen'):
        for name in ('RESULT.json','DATA_MANIFEST.json','PROTOCOL.json','SOURCE_LOCK.json'):
            target=HERE/'confirmation_final'/split/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(CONFIRM/split/name,target)
    for name in ('STATUS.json','PROTOCOL.json','DATA_MANIFEST.json','SPLIT_AUDIT.json','SOURCE_LOCK.json','CPU_TEST_RESULT.json'):
        target=HERE/'extension_snapshot'/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(NEW/name,target)
    if (NEW/'mechanism/RESULT.json').exists():
        for name in ('RESULT.json','STATUS.json','REPORT_ZH.md'):
            target=HERE/'mechanism'/name;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(NEW/'mechanism'/name,target)
        with (HERE/'mechanism/TRAJECTORY_RESULTS.jsonl').open('w') as f:
            records=list((NEW/'mechanism').glob('*/*.json'))
            assert len(records)==384
            for p in sorted(records):
                f.write(json.dumps(dict(arm=p.parent.name,**read(p)),ensure_ascii=False)+'\n')
    snapshot=dict(unix=time.time(),confirmation_groups=200,confirmation_executions=1400,
        verified_log_files=len(files),log_archive_sha256=sha(HERE/'SEALED_CONFIRMATION_LOGS.tar.gz'),
        extension_status=read(NEW/'STATUS.json'),extension_planned_groups=369,extension_planned_executions=2583,
        tensors_uploaded=False,raw_rgb_uploaded=False,scene_assets_uploaded=False,
        reproducibility='Source, sealed logs and hashes; requires separately acquired local model, environment and feature assets.')
    (HERE/'ARCHIVE_STATE.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    selected=list((BASE/'history_validation_v4').glob('*.py'))+[BASE/'history_validation_v4/README_ZH.md',BASE/'history_validation_v4/HANDOFF.json',BASE/'MONITOR_HANDOFF.json']
    selected+=list((NEW/'fixtures').iterdir())
    selected += [NEW/n for n in ('PROTOCOL.json','DATA_MANIFEST.json','SOURCE_LOCK.json','SPLIT_AUDIT.json','CPU_TEST_RESULT.json')]
    selected += [p for p in HERE.rglob('*') if p.is_file() and p.name not in ('FILES.json','SELECTED_PATHS.json')]
    paths=sorted({str(p.relative_to(REPO)) for p in selected})
    (HERE/'FILES.json').write_text(json.dumps([dict(path=p,sha256=sha(REPO/p),bytes=(REPO/p).stat().st_size) for p in paths],indent=2)+'\n')
    paths += [str((HERE/n).relative_to(REPO)) for n in ('FILES.json','SELECTED_PATHS.json')]
    (HERE/'SELECTED_PATHS.json').write_text(json.dumps(paths,indent=2)+'\n')
    print(json.dumps(dict(files=len(paths),archive_bytes=(HERE/'SEALED_CONFIRMATION_LOGS.tar.gz').stat().st_size)))


if __name__=='__main__':
    main()
