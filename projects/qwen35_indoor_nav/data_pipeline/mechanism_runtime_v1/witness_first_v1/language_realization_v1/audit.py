"""Snapshot two existing batch configs; never edits or launches either batch."""
import json
from pathlib import Path
import hashlib
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verbalizer as v


def stable(path):
    path=path.resolve(strict=True)
    assert path.is_relative_to(v.ROOT)
    before=path.stat();raw=path.read_bytes();after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return raw,hashlib.sha256(raw).hexdigest()


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    locks={};batches=[]
    for batch in ('batch_00','batch_01'):
        path=v.WF/f'batch_execution_v1/{batch}/run_v1/EXECUTION_CONFIG.json'
        raw,h=stable(path);cfg=json.loads(raw);locks[str(path.relative_to(v.ROOT))]=h
        revisions=[v.propose_revision(row) for row in cfg['candidates']]
        batches.append(dict(batch=batch,source=str(path.relative_to(v.ROOT)),source_sha256=h,
            candidate_count=len(revisions),changed_candidates=sum(bool(r['changes']) for r in revisions),
            changed_instructions=sum(len(r['changes']) for r in revisions),revisions=revisions,
            original_batch_modified=False))
    references=[v.PLANNER,v.RUNTIME/'feedback_generation_v1/prepare.py',v.RUNTIME.parent/'mechanism_factory_v2/compiler.py']
    for p in list(HERE.glob('*.py'))+[HERE/'SCHEMA.json',HERE/'README_ZH.md']+references:
        _,h=stable(p);locks[str(p.relative_to(v.ROOT))]=h
    for path,h in locks.items():assert stable(v.ROOT/path)[1]==h
    save(HERE/'INPUT_LOCK.json',locks)
    save(HERE/'PROPOSED_REVISIONS.json',dict(schema_version=v.VERSION,batches=batches,
        runtime_allowed=False,executable=False,scientific_pass=False,
        scope='Surface wording proposals only; original configs and completed families unchanged'))
    result=dict(status='CPU_LANGUAGE_REVISION_PROPOSED_NOT_APPLIED',batches=[{k:x for k,x in b.items() if k!='revisions'} for b in batches],
        original_configs_modified=0,gpu_operations=0,training_started=False,scientific_pass=False)
    save(HERE/'result.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
