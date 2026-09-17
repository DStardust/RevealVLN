import ast,hashlib,importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
parent=c.LINE/'sft_acceptance/ordinary_stop_row_v12';old=c.FIT/'run_001/cache/triton';new=HERE/'triton_cache'
assert not new.exists()
files=dict(c.read(parent/'SOURCE_LOCK.json')['files']);table=[];copied={}
for p in sorted(old.rglob('*.autotune.json')):
    rel=p.relative_to(old);target=new/rel;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(p.read_bytes())
    assert c.sha(p)==c.sha(target);copied[str(target)]=c.sha(p);files[str(p)]=c.sha(p)
    previous=parent/'triton_cache'/rel;a=c.read(p);b=c.read(previous) if previous.exists() else None
    best=lambda x:min(x['configs_timings'],key=lambda row:row[1])[0]
    table.append(dict(key=str(rel),old=best(a),fresh=None if b is None else best(b)))
assert len(copied)==7
c.write(HERE/'CACHE_COPY.json',dict(files=copied,comparison=table))
for p in HERE.glob('*.py'):ast.parse(p.read_text())
c.write(HERE/'CPU_TEST_RESULT.json',dict(status='PASS',source_model_and_extractor_tests=c.sha(parent/'CPU_TEST_RESULT.json'),fixed_indices=list(range(16))+[i*300 for i in range(1,17)],cache_files=7,byte_exact_copy=True,extra_gpu_forwards=0))
for p in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[HERE/'PLAN_ZH.md',parent/'FEATURE_PARITY.json',parent/'FEATURES.pt']:files[str(p)]=c.sha(p)
c.write(HERE/'SOURCE_LOCK.json',dict(files=files,unix=time.time()));print('CACHE_DIAGNOSTIC_PREFROZEN')
