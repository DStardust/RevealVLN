"""Fresh-only worker ledger binding; original factory, runner and supervisor."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
BE=HERE.parent
WF=BE.parent
RUNTIME=WF.parent
HASHES={
 'shared.py':'3b717bdf07a53ce06191a2445a20e86982073c17b43412b30b1332c8e01a53b0',
 'prepare.py':'14b0288ccdc027df9556ef3ff9cb1b478c3fd0c9158e6587a1a6432d8278d718',
 'readiness_v1.py':'6ad5b369845917a57963ce2fb4718bf2580b8d87a408c8f467cb70ddc61b7600'}
CLOCK=WF/'budget_clock_batching_cpu_v1/ledger.py'
CLOCK_SHA='c1d1b9eaecdcb387e2e4c252e76d9f61632eb649865b1d15b14f154b2a79fa44'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def base(name):
    p=BE/name
    assert sha(p)==HASHES[name],'FROZEN_TRANSPORT_SOURCE_CHANGED'
    return load('clock_transport_'+name.replace('.','_'),p)

def check_config(cfg):
    assert cfg['budget_transport']=='clock_batching_fresh_only_v1'
    assert cfg['fresh_only'] is True and cfg['resume_allowed'] is False
    assert cfg['runtime_allowed'] is True and cfg['executable'] is True and cfg['training_allowed'] is False
    assert cfg['gpu_device']==2 and cfg['gpu_uuid']=='GPU-be1b30d0-517b-b079-871b-de195d35a1a2'
    assert cfg['supervision_wall_seconds']==3900
    assert cfg['budget']==dict(total_actions=60000,total_seconds=3600,discovery_actions=15000,
        discovery_seconds=1000,certification_actions=20000,certification_seconds=1500)

def check_inputs(batch,worker=False):
    given=Path(batch).absolute();batch=given.resolve(strict=True)
    assert batch==given and batch==BE/'batch_07','EXACT_FRESH_BATCH07_ONLY'
    out=batch/'run_v1';assert out==out.resolve() and out.is_dir()
    cfg=json.loads((out/'EXECUTION_CONFIG.json').read_text());check_config(cfg)
    lock=json.loads((out/'INPUT_LOCK.json').read_text())
    assert 1<=len(lock)<=1024
    for key,value in lock.items():
        p=Path(key);assert p.absolute()==p.resolve() and p.is_relative_to(WF.parents[4])
        assert sha(p)==value, key
    for p in (HERE/'transport.py',HERE/'prepare.py',HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',CLOCK,
              batch/'worker.py',batch/'run.py',out/'EXECUTION_CONFIG.json'):
        assert lock.get(str(p))==sha(p),'MISSING_NEW_RUNTIME_CLOSURE'
    if worker:
        assert not any((out/p).exists() for p in ('bundles','content','journal','PROGRESS.json','result.json','BUDGET_FINAL.json')),'WORKER_FRESH_ONLY'
    return cfg

def bind_budget(journal,limits,*,clock,journal_class):
    assert isinstance(journal,journal_class),'VERIFIED_JOURNAL_REQUIRED'
    assert len(journal._records)==1 and journal._records[0]['kind']=='__config__','FRESH_GENESIS_ONLY'
    assert sha(CLOCK)==CLOCK_SHA,'CLOCK_SOURCE_CHANGED'
    m=load('production_clock_batching',CLOCK)
    # No state/resume argument is accepted. Original Journal append is sync.
    return m.ClockBatchingBudgetLedger(limits,clock=clock,persist=lambda s:journal.append('budget',s))

def build_worker(batch,cfg):
    shared=base('shared.py');path=WF/'assembly_v1/worker.py'
    shared.load('clock_transport_original_imports',WF/'short_revisit_v2/method.py')
    module=types.ModuleType('clock_batched_original_worker');module.__file__=str(path)
    exec(compile(shared.worker_source(path.read_text()),str(BE/'shared.py'),'exec'),module.__dict__)
    assert cfg['factory_variant']=='winding_v1'
    method=shared.load('clock_transport_original_factory',WF/'winding_balance_v1/method.py')
    module.HERE=Path(batch);module.WitnessFactory=method.BalancedFactory
    module.durable_budget=lambda journal,limits,*,clock:bind_budget(journal,limits,clock=clock,journal_class=module.Journal)
    return module

def worker_main(batch):
    cfg=check_inputs(batch,worker=True)
    build_worker(batch,cfg).main()

def run_main(batch):
    check_inputs(batch)
    # Unmodified original readiness + supervisor and all resource/cleanup rules.
    base('readiness_v1.py').run_main(batch)
