"""Versioned pre-replay surface realization; never rewrites an attempted batch."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WF=HERE.parent
ROOT=WF.parents[4]
def main():
    spec=importlib.util.spec_from_file_location('language_batch_realize',WF/'language_realization_v1/verbalizer.py')
    lang=importlib.util.module_from_spec(spec);spec.loader.exec_module(lang)
    spec2=importlib.util.spec_from_file_location('language_batch_hash',HERE/'combine.py')
    util=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(util)
    source=HERE/'composite_01'
    cfg=json.loads((source/'CONFIG_DRAFT.json').read_text())
    lock=json.loads((source/'SOURCE_LOCK.json').read_text())
    for p,h in lock.items():assert util.sha(p)==h,p
    for name,h in json.loads((WF/'language_realization_v1/INPUT_LOCK.json').read_text()).items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        assert util.sha(path)==h,name
        lock[str(path)]=h
    revisions=[]
    for row in cfg['candidates']:
        revision=lang.propose_revision(row);revisions.append(revision)
        row['tasks']=revision['proposed_tasks'];row['candidate_id']+='_LR1'
        row['language_version']=lang.VERSION
        row['component_provenance']['language_revision']=revision
    cfg['language_version']=lang.VERSION
    cfg['source_tasks_and_actions_unchanged_except_surface_language']=True
    output=HERE/'composite_01_language_v1';output.mkdir(exist_ok=False)
    for p in [HERE/'realize.py',source/'CONFIG_DRAFT.json',WF/'language_realization_v1/INPUT_LOCK.json',WF/'language_realization_v1/SCHEMA.json']:
        lock[str(p)]=util.sha(p)
    for name,value in [('CONFIG_DRAFT.json',cfg),('SOURCE_LOCK.json',lock),('LANGUAGE_REVISIONS.json',revisions)]:
        with (output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps(dict(prepared=True,output=str(output),tasks_changed=sum(len(r['changes']) for r in revisions),
                         structure_unchanged=True,physical_replay_executed=False)))
if __name__=='__main__':main()
