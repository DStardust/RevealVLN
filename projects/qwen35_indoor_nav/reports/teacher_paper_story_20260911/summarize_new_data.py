"""Read-only cohort accounting; new path variants are not new raw houses."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import time

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
NEW=Path('/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/witness_first_v1/diversity_v1r2')

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def physical(m):
    c=m['candidate']
    return digest(dict(house=c['context']['house_id'],position=c['position'],yaw=c['yaw_bin'],
                       histories=c['histories'],continuations=c['continuations']))

def main():
    result=read(NEW/'audit_v1/RESULT.json')
    cfg=read(NEW/'run_v1/EXECUTION_CONFIG.json')
    old=read(LINE/'reports/opencode_navbench_fusion_20260911/LIVE_READONLY_SNAPSHOT.json')
    old_hubs={};old_signatures=set()
    for row in old['special']['export_manifests']:
        p=Path(row['path']);assert sha(p)==row['sha256']
        m=read(p);assert m['candidate']['public_tail']=='LRLRLRLR'
        old_hubs.setdefault(row['house'],set()).add(tuple(m['candidate']['position']))
        old_signatures.add(physical(m))
    accepted=[r for r in result['attempts'] if r.get('quality_pass')]
    by_id={r['candidate_id']:r for r in cfg['candidates']}
    rows=[];signatures=set();hubs={}
    for report in accepted:
        aid=report['attempt_id'];source=by_id[aid]
        p=NEW/'run_v1/bundles'/aid/'export_v4/MANIFEST.json';m=read(p)
        position=report['actual_decision_hub'];house=report['house_id']
        sig=physical(m);signatures.add(sig);hubs.setdefault(house,set()).add(tuple(position))
        nearest=min(math.dist(position,oldp) for oldp in old_hubs[house])
        rows.append(dict(id=aid,house=house,source_family=source['source_family_id'],
                         split_group=house,ancestor_group=source['source_family_id'],
                         manifest_sha256=sha(p),strong_audit_sha256=sha(NEW/'audit_v1'/aid/'DIVERSITY_REPORT.json'),
                         physical_signature=sig,same_physical_signature_as_old=sig in old_signatures,
                         actual_decision_hub=position,nearest_old_exact_hub_m=nearest,
                         language_variant=source['language_variant'],
                         task_language_variants=[(source['language_variant']+i)%5 for i in range(len(m['compiler_config']['tasks']))],cells=18,
                         ce_within_family_owners=m['ce_unique_owners'],training_admission=False))
    data=dict(created_unix=time.time(),run=str(NEW),registered_candidates=len(cfg['candidates']),
              accepted_variants=len(rows),accepted_houses=len(hubs),new_raw_houses=0,
              old_accepted_families=len(old['special']['export_manifests']),
              old_houses=len(old_hubs),old_exact_positions=sum(map(len,old_hubs.values())),
              new_exact_decision_hubs=sum(map(len,hubs.values())),
              new_hubs_at_least_1m_from_any_old_same_house_hub=sum(r['nearest_old_exact_hub_m']>=1 for r in rows),
              physical_signatures=len(signatures),old_physical_signatures=len(old_signatures),
              old_physical_signature_overlap=sum(r['same_physical_signature_as_old'] for r in rows),
              observed_new_language_variants=len({v for r in rows for v in r['task_language_variants']}),
              implemented_language_variants=5,task_program_types=1,
              old_public_tail='LRLRLRLR',new_public_tail='FFFFFFFF',
              cells=len(rows)*18,ce_within_family_owner_sum=sum(r['ce_within_family_owners'] for r in rows),
              global_training_owner_dedup_complete=False,training_admission=False,
              new_rows=rows,unaccepted=[r for r in result['attempts'] if not r.get('quality_pass')],
              small_batch_acceptance=result['small_batch_acceptance'],
              notes=['Path variants share source ancestry; do not report independent new tasks or unseen houses.',
                     'Counts include all preregistered candidates; illustration selection does not change denominator.',
                     'Language variants and ancestor paths must remain in the same house group for splits.'])
    with (HERE/'DIVERSITY_COUNTS.json').open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in data.items() if k not in ('new_rows','unaccepted')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
