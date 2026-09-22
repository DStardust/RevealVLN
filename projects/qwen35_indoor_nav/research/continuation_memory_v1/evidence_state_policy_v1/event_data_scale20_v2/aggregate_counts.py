"""Never count variant expansion as new parents; merge disjoint physical-house batches."""
from pathlib import Path
from shared import *
from quality import counts as own_counts

def combine(first,second):
    ids=[f['family_id'] for f in first+second]
    if len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_FAMILY_ACROSS_BATCHES')
    if {f['house'] for f in first}&{f['house'] for f in second}:raise ValueError('BATCH_HOUSE_OVERLAP')
    return first+second

def target_met(prior_parents,new_parents,target=800):
    return prior_parents+new_parents>=target

def counts(run):
    cfg=read(run/'PROTOCOL.json');stats,own=own_counts(run);prior=Path(cfg['prior_run'])
    previous,old=own_counts(prior);families=combine(old,own)
    total=previous['new_parents']+stats['new_parents']
    value=dict(stats,batch2_new_parents=stats['new_parents'],batch1_new_parents=previous['new_parents'],
        batch2_new_variants=stats['new_variants'],batch1_new_variants=previous['new_variants'],
        new_parents=total,new_variants=len(families),new_histories=4*len(families),
        certified_physical_cross_executions=12*len(families),crossed_labels=24*len(families),
        expansion_vs_old_total=total/40,expansion_vs_old_FIT=total/32,
        target_new_parents=800,planned_capacity=cfg['registered_prior_parents']+cfg['planned_parent_capacity'],
        prior_batch_complete=(prior/'DATASET.json').exists(),houses={**previous['houses'],**stats['houses']})
    for h,row in value['houses'].items():
        row['batch']=1 if h in previous['houses'] else 2
        row['target_parents']=64 if row['batch']==1 else cfg['families_per_house']
    return value,families

