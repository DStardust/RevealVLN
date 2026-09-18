"""Post-run descriptive uncertainty and training/runtime coverage; never select policy."""
import json
from pathlib import Path
import random
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import review
c=review.c
MEMORY=HERE.parents[1]/'research/continuation_memory_v1/cold_start_repair_v8'


def interval(values):
    values=sorted(values)
    return [values[int(.025*len(values))],values[int(.975*len(values))-1]]


def main():
    found=review.committed()
    rows=[found[k] for k in sorted(found)]
    assert len(rows)==100,'FULL_DENOMINATOR_REQUIRED'
    for row in rows:
        session=Path(row['path']).parents[2]
        assert row['protocol_sha256']==c.sha(session/'source/PROTOCOL.json'),'PROTOCOL_IDENTITY_MISMATCH'
        assert row['method_identity_sha256']==c.sha(session/'METHOD_IDENTITY.json'),'METHOD_IDENTITY_MISMATCH'
    coverage=c.read(MEMORY/'MECHANISM_DIAGNOSIS.json')['action_supervision_ages']['fit']
    first_index=coverage['zero_based_min']
    arms={}
    for arm in ('A','B','C','D'):
        early=late=early_stop=late_stop=0
        first=[];early_stop_loss_ids=[];full_repeats=0
        for row in rows:
            steps=c.records(Path(row['path']).parent/arm/'POLICY_STEPS.jsonl')
            seen=set()
            for decision in steps:
                memory=decision['memory_fingerprint']
                key=(decision['raw']['input_key'],memory['sha256'] if memory is not None else None)
                full_repeats+=int(key in seen);seen.add(key)
            overrides=[d for d in steps if d['override']]
            if overrides:first.append(overrides[0]['step'])
            for decision in overrides:
                # Log step is one-based; training action indices are zero-based.
                before_support=decision['step']<=first_index
                early+=int(before_support);late+=int(not before_support)
                extra_stop=decision['executed_action']=='STOP'
                early_stop+=int(before_support and extra_stop)
                late_stop+=int(not before_support and extra_stop)
                if (before_support and extra_stop and row['episodes']['A']['success']
                        and not row['episodes'][arm]['success']):
                    early_stop_loss_ids.append(row['episode_id'])
        arms[arm]=dict(full_raw_window_and_memory_input_repeats=full_repeats,overrides_before_first_training_action=early,overrides_in_later_steps=late,
            extra_stop_before_first_training_action=early_stop,extra_stop_in_later_steps=late_stop,
            first_override_step_median=statistics.median(first) if first else None,
            native_success_losses_with_early_extra_stop=early_stop_loss_ids)
    uncertainty={}
    for left,right in (('A','B'),('A','C'),('A','D'),('B','C'),('D','C'),('D','B')):
        differences=[row['episodes'][right]['success']-row['episodes'][left]['success'] for row in rows]
        houses={}
        for row,value in zip(rows,differences):houses.setdefault(row['episodes'][left]['house'],[]).append(value)
        rng=random.Random(1209)
        episode_draws=[];house_draws=[];groups=list(houses.values())
        for _ in range(5000):
            episode_draws.append(statistics.mean(rng.choices(differences,k=len(differences))))
            selected=rng.choices(groups,k=len(groups))
            house_draws.append(sum(map(sum,selected))/sum(map(len,selected)))
        uncertainty[left+right]=dict(episode_resample_95pct=interval(episode_draws),
            house_cluster_resample_95pct=interval(house_draws),houses=len(groups))
    sessions={}
    for path in sorted((HERE/'sessions').glob('session_*')):
        samples=c.records(path/'RESOURCE.jsonl')
        sessions[path.name]=dict(launch=c.read(path/'LAUNCH_RESULT.json'),
            peak_own_gpu_mib=max(r['own_memory_mib'] for r in samples),
            peak_rss_bytes=max(r['rss_bytes'] for r in samples),
            peak_output_bytes=max(r['output_bytes'] for r in samples),
            resource_competition_samples=sum(bool(r['foreign']) for r in samples))
    c.write(HERE/'DIAGNOSIS.json',dict(status='FULL_RUN_READ_ONLY_DIAGNOSIS',
        source_sha256=c.sha(Path(__file__)),complete_quartets=len(rows),
        training_action_coverage=coverage,first_training_action_one_based=first_index+1,
        arms=arms,descriptive_delta_sr_intervals=uncertainty,bootstrap_draws=5000,bootstrap_seed=1209,
        uncertainty_limit='Episode resampling ignores house dependence; five-house cluster resampling is also descriptive and cannot establish broad generalization. No interval selects a policy.',
        causality_limit='Early-override/STOP co-occurrence is diagnostic, not a counterfactual ablation of that mechanism.',
        repeat_metric_scope='RESULT repeated_decisions counts original instruction/RGB/action-window repeats, excluding memory. Full-input repeats include the logged post-update memory fingerprint. Merely changing memory values does not establish less navigation waste.',
        sessions=sessions),True)
    print('Completed full-denominator coverage, resource and descriptive uncertainty report.')


if __name__=='__main__':main()
