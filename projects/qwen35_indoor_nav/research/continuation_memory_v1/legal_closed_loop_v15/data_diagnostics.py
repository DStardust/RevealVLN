"""Actual data counts and the finite short-window information ceiling; no training."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c


def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    data=c.read(HERE/'DATA.json');config=c.read(LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json')
    counts={};traces=set();histories=set();hubs=set();starts=set();instructions=set();costs=[]
    for family in config['families']:
        candidate=family['candidate'];hubs.add(key((family['house'],candidate['position'],candidate['yaw_bin'])))
        for h,actions in candidate['histories'].items():
            trace=c.read(LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008'/family['family_id']/(h+'__C0.json'))
            base=(family['house'],trace['observations'][0]['pose']);starts.add(key(base))
            histories.add(key((base,actions)))
            for suffix in candidate['continuations'].values():traces.add(key((base,actions+suffix)))
        instructions.update(t['instruction'] for t in family['compiler']['tasks'].values())
    for split in ('fit','check'):
        families=[f for f in data['families'] if f['split']==split];groups=defaultdict(Counter);labels=Counter()
        for f in families:
            for cell in f['cells']:
                prefix=f['prefixes'][cell['prefix']]
                group=key((prefix['features'][-1],cell['query_context']))
                groups[group][cell['y']]+=1;labels[cell['y']]+=1
        counts[split]=dict(families=len(families),houses=len({f['house'] for f in families}),
            cells=sum(labels.values()),label_counts=dict(labels),
            identical_current_input_and_query_groups=len(groups),conflicting_groups=sum(len(x)>1 for x in groups.values()),
            short_window_query_only_empirical_upper_bound_correct=sum(max(x.values()) for x in groups.values()),
            majority_label_correct=max(labels.values()),
            caveat='In-sample information ceiling from registered identical inputs, not a trained baseline or a generalization result.')
    for family in data['families']:
        for task in ('task_A','task_B','task_T'):
            prefix_ids=[i for i,p in enumerate(family['prefixes']) if p['task_id']==task]
            cells=[cell for cell in family['cells'] if cell['prefix'] in prefix_ids]
            costs_with_history=[min(len(cell['tail']) for cell in cells if cell['prefix']==i and cell['y']) for i in prefix_ids]
            universal=[q for q in {cell['continuation_id'] for cell in cells}
                if all(cell['y'] for cell in cells if cell['continuation_id']==q)]
            universal_cost=min(len(cell['tail']) for cell in cells if cell['continuation_id'] in universal) if universal else None
            costs.append(dict(family_id=family['family_id'],task=task,universal_tested_PASS_suffixes=universal,
                best_universal_tested_suffix_actions=universal_cost,
                mean_history_conditioned_best_tested_suffix_actions=sum(costs_with_history)/len(costs_with_history),
                scope='Only the three actually executed suffixes; no global optimality or learned policy claim.'))
    result=dict(counts=counts,complete_execution_records=72,unique_physical_full_action_sequences=len(traces),
        unique_physical_histories=len(histories),distinct_actual_start_states=len(starts),coarse_hub_proposals=len(hubs),
        initial_state_counting='Use actual recorded initial agent/sensor poses, not coarse candidate position/yaw; starts may differ by small calibrated offsets between histories. Every history is executed in full and joins exactly at the registered cutoff.',
        ordered_task_instructions=len(instructions),terminal_only_instruction_count=len({f['task_terminal_only']['instruction'] for f in config['families']}),
        independence='Families, crossed labels, and repeated seeds are not independent houses or independent trajectories.',
        next_data_axis='Additional valid houses and event/route/wording families, not relabelling or multiplying the same crossed cells.',
        finite_suffix_costs=costs,
        success_identifiability_limit='Every registered task has a tested continuation that passes after every history. Conditional teacher forks demonstrate a finite-set efficiency opportunity; they do not prove that immediate opposite actions or memory are necessary for any eventual success.',
        model_results_used=False,data_sha256=c.sha(HERE/'DATA.json'))
    c.write(HERE/'DATA_INFORMATION_AUDIT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
