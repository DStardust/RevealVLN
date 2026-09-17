"""Read-only action/source exposure audit of the existing ordinary training pool."""
from collections import Counter, defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]


def main():
    rows = [json.loads(x) for x in (LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/TRAINING_INDEX.jsonl').read_text().splitlines()]
    result = {}
    for name, path, paired in (
        ('full_pool', LINE/'sft_acceptance/ordinary_expanded_v1/SAMPLE_INDEX.jsonl', False),
        ('history_pair_4000_steps', LINE/'sft_acceptance/ordinary_history8_paired_train_v1/SELECTED_SAMPLES.jsonl', True)):
        counts = defaultdict(lambda: dict(actions=[0]*4,weighted_actions=[0.]*4,records=set(),houses=set()))
        observed = Counter()
        with path.open() as stream:
            for line in stream:
                value = json.loads(line)
                index,t,target,weight,_ = value['entry'] if paired else value
                row = rows[index]
                assert row['split']=='FIT' and 0<=t<row['decisions'] and 0<=target<4
                assert (target==3)==(t==row['decisions']-1)
                out = counts[row['source']]
                out['actions'][target]+=1;out['weighted_actions'][target]+=weight
                out['records'].add(index);out['houses'].add(row['scene_group'])
                observed[index]+=1
        for out in counts.values():
            out['records']=len(out['records']);out['houses']=len(out['houses'])
            out['total_actions']=sum(out['actions'])
            out['stop_fraction']=out['actions'][3]/out['total_actions']
        if not paired:
            assert all(observed[i]==row['decisions'] for i,row in enumerate(rows))
        result[name] = dict(source=str(path.relative_to(LINE)),groups=dict(counts),
                           total_reads=sum(observed.values()))
    result['interpretation'] = 'Descriptive composition, not a causal attribution of navigation regression; no label rewriting or new selection.'
    (HERE/'DATA_MIX.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
