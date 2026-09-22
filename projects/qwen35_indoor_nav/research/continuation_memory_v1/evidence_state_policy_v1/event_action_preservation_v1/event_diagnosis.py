"""Read-only event calibration after the fixed full-policy updates."""
from collections import defaultdict
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import *
from event_loss import metrics, predictions


def main(run):
    cfg = config(run); torch.set_num_threads(2)
    index = read(Path(read(run/'PRESERVATION_POOL.json')['source']))['records']
    data = read(run/'DATA.json')
    cache = {k: v.float() for k, v in torch.load(run/'features/FEATURES.pt', map_location='cpu', weights_only=True).items()}
    groups = defaultdict(list)
    for record in index:
        groups[(record['split'], record['house'])].append(record)
    result = []
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            tag = f'{arm}_{seed}'; folder = run/'train'/tag
            if sha(folder/'FINAL.pt') != read(folder/'RESULT.json')['checkpoint_sha256']:
                raise ValueError('DIAGNOSIS_MODEL_CHANGED')
            net = make_head(tag)
            net.load_state_dict(torch.load(folder/'FINAL.pt', map_location='cpu', weights_only=True)); net.eval()
            counts = defaultdict(lambda: defaultdict(int))
            with torch.no_grad():
                scores = [dict(split=split, house=house, metrics=metrics(predictions(net, cache, rows).sigmoid().tolist(), rows))
                          for (split, house), rows in sorted(groups.items())]
                # Deduplicate cutoff histories across repeated continuation labels.
                seen = set()
                for family in data['families']:
                    for row in family['sequences']:
                        if row['task'] != 'task_A':
                            continue
                        ids = row['features'][:row['cutoff']+1]
                        key = (family['house'], tuple(ids))
                        if key in seen:
                            continue
                        seen.add(key)
                        alive = torch.ones(1, len(ids), dtype=torch.bool)
                        output = net(cache['features'][ids][None], cache['logits'][ids][None], alive)
                        probability = float(output['state'][0, -1, 1])
                        truth = row['state_targets'][row['cutoff']][1]
                        cell = counts[(family['split'], family['house'])]
                        cell['positive' if truth else 'negative'] += 1
                        cell['tp' if truth and probability >= .5 else 'fn' if truth else 'fp' if probability >= .5 else 'tn'] += 1
            result.append(dict(model=tag, event=scores, causal_cutoff=[dict(split=k[0], house=k[1], **v) for k, v in sorted(counts.items())]))
    immutable(run/'PRESERVATION_DIAGNOSIS.json', dict(models=result, new_updates=0, base_loaded=False,
        checkpoint_selection=False, independent_test_access=False,
        note='Fixed-final-head diagnostic on existing FIT/DEV. Event and cutoff metrics do not replace sealed autonomous navigation.'))
    print('Six fixed heads diagnosed; no model selection.', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
