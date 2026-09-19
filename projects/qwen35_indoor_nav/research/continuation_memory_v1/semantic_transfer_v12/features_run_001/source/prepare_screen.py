"""Fixed low-cost current-witness screen before full-sequence feature extraction."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
prior = HERE.parent/'query_semantics_v11'
spec = importlib.util.spec_from_file_location('v12_witness_labels', prior/'probe_witness.py')
witness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(witness)
c = witness.c


def select(rows, features, per_class=8):
    selected = set()
    for family in sorted({r['family_id'] for r in rows}):
        for target in range(2):
            for label in (0, 1):
                candidates = [i for i, r in enumerate(rows) if r['family_id']==family
                              and r['mask'][target] and r['y'][target]==label]
                candidates.sort(key=lambda i: hashlib.sha256(
                    ('v12-current-witness:'+features[rows[i]['feature']]['key']).encode()).hexdigest())
                selected.update(candidates[:per_class])
    return [rows[i] for i in sorted(selected)]


def main():
    data = c.read(HERE/'DATA.json')
    scope = c.read(HERE/'POOL_PROTOCOL.json')
    protocol = dict(status='FROZEN_BEFORE_SCREEN_ENCODER_OR_PREDICTOR_RUN',
        data_sha256=c.sha(HERE/'DATA.json'), pool_protocol_sha256=c.sha(HERE/'POOL_PROTOCOL.json'),
        source_sha256=c.sha(Path(__file__)), witness_label_source_sha256=c.sha(prior/'probe_witness.py'),
        selection='Union of up to8 positive and8 negative unique causal inputs for each of two targets in every family, stable input-key hash order. All families and splits retained; no score selection.',
        target='Original current two-frame anchor/terminal SEE2 witnesses, not cumulative state or room visit.',
        heads=['linear','mlp128'], seeds=[1209,1210,1211], steps=600, batch_size=256,
        learning_rate=.001, weight_decay=.01, threshold=.5,
        partitions=scope['house_partition'],
        policy_inputs='Original instruction, at most2 RGB and8 executed actions, frozen best4k.',
        representation='Unchanged action-head input, no semantic truth, future query or task state in encoder.',
        interpretation='Balanced diagnostic screen, not population prevalence or closed-loop method benefit.',
        original_training_admission=False, gpu_optimizer_updates=0)
    c.write(HERE/'SCREEN_PROTOCOL.json', protocol, True)
    rows, audit = witness.examples(data)
    selected = select(rows, data['features'])
    features, lookup = [], {}
    for row in selected:
        old_index = row['feature']
        if old_index not in lookup:
            lookup[old_index] = len(features)
            features.append(data['features'][old_index])
        row['source_feature_index'] = old_index
        row['feature'] = lookup[old_index]
        row['partition'] = scope['house_partition'][row['house']]
    references = {r for f in features for r in f['rgb_refs']}
    old = c.read(HERE.parent/'multifamily_v7/DATA.json')
    cached = {f['key']: i for i,f in enumerate(old['features'])}
    reuse = {str(i): cached[f['key']] for i,f in enumerate(features) if f['key'] in cached}
    output = dict(rows=selected, features=features,
        contents={key:data['contents'][key] for key in sorted(references)},
        reused_v7_feature_indices=reuse, original_row_count=len(rows), audit=audit,
        selection_before_model_outcomes=True, original_training_admission=False)
    c.write(HERE/'SCREEN.json', output, True)
    print(json.dumps(dict(rows=len(selected),unique_causal_inputs=len(features),
        existing_cache_rows=len(reuse),new_forwards_needed=len(features)-len(reuse)), ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
