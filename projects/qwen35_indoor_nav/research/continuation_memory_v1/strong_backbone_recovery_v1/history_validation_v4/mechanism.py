"""Fixed-weight causal history interventions with matched state magnitude."""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent/'recovery_confirmation_v2')]
import common as u
import torch
import torch.nn.functional as F
from confirmation_model import ConfirmationMemory

CONTROLS = ('FULL', 'SHAM', 'RECENT8_RAW', 'RECENT8_NORM', 'REVERSED_OLD_NORM', 'CURRENT_NORM')


def match_norm(candidate, reference):
    a, b = candidate.norm(), reference.norm()
    if not torch.isfinite(a+b):
        raise ValueError('NONFINITE_MEMORY')
    if b == 0:
        return torch.zeros_like(candidate)
    if a == 0:
        raise ValueError('ZERO_DIRECTION_CANNOT_MATCH_NONZERO_NORM')
    return candidate * (b/a)


def unroll(model, features):
    state = model.reset()
    for x in features:
        state = model.update(x[None], state)
    return state


def interventions(model, prefix, full):
    recent = unroll(model, prefix[-8:])
    cut = max(0, len(prefix)-8)
    reverse = unroll(model, torch.cat((prefix[:cut].flip(0), prefix[cut:])))
    current = unroll(model, prefix[-1:])
    return dict(FULL=full, SHAM=full.clone(), RECENT8_RAW=recent,
                RECENT8_NORM=match_norm(recent, full),
                REVERSED_OLD_NORM=match_norm(reverse, full), CURRENT_NORM=match_norm(current, full))


def main(run):
    torch.set_num_threads(2)
    u.verify_sources(run)
    out = run/'mechanism'
    if (out/'RESULT.json').exists():
        return
    p = u.read(run/'PROTOCOL.json')
    source = Path(p['source_run'])
    a = u.read(source/'data/42/ADMISSION.json')
    pool = Path(a['pools_path'])
    assert u.sha(pool) == a['pools_sha256']
    rows = [r for r in torch.load(pool, map_location='cpu', weights_only=True)['rows'] if r['partition']=='DEV']
    assert len(rows) == 64
    began = time.time()
    summaries = {}
    for arm, entry in p['heads'].items():
        assert u.sha(entry['path']) == entry['sha256']
        head = ConfirmationMemory(3584, entry['architecture']).eval()
        head.load_state_dict(torch.load(entry['path'], map_location='cpu', weights_only=False)['model'])
        head.requires_grad_(False)
        state_before = {k: v.clone() for k, v in head.state_dict().items()}
        records = []
        with torch.inference_mode():
            for row in rows:
                path = out/arm/f'{row["id"]}.json'
                if path.exists():
                    saved = u.read(path)
                    assert saved['checkpoint_sha256'] == entry['sha256'] and saved['pool_sha256'] == a['pools_sha256']
                    records.append(saved)
                    continue
                queries = {int(t): q for q,t in enumerate(row['query_steps'])}
                full = head.reset()
                logits = {k: [] for k in CONTROLS}
                measurements = []
                for t, x in enumerate(row['memory_features']):
                    full = head.update(x[None], full)
                    if t not in queries:
                        continue
                    q = queries[t]
                    states = interventions(head, row['memory_features'][:t+1], full)
                    norms = {k: float(m.norm()) for k,m in states.items()}
                    for k, m in states.items():
                        value = row['base_logits'][q:q+1].float()+head.action_delta(row['actor_features'][q:q+1], m)
                        assert torch.isfinite(value).all()
                        logits[k].append(value[0])
                    measurements.append(dict(step=t, norms=norms,
                        cosine={k: float(F.cosine_similarity(full.flatten()[None], m.flatten()[None])) for k,m in states.items()}))
                values = {k: torch.stack(v) for k,v in logits.items()}
                assert torch.equal(values['FULL'], values['SHAM']), 'SHAM_CHANGED_OUTPUT'
                targets = row['targets']
                masks = dict(all_known=row['known'], beyond8_known=row['known'] & (row['query_steps']>=8))
                counts = {}
                for scope, mask in masks.items():
                    counts[scope] = {}
                    f = values['FULL'].argmax(-1)
                    for k, v in values.items():
                        pred = v.argmax(-1)
                        counts[scope][k] = dict(n=int(mask.sum()), correct=int(((pred==targets)&mask).sum()),
                            full_helped=int(((f==targets)&(pred!=targets)&mask).sum()),
                            full_hurt=int(((f!=targets)&(pred==targets)&mask).sum()),
                            changes=int(((pred!=f)&mask).sum()),
                            ce_sum=float(F.cross_entropy(v[mask], targets[mask], reduction='sum')) if mask.any() else 0.)
                record = dict(id=row['id'], house=row['house'], kind=row['kind'], checkpoint_sha256=entry['sha256'],
                    pool_sha256=a['pools_sha256'], counts=counts, queries=row['query_steps'].tolist(),
                    targets=targets.tolist(), known=row['known'].tolist(), logits={k:v.tolist() for k,v in values.items()},
                    measurements=measurements)
                u.write(path, record)
                records.append(record)
                u.write(out/'STATUS.json', dict(status='RUNNING', arm=arm, model_rows=len(summaries)*64+len(records), planned_model_rows=384))
                if time.time()-began > 3600:
                    raise RuntimeError('CPU_MECHANISM_TIME_LIMIT')
        assert all(torch.equal(v, head.state_dict()[k]) for k,v in state_before.items()), 'HEAD_CHANGED'
        summaries[arm] = {}
        for kind in ('RECOVERY', 'PRESERVATION'):
            selected = [r for r in records if r['kind']==kind]
            summaries[arm][kind] = {scope:{k:{field:sum(r['counts'][scope][k][field] for r in selected)
                for field in ('n','correct','full_helped','full_hurt','changes','ce_sum')} for k in CONTROLS}
                for scope in ('all_known','beyond8_known')}
        errors = [abs(m['norms'][k]-m['norms']['FULL']) for r in records for m in r['measurements'] for k in CONTROLS if k.endswith('_NORM')]
        assert max(errors, default=0) < 1e-4, 'NORM_MATCH_FAILED'
        summaries[arm]['max_norm_error'] = max(errors, default=0)
        if entry['architecture']=='LOCAL':
            assert all(r['counts']['all_known'][k]['changes']==0 for r in records for k in CONTROLS), 'LOCAL_HISTORY_CONTROL_CHANGED_ACTION'
    result = dict(status='COMPLETE', models=summaries, model_rows=384, optimizer_updates=0, gpu_hours=0,
                  cpu_seconds=time.time()-began, navigation_sr=False,
                  limitation='L2 matching controls magnitude, not state distribution. Recorded teacher traces and interventions do not prove autonomous or semantic-history benefit.')
    u.write(out/'RESULT.json', result)
    u.write(out/'STATUS.json', dict(status='COMPLETE', model_rows=384, planned_model_rows=384))
    lines = ['COMPLETE', '', '固定权重、同一动作特征与原生 logits，比较完整记忆和等范数历史干预；仅为已记录 DEV 轨迹上的动作诊断。',
             '同范数排除整体幅度差异，不能排除分布偏移；近期特征可能已携带当前任务线索。不能把标签准确率当导航 SR。', '']
    for arm, v in summaries.items():
        for kind in ('RECOVERY','PRESERVATION'):
            d = v[kind]['beyond8_known']
            lines.append(f"{arm} {kind}（旧历史超过8步）: "+'; '.join(f"{k} {r['correct']}/{r['n']}" for k,r in d.items()))
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
