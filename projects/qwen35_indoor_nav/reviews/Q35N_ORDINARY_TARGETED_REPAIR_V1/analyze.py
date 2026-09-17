"""Reproducible offline failure accounting; never feeds privileged data to policy."""
import collections
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
BENCH = LINE / 'closed_loop_bench'
CASES = dict(before=BENCH/'ordinary_expanded_dev_before_v1', after_batch8=BENCH/'ordinary_expanded_dev_after_v1',
             after_matched_single=BENCH/'ordinary_expanded_dev_after_single_v1', guard=BENCH/'ordinary_visual_stall_guard_v1')


def read(p): return json.loads(p.read_text())
def records(p):
    with p.open() as f: return [json.loads(line) for line in f if line.strip()]
def save(p, value):
    with p.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def case_report(case):
    result = read(case/'run_001/RESULT.json')
    assert result['status'] == 'COMPLETE' and result['completed'] == 100 and result['trace_audit_passed']
    episodes = []; total_stagnant = total_forward = 0
    for folder in sorted((case/'run_001/lanes').glob('lane_*')):
        starts = {r['index']: r for r in records(folder/'INTERFACE.jsonl')}
        policies = collections.defaultdict(list); steps = collections.defaultdict(list)
        for r in records(folder/'POLICY_STEPS.jsonl'): policies[r['index']].append(r)
        for r in records(folder/'STEPS_PRIVILEGED.jsonl'): steps[r['index']].append(r)
        for file in sorted(folder.glob('episode_*.json')):
            e = read(file); i = e['index']; ps = policies[i]; ss = steps[i]
            assert len(ps) == len(ss) == e['steps']
            prior_rgb = starts[i]['rgb_sha256']; prior_position = e['positions'][0]
            observations = [prior_rgb]; actions = []; previous_input = None
            stagnant = longest_stagnant = stagnant_total = same_input_run = longest_same_input = 0
            first_repeat_step = None
            for p, s in zip(ps, ss):
                assert p['step'] == s['step'] and p['action'] == s['action']
                # Complete causal *content* fingerprint: original instruction, <=2 RGB frames, <=8 executed actions.
                key = hashlib.sha256(json.dumps([e['instruction'], observations[-2:], actions[-8:]], ensure_ascii=False).encode()).hexdigest()
                same_input_run = same_input_run + 1 if key == previous_input and s['action'] == 'move_forward' else 0
                if same_input_run and first_repeat_step is None: first_repeat_step = s['step']
                longest_same_input = max(longest_same_input, same_input_run)
                no_motion = (s['action'] == 'move_forward' and s['position'] == prior_position and s['rgb_sha256'] == prior_rgb)
                stagnant = stagnant + 1 if no_motion else 0
                stagnant_total += int(no_motion); longest_stagnant = max(longest_stagnant, stagnant)
                total_forward += int(s['action'] == 'move_forward')
                previous_input = key; prior_rgb = s['rgb_sha256']; prior_position = s['position']
                observations = (observations + [prior_rgb])[-2:]; actions = (actions + [s['action']])[-8:]
            total_stagnant += stagnant_total
            episodes.append(dict(index=i, episode_id=e['episode_id'], house=e['house'], success=e['success'], spl=e['spl'],
                ndtw=e['ndtw'], oracle_success=e['oracle_success'], steps=e['steps'], stopped=e['stopped'],
                failure_category=e['failure_category'], stagnant_forward_steps=stagnant_total,
                longest_stagnant_forward_run=longest_stagnant,
                longest_consecutive_repeated_causal_input_forward_decisions=longest_same_input,
                first_repeated_input_step=first_repeat_step))
    assert len(episodes) == 100 and len({x['index'] for x in episodes}) == 100
    stuck = [e for e in episodes if e['longest_stagnant_forward_run'] >= 8]
    return dict(result=result, episodes=sorted(episodes, key=lambda e:e['index']),
                stagnant_forward_steps=total_stagnant, forward_attempts=total_forward,
                stagnant_forward_fraction=total_stagnant/total_forward,
                episodes_with_stagnant_run_at_least_8=len(stuck),
                stagnant_episodes_unsuccessful_timeouts=sum(not e['success'] and not e['stopped'] and e['steps']==500 for e in stuck),
                episodes_with_stagnant_run_at_least_20=sum(e['longest_stagnant_forward_run']>=20 for e in episodes),
                episodes_with_repeated_full_causal_input_at_least_8=sum(e['longest_consecutive_repeated_causal_input_forward_decisions']>=8 for e in episodes))


def paired(a, b):
    pairs = list(zip(a['episodes'], b['episodes']))
    assert all(x['index']==y['index'] and x['episode_id']==y['episode_id'] and x['house']==y['house'] for x,y in pairs)
    houses = sorted({x['house'] for x,y in pairs})
    def stats(ps):
        return {k:statistics.mean(y[k]-x[k] for x,y in ps) for k in ('success','spl','ndtw')}
    house = {h:stats([(x,y) for x,y in pairs if x['house']==h]) for h in houses}
    rng = random.Random(1209); draws = collections.defaultdict(list)
    for _ in range(2000):
        sample = [pair for h in rng.choices(houses, k=len(houses)) for pair in pairs if pair[0]['house']==h]
        for k,v in stats(sample).items(): draws[k].append(v)
    return dict(delta=stats(pairs), by_house_delta=house,
        wins=sum(not x['success'] and y['success'] for x,y in pairs),
        losses=sum(x['success'] and not y['success'] for x,y in pairs),
        both_success=sum(x['success'] and y['success'] for x,y in pairs),
        both_fail=sum(not x['success'] and not y['success'] for x,y in pairs),
        leave_one_house_out={h:stats([(x,y) for x,y in pairs if x['house']!=h]) for h in houses},
        house_bootstrap_95_interval={k:[sorted(v)[49],sorted(v)[1949]] for k,v in draws.items()},
        uncertainty_note='Descriptive paired 5-house bootstrap; exploratory exposed development data, not independent confirmation',
        matched_inference_batch=a['result']['selected_batch_size']==b['result']['selected_batch_size'],
        scientific_gain_verified=False)


def main():
    stage = sys.argv[1] if len(sys.argv)>1 else 'initial'
    assert stage in ('initial','matched','guard')
    keys = ['before','after_batch8'] + (['after_matched_single'] if stage in ('matched','guard') else []) + (['guard'] if stage=='guard' else [])
    cases = {k:case_report(CASES[k]) for k in keys}
    pairs = {'original_unmatched':paired(cases['before'],cases['after_batch8'])}
    if 'after_matched_single' in cases:
        pairs['corrected_model_comparison'] = paired(cases['before'],cases['after_matched_single'])
        pairs['same_checkpoint_batch_sensitivity'] = paired(cases['after_batch8'],cases['after_matched_single'])
    if 'guard' in cases: pairs['guard_vs_pure_model'] = paired(cases['after_matched_single'],cases['guard'])
    obj = dict(stage=stage, unix=time.time(), cases=cases, pairs=pairs,
               training_updates_added_by_this_review=0, data_added_to_training=0, scientific_gain_verified=False)
    save(HERE/f'ANALYSIS_{stage.upper()}.json',obj)
    print(json.dumps(dict(stage=stage,diagnostics={k:{x:v[x] for x in ('stagnant_forward_steps','forward_attempts','episodes_with_stagnant_run_at_least_8','episodes_with_repeated_full_causal_input_at_least_8')} for k,v in cases.items()},pairs=pairs),ensure_ascii=False))


if __name__=='__main__': main()
