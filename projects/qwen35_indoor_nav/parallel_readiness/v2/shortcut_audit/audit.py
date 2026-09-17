"""CPU-only, standard-library readback diagnostics; never imports simulator/model."""
import collections
import copy
import hashlib
import json
from pathlib import Path
import unittest

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[2]
SRC = LINE / 'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1'
THRESHOLDS = (192, 224, 256, 288, 320)

def read(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def verify_source():
    evidence = []
    for line in (SRC / 'SHA256SUMS').read_text().splitlines():
        expected, rel = line.split(None, 1)
        target = (SRC / rel).resolve()
        assert target.is_relative_to(SRC.resolve()), rel
        actual = digest(target)
        assert actual == expected, rel
        evidence.append({'file': rel, 'sha256': actual})
    return {'pass': True, 'manifest_sha256': digest(SRC / 'SHA256SUMS'), 'files': evidence}

def semantic_query(q):
    return {k: q[k] for k in ('query_schema_version', 'coordinate_frame', 'sequence')}

def atoms(obs, eligible, threshold):
    events = [{k: [] for k in eligible}]
    for prev, cur in zip(obs, obs[1:]):
        known = prev['evidence_complete'] and cur['evidence_complete']
        events.append({k: ([str(i) for i in ids if prev['pixels'].get(str(i), 0) >= threshold
                           and cur['pixels'].get(str(i), 0) >= threshold] if known else None)
                       for k, ids in eligible.items()})
    return events

def evaluate(trace, anchor, eligible, threshold):
    obs, acts = trace['observations'], trace['actions']
    if (not trace['complete'] or trace['collisions'] or not all(o['evidence_complete'] for o in obs)
            or any(a not in ('F', 'L', 'R', 'S') for a in acts) or 'S' in acts[:-1]
            or len(obs) != len(acts) + (0 if acts[-1:] == ['S'] else 1)
            or [o['step'] for o in obs] != list(range(len(obs)))): return 'unknown'
    ev = atoms(obs, eligible, threshold)
    return 'pass' if acts[-1:] == ['S'] and ev[-1]['B'] and any(e[anchor] for e in ev[:-1]) else 'fail'

def oracle(state, query, program):
    if state == 'UNKNOWN': return 'unknown'
    history_seen = state in ('WAIT_TERMINAL_WITNESS', 'READY_TO_STOP')
    step, stop = 0, False
    event_steps = collections.defaultdict(set)
    for item in query['sequence']:
        if item['kind'] == 'movement': step += item['repeat']
        elif item['kind'] == 'observe': event_steps[step].add((item['object_category'], item['room_category']))
        else: stop = item['action'] == 'STOP'
    anchor = (program['anchor_object_category'], program['anchor_room_category'])
    terminal = (program['terminal_object_category'], program['terminal_room_category'])
    return 'pass' if stop and terminal in event_steps[step] and (history_seen or any(anchor in es for t, es in event_steps.items() if t < step)) else 'fail'

def purity(features, labels, ids):
    groups = collections.defaultdict(list)
    for x, y, i in zip(features, labels, ids): groups[canon(x)].append({'sample_id': i, 'outcome': y})
    table = []
    for key, members in groups.items():
        counts = dict(collections.Counter(v['outcome'] for v in members))
        table.append({'feature': json.loads(key), 'counts': counts, 'members': members})
    correct = sum(max(g['counts'].values()) for g in table)
    return {'groups': len(table), 'mixed_groups': sum(len(g['counts']) > 1 for g in table),
            'fit_upper_correct': correct, 'denominator': len(labels), 'fit_upper_fraction': correct / len(labels), 'evidence': table}

class Tests(unittest.TestCase):
    def test_purity(self):
        self.assertEqual(purity([0, 0, 1], ['pass', 'fail', 'pass'], ['a','b','c'])['fit_upper_correct'], 2)
    def test_threshold_boundary(self):
        obs = [{'pixels': {'1': 256}, 'evidence_complete': True}] * 2
        self.assertEqual(atoms(obs, {'D':[1]}, 256)[1]['D'], ['1'])
        self.assertEqual(atoms(obs, {'D':[1]}, 257)[1]['D'], [])
    def test_instance_identity(self):
        obs = [{'pixels': {'1': 500}, 'evidence_complete': True}, {'pixels': {'2':500}, 'evidence_complete': True}]
        self.assertFalse(atoms(obs, {'D':[1,2]}, 256)[1]['D'])
    def test_unknown(self):
        obs = [{'pixels': {}, 'evidence_complete': False}] * 2
        self.assertIsNone(atoms(obs, {'D':[1]}, 256)[1]['D'])
    def test_query_projection(self):
        q = {'query_schema_version': 'v2', 'coordinate_frame':'relative', 'sequence':[], 'action_trace_ref':'secret'}
        a = semantic_query(q); q['action_trace_ref']='renamed'; self.assertEqual(a, semantic_query(q))
    def test_oracle_and_order(self):
        p = {'anchor_object_category':'sink','anchor_room_category':'kitchen','terminal_object_category':'bed','terminal_room_category':'bedroom'}
        q = {'sequence':[{'kind':'observe','object_category':'bed','room_category':'bedroom'}, {'kind':'act','action':'STOP'}]}
        self.assertEqual(oracle('WAIT_TERMINAL_WITNESS',q,p), 'pass')
        self.assertEqual(oracle('WAIT_ANCHOR',q,p), 'fail')
        q['sequence'].insert(0, {'kind':'observe','object_category':'sink','room_category':'kitchen'})
        self.assertEqual(oracle('WAIT_ANCHOR',q,p), 'fail')
        q['sequence'].insert(1, {'kind':'movement','repeat':1})
        self.assertEqual(oracle('WAIT_ANCHOR',q,p), 'pass')

def main():
    save('CODE_LOCK.json', {p.name: digest(p) for p in (OUT/'SPEC_ZH.md', OUT/'audit.py')})
    before = verify_source(); save('SOURCE_BEFORE.json', before)
    result = unittest.TestResult(); unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(result)
    save('UNIT_TESTS.json', {'run': result.testsRun, 'failures': result.failures, 'errors': result.errors, 'pass': result.wasSuccessful()})
    assert result.wasSuccessful()
    sup = rows(SRC/'SUPERVISION_ONLY.jsonl'); pol = {r['sample_id']:r for r in rows(SRC/'POLICY_INPUT.jsonl')}
    traces = {p.stem:read(p) for p in sorted((SRC/'physical_traces').glob('*.json'))}
    by_hash = {t['trace_hash']:t for t in traces.values()}
    roles = read(SRC/'TASK_ROLE_INVENTORY.json'); eligible = roles['eligible']; kinds = roles['kinds']
    labels, ids = [r['outcome'] for r in sup], [r['sample_id'] for r in sup]
    features = collections.defaultdict(list); cell_evidence = []
    for s in sup:
        p = pol[s['sample_id']]; tr = by_hash[s['full_log_ref'][7:]]; cutoff=s['prefix_cutoff_step']
        q=semantic_query(s['query']); task=p['instruction']; obs=tr['observations'][:cutoff+1]
        ev=atoms(obs,eligible,256); counts=[sum(bool(e[k]) for e in ev) for k in kinds]
        eventset=[kinds[k] for k in kinds if any(e[k] for e in ev)]
        acount=dict(sorted(collections.Counter(tr['actions'][:cutoff]).items())); aset=sorted(acount)
        two=[o['rgb_ref'] for o in p['observations']]; recent=[a['action'] for a in p['executed_actions']]
        xs={'task_only': task,'query_only':q,'task_query':[task,q], 'current_rgb_only':two[-1],
            'two_rgb_only':two,'recent8_actions_only':recent,'history_length_only':cutoff,
            'short_context_task_query':[task,q,two,recent,cutoff],
            'action_counts_task_query':[task,q,acount],'action_set_task_query':[task,q,aset],
            'event_counts_task_query':[task,q,counts],'event_set_task_query':[task,q,eventset]}
        state=s['m2_program_state'][cutoff]['state']; pred=oracle(state,q,s['task_program'])
        xs['m2_oracle_state_task_query']=[task,q,state]
        for name,x in xs.items(): features[name].append(x)
        cell_evidence.append({'sample_id':s['sample_id'],'history_length':cutoff,'action_counts':acount,
                              'event_counts':dict(zip(kinds,counts)),'event_set':eventset,'m2_state':state,
                              'oracle_prediction':pred,'actual_outcome':s['outcome'],'oracle_correct':pred==s['outcome']})
    pure={name:purity(values,labels,ids) for name,values in features.items()}
    save('GROUP_PURITY.json',pure); save('CELL_DIAGNOSTICS.json',cell_evidence)
    sensitivity=[]
    anchor_for={s['task_id']: next(k for k,v in kinds.items() if v==[s['task_program']['anchor_object_category'],s['task_program']['anchor_room_category']]) for s in sup}
    for threshold in THRESHOLDS:
        details=[]
        for tid,tr in traces.items():
            normalized=atoms(tr['observations'],eligible,threshold)
            baseline=atoms(tr['observations'],eligible,256)
            raw=copy.deepcopy(tr)
            for n in raw['normalization_events']: raw['observations'][n['step']]=dict(n['raw_record'], step=n['step'])
            re=atoms(raw['observations'],eligible,threshold)
            y={task:evaluate(tr,a,eligible,threshold) for task,a in anchor_for.items()}
            yr={task:evaluate(raw,a,eligible,threshold) for task,a in anchor_for.items()}
            yb={task:evaluate(tr,a,eligible,256) for task,a in anchor_for.items()}
            edges=sorted({i for n in tr['normalization_events'] for i in (n['step'],n['step']+1) if i<len(normalized)})
            details.append({'trace':tid,'normalized_outcomes':y,'raw_boundary_replacement_outcomes':yr,
                'baseline_256_outcomes':yb,'outcomes_changed_from_256':sum(y[t]!=yb[t] for t in y),
                'raw_vs_normalized_outcome_changes':sum(y[t]!=yr[t] for t in y),
                'event_time_records_changed_from_256':sum(a!=b for a,b in zip(normalized,baseline)),
                'normalization_boundary_event_changes':sum(normalized[i]!=re[i] for i in edges),
                'boundary_evidence':[{'step':i,'normalized':normalized[i],'raw_replacement':re[i]} for i in edges]})
        sensitivity.append({'threshold':threshold,'physical_traces':len(details),'task_evaluations':len(details)*2,
          'outcome_changes_from_256':sum(d['outcomes_changed_from_256'] for d in details),
          'raw_boundary_outcome_changes':sum(d['raw_vs_normalized_outcome_changes'] for d in details),
          'normalization_boundary_event_changes':sum(d['normalization_boundary_event_changes'] for d in details),
          'event_time_records_changed_from_256':sum(d['event_time_records_changed_from_256'] for d in details), 'details':details})
    save('THRESHOLD_SENSITIVITY.json',sensitivity)
    # Independent default-threshold re-evaluation against every exported cell.
    for s in sup: assert evaluate(by_hash[s['full_log_ref'][7:]],anchor_for[s['task_id']],eligible,256)==s['outcome']
    after=verify_source(); save('SOURCE_AFTER.json',after); assert before==after
    save('result.json', {'decision':'AUDIT_COMPLETE_PENDING_MAIN_AGENT_REVIEW','scientific_pass':False,
        'independent_families':1,'cells':len(sup),'label_counts':dict(collections.Counter(labels)),
        'fit_upper_bounds':{k:{n:v[n] for n in ('groups','mixed_groups','fit_upper_correct','denominator','fit_upper_fraction')} for k,v in pure.items()},
        'm2_oracle_correct':sum(x['oracle_correct'] for x in cell_evidence),
        'threshold_summary':[{k:v for k,v in x.items() if k!='details'} for x in sensitivity],
        'source_hashes_unchanged':True,'source_files_verified':len(before['files']),'unit_tests_pass':True,
        'new_model_runs':0,'new_simulator_runs':0,'new_training_runs':0,
        'scope':'Empirical single-family separability and label sensitivity, not model accuracy/generalization/contribution evidence'})
    print(canon(read(OUT/'result.json')))

if __name__=='__main__': main()
