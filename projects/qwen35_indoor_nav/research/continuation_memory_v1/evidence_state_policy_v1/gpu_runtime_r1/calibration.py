"""Read-only autonomous state/event calibration; no thresholds or policy changes."""
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from evaluate_continuations import registry,admitted
from evaluator_v16 import legacy,state_sequence

def summarize(values):
    bins=[];n=len(values)
    for k in range(10):
        items=[(p,y) for p,y in values if min(int(p*10),9)==k]
        if items:bins.append(dict(bin=k,n=len(items),mean_probability=sum(p for p,y in items)/len(items),frequency=sum(y for p,y in items)/len(items)))
    return dict(n=n,brier=sum((p-y)**2 for p,y in values)/n if n else None,
                ece=sum(b['n']*abs(b['mean_probability']-b['frequency']) for b in bins)/n if n else None,bins=bins)

def main(run):
    reg=registry(run);groups=admitted(run,reg);raw={f['family_id']:f for f in read(run/'DATA.json')['raw_families']};buckets=defaultdict(list)
    for slot in reg['slots']:
        if slot['condition'] not in groups:continue
        cond=reg['conditions'][slot['condition']];family=raw[cond['family_id']]
        folder=groups[slot['condition']]/'rollouts'/f"{slot['rank']:04d}"
        trace=read(folder/'TRACE_PRIVILEGED.json');compiler=legacy.Compiler(**family['compiler'])
        atoms=compiler.atoms(trace['observations']);states=state_sequence(compiler,trace['observations'],cond['task_id'])
        for row in c.records(folder/'POLICY_STEPS.jsonl'):
            t=row['decision'];scope='active_stop' if row['executed_action']==c.ACTIONS[-1] else 'motion'
            for bit,name in enumerate(('past_anchor','seen_after','current_terminal','ready')):
                buckets[(slot['arm'],name,scope)].append((row['predicted_state'][bit],states[t][bit]))
            for bit,name in enumerate(('anchor','terminal')):
                if name=='anchor' and cond['task_id']=='task_T':continue
                key=compiler.tasks[cond['task_id']]['anchor'] if name=='anchor' else 'terminal'
                truth=atoms[t][key]
                if truth is not None:buckets[(slot['arm'],'event_'+name,scope)].append((row['event_probabilities'][bit],int(truth)))
    immutable(run/'CALIBRATION.json',dict(status='READ_ONLY_DIAGNOSTIC',groups=len(groups),rows=[dict(arm=a,target=t,scope=s,**summarize(v)) for (a,t,s),v in sorted(buckets.items())],
              thresholds_selected=0,policy_changes=0,external_api_calls=0,
              caveats=['Repeated decisions in one exposed DEV house are correlated, not independent calibration samples.',
                       'Weighted BCE outputs need not be prevalence-calibrated; this is measured, not assumed.',
                       'Action softmax STOP is not interpreted as probability of task completion.',
                       'State/event correctness differs from collision-free task success.']))

if __name__=='__main__':main(Path(sys.argv[1]))
