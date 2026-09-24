"""Additional read-only FIT sanity measurement; no evaluation scores or updates."""
from pathlib import Path
import json,sys,torch
torch.set_num_threads(4)
HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'PROTOCOL.json').read_text());run=HERE/'runs/survival_001';source=Path(cfg['upstream_run'])
models={key:torch.load(path,map_location='cpu',weights_only=True)['trainable'] for key,path in [('V13',cfg['reference_checkpoint']),('V17',run/'CANDIDATE.pt')]}
out={key:dict(positive_inputs=0,positive_stops=0,negative_inputs=0,negative_stops=0,terminal_examples=0,terminal_stops=0,trajectories_any_positive_stop=0) for key in models}
for p in source.glob('collect/sessions/*/pairs/*/COLLECT.json'):
 s=json.loads(p.read_text());f=p.parent/'C';cache=torch.load(f/'FEATURES.pt',map_location='cpu',weights_only=True);labels=[json.loads(x) for x in (f/'SUPERVISION_ONLY.jsonl').read_text().splitlines()];y=torch.tensor([x['target'] for x in labels]).bool();h=cache['features'].float()
 terminal=s['termination']=='STOP' and labels[-1]['target']==1
 for key,model in models.items():
  z=h@model['action_head.weight'].T+model['action_head.bias'];stop=z.argmax(1)==3;r=out[key]
  r['positive_inputs']+=int(y.sum());r['positive_stops']+=int((y&stop).sum());r['negative_inputs']+=int((~y).sum());r['negative_stops']+=int((~y&stop).sum());r['terminal_examples']+=int(terminal);r['terminal_stops']+=int(terminal and stop[-1]);r['trajectories_any_positive_stop']+=int((y&stop).any())
(HERE/'TERMINAL_SANITY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
