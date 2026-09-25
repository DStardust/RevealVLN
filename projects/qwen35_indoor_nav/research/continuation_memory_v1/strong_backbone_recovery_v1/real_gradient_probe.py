"""One debug update on real frozen StreamVLN features, outside matched models."""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from memory_v2 import ExecutionMemory

torch.set_num_threads(4);torch.manual_seed(42)
fid='V15_NEW_0757bea729cfcf291401'
path=u.HERE/'method_runs/pilot_001/features'/fid/'H_A__C0__task_A.pt'
data=torch.load(path,weights_only=True)
family=u.read(u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json')['families'][0]
trace_path=u.HERE/'data_runs/moving_003'/fid/'H_A__C0.json';trace=u.read(trace_path)
spec=importlib.util.spec_from_file_location('strong_gradient_checker',u.PROJECT/'data_pipeline/mechanism_factory_v2/compiler.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
compiler=checker.Compiler(**family['compiler']);events=compiler.atoms(trace['observations'])
assert compiler.evaluate(trace,'task_A')=='pass'
anchor=family['compiler']['tasks']['task_A']['anchor'];witness=next(t for t,e in enumerate(events) if e[anchor])
x=data['memory_features'][None].clone().requires_grad_(True);n=x.shape[1]
actor=torch.zeros_like(x);actor[0,data['query_steps']]=data['features']
model=ExecutionMemory(x.shape[-1]);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
before=model.writer.weight.detach().clone()
out=model(x,torch.tensor([n]),actor)
# The final exact state depends on the witnessed old event. The target is loss
# only; no task-checker output is supplied to either recurrent or actor inputs.
target=torch.ones(1,4)
state=model.state_reader(out['memory'][:,-1].flatten(1))
loss=torch.nn.functional.binary_cross_entropy_with_logits(state,target)
optimizer.zero_grad();loss.backward()
gradient=float(x.grad[0,witness].norm());writer_gradient=float(model.writer.weight.grad.norm())
assert torch.isfinite(loss) and gradient>0 and writer_gradient>0
optimizer.step();change=float((model.writer.weight.detach()-before).norm());assert change>0
u.write(u.HERE/'REAL_GRADIENT_PROBE.json',dict(status='REAL_FEATURE_FORWARD_BACKWARD_UPDATE_MEASURED',
    feature_path=str(path),feature_sha256=u.sha(path),trace_sha256=u.sha(trace_path),
    sequence_length=n,old_event_step=witness,supervision_step=n-1,old_event_feature_gradient_norm=gradient,
    writer_gradient_norm=writer_gradient,writer_parameter_delta_norm=change,loss=float(loss.detach()),
    debug_updates=1,registered_comparison_updates=0,base_updates=0,device='cpu',
    scope='Actual public-backbone frozen features; one auxiliary-state debug update. Not Ours benefit or closed-loop success.',
    checkpoint_saved=False))
print((u.HERE/'REAL_GRADIENT_PROBE.json').read_text())
