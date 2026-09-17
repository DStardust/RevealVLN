"""CPU regression for aliased optimizer step; no CUDA initialization."""
import copy
import importlib.util
from pathlib import Path
import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch CPU dependency unavailable')
class OwnershipTest(unittest.TestCase):
    def run_case(self, protected):
        spec=importlib.util.spec_from_file_location('execution',Path(__file__).parent/'execute.py')
        execution=importlib.util.module_from_spec(spec);spec.loader.exec_module(execution)
        p=torch.nn.Parameter(torch.tensor([1.0,2.0]))
        opt=torch.optim.AdamW([p],lr=1e-4)
        def update():
            opt.zero_grad();p.grad=torch.tensor([0.3,-0.2]);opt.step()
        update()
        saved={'parameter':p.detach().clone(),'optimizer':copy.deepcopy(opt.state_dict())}
        class Backend:
            def restore(self,state):
                with torch.no_grad():p.copy_(state['parameter'])
                opt.load_state_dict(state['optimizer'])
                # Mimic GPU load: moment buffers move/copy; scalar step remains CPU.
                opt.state[p]['exp_avg']=opt.state[p]['exp_avg'].clone()
                opt.state[p]['exp_avg_sq']=opt.state[p]['exp_avg_sq'].clone()
        backend=Backend()
        if protected:execution.install_owned_restore(backend)
        backend.restore(saved);update();first=p.detach().clone()
        saved_step=float(next(iter(saved['optimizer']['state'].values()))['step'])
        backend.restore(copy.deepcopy(saved));update();second=p.detach().clone()
        self.assertFalse(torch.cuda.is_initialized())
        return saved_step,torch.equal(first,second)
    def test_unprotected_control_reproduces(self):
        self.assertEqual(self.run_case(False),(2.0,False))
    def test_owned_restore_preserves_source_and_update(self):
        self.assertEqual(self.run_case(True),(1.0,True))


if __name__=='__main__':unittest.main()
