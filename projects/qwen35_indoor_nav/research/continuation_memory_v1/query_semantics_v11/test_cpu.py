"""Real-cache input firewall, shared initialization and full-prefix gradient checks."""
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
c=train.c


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        prior=HERE.parent/'multifamily_v7'
        cls.data=c.read(prior/'DATA.json')
        cls.cache={k:v.float() for k,v in torch.load(prior/'run_001/FEATURES.pt',map_location='cpu',weights_only=True).items()}

    def network(self,seed=1209):
        torch.manual_seed(seed)
        return train.models.MemoryPolicy(2048,len(self.data['query_vocabulary']),8,64,.99)

    def test_runtime_initialization_matches_v10_exactly(self):
        for seed in (1209,1210,1211):
            net=self.network(seed)
            old=torch.load(HERE.parent/'contextual_readout_v10/run_001'/f'INITIAL_{seed}.pt',map_location='cpu',weights_only=True)
            for name,value in net.state_dict().items():
                if not name.startswith('result_head.'):
                    self.assertTrue(torch.equal(value,old[name]),name)

    def test_all_queries_are_history_independent_and_exact_state_composes(self):
        cells=0
        for family in self.data['families']:
            batch=train.special.tensors(family,'cpu')
            q=train.special.query_features(batch,'cpu')
            lookup={}
            for i,cell in enumerate(family['cells']):
                prefix=family['prefixes'][cell['prefix']]
                key=(prefix['task_id'],cell['query'])
                value=q[i].tolist()
                self.assertEqual(lookup.setdefault(key,value),value)
                z=prefix['state_targets'][-1]
                at_cut,stops,anchor,terminal=map(bool,value)
                y=stops and (z[3] if at_cut else terminal and (z[1] or anchor))
                if cell['mask']:
                    self.assertEqual(int(y),cell['y']);cells+=1
        self.assertEqual(cells,450)

    def test_query_changes_do_not_change_causal_memory_or_actions(self):
        net=self.network().eval()
        batch=train.special.tensors(self.data['families'][0],'cpu')
        with torch.no_grad():
            states,_=net.encode(self.cache['features'][batch['prefix_indices']])
            final=states[:,-1];before=final.clone()
            feature=self.cache['features'][batch['prefix_indices'][:,-1]]
            base=self.cache['logits'][batch['prefix_indices'][:,-1]]
            action=net.action_logits(final,base,feature)
            a=net.reader(final,torch.zeros(len(final),4))
            b=net.reader(final,torch.ones(len(final),4))
            self.assertFalse(torch.equal(a,b))
            self.assertTrue(torch.equal(before,final))
            self.assertTrue(torch.equal(action,net.action_logits(final,base,feature)))
            self.assertTrue(torch.equal(action,base))

    def test_real_updates_and_old_write_gradients_both_arms(self):
        family=next(f for f in self.data['families'] if f['split']=='fit')
        batch=train.special.tensors(family,'cpu')
        for arm in ('B2','Ours'):
            net=self.network()
            initial={k:v.detach().clone() for k,v in net.state_dict().items()}
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss,_,_=train.special.losses(net,self.cache,batch,arm)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward();optimizer.step()
            audit=train.gradient_audit(net,self.cache,batch,arm)
            self.assertGreater(audit['critical_writer_auxiliary_gradient_norm'],0)
            self.assertGreater(audit['action_memory_gradient_norm'],0)
            for name in ('writer.weight','recurrent.weight','action.0.weight','action.2.weight'):
                self.assertFalse(torch.equal(initial[name],net.state_dict()[name]),name)


if __name__=='__main__':unittest.main()
