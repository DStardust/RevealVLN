"""Real sealed FIT data: loss masking, physical history gradients and resume."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from objective import legacy,first_positions,objective
torch=legacy.torch
EVIDENCE={}


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


class Alignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.old=legacy.u.read(HERE.parent/'runs/train_001/TRAINING_CONFIG.json')
        fit=legacy.load_rows(cls.old['capture_run'],'FIT')
        cls.r=next(r for r in fit if r['kind']=='RECOVERY' and r['cutoff']>4)
        cls.o=next(r for r in fit if r['kind']=='PRESERVATION')
        cls.initial=torch.load(cls.old['initial_files']['42']['path'],map_location='cpu',weights_only=True)
        cls.weights=torch.tensor(cls.old['class_weights'])
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        EVIDENCE.update(real_fit_ids=[cls.r['id'],cls.o['id']],gpu_hours=0,base_updates=0,
            data_scope='REAL_SEALED_ALL_TOKEN_RECAPTURE',test_scope='CPU_IMPLEMENTATION_NOT_NAVIGATION',
            trajectories=[dict(id=r['id'],cache_path=r['cache_path'],cache_sha256=r['cache_sha256']) for r in [cls.r,cls.o]])

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def model(self):
        m=legacy.ExecutionAdaptation(3584,'DELTA');m.load_state_dict(self.initial);return m

    def test_position_projection_keeps_real_physical_history(self):
        for r in [self.r,self.o]:
            v=first_positions(r);mask=r['action_token_offsets']==0
            self.assertIs(v['memory_features'],r['memory_features']);self.assertIs(v['executed_actions'],r['executed_actions'])
            self.assertTrue(torch.equal(v['actor_context_steps'],r['actor_context_steps'][mask]))
            self.assertTrue(torch.equal(v['known'],r['known'][mask]));self.assertTrue(bool((v['action_token_offsets']==0).all()))

    def test_inactive_labels_and_actor_features_do_not_train(self):
        m=self.model();loss,info=objective(m,self.r,self.o,self.weights,'cpu');loss.backward()
        before={n:p.grad.clone() if p.grad is not None else None for n,p in m.named_parameters()}
        altered=[]
        for r in [self.r,self.o]:
            x=dict(r,targets=r['targets'].clone(),actor_features=r['actor_features'].clone())
            inactive=r['action_token_offsets']!=0;x['targets'][inactive]=999;x['actor_features'][inactive]=float('nan')
            x['targets'][~r['known']]=999;altered.append(x)
        m.zero_grad(set_to_none=True);other,other_info=objective(m,*altered,self.weights,'cpu');other.backward()
        self.assertTrue(torch.equal(loss,other));self.assertEqual(info,other_info)
        self.assertTrue(equal(before,{n:p.grad for n,p in m.named_parameters()}))
        EVIDENCE['inactive_actor_and_unknown_label_loss_gradient_invariant']=True

    def test_real_long_history_gradient_and_update(self):
        m=self.model();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        loss,_=objective(m,self.r,self.o,self.weights,'cpu');loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        row=dict(self.r,memory_features=self.r['memory_features'].clone().requires_grad_(True))
        loss,_=objective(m,row,self.o,self.weights,'cpu');loss.backward()
        grad=row['memory_features'].grad;indices=torch.arange(len(grad))
        hidden_old=(indices<self.r['cutoff']) & (~torch.isin(indices,self.r['query_steps']))
        norm=float(grad[hidden_old].norm());self.assertGreater(norm,0)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
        self.assertTrue(any(not torch.equal(p,self.initial[n]) for n,p in m.state_dict().items()))
        EVIDENCE['non_query_old_observation_gradient_norm']=norm;EVIDENCE['real_parameter_update']=True

    def test_dev_rejected(self):
        with self.assertRaisesRegex(ValueError,'FIT_ONLY_OPTIMIZATION'):
            objective(self.model(),dict(self.r,partition='DEV'),self.o,self.weights,'cpu')

    def test_real_trainer_resume_exact(self):
        legacy.objective=objective;rows={r['id']:r for r in [self.r,self.o]}
        def config(name):
            root=self.root/name;root.mkdir();c=copy.deepcopy(self.old)
            c.update(device='cpu',steps=3,checkpoint_every=1,modes=['DELTA'],seeds=[42],
                schedules={'42':[[self.r['id'],self.o['id']]]*3},action_scope='FIRST_ACTION_TOKENS_QUERY_START_MEMORY',
                source_files={str(HERE/'objective.py'):legacy.u.sha(HERE/'objective.py')},
                input_files={r['cache_path']:r['cache_sha256'] for r in rows.values()},cpu_contract_test_only=True)
            legacy.u.write(root/'TRAINING_CONFIG.json',c);return root,c
        full,c1=config('full');resume,c2=config('resume')
        legacy.train_one(full,c1,rows,42,'DELTA')
        legacy.train_one(resume,c2,rows,42,'DELTA',stop_after=1)
        legacy.train_one(resume,c2,rows,42,'DELTA',resume=True)
        a=torch.load(full/'DELTA_s42/FINAL.pt',map_location='cpu',weights_only=False)
        b=torch.load(resume/'DELTA_s42/FINAL.pt',map_location='cpu',weights_only=False)
        self.assertTrue(equal(a['model'],b['model']));self.assertTrue(equal(a['optimizer'],b['optimizer']))
        EVIDENCE['exact_model_optimizer_resume']=True;EVIDENCE['resume_test_updates']=6


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Alignment))
    EVIDENCE.update(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful())
    (HERE/'CPU_TEST_RESULT.json').write_text(json.dumps(EVIDENCE,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
