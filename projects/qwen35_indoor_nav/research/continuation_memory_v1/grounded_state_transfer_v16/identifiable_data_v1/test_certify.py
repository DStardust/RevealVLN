"""CPU counterexamples for admission, not physically generated sample evidence."""
import copy
import unittest
from certify import certify, legacy, HISTORIES, QUERIES


class Admission(unittest.TestCase):
    def setUp(self):
        self.compiler=legacy.Compiler({'anchor':['plant','room'],'terminal':['sink','room']},
            {'task_A':dict(anchor='anchor',terminal='terminal',instruction='see plant then sink')},
            {'anchor':[1],'terminal':[2]})
        words=['LLRRLR','LRLLRR','RRLLRL','RLRRLL']
        self.histories={h:list(w)+list('LR'*4) for h,w in zip(HISTORIES,words)}
        self.suffixes=dict(direct_stop=['S'],acquire_anchor=list('LLRR')+['S'],wrong_terminal=['L','L','S'])
        self.traces={}
        pose=dict(position=[0,0,0],rotation=[1,0,0,0],sensors={})
        def observation(step,anchor=False,terminal=False):
            pixels={'0':50176}
            if anchor:pixels['1']=256;pixels['0']-=256
            if terminal:pixels['2']=256;pixels['0']-=256
            return dict(step=step,evidence_complete=True,pixels=pixels,rgb_hash=str((anchor,terminal)),
                        semantic_hash=str((anchor,terminal)),pose=copy.deepcopy(pose))
        for h in HISTORIES:
            cutoff=len(self.histories[h]);obs=[observation(t,h.startswith('seen') and t in (1,2),t>=cutoff-1) for t in range(cutoff+1)]
            for q in QUERIES:
                tail=[observation(cutoff+i+1,q=='acquire_anchor' and i in (0,1),q=='acquire_anchor' and i>=2)
                      for i in range(len(self.suffixes[q])-1)]
                self.traces[h+'__'+q]=dict(actions=self.histories[h]+self.suffixes[q],observations=copy.deepcopy(obs)+tail,
                     collisions=0,complete=True,interior_state_assignments=0)

    def test_complete_cross(self):
        result=certify(self.compiler,self.histories,self.suffixes,self.traces)
        self.assertEqual(result['crossed_labels'],24)
        self.assertEqual(result['labels']['missing__direct_stop__task_A']['label'],'FAIL')
        self.assertEqual(result['labels']['missing__direct_stop__task_T']['label'],'PASS')

    def test_missing_cell(self):
        del self.traces['missing__acquire_anchor']
        with self.assertRaises(ValueError):certify(self.compiler,self.histories,self.suffixes,self.traces)

    def test_reject_collision(self):
        self.traces['seen__direct_stop']['collisions']=1
        with self.assertRaises(ValueError):certify(self.compiler,self.histories,self.suffixes,self.traces)

    def test_reject_pose_change(self):
        self.traces['missing__direct_stop']['observations'][-1]['pose']['position'][0]=0.001
        with self.assertRaises(ValueError):certify(self.compiler,self.histories,self.suffixes,self.traces)

    def test_reject_teleport(self):
        self.traces['seen__acquire_anchor']['interior_state_assignments']=1
        with self.assertRaises(ValueError):certify(self.compiler,self.histories,self.suffixes,self.traces)

    def test_stop_cannot_add_observation(self):
        self.traces['seen__direct_stop']['observations'].append(copy.deepcopy(self.traces['seen__direct_stop']['observations'][-1]))
        with self.assertRaises(ValueError):certify(self.compiler,self.histories,self.suffixes,self.traces)

    def test_fake_missing_negative(self):
        self.traces['missing__direct_stop']['observations'][1]['pixels']['1']=256
        self.traces['missing__direct_stop']['observations'][2]['pixels']['1']=256
        with self.assertRaises(ValueError):certify(self.compiler,self.histories,self.suffixes,self.traces)


if __name__=='__main__':unittest.main()
