"""Synthetic CPU counterexamples; not physical-data or method-gain evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from certify import certify, legacy, HISTORIES, QUERIES


class Coverage(unittest.TestCase):
    def setUp(self):
        self.compiler=legacy.Compiler({'anchor':['plant','room'],'terminal':['sink','room']},
            {'task_A':dict(anchor='anchor',terminal='terminal',instruction='see plant then sink')},
            {'anchor':[1],'terminal':[2]})
        self.histories={h:list(w+'LR'*4) for h,w in zip(HISTORIES,('LLRRLR','LRLLRR','RRLLRL','RLRRLL'))}
        self.suffixes=dict(direct_stop=['S'],return_terminal=list('LR')+['S'],acquire_anchor=list('LLRR')+['S'])
        self.traces={}
        def obs(t,anchor=False,terminal=False):
            pixels={'0':50176}
            for key,active in (('1',anchor),('2',terminal)):
                if active:pixels[key]=256;pixels['0']-=256
            return dict(step=t,evidence_complete=True,pixels=pixels,rgb_hash=str((anchor,terminal)),
                semantic_hash=str((anchor,terminal)),pose=dict(position=[0,0,0],rotation=[1,0,0,0],sensors={}))
        for h in HISTORIES:
            cutoff=len(self.histories[h])
            prefix=[obs(t,h.startswith('seen') and t in (1,2)) for t in range(cutoff+1)]
            for q in QUERIES:
                tail=[obs(cutoff+i+1,q=='acquire_anchor' and i in (0,1),
                          q=='return_terminal' or (q=='acquire_anchor' and i>=2)) for i in range(len(self.suffixes[q])-1)]
                self.traces[h+'__'+q]=dict(actions=self.histories[h]+self.suffixes[q],observations=copy.deepcopy(prefix)+tail,
                    collisions=0,complete=True,interior_state_assignments=0)

    def certify(self):return certify(self.compiler,self.histories,self.suffixes,self.traces)

    def test_exact_cross_pattern(self):
        labels=self.certify()['labels']
        self.assertEqual(len(labels),24)
        for h in HISTORIES:
            self.assertEqual(labels[h+'__direct_stop__task_T']['label'],'FAIL')
            self.assertEqual(labels[h+'__return_terminal__task_T']['label'],'PASS')
            self.assertEqual(labels[h+'__return_terminal__task_A']['label'],'PASS' if h.startswith('seen') else 'FAIL')
            self.assertEqual(labels[h+'__acquire_anchor__task_A']['label'],'PASS')
            self.assertEqual(labels[h+'__direct_stop__task_A']['state'][2],0)

    def test_missing_cell(self):
        del self.traces['missing__acquire_anchor']
        with self.assertRaises(ValueError):self.certify()

    def test_collision_rejected(self):
        self.traces['seen__return_terminal']['collisions']=1
        with self.assertRaises(ValueError):self.certify()

    def test_unknown_not_negative(self):
        self.traces['missing__direct_stop']['observations'][0]['evidence_complete']=False
        with self.assertRaises(ValueError):self.certify()

    def test_stop_cannot_add_frame(self):
        self.traces['seen__direct_stop']['observations'].append(copy.deepcopy(self.traces['seen__direct_stop']['observations'][-1]))
        with self.assertRaises(ValueError):self.certify()

    def test_pose_mismatch_rejected(self):
        self.traces['missing__direct_stop']['observations'][-1]['pose']['position'][0]=.001
        with self.assertRaises(ValueError):self.certify()

    def test_interior_teleport_rejected(self):
        self.traces['seen__acquire_anchor']['interior_state_assignments']=1
        with self.assertRaises(ValueError):self.certify()

    def test_500_budget_includes_stop(self):
        self.traces['seen__acquire_anchor']['actions']=['L']*500+['S']
        with self.assertRaises(ValueError):self.certify()

    def test_no_forged_absent_anchor(self):
        for q in QUERIES:
            for t in (1,2):self.traces['missing__'+q]['observations'][t]['pixels']['1']=256
        with self.assertRaises(ValueError):self.certify()

    def test_terminal_present_rejected(self):
        cutoff=len(self.histories['seen'])
        for trace in self.traces.values():
            for t in (cutoff-1,cutoff):trace['observations'][t]['pixels']['2']=256
        with self.assertRaisesRegex(ValueError,'TERMINAL_MUST_BE_ABSENT'):self.certify()


if __name__=='__main__':unittest.main()
