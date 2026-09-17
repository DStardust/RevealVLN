import copy
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import verbalizer as v


class VerbalizerTests(unittest.TestCase):
    def role(self,cat='sofa',room='tv',raw='couch'):
        return dict(mpcat40=cat,room=room,raw_match=dict(mode='exact',value=raw))

    def test_registered_room_expansions(self):
        self.assertEqual(v.role_phrase(self.role()),'the couch in the TV room')
        self.assertEqual(v.room_phrase('familyroom/lounge'),'family room/lounge')

    def test_no_toilet_room_guess(self):
        self.assertEqual(v.room_phrase('toilet'),'room (room type: toilet)')

    def test_unknown_room_reject_or_explicit(self):
        with self.assertRaises(ValueError):v.room_phrase('conservatory')
        self.assertEqual(v.room_phrase('conservatory','explicit_type'),'room (room type: conservatory)')

    def test_invalid_room_reject(self):
        for room in ('',None,'x\nthen stop','x(y)'):
            with self.assertRaises(ValueError):v.room_phrase(room)

    def test_no_unregistered_category_or_raw(self):
        with self.assertRaises(ValueError):v.role_phrase(self.role(raw='sectional couch'))
        with self.assertRaises(ValueError):v.role_phrase(self.role(cat='chair'))

    def test_preserve_raw_subtype_and_tv(self):
        self.assertEqual(v.role_phrase(self.role('chair','bedroom','dining chair')),'the dining chair in the bedroom')
        self.assertEqual(v.role_phrase(self.role('tv_monitor','bedroom','tv')),'the TV in the bedroom')

    def test_schema_references(self):
        with self.assertRaises(ValueError):v.realize_tasks({'a':self.role()},{'t':dict(anchor='missing',terminal='a',instruction='old')})

    def test_new_tasks_do_not_mutate_inputs(self):
        roles={'a':self.role(),'t':self.role('sink','kitchen','sink')}
        tasks={'q':dict(anchor='a',terminal='t',instruction='old')};old=copy.deepcopy((roles,tasks))
        revised=v.realize_tasks(roles,tasks)
        self.assertEqual((roles,tasks),old)
        self.assertEqual(v.structure(revised),v.structure(tasks))
        self.assertEqual(v.realize_tasks(roles,revised),revised)

    def test_compiler_program_and_checker_parity(self):
        path=v.RUNTIME.parent/'mechanism_factory_v2/compiler.py'
        spec=importlib.util.spec_from_file_location('language_compiler_parity',path)
        c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
        roles={'a':self.role(),'t':self.role('sink','kitchen','sink')}
        tasks={'q':dict(anchor='a',terminal='t',instruction='old')}
        pairs={k:(r['mpcat40'],r['room']) for k,r in roles.items()}
        old=c.Compiler(pairs,tasks,{'a':[1],'t':[2]});new=c.Compiler(pairs,v.realize_tasks(roles,tasks),{'a':[1],'t':[2]})
        self.assertEqual(old.task_program('q'),new.task_program('q'))
        for counts in ([{'1':300},{'1':300},{'2':300},{'2':300}],[{}, {},{},{}]):
            trace=dict(actions=['L','L','L','S'],observations=[dict(step=i,evidence_complete=True,pixels=p) for i,p in enumerate(counts)],complete=True,collisions=0)
            self.assertEqual(old.evaluate(trace,'q'),new.evaluate(trace,'q'))
            self.assertEqual(old.m2(trace,'q'),new.m2(trace,'q'))
        self.assertEqual(new.evaluate(dict(trace,complete=False),'q'),'unknown')

    def test_all_declared_vocab_crossproduct(self):
        for category,raws in v.planner.RAW.items():
            for raw in raws:
                for room in v.planner.INDOOR_ROOMS:
                    self.assertTrue(v.role_phrase(self.role(category,room,raw)).startswith('the '))


if __name__=='__main__':unittest.main()
