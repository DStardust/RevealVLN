import importlib.util,math,unittest,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('recovery_contract_tests',HERE/'contracts.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
class Tests(unittest.TestCase):
    def test_stop_strict(self):
        self.assertTrue(c.stop_at(2.999));self.assertFalse(c.stop_at(3.0))
        with self.assertRaises(AssertionError):c.stop_at(float('nan'))
    def test_whitelist(self):
        p=c.payload('Go to the room.',['a'*64],[]);self.assertEqual(len(c.key(p)),64)
        with self.assertRaises(AssertionError):c.key(dict(p,goal=[1,2,3]))
        with self.assertRaises(AssertionError):c.payload('x',['a'*64],['STOP'])
    def test_dedup(self):
        p=c.payload('x',['a'*64,'b'*64],['move_forward'])
        self.assertEqual(c.key(p),c.key(dict(reversed(list(p.items())))))
        self.assertNotEqual(c.key(p),c.key(c.payload('y',['a'*64,'b'*64],['move_forward'])))
    def test_conflicts(self):
        self.assertTrue(c.admissible_group([dict(target=1),dict(target=1)]))
        self.assertFalse(c.admissible_group([dict(target=1),dict(target=2)]))
        self.assertFalse(c.admissible_group([dict(target=1),dict(target=None)]))
    def test_movement(self):
        before=dict(position=[0,0,0],rotation=[1,0,0,0])
        after=dict(position=[0,0,-.25],rotation=[1,0,0,0])
        self.assertTrue(c.movement_valid('move_forward',before,after,False,5,4.75))
        self.assertFalse(c.movement_valid('move_forward',before,after,True,5,4.75))
        self.assertFalse(c.movement_valid('move_forward',before,before,False,5,5))
        turn=dict(position=[0,0,0],rotation=[math.cos(math.radians(7.5)),0,math.sin(math.radians(7.5)),0])
        self.assertTrue(c.movement_valid('turn_left',before,turn,False,5,5))
    def test_parse(self):
        for p in HERE.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
