"""CPU checks for deterministic semantic-screen sampling and split preservation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_screen as screen


class ScreenTest(unittest.TestCase):
    def test_full_family_and_label_coverage(self):
        rows=[];features=[]
        for family,split in [('fit_family','fit'),('held_family','check')]:
            for label in (0,1):
                for i in range(20):
                    rows.append(dict(family_id=family,split=split,feature=len(features),
                                     y=[label,1-label],mask=[True,True]))
                    features.append(dict(key=f'{family}-{label}-{i}'))
        selected=screen.select(rows,features)
        self.assertEqual(selected,screen.select(rows,features))
        self.assertEqual(len(selected),32)
        for family,split in [('fit_family','fit'),('held_family','check')]:
            group=[r for r in selected if r['family_id']==family]
            self.assertEqual({r['split'] for r in group},{split})
            self.assertEqual(sum(r['y'][0] for r in group),8)
            self.assertEqual(len(group),16)

    def test_unknown_is_not_a_negative(self):
        rows=[dict(family_id='f',feature=0,y=[0,0],mask=[False,False]),
              dict(family_id='f',feature=1,y=[1,0],mask=[True,False])]
        self.assertEqual(screen.select(rows,[dict(key='unknown'),dict(key='known')]),[rows[1]])


if __name__=='__main__':
    unittest.main()
