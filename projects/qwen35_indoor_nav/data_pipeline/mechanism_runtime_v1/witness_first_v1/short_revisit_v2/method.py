import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('short_revisit_v2_parent', HERE.parent/'revisit_v1/method.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)

class ShortRevisitFactory(parent.RevisitFactory):
    def continuations(self, position, yaw):
        c = self.components
        return {'C0':c['terminal']+['S'],
                'C_A':c['short_a']+c['terminal']+['S'],
                'C_B':c.get('short_b',c['b'])+c['terminal']+['S']}
