"""Two RGB views aligned to the existing eight executed actions; causal and finite."""
import hashlib
STRIDE=8
def indices(t):
    assert type(t) is int and t>=0
    return [0] if t==0 else [max(0,t-STRIDE),t]
def choose(sequence,t):
    assert len(sequence)>t
    return [sequence[i] for i in indices(t)]
def fingerprint(instruction,rgbs,actions):
    import json
    return hashlib.sha256(json.dumps([instruction,list(rgbs),list(actions)],ensure_ascii=False).encode()).hexdigest()
def window_class(parent):
    class ActionAlignedWindow(parent):
        def __init__(self):
            super().__init__();self.frames=[];self.decision_t=None
        def receive(self,payload,executed=None):
            alive=super().receive(payload,executed)
            if not alive:return False
            latest=self.images[-1]
            if executed is None:self.frames=[latest];self.decision_t=0
            else:self.frames=(self.frames+[latest])[-(STRIDE+1):];self.decision_t+=1
            self.images=[self.frames[-1]] if self.decision_t==0 else [self.frames[0],self.frames[-1]]
            assert len(self.images)==min(2,self.decision_t+1) and len(self.frames)<=STRIDE+1
            return True
        def input_audit(self):
            return dict(input_frame_indices=indices(self.decision_t),
                        input_rgb_sha256=[hashlib.sha256(x).hexdigest() for x in self.images],
                        frame_stride=STRIDE)
    return ActionAlignedWindow
