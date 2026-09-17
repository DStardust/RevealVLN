"""A standard execution repair for exact input cycles, not a learned contribution."""
import hashlib
import json

ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'STOP')


class CycleRecovery:
    def __init__(self):
        self.counts = {}

    def reset(self):
        self.counts.clear()

    def choose(self, instruction, images, executed, logits):
        """Inputs are only the original causal policy payload and its four logits.

        On a repeated complete input, try the least used movement at that input;
        model logits break ties. Preserve a native STOP. The caller accounts for
        the selected primitive and adds its actual execution to the next history.
        """
        assert len(logits)==4 and 1<=len(images)<=2 and len(executed)<=8
        assert all(a in ACTIONS[:3] for a in executed)
        key = hashlib.sha256(json.dumps([instruction,
            [hashlib.sha256(image).hexdigest() for image in images],list(executed)],
            ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        native = max(range(4),key=lambda i:logits[i])
        repeated = key in self.counts
        counts = self.counts.setdefault(key,[0,0,0])
        choice = native
        if repeated and native!=3:
            choice = max(range(3),key=lambda i:(-counts[i],logits[i]))
        if choice!=3:
            counts[choice]+=1
        return ACTIONS[choice],dict(native_action=ACTIONS[native],cycle_override=choice!=native,
                                   repeated_input=repeated,input_key=key)
