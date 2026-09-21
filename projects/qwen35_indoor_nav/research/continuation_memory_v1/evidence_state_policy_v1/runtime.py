"""One isolated causal stream; simulator/checker integration stays outside the actor."""
import torch
from model import select_action


class RuntimePolicy:
    def __init__(self, model, budget=500):
        if budget != 500:
            raise ValueError('REGISTERED_BUDGET_IS_500')
        self.model = model
        self.reset()

    def reset(self):
        self.state = self.model.reset(1, next(self.model.parameters()).device)
        self.decisions = 0
        self.history = []
        self.pending = None
        self.stopped = False

    @torch.inference_mode()
    def propose(self, feature, native_logits):
        if self.stopped or self.decisions >= 500 or self.pending is not None:
            raise ValueError('EPISODE_FINISHED_OR_ACTION_NOT_COMMITTED')
        logits, self.state, detail = self.model.step(feature, native_logits, self.state)
        action = int(select_action(logits)[0])
        self.pending = action
        return dict(action=action, native_action=int(select_action(native_logits)[0]),
                    method_logits=logits[0].tolist(), state=detail['state'][0].tolist())

    def commit(self, executed_action):
        if self.pending is None or executed_action != self.pending:
            raise ValueError('EXECUTED_ACTION_MISMATCH')
        self.decisions += 1
        self.stopped = executed_action == 3
        if not self.stopped:
            self.history.append(('F','L','R')[executed_action])
            self.history = self.history[-8:]
        self.pending = None
