"""Added recurrence ablation; the frozen StreamVLN history is unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'recovery_action_v1'))
from recovery_model import RecoveryMemory, sequence_logits
import torch


class ConfirmationMemory(RecoveryMemory):
    def __init__(self,width,architecture):
        assert architecture in ('CONCAT','LOCAL')
        super().__init__(width,'CONCAT')
        self.architecture=architecture

    def update(self,feature,memory):
        # Unit-gain current embedding avoids shrinking the LOCAL control by 100x.
        if self.architecture=='LOCAL':return torch.tanh(self.writer(self.norm(feature))).reshape(-1,8,64)
        return super().update(feature,memory)
