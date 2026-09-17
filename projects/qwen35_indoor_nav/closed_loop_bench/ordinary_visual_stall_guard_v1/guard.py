"""Opt-in causal visual-stagnation controller. No simulator or model imports."""
import hashlib

ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'STOP')


class VisualStallGuard:
    def __init__(self):
        self.last_rgb_sha256 = None
        self.stagnant_forwards = 0
        self.interventions = 0

    def observe_rgb(self, rgb, executed=None):
        if not isinstance(rgb, bytes) or len(rgb) != 224 * 224 * 3:
            raise ValueError('EXPECTED_CAUSAL_RGB_BYTES')
        digest = hashlib.sha256(rgb).hexdigest()
        if executed is None:
            self.last_rgb_sha256 = digest
            self.stagnant_forwards = self.interventions = 0
            return
        if self.last_rgb_sha256 is None or executed not in ACTIONS[:-1]:
            raise ValueError('EXPECTED_ALREADY_EXECUTED_MOTION')
        self.stagnant_forwards = min(8, self.stagnant_forwards + 1) if (
            executed == 'move_forward' and digest == self.last_rgb_sha256) else 0
        self.last_rgb_sha256 = digest

    def choose(self, proposed_action):
        if proposed_action not in ACTIONS or self.last_rgb_sha256 is None:
            raise ValueError('INVALID_PROPOSAL_OR_NO_OBSERVATION')
        streak = self.stagnant_forwards
        used_before = self.interventions
        intervene = proposed_action == 'move_forward' and streak >= 8 and used_before < 4
        executed_action = 'turn_left' if intervene else proposed_action
        if intervene:
            self.interventions += 1
        return dict(proposed_action=proposed_action, action=executed_action, intervened=intervene,
                    stagnant_forwards_before=streak, interventions_before=used_before,
                    interventions_after=self.interventions, rgb_sha256=self.last_rgb_sha256,
                    reason='EIGHT_IDENTICAL_FORWARD_OBSERVATIONS' if intervene else None)
