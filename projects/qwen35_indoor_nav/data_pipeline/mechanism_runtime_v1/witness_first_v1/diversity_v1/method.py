"""New physical tails with unchanged sealed matrix checker and task semantics."""
from collections import Counter
import copy
import math
import sys

from common import RUNTIME
sys.path.insert(0, str(RUNTIME))
from core_bridge import FamilyFactory, Reject, factory


def reorient_histories(source, yaw):
    before = factory.rotations(source['yaw_bin'] - yaw)
    after = factory.rotations(yaw - source['yaw_bin'])
    assert source['public_tail'] == 'LRLRLRLR'
    rows = {name: before + list(actions[:-8]) + after
            for name, actions in source['histories'].items()}
    if max(map(len, rows.values())) > 504:
        raise Reject('NEW_HISTORY_CAP')
    counts = {tuple(Counter(actions)[a] for a in 'FLR') for actions in rows.values()}
    if len(counts) != 1:
        raise Reject('NEW_HISTORY_COUNTS')
    return rows


class DiversityFactory(FamilyFactory):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = copy.deepcopy(source)

    def construct(self, config):
        self.config = config
        # Cheap actual tail probe before replaying long component histories.
        tail = self.runner.run(config['u_position'], config['yaw_bin'], list('FFFFFFFF'))
        if not self.pattern(tail, [], [self.a, self.b, self.end]):
            raise Reject('NEW_MOVING_TAIL_EVENT_OR_COLLISION')
        distance = math.dist(tail['observations'][0]['pose']['position'],
                             tail['observations'][-1]['pose']['position'])
        if not 1.95 <= distance <= 2.05:
            raise Reject('NEW_MOVING_TAIL_DISPLACEMENT', distance=distance)
        return super().construct(config)

    def histories(self, position, yaw):
        initial = self.runner.run(position, yaw, [])['observations'][0]['pose']
        rows = reorient_histories(self.source, yaw)
        for name, actions in rows.items():
            trace = self.runner.run(position, yaw, actions)
            required = [self.b] if name == 'H_B' else [self.a, self.irrelevant]
            forbidden = [self.a] if name == 'H_B' else [self.b]
            if not self.pattern(trace, required, forbidden):
                raise Reject('NEW_HISTORY_EVENT', history=name)
            final = trace['observations'][-1]['pose']
            for a, b in [(initial, final)] + [(initial['sensors'][s], final['sensors'][s]) for s in ('rgb', 'semantic')]:
                if max(factory.pose_distance(a, b)) > 1e-5:
                    raise Reject('NEW_HISTORY_CLOSURE', history=name)
        return rows, dict(step=len(rows['H_A']), position=position, yaw_bin=yaw,
                          target_pose=initial, registered_histories=list(rows.values()))

    def continuations(self, position, yaw):
        bridge = ['L'] * 12 + ['F'] * 8 + ['R'] * 12
        bridge += factory.rotations(self.source['yaw_bin'] - yaw)
        result = {k: bridge + list(v) for k, v in self.source['continuations'].items()}
        if max(map(len, result.values())) > 160:
            raise Reject('NEW_CONTINUATION_CAP')
        return result
