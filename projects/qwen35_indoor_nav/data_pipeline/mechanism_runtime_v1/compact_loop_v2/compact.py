"""Validate the actual compact return, never an unexecuted geometric path.

The old expanded inverse is a different, unused trajectory. This version does
not execute it as a prerequisite. Every accepted compact loop still undergoes
actual replay, complete event checks and full agent/sensor closure. All family
matrix, join and repeated replay checks remain inherited and unchanged.
"""
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RUNTIME / 'feedback_generation_v1'), str(RUNTIME)]
from feedback import FeedbackFactory, candidate_targets
from factory import inverse, pose_distance, Reject


def closure_report(first, last):
    if set(first.get('sensors', {})) != {'rgb', 'semantic'} or set(last.get('sensors', {})) != {'rgb', 'semantic'}:
        raise Reject('LOOP_SENSOR_SET')
    values = {'agent': pose_distance(first, last)}
    values.update({name: pose_distance(first['sensors'][name], last['sensors'][name])
                   for name in ('rgb', 'semantic')})
    return {'distances': values, 'pass': all(max(value) <= 1e-5 for value in values.values())}


class CompactLoopFactory(FeedbackFactory):
    def construct(self, config):
        # A common public tail containing an anchor/terminal can never pass the
        # inherited matrix contract. Reject it before expensive loop discovery.
        trace = self.runner.run(config['u_position'], config['yaw_bin'], list(config['public_tail']))
        if not self.pattern(trace, [], [self.a, self.b, self.end]):
            raise Reject('INITIAL_PUBLIC_TAIL_NOT_NEUTRAL')
        return super().construct(config)

    def loop(self, position, yaw, role, forbidden):
        for outbound in self.routes(position, yaw, role):
            outgoing = self.runner.run(position, yaw, outbound)
            if not self.pattern(outgoing, [role], forbidden):
                self.emit('route_rejected', {'role': role, 'reason': 'OUTBOUND_EVENT_OR_LEGALITY'})
                continue
            expanded, compact = inverse(outbound)
            actions = outbound + compact
            if len(actions) > 504:
                self.emit('route_rejected', {'role': role, 'reason': 'HISTORY_LENGTH'})
                continue
            trace = self.runner.run(position, yaw, actions)
            if not self.pattern(trace, [role], forbidden):
                self.emit('route_rejected', {'role': role, 'reason': 'ACTUAL_COMPACT_EVENT_OR_LEGALITY'})
                continue
            report = closure_report(trace['observations'][0]['pose'], trace['observations'][-1]['pose'])
            self.emit('compact_loop_closure', dict(report, role=role, executed_loop_actions=len(actions),
                unused_expanded_trajectory_actions=len(outbound)+len(expanded),
                unused_expanded_trajectory_executed=False, trace_hash=trace['trace_hash']))
            if report['pass']:
                return actions
        raise Reject('NO_VALID_LOOP', role=role)


def rank_reachable_positions(backend, positions, limit=4):
    """Geometry-only prefilter; not a visibility or family certificate.

    No source coordinate is snapped into a valid trajectory. Candidate targets
    may be snapped as proposals, exactly as in V1. All four roles need a path.
    """
    rows = []
    for position in positions:
        snapped = backend.snap_position(position)
        if snapped is None:
            continue
        import math
        if math.dist(position, snapped) > 1e-5:
            continue
        costs = {}
        for role in sorted(backend.eligible):
            targets = candidate_targets(backend, position, role)
            if not targets:
                break
            costs[role] = targets[0]['distance']
        if set(costs) == set(backend.eligible):
            rows.append({'position': list(position), 'role_min_geodesic_m': costs})
    rows.sort(key=lambda row: (max(row['role_min_geodesic_m'].values()),
                              sum(row['role_min_geodesic_m'].values()), row['position']))
    return rows[:limit], {'source_positions': len(positions), 'all_roles_reachable_positions': len(rows)}
