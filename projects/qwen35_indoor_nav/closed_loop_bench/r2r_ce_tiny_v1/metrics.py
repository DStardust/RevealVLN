"""Run the unmodified official Habitat v0.1.7 metric class bodies.

Only registry/base-class plumbing is supplied locally; no habitat-lab dependency
installation or old environment execution. Original MIT license is fingerprinted
and copied into this artifact directory by prepare.py.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'third_party/habitat-lab/habitat/tasks/nav/nav.py'


class Measure:
    def __init__(self, **kwargs):
        self.uuid = self._get_uuid()
        self._metric = None
    def get_metric(self):
        return self._metric


def official_classes():
    tree = ast.parse(SOURCE.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name in ('DistanceToGoal', 'Success', 'SPL')]
    assert len(selected) == 3
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    ns = {'np': np, 'Measure': Measure, 'registry': NS(register_measure=lambda cls: cls)}
    exec(compile(module, str(SOURCE), 'exec'), ns)
    return ns


class OfficialMetrics:
    def __init__(self, sim, goal):
        classes = official_classes()
        self.episode = NS(goals=[NS(position=goal)])
        config = NS(DISTANCE_TO='POINT', SUCCESS_DISTANCE=3.)
        self.values = {name: classes[cls](sim=sim, config=config) for name, cls in
                       [('distance_to_goal','DistanceToGoal'), ('success','Success'), ('spl','SPL')]}
        def check(uuid, dependencies):
            assert all(k in self.values for k in dependencies)
        self.task = NS(is_stop_called=False,
                       measurements=NS(measures=self.values, check_measure_dependencies=check))
        for measure in self.values.values():
            measure.reset_metric(episode=self.episode, task=self.task)

    def update(self, stopped):
        self.task.is_stop_called = stopped
        for measure in self.values.values():
            measure.update_metric(episode=self.episode, task=self.task)
        return self.get()

    def get(self):
        return {k: float(m.get_metric()) for k, m in self.values.items()}
