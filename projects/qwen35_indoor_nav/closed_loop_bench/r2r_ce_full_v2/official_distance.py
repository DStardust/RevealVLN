"""Unmodified Habitat-Lab 0.1.7 geodesic_distance function, isolated plumbing."""
import ast
from collections.abc import Sequence
from pathlib import Path
import numpy as np
import habitat_sim

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[3]/'third_party/habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py'


def extract():
    tree=ast.parse(SOURCE.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='HabitatSim')
    function=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='geodesic_distance')
    module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[]))
    namespace=dict(habitat_sim=habitat_sim,np=np,Sequence=Sequence)
    exec(compile(module,str(SOURCE),'exec'),namespace)
    return namespace['geodesic_distance']


official_geodesic_distance=extract()


def measure(pathfinder,position,goals,episode):
    # The tiny metrics harness uses SimpleNamespace for Episode. Supply only the
    # cache slot expected by the actual official Episode class.
    from types import SimpleNamespace
    if not hasattr(episode,'_shortest_path_cache'):episode._shortest_path_cache=None
    return float(official_geodesic_distance(SimpleNamespace(pathfinder=pathfinder),position,goals,episode))
