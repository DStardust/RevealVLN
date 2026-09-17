"""Explicit, count-checked reuse of the sealed worker in a new output version."""
import ast
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'feedback_generation_v1'
sys.path[:0] = [str(HERE), str(OLD), str(HERE.parent)]
from compact import CompactLoopFactory, rank_reachable_positions


def adapted_source():
    source = (OLD / 'worker.py').read_text()
    old = """                positions=[p for p in row['source_positions'] if (backend.snap_position(p) is not None and math.dist(p,backend.snap_position(p))<=1e-5)]
                positions=sorted(positions,key=lambda p:(sum(math.dist(p,c) for c in centers),p))[:4]"""
    new = """                ranked, geometry = rank_reachable_positions(backend, row['source_positions'])
                save(folder/'GEOMETRY_REACHABILITY.json', dict(geometry, selected=ranked))
                positions = [entry['position'] for entry in ranked]"""
    assert source.count(old) == 1, 'OLD_POSITION_SELECTION_CHANGED'
    source = source.replace(old, new)
    ast.parse(source)
    return source


def main():
    module = types.ModuleType('q35n_compact_loop_v2_worker')
    module.__file__ = str(OLD / 'worker.py')
    exec(compile(adapted_source(), str(OLD / 'worker.py'), 'exec'), module.__dict__)
    module.HERE = HERE
    module.FeedbackFactory = CompactLoopFactory
    module.rank_reachable_positions = rank_reachable_positions
    module.main()

if __name__ == '__main__': main()
