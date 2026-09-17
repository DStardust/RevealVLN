"""Same STOP-only prefixes, reproducing the original float32 official SPL."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]


def load(name,path,expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,path
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


old=load('unchanged_stop_prefix',HERE/'counterfactual.py','550683db2707bcf49006ca669fb9a0c3955c3efd1d8b14e5673da23188e7f82d')
official=load('original_spl_class',LINE/'closed_loop_bench/r2r_ce_tiny_v1/metrics.py','2433cdaec622daed5b3aad921378e99ef55342dee6c4325e5e2bd5d181c04dc7')
assert hashlib.sha256(official.SOURCE.read_bytes()).hexdigest()=='7bc74c490715dd67d42afa93b8703b582eff56d394cd133cce9bfac503b16216'
_distance=official.official_classes()['SPL']._euclidean_distance
ACTIONS=old.ACTIONS;GRID=old.GRID;choose=old.choose;ndtw=old.ndtw;stats=old.stats;assess=old.assess;select=old.select


def official_traveled(positions):
    previous=np.asarray(positions[0],dtype=np.float32)
    traveled=0.0
    for position in positions[1:]:
        current=np.asarray(position,dtype=np.float32)
        traveled+=_distance(None,current,previous)
        previous=current
    return traveled


def prefix(episode,decisions,reference,bias):
    result=old.prefix(episode,decisions,reference,bias)
    if result['stopped']:
        j=result['steps']-1
        positions=episode['positions'][:j+1]+[episode['positions'][j]]
    else:positions=episode['positions']
    traveled=official_traveled(positions)
    distance=episode['distances'][0]
    result['spl']=float(result['success']*(distance/max(distance,traveled)))
    return result
