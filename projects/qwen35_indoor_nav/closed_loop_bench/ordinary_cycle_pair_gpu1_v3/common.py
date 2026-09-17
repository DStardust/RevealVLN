"""Versioned full-evaluation control. Frozen tiny interface reused read-only."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1];ROOT=LINE.parents[1]
TINY=HERE.parent/'r2r_ce_tiny_v1'
TRAIN=LINE/'sft_acceptance/ordinary_sync_recovery_v1'
s=importlib.util.spec_from_file_location('tiny_common',TINY/'common.py')
old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
sha=old.sha;load=old.load;write=old.write;append=old.append
Window=old.Window;advance=old.advance;xyzw_to_wxyz=old.xyzw_to_wxyz
ACTIONS=old.ACTIONS;HABITAT_IDS=old.HABITAT_IDS


def verify_lock():
    lock=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    for path,expected in lock['files'].items():
        p=Path(path).resolve()
        assert p.is_relative_to(ROOT) and sha(p)==expected,'LOCK_MISMATCH:'+path
    return lock


def schedules(count,lanes):
    assert count>0 and lanes>0
    result=[list(range(i,count,lanes)) for i in range(lanes)]
    assert sorted(x for row in result for x in row)==list(range(count))
    return result


def parity_accept(rows):
    return len(rows)==2 and all(r['action_match'] and r['max_abs']<=.15 and r['relative_l2']<=.03 for r in rows)

