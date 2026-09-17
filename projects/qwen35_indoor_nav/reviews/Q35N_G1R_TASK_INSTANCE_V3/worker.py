import importlib.util
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
s = importlib.util.spec_from_file_location('numerical', OUT.parent/'Q35N_G1R_NUMERICAL_JOIN_V1/worker.py')
num = importlib.util.module_from_spec(s); s.loader.exec_module(num)
core = num.core; core.OUT = OUT
ORIGINAL = num.ORIGINAL
OLD_KINDS = dict(core.KINDS)
NEW_KINDS = {'D': ('tv_monitor', 'living room'), 'K': ('sink', 'kitchen'),
             'B': ('bed', 'bedroom'), 'L': ('chair', 'dining room')}
ALIASES = {'g_D_v2': 'g_T_v3', 'g_K_v2': 'g_K_v3', 'H_D': 'H_T', 'H_D_L': 'H_T_I', 'C_D': 'C_T'}
REVERSE = {v: k for k, v in ALIASES.items()}
old_expected = core.expected
core.expected = lambda task, h, c: old_expected(REVERSE.get(task, task), REVERSE.get(h, h), REVERSE.get(c, c))


def rename(value):
    if isinstance(value, dict): return {ALIASES.get(k, k): rename(v) for k, v in value.items()}
    if isinstance(value, list): return [rename(v) for v in value]
    if isinstance(value, str): return ALIASES.get(value, value)
    return value


class Engine(num.Engine):
    def __init__(self, phase='task_instance_v3'):
        core.KINDS = OLD_KINDS.copy()
        super().__init__(phase)
        self.eligible['D'], self.eligible['L'] = self.eligible['L'], self.eligible['D']
        core.KINDS = NEW_KINDS.copy()
        core.save('TASK_ROLE_INVENTORY.json', {'kinds': NEW_KINDS, 'eligible': self.eligible,
            'object_mapping_unchanged': True, 'original_eligible_filters': 'TV/raw tv; chair/raw word chair; sink/raw sink; bed/raw bed'})

    def discovery(self):
        pool = self.u_pool(); u = pool[22]; yaw = 0; tail = 'LRLRLRLR'
        base = self.candidate_base(u['position'], yaw)
        core.save('RAW_BASE.json', base)
        candidate = self.complete_candidate(u, yaw, base, tail)
        candidate.pop('frozen_candidate_hash')
        candidate = rename(candidate)
        candidate['family_id'] = 'F17_TK_B_numeric_v3'
        candidate['task_configuration'] = {'task_revision': 'observable_tv_sink_then_stop.v3',
            'tasks': {'g_T_v3': '先连续两帧看见客厅内的一台电视，再连续两帧看见卧室内的一张床，然后停止。',
                      'g_K_v3': '先连续两帧看见厨房内的水槽，再连续两帧看见卧室内的一张床，然后停止。'},
            'task_anchor_codes': {'g_T_v3': 'D', 'g_K_v3': 'K'}, 'event_code_categories': NEW_KINDS,
            'id_aliases_from_internal_constructor': ALIASES,
            'not_a_pass_for_original_dining_chair_task': True}
        candidate['frozen_candidate_hash'] = core.digest(candidate)
        core.save('FROZEN_CANDIDATE.json', candidate)
        return candidate


def main():
    assert not (OUT/'result.json').exists()
    core.save('SOURCE_DERIVATION.json', {'numerical_worker_sha256': core.digest((OUT.parent/'Q35N_G1R_NUMERICAL_JOIN_V1/worker.py').read_bytes()),
        'original_sha256': core.digest(ORIGINAL.read_bytes()), 'task_roles': NEW_KINDS, 'output_ids': ALIASES,
        'old_task_route_cache_reused': False})
    eng = None; started = time.time()
    try:
        eng = Engine(); eng.preview(); candidate = eng.discovery()
        core.save('result.json', {'decision': 'V3_REAL_FAMILY_CANDIDATE_READY_FOR_CERTIFICATION',
            'candidate_hash': candidate['frozen_candidate_hash'], 'family_certified': False,
            'raw_exact_merge_pass': False, 'original_dining_task_pass': False, 'scientific_pass': False})
    except core.Reject as e:
        core.save('result.json', {'decision': e.code, 'detail': e.detail, 'family_certified': False, 'scientific_pass': False})
    finally:
        if eng: eng.close()
        core.save('WALL.json', {'seconds': time.time()-started})


if __name__ == '__main__': main()
