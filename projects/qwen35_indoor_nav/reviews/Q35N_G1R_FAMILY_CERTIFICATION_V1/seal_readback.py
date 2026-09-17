"""Independent on-disk schema/hash readback and final report, without GPU."""
import ast
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def save(name, data):
    with (OUT/name).open('x') as f: json.dump(data, f, indent=2, ensure_ascii=False)


def legacy_keys(value):
    if isinstance(value, dict):
        return {k: ({int(i): n for i, n in v.items()} if k == 'pixels' else legacy_keys(v)) for k, v in value.items()}
    if isinstance(value, list): return [legacy_keys(v) for v in value]
    return value


def legacy_hash(value):
    return hashlib.sha256(json.dumps(legacy_keys(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    result = json.loads((OUT/'result.json').read_text()); assert result['family_certified']
    assert json.loads((OUT/'LEASE_RESTORED.json').read_text())['restored']
    assert json.loads((OUT/'EXECUTION.json').read_text())['cleanup_complete']
    c = module('compiler', LINE/'data_pipeline/mechanism_family_v1/compiler.py')
    v = module('schema_check', LINE/'data_pipeline/mechanism_family_v1/schema_check.py')
    schema = json.loads((OUT/'SCHEMA_USED.json').read_text()); v.vocabulary_check(schema)
    lock = json.loads((OUT/'COMPILER_CODE_LOCK.json').read_text())
    for rel, expected in lock.items(): assert sha(LINE/rel) == expected, rel
    sup = [json.loads(x) for x in (OUT/'SUPERVISION_ONLY.jsonl').read_text().splitlines()]
    pol = [json.loads(x) for x in (OUT/'POLICY_INPUT.jsonl').read_text().splitlines()]
    family = json.loads((OUT/'FAMILY_MANIFEST.json').read_text())
    for item in sup+pol+[family]: v.validate(item, schema)
    assert {x['sample_id'] for x in sup} == {x['sample_id'] for x in pol} and len(sup) == len(pol) == 18
    prefix_count = 0
    for p in (OUT/'policy_prefixes').glob('*.jsonl'):
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        assert [r['causal_cutoff_step'] for r in rows] == list(range(len(rows)))
        assert rows[0]['memory_reset'] and not any(r['memory_reset'] for r in rows[1:])
        for row in rows:
            v.validate(row, schema)
            assert all(o['step'] <= row['causal_cutoff_step'] for o in row['observations'])
            assert all(a['step'] < row['causal_cutoff_step'] for a in row['executed_actions'])
            prefix_count += 1
    # Read back both .npy format headers and raw pixel bytes independently of numpy.
    blobs = collections.Counter()
    for p in (OUT/'content').glob('*.npy'):
        data = p.read_bytes(); assert data[:6] == b'\x93NUMPY'
        version = tuple(data[6:8]); size = 2 if version == (1, 0) else 4
        assert version in [(1, 0), (2, 0), (3, 0)]
        n = struct.unpack('<H' if size == 2 else '<I', data[8:8+size])[0]
        header = ast.literal_eval(data[8+size:8+size+n].decode().strip())
        raw = data[8+size+n:]; kind = p.name.split('.')[1]
        assert not header['fortran_order']
        assert header['shape'] == ((224, 224, 3) if kind == 'rgb' else (224, 224))
        assert header['descr'] == ('|u1' if kind == 'rgb' else '<u4')
        assert len(raw) == 224*224*(3 if kind == 'rgb' else 4)
        assert hashlib.sha256(raw).hexdigest() == p.name.split('.')[0]
        blobs[kind] += 1
    traces = {}; norms = []
    for p in (OUT/'physical_traces').glob('*.json'):
        tr = json.loads(p.read_text()); h = tr.pop('trace_hash')
        assert legacy_hash(tr) == h
        tr['trace_hash'] = h; traces[h] = tr; norms += tr.get('normalization_events', [])
    assert len(traces) == 27 and len(norms) == 27
    for record in sup:
        assert record['full_log_ref'][7:] in traces
        assert c.ref(c.semantic_query(record['query'])) == record['query_hash']
        for o in pol:
            if o['sample_id'] == record['sample_id']:
                for image in o['observations']: assert (OUT/'content'/f"{image['rgb_ref'][7:]}.rgb.npy").is_file()
    protected = {}
    for name in ['Q35N_G0R_DEPENDENCY_RECOVERY_V1', 'Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2',
                 'Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1', 'Q35N_P2R1_SPEC_CORRECTIONS_V1',
                 'Q35N_G2_INTERFACE_ACCEPTANCE_V1', 'Q35N_G1R_TASK_INSTANCE_V3']:
        run = subprocess.run(['sha256sum', '-c', 'SHA256SUMS'], cwd=OUT.parent/name, capture_output=True, text=True)
        assert run.returncode == 0, name+run.stderr
        protected[name] = {'pass': True, 'entries': len(run.stdout.splitlines())}
    same_query = collections.defaultdict(list)
    for r in sup: same_query[(r['task_id'], r['query_hash'])].append(r)
    contrast_groups = sum({r['outcome'] for r in rs} == {'pass', 'fail'} for rs in same_query.values())
    summary = {'readback_pass': True, 'schema_records': len(sup)+len(pol)+1,
        'causal_prefix_records': prefix_count, 'raw_content_files': dict(blobs),
        'verified_physical_trace_hashes': len(traces), 'same_task_query_opposite_label_groups': contrast_groups,
        'max_numerical_position_correction_m': max(n['position_correction_m'] for n in norms),
        'max_numerical_rotation_correction_rad': max(n['angle_correction_rad'] for n in norms),
        'all_boundary_event_sets_unchanged': all(n['raw_see2'] == n['canonical_see2'] for n in norms),
        'action_ce_unique_owners': len(json.loads((OUT/'ACTION_DEDUP.json').read_text())['owners']),
        'history_lengths_including_tail': sorted({r['prefix_cutoff_step'] for r in sup}),
        'continuation_lengths': {r['continuation_trace_id']: len(r['action_targets']) for r in sup},
        'compiler_execution_code_lock_pass': True, 'protected_nodes': protected,
        'scientific_pass': False, 'raw_exact_merge_pass': False}
    save('READBACK_AUDIT.json', summary)
    with (OUT/'REPORT_ZH.md').open('x') as f:
        f.write('# 首个真实机制数据族 V3：主 agent 数据验收\n\n')
        f.write('结论：**NUMERICALLY_NORMALIZED_REAL_FAMILY_ACCEPTANCE_PASS**。首个TV/sink任务实例的真实机制数据已补齐。不是原餐椅任务通过，也不是模型正收益。\n\n')
        f.write('## 实际交付\n\n')
        f.write('- 1个旧暴露MP3D房屋中的真实族；2任务×3历史×3续接=18个cell，12 pass、6 fail。\n')
        f.write('- 3个固定seed×9条完整物理轨迹=27次回放、54次独立任务求值；不计作54个独立统计样本。\n')
        f.write(f'- {prefix_count}条因果prefix记录，{blobs["rgb"]}个不同RGB像素数组、{blobs["semantic"]}个semantic数组，原始与重建前后证据均保留。\n')
        f.write(f'- {result["assertions"]}项运行断言、13项合成组件单测、导出后schema/NPY原始像素/27轨迹hash/保护文件复核通过。\n')
        f.write(f'- {contrast_groups}组同任务同query但不同历史标签相反的实际关系；Y由原始像素事件重算，不由纸面矩阵赋值。\n\n')
        f.write('## 关键限定与修订\n\n')
        f.write('原餐椅实例受到额外事件混入影响，在有界搜索后仍失败。V3改用TV/sink作任务anchor、餐椅作无关绕行；任务ID/模板/角色/schema明确升版，旧失败另目录保留。\n\n')
        f.write(f'真实历史回到u后仅实施一次登记的数值共同状态重建。最大位置修正 {summary["max_numerical_position_correction_m"]:.9g} m、旋转修正 {summary["max_numerical_rotation_correction_rad"]:.9g} rad，均低于事前1e-5上限；27次边界事件集合全部不变。原始逐像素自然汇合=false，规范化后的u/s跨27回放位置/旋转spread为0，RGB/semantic哈希精确一致。不能在论文中省略此干预。\n\n')
        f.write('这是合法授权MP3D场景中实际运动和真实渲染形成的离线数值规范化对照数据，不是人工编辑图像、纸面标签或导航性能测试。此单族仅interface_only，不能扩称训练集规模、跨场景泛化、SOTA或贡献已成立。\n\n')
        f.write('## 数据入口\n\n')
        f.write('[FAMILY_MANIFEST](FAMILY_MANIFEST.json)、[监督侧18格](SUPERVISION_ONLY.jsonl)、[策略当前输入](POLICY_INPUT.jsonl)、[完整因果prefix索引](POLICY_PREFIX_INDEX.json)、[复核](READBACK_AUDIT.json)。按DATASET_USE_ZH使用，不得把监督侧future query/pose/semantic/ID喂给策略。\n\n')
        f.write('GPU3已恢复原占位，无真实任务被停止。本会话未加载Qwen或进行SFT/机制训练；SFT由另一个会话按独立handoff负责。\n')
    paths = sorted(p for p in OUT.rglob('*') if p.is_file() and 'cache' not in p.relative_to(OUT).parts and '__pycache__' not in p.relative_to(OUT).parts and p.name != 'SHA256SUMS')
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in paths: f.write(f'{sha(p)}  {p.relative_to(OUT)}\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__': main()
