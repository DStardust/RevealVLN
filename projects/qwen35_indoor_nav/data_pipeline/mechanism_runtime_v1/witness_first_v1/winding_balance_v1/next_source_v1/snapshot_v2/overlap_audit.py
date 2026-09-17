"""CPU planning annotations only: same-house Euclidean hub distance, no filtering."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if p.name=='vla')
WF=ROOT/'projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/witness_first_v1'


def read(path):
    before=path.stat();raw=path.read_bytes();after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def main():
    locks={};old=[]
    for batch in ('batch_00','batch_01r1'):
        path=WF/f'batch_execution_v1/{batch}/run_v1/EXECUTION_CONFIG.json'
        cfg,h=read(path);locks[str(path.relative_to(ROOT))]=h
        old.extend(dict(batch=batch,candidate_id=r['candidate_id'],house_id=r['house_id'],
            hub_index=r['hub_index'],position=r['configuration']['u_position']) for r in cfg['candidates'])
    cfg,h=read(HERE/'CONFIG_DRAFT.json');locks[str((HERE/'CONFIG_DRAFT.json').relative_to(ROOT))]=h
    result,h=read(HERE/'result.json');locks[str((HERE/'result.json').relative_to(ROOT))]=h
    summaries={(r['house_id'],r['hub_index']):r for r in result['hubs']}
    table=[]
    for index,r in enumerate(cfg['candidates']):
        references=[]
        for row in old:
            if row['house_id']!=r['house_id']:continue
            distance=math.dist(row['position'],r['configuration']['u_position'])
            references.append(dict(row,distance_m=distance,within_strict_1m=distance<1.0))
        near=[x for x in references if x['within_strict_1m']]
        summary=summaries[(r['house_id'],r['hub_index'])]
        table.append(dict(config_index=index,house_id=r['house_id'],hub_index=r['hub_index'],
            candidate_id=r['candidate_id'],position=r['configuration']['u_position'],
            count_matched_candidates=summary['count_matched_candidates'],
            balanced_history_actions=r['history_actions_before_public_tail'],
            history_exact_FLR_counts=r['expected_history_action_counts'],
            max_continuation_actions=r['max_continuation_actions'],balance=r['balance'],
            required_neutral_spin_directions=r['required_neutral_spin_directions'],
            reference_hubs_same_house=references,overlaps_preselected_hub_within_1m=bool(near),
            planning_status='ALREADY_SELECTED_HUB_RETAINED' if near else 'NO_PRIOR_SELECTED_HUB_WITHIN_1M',
            physical_family_certified=False,executable=False))
    assert len(table)==len(summaries)==6
    for path,h in locks.items():assert read(ROOT/path)[1]==h
    out=dict(status='CPU_PLANNING_ANNOTATIONS_NOT_APPLIED',rows=table,
        previous_batches_read_only=old,comparison='same_house_euclidean_m_strict_lt_1_no_cross_house_distance',
        total_hubs=len(table),near_preselected_hubs=sum(r['overlaps_preselected_hub_within_1m'] for r in table),
        kept_all_candidates=True,no_failed_outcome_replacement=True,source_sha256=locks,
        runtime_executed=False,new_physical_families=0,scientific_pass=False)
    with (HERE/'HUB_OVERLAP_PLANNING.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    lines=['# 六汇合点动作计数匹配规划（未执行）','',
        '原封存prepare生成新snapshot_v2；不覆盖snapshot_v1，不改旧批次，不以新候选替换或隐去失败。全部新cfg仍须主agent单独冻结预算、应用已登记语言规范并完整重放。', '',
        '|配置索引|房屋|hub|计数匹配备选数|历史动作|最长continuation|与已选hub<1m|',
        '|---:|---|---:|---:|---:|---:|---|']
    for row in table:
        overlaps=', '.join(x['batch']+':'+x['candidate_id']+f" ({x['distance_m']:.6f} m)" for x in row['reference_hubs_same_house'] if x['within_strict_1m']) or '无'
        lines.append(f"|{row['config_index']}|{row['house_id']}|{row['hub_index']}|{row['count_matched_candidates']}|{row['balanced_history_actions']}|{row['max_continuation_actions']}|{overlaps}|")
    lines+=['','距离仅在相同房屋内计算，阈值严格小于1m；不将不同屋的坐标作距离比较。标注只是已选位置关系，不代表任务语义相同或已取得证书。候选选择顺序原样保留，不删除重叠hub。',
        '', '六hub来自三个房屋，不能当六个独立房屋。count-match仅在每hub最多1000保留proposal内成立，截断量详见原result.json；没有声明穷尽可行组合。',
        '', '自然语言修订需在下一新配置应用language_realization_v1。此快照原任务文案为封存算法输出，未在本节点暗改。']
    with (HERE/'HUB_PLANNING_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(total_hubs=6,near_preselected_hubs=out['near_preselected_hubs'],rows=table)))


if __name__=='__main__':main()
