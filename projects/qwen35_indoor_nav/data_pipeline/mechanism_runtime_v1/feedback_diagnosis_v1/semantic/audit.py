"""Read-only evidence audit; no Habitat, GPU, model, or writable old journal."""
import collections
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
SOURCE=RUNTIME/'feedback_generation_v1'
RUN=SOURCE/'run_v1'
ROOT=RUNTIME.parents[3]

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_journal(path,head,config):
    previous='0'*64;count=0;size=0;retained=[];kinds=collections.Counter()
    with path.open('rb') as handle:
        for seq,line in enumerate(handle):
            size+=len(line)
            assert line.endswith(b'\n'),'TRUNCATED_RECORD'
            row=json.loads(line)
            assert canonical(row)+b'\n'==line,'NONCANONICAL'
            assert row['seq']==seq and row['prev']==previous,'CHAIN_ORDER'
            body={k:v for k,v in row.items() if k!='hash'}
            assert hashlib.sha256(canonical(body)).hexdigest()==row['hash'],'CHAIN_HASH'
            previous=row['hash'];count+=1;kinds[row['kind']]+=1
            if row['kind']=='__config__':assert seq==0 and row['payload']==config
            if row['kind'] not in ('__config__','budget','freeze'):retained.append(row)
    assert count==head['count'] and size==head['byte_length'] and previous==head['last_hash']
    assert head['config_hash']==hashlib.sha256(canonical(config)).hexdigest()
    return retained,dict(kinds)

def action_stats(trace):
    actions=trace['actions'];observations=trace['observations']
    assert len(observations)==len(actions)+1
    return dict(length=len(actions),actions=dict(collections.Counter(actions)),
        observations=len(observations),complete=trace['complete'],collisions=trace['collisions'],
        kind=trace.get('discovery_kind','factory_replay'),failure_reason=trace.get('failure_reason'),
        initial_position=list(trace['initial_position']),initial_yaw=trace['initial_yaw_bin'],
        reset_position_error=math.dist(trace['initial_position'],observations[0]['pose']['position']))

def summarize_target_calls(rows):
    by_role={}
    for role in sorted({r['role'] for r in rows}):
        matching=[r for r in rows if r['role']==role]
        by_role[role]=dict(calls=len(matching),empty=sum(not r['targets'] for r in matching),
            nonempty=sum(bool(r['targets']) for r in matching),accepted_targets=sum(len(r['targets']) for r in matching))
    return by_role

def geometry(configs,row,inventory):
    starts=sorted({tuple(c['u_position']) for c in configs})
    objects=inventory['objects'];eligible=inventory['eligible']
    centers={role:[objects[str(i)]['center'] for i in ids] for role,ids in eligible.items()}
    details=[]
    for p in starts:
        by_role={}
        for role,points in centers.items():
            ring=[[c[0]+r*math.cos(a*math.pi/4),c[1],c[2]+r*math.sin(a*math.pi/4)]
                  for c in points for r in (.75,1.25) for a in range(8)]
            by_role[role]=dict(min_center_euclidean_m=min(math.dist(p,c) for c in points),
                min_unsnapped_ring_euclidean_m=min(math.dist(p,t) for t in ring),
                all_unsnapped_ring_points_beyond_8m=all(math.dist(p,t)>8 for t in ring))
        details.append(dict(position=list(p),roles=by_role))
    source_positions=row['source_positions']
    return dict(unique_selected_starts=len(starts),selected_start_y=sorted({p[1] for p in starts}),
        source_position_count=len(source_positions),source_y_range=[min(p[1] for p in source_positions),max(p[1] for p in source_positions)],
        shared_metadata_levels=row.get('shared_metadata_levels'),
        role_object_region_ids={role:sorted({objects[str(i)]['region_id'] for i in ids}) for role,ids in eligible.items()},
        role_center_y_ranges={role:[min(p[1] for p in points),max(p[1] for p in points)] for role,points in centers.items()},
        starts=details,
        closest_source_position_to_anchor_A=min(source_positions,key=lambda p:min(math.dist(p,c) for c in centers['anchor_A'])),
        closest_source_anchor_A_center_distance_m=min(math.dist(p,c) for p in source_positions for c in centers['anchor_A']),
        target_snap_distances_recorded=False,target_filter_reason_counters_recorded=False,
        global_reachability_or_wrong_floor_proven=False,
        scope='Euclidean pre-snap geometry and stored metadata only; not navmesh/geodesic evaluation')

def save(name,value):
    path=HERE/name
    content=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==content,('NEW_VERSION_REQUIRED',name)
    else:
        with path.open('x') as handle:handle.write(content)

def main():
    inputs={}
    def read(path):
        assert path.resolve().is_relative_to(ROOT)
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    config=read(RUN/'EXECUTION_CONFIG.json');head=read(RUN/'journal/HEAD.json')
    inputs[str((RUN/'journal/events.jsonl').relative_to(ROOT))]=sha(RUN/'journal/events.jsonl')
    events,kinds=verify_journal(RUN/'journal/events.jsonl',head,config)
    global_result=read(RUN/'result.json')
    results=[]
    for candidate in config['candidates']:
        cid=candidate['candidate_id'];folder=RUN/'bundles'/cid
        outcome=read(folder/'result.json');configs=read(folder/'FROZEN_GEOMETRY_CONFIGS.json')
        inventory=read(folder/'SEMANTIC_INVENTORY.json')
        selected=[e for e in events if e['payload'].get('bundle')==cid]
        targets=[e['payload']['value'] for e in selected if e['kind']=='feedback_targets']
        trace_events=[e for e in selected if e['kind']=='trace_saved']
        traces=[]
        for event in trace_events:
            value=event['payload'];path=folder/'traces'/('%06d.json'%value['index'])
            trace=read(path);assert sha(path)==value['sha256'],'TRACE_JOURNAL_HASH'
            assert trace['complete']==value['complete']
            traces.append(dict(index=value['index'],journal_seq=event['seq'],**action_stats(trace)))
        assert len(traces)==len(list((folder/'traces').glob('*.json')))==outcome['trace_count']
        journal_actions=[e for e in selected if e['kind']=='action_completed']
        last_seq=max((t['journal_seq'] for t in traces),default=-1)
        unsealed_tail=[e for e in journal_actions if e['seq']>last_seq]
        rejects=[]
        for event in selected:
            if event['kind'] in ('route_rejected','discovery_rejected'):
                rejects.append(dict(journal_seq=event['seq'],kind=event['kind'],**event['payload']['value']))
        action_count=collections.Counter()
        for tr in traces:action_count.update(tr['actions'])
        feedback_keys={(tuple(t['position']),t['yaw']) for t in targets}
        config_keys={(tuple(c['u_position']),c['yaw_bin']) for c in configs}
        rejected_configs={(tuple(r['config']['u_position']),r['config']['yaw_bin']) for r in rejects if r['kind']=='discovery_rejected'}
        results.append(dict(candidate_id=cid,house_id=candidate['house_id'],outcome=outcome,
            frozen_configurations=len(configs),unique_feedback_configurations=len(feedback_keys&config_keys),
            explicitly_rejected_configurations=len(rejected_configs),
            target_calls_by_role=summarize_target_calls(targets),target_events=targets,
            rejection_counts=dict(collections.Counter(r['kind']+':'+r['reason']+':'+str(r.get('role',r.get('details',{}).get('role'))) for r in rejects)),
            rejection_events=rejects,traces=traces,trace_count=len(traces),
            complete_trace_count=sum(t['complete'] for t in traces),zero_action_trace_count=sum(t['length']==0 for t in traces),
            trace_action_count=sum(t['length'] for t in traces),trace_action_composition=dict(action_count),
            trace_length_range=[min(t['length'] for t in traces),max(t['length'] for t in traces)],
            journal_completed_actions=len(journal_actions),journal_actions_without_saved_trace=len(journal_actions)-sum(t['length'] for t in traces),
            action_events_after_last_saved_trace=len(unsealed_tail),
            unsealed_tail_action_composition=dict(collections.Counter(e['payload']['value']['action'] for e in unsealed_tail)),
            geometry=geometry(configs,candidate,inventory)))
    for path in (SOURCE/'feedback.py',SOURCE/'worker.py',RUNTIME/'runtime_journal.py',RUNTIME.parent/'mechanism_factory_v2/factory.py'):
        inputs[str(path.relative_to(ROOT))]=sha(path)
    output=dict(status='CPU_FAILURE_EVIDENCE_AUDIT_COMPLETE',journal_chain_and_head_pass=True,
        journal_records=head['count'],journal_kind_counts=kinds,candidates=results,
        observed_completed_actions=sum(r['journal_completed_actions'] for r in results),
        saved_trace_actions=sum(r['trace_action_count'] for r in results),
        global_result=global_result,gpu_operations=0,old_files_modified=False,scientific_pass=False)
    assert output['observed_completed_actions']==global_result['counts']['actual_actions']
    save('result.json',output);save('SOURCE_HASHES.json',inputs)
    lines=['# 特殊数据 CPU 失败审计','',
        '只读取旧日志/trace/config，不运行 navmesh、仿真或 GPU；旧结果保持原样。',
        f'日志链及 HEAD 通过：{head["count"]} 条。141 条 trace 均通过 journal SHA256 核验。','',
        '| 房屋 | 配置 | 空/非空 target 调用 | 保存 trace/动作 | 未封存尾动作 | 终态 |',
        '|---|---:|---:|---:|---:|---|']
    for r in results:
        roles=r['target_calls_by_role'];empty=sum(x['empty'] for x in roles.values());nonempty=sum(x['nonempty'] for x in roles.values())
        lines.append(f'| {r["house_id"]} | {r["frozen_configurations"]} | {empty}/{nonempty} | {r["trace_count"]}/{r["trace_action_count"]} | {r["journal_actions_without_saved_trace"]} | {r["outcome"]["status"]} |')
    lines += ['', '## 可证结论与界限','',
        '- 1LXtFkjw3qL：16 配置全部在 anchor_A 的 target 生成处返回空；全部 NO_VALID_LOOP(anchor_A)，其他 role 未尝试。只有 16 个零动作初始 trace。证实的是当前生成器无候选，不是场景没有可达目标。',
        '- 该屋选择 4 个起点，同一 y=0.084411；chair 对象中心 y 约 1.05，metadata level=1。对象中心高度不等于地面高度，level 编号也不等于 y 坐标；没有证据确证楼层错配。代码要求起点 snap 偏移不超过 1e-5，实际 reset 偏差已逐 trace 统计；没有保存起点筛选时逐点 snap 结果。',
        '- 3 个选定起点的全部原始（未 snap）anchor_A 环采样点都超过 8 m；另一个有小于 8 m 的环点。距离上限是具体可疑限制，但未记录 snap/find_path/距离超限计数，因此不能把全部空候选归因到单一条件，更不能宣称调整距离已经有效。',
        '- 17DRP5sb8fy 与 1pXnuDYAj8r 有非空目标和完整动作 trace，不能归因于统一“走不动”；显式拒绝集中在 anchor_B 的 OUTBOUND_EVENT_OR_LEGALITY / NO_VALID_LOOP。原 reason 合并了事件与合法性，需要下一版本记录细分谓词。',
        '- journal 完成动作共 14205，保存 trace 动作共 14144；相差 61。差额逐屋与最后一个 trace 后的动作尾一致，是资源截断后的未封存轨迹，不可作为合格训练动作。141/141 complete 是“已保存 trace”的条件统计，不代表全部实际尝试都完整。',
        '- 下一版本至少增加：target 筛选原因计数、起点/目标 snap 偏移与 floor/geodesic 证据、role-specific 拒绝谓词、未完成 trace 持久化；本审计没有实现或授权新生产。','']
    report='\n'.join(lines)
    path=HERE/'REPORT_ZH.md'
    if path.exists():assert path.read_text()==report
    else:
        with path.open('x') as handle:handle.write(report)
    print(json.dumps(dict(journal_records=head['count'],trace_count=sum(r['trace_count'] for r in results),
        saved_actions=output['saved_trace_actions'],journal_actions=output['observed_completed_actions'],
        candidates=[{k:r[k] for k in ('house_id','target_calls_by_role','trace_count','trace_action_count','journal_actions_without_saved_trace','action_events_after_last_saved_trace')} for r in results]),indent=2))

if __name__=='__main__':main()
