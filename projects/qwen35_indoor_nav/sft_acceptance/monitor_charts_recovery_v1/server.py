"""Read-only recovery adapter for the existing 18766 charts; no trainer writes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
SEALED = HERE.parent / 'monitor_charts_v3'
RECOVERY = HERE.parent / 'ordinary_sync_recovery_v1'
FORMAL = RECOVERY / 'formal'
for name, digest in json.loads((SEALED / 'CODE_SEAL.json').read_text()).items():
    assert hashlib.sha256((SEALED / name).read_bytes()).hexdigest() == digest
spec = importlib.util.spec_from_file_location('sealed_monitor_v3', SEALED / 'server.py')
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
b = v3.b


def points_for_segments(segments):
    """Never infer throughput across recovery/reserve jumps or counter rollback."""
    out = []
    for segment, rows in segments:
        pts = v3.chart_points_v3(rows)
        start = 0
        for i, point in enumerate(pts):
            if i == 0 or point['discontinuity']:
                point['discontinuity'] = True
                start = i
            prev = pts[max(start, i - 10)]
            dt = point['unix'] - prev['unix']
            dd = point['decisions'] - prev['decisions']
            point['throughput'] = dd / dt if dt > 0 and dd >= 0 else None
            point['segment'] = segment
        out.extend(pts)
    return out[-1600:]


def resolve_run(status):
    value = status.get('run_dir')
    if not isinstance(value, str):
        return FORMAL / 'NO_CURRENT_RUN'
    run = Path(value).resolve()
    if run.parent != FORMAL.resolve() or not run.name.startswith('attempt_'):
        raise ValueError('UNEXPECTED_CURRENT_RUN')
    return run


def collect(include_gpu=True):
    d = v3.original_collect(execution=FORMAL, include_gpu=include_gpu)
    now = time.time()
    status = d['status']['data'] or {}
    run = resolve_run(status)
    progress = b.read_json(run / 'PROGRESS.json', now=now)
    result = b.read_json(run / 'RESULT.json', now=now)
    history = b.tail_records(run / 'PROGRESS.jsonl')
    rows = history.pop('records')
    old = b.tail_records(v3.EXECUTION / 'run_0001/PROGRESS.jsonl', max_points=1000)
    segments = [('legacy_v3', old['records'])]
    for attempt in sorted(FORMAL.glob('attempt_*'))[:3]:
        if attempt.is_dir() and not attempt.is_symlink():
            segments.append((attempt.name, rows if attempt == run else
                             b.tail_records(attempt / 'PROGRESS.jsonl')['records']))
    points = points_for_segments(segments)
    p = progress['data'] or {}
    c, budget = p.get('cursor', {}), p.get('budget', {})
    charged = p.get('cumulative_compute', {}).get('decisions')
    done = p.get('global_plan_decisions')
    limit = budget.get('max_decisions')
    current_points = [x for x in points if x['segment'] == run.name and x['throughput'] is not None]
    speed = current_points[-1]['throughput'] if current_points else None
    wall = max(0, p.get('wall_deadline_unix', status.get('deadline_unix', now)) - now)
    eta = max(0, limit - charged) / speed if None not in (limit, charged, speed) and speed > 0 else None
    stale = progress['stale'] or d['status']['stale']
    live = status.get('status') == 'TRAINING' and not stale
    state = 'STALE_NO_CONFIRMED_PROGRESS' if status.get('status') == 'TRAINING' and stale else status.get('status', 'UNKNOWN')
    quality = v3.recent_quality_v3(rows, 4000)
    actions = b.action_summary({'confusion': quality.get('confusion')})
    d.update(run_name=run.name, progress=progress, result=result, history=history,
             points=points, actions=actions, recent_quality=quality, monitor_version='recovery_v1',
             decisions=done, charged_compute_decisions=charged, decision_limit=limit,
             training_soft_limit=None, epoch_eta_seconds=None, wall_remaining_seconds=wall,
             wall_grace_seconds=0, budget_eta_seconds=eta if live else None,
             estimated_segment_remaining_seconds=min(wall, eta) if live and eta is not None else None,
             updates_remaining=max(0, budget.get('max_updates', 0) - c.get('updates', 0)),
             steady_decisions_per_second=speed if live else None, display_state=state,
             checkpoint_name=Path(p['latest_checkpoint']).name if p.get('latest_checkpoint') else None,
             legacy_history_available=bool(old['records']), probe={'data': None, 'error': None},
             eta_note='Budget boundary estimate, not completion promise. Charge includes reserve; curves break at recovery.',
             metric_scope='RECENT_ONLINE_FIT_NOT_DEV_NOT_NAVIGATION_SUCCESS')
    return d


def html():
    source = (v3.BASE / 'index.html').read_text()
    replacements = {
        '曲线仅使用日志自带时间与 cursor；无数据处保持缺失。':
        '已接入修复后的续训，地址不变。旧曲线保留；恢复处分段，不跨预算预留跳变计算速度。计费量含保守预留，不是独立样本数。',
        '动作交叉熵 · 随训练决策数': '动作交叉熵 · 随优化器更新数',
        '记录中的 last chunk CE': '每次报告窗口的 CE',
        '吞吐是生产者记录的累计速率，非瞬时 GPU 性能。': '吞吐按同一运行段最近约10个报告窗口重新计算，不混入恢复前计算量或预算预留。',
        '四类动作 · 累计训练目标与预测': '四类动作 · 最近至少4000个训练决策',
        '尚无四类累计混淆矩阵': '尚无近期动作统计',
        "p.status||status.status||'未知'": "d.display_state||'未知'",
        "card('已训练决策',fmt(c.decisions),'实际指令条件动作累计')": "card('计划已完成决策',fmt(d.decisions),'三卡确定性计划计数，含多个epoch；非独立样本')",
        "fmt(m.mean_ce,4),'本训练累计（含断点恢复）'": "fmt(d.recent_quality.mean_ce,4),'近期训练窗口加权均值；非验证集'",
        "fmt(p.throughput,2),'decisions / second'": "fmt(d.steady_decisions_per_second,2),'近期全局 decisions / second'",
        "lineChart('loss',d.points,'decisions'": "lineChart('loss',d.points,'updates'",
        "'本段计费决策预算（含probe）'": "'总计费预算（含旧失败尾部保守预留）'",
        "progressBar($('progress'),'训练cursor软停止界限',d.decisions,d.training_soft_limit,'#ffbb70');": '',
        "const epochDone=N(d.decisions)&&N(d.epoch_total_decisions)&&d.epoch_total_decisions>0?d.decisions%d.epoch_total_decisions:null;progressBar($('progress'),'当前 epoch 决策参考',epochDone,d.epoch_total_decisions,'#a4e391');": '',
        '指令顺序位置 ${fmt(c.route_position)} · 路线内步 ${fmt(c.step)}': '本轮batch位置 ${fmt(c.position)}',
        '本段剩余wall（预留600秒收尾）': '距原协议绝对墙钟上限',
        '完成当前 epoch 的参考耗时': '当前epoch精确ETA（未提供）',
        '· 决策 ${p.decisions}': '· 更新 ${p.updates} · 预算计数 ${p.decisions}',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, f'UI_BASE_CHANGED: {old}'
        source = source.replace(old, new)
    return source.encode()


b.collect = collect
BaseHandler = b.make_handler()


class Handler(BaseHandler):
    def do_GET(self):
        if self.path == '/':
            self.respond(200, html(), 'text/html; charset=utf-8')
        else:
            super().do_GET()
    do_HEAD = do_GET


if __name__ == '__main__':
    import sys
    html()
    if '--check' in sys.argv:
        d = collect(include_gpu=False)
        print(json.dumps({k: d[k] for k in ('monitor_version', 'run_name', 'display_state', 'decisions',
            'charged_compute_decisions', 'checkpoint_name', 'steady_decisions_per_second')}, ensure_ascii=False))
    else:
        with b.ThreadingHTTPServer(('127.0.0.1', 18766), Handler) as server:
            server.daemon_threads = True
            print('Read-only existing charts on 127.0.0.1:18766', flush=True)
            server.serve_forever(poll_interval=0.5)
