"""Read-only recapture card; preserve the existing evaluation monitor routes."""
import argparse
import importlib.util
import json
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_history_monitor', HERE.parents[1] / 'history_validation_v4/monitor.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
RUN = HERE / 'runs/capture_001'

CARD = r'''<section style="margin-bottom:20px;background:#21394d;border-color:#90cdf4">
<h1>动作特征补采进度</h1><pre id="recapture-live">加载中…</pre>
<p>先等待现有评测结束，再使用 GPU 回放已登记轨迹。等待不计 GPU 工作；补采完成不等于训练完成或 SR 提升。</p>
<a href="/previous">此前训练与评测仪表盘</a> · <a href="/api/observation_recollection">补采原始进度</a></section>
<script>
async function recaptureUpdate(){
try{
 let d=await(await fetch('/api/observation_recollection',{cache:'no-store'})).json();
 let s=d.status||{},p=d.protocol||{},r=d.result||{};
 let count=(key,fallback)=>s[key]??p[key]??fallback;
 let lines=[(s.status||'NOT_STARTED')+' · '+(s.phase||'PREPARING'),
  '完整轨迹：记录 '+(s.recorded_trajectories??0)+' / '+count('planned_trajectories',334)+'；已封存 '+(s.sealed_trajectories??0),
  '原缺位特征：'+(s.captured_missing??0)+' / '+count('planned_missing',21396),
  '原缺位 STOP：'+(s.captured_stop??0)+' / '+count('planned_stop',133),
  '本轮 GPU 会话小时：'+Number(s.gpu_hours??0).toFixed(3)];
 if(s.phase==='WAIT_DEPENDENCY')lines.push('等待旧 V4 评测服务结束；尚未启动 GPU 补采。');
 if(!d.status)lines.push('运行尚未登记，以上分母为已计划数量。');
 for(let w of s.workers||[]){
  let q=w.progress||{};
  lines.push('GPU '+(w.gpu??'—')+' · 轨迹 '+(q.trajectory_id??q.id??q.current_id??'—')+
   ' · query '+(q.query??q.captured_queries??'—')+' · 动作步 '+(q.currentstep??q.step??q.environment_step??'—')+
   ' · 动作 '+(q.action??q.last_action??q.executed_action??'—')+(w.progress?'':' · 加载/初始化中'));
 }
 if(r.status)lines.push('补采结果：'+r.status);
 if(s.error)lines.push('错误：'+s.error);
 if(d.read_errors.length)lines.push('读取错误：'+d.read_errors.join('；'));
 document.getElementById('recapture-live').textContent=lines.join('\n');
}catch(e){document.getElementById('recapture-live').textContent='读取失败 '+e;}
}
recaptureUpdate();setInterval(recaptureUpdate,5000);
</script>'''


def payload():
    errors = []
    def read(name):
        try:
            return json.loads((RUN / name).read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            errors.append(name + ': ' + str(error))
            return None
    result = dict(status=read('STATUS.json'), protocol=read('PROTOCOL.json'), result=read('RESULT.json'))
    result['read_errors'] = errors
    return result


class Handler(old.Handler):
    def do_GET(self):
        if self.path == '/api/observation_recollection':
            body = json.dumps(payload(), ensure_ascii=False).encode()
            mime = 'application/json; charset=utf-8'
        elif self.path == '/':
            page = old.PAGE.replace('<section>', CARD + '<section>', 1)
            page = page.replace('Q35N 历史机制与新增路线复验', 'Q35N 数据补采与评测进度', 1)
            page = page.replace('<h1>当前：冻结六模型，新增路线复验', '<h1>既有评测：冻结六模型，新增路线复验', 1)
            body = page.encode()
            mime = 'text/html; charset=utf-8'
        else:
            return super().do_GET()
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('port', type=int, nargs='?', default=18770)
    parser.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run.resolve()
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
