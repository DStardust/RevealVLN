"""Read-only strong-backbone progress on the user's existing dashboard."""
import importlib.util
import json
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer
_spec=importlib.util.spec_from_file_location('strong_monitor_common',Path(__file__).with_name('common.py'))
u=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(u)
spec=importlib.util.spec_from_file_location('strong_previous_monitor',u.PROJECT/'sft_acceptance/ordinary_branch_decision_v21/monitor_r1.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

CARD='''<section style="margin:20px;padding:20px;background:#173e36;color:#eef7ff;border:2px solid #68dfb0;border-radius:12px">
<h2>公开强导航模型 · StreamVLN</h2><p id="strong-status">读取状态…</p><pre id="strong-detail" style="white-space:pre-wrap"></pre>
<p>先核验零增量不改变原模型，再完成原生100条开发导航。原模型成绩与新增记忆方法收益分开显示。</p></section>
<script>async function strongUpdate(){try{let d=await(await fetch('/api/strong',{cache:'no-store'})).json(),s=d.status;
document.getElementById('strong-status').textContent=d.run+' · '+s.status+' · '+s.phase;
let lines=['已记录原生导航：'+(s.recorded_native||s.completed_native||0)+'/100；完成会话封存：'+(s.completed_native||0),
'累计GPU会话小时：'+Number(s.gpu_hours||0).toFixed(3), '新增方法训练：'+(s.method_training_started?'已启动':'尚未启动'),
'新增方法收益：'+(s.method_benefit||'未测量')];
for(let w of s.workers||[])lines.push('GPU'+w.gpu+' '+w.phase+' · '+(w.progress? w.progress.complete.length+'/'+w.progress.planned.length:'正在加载或启动'));
if(d.method){let m=d.method;lines.push('记忆对照：'+m.status.status+' · '+(m.status.phase||m.status.error||''));lines.push('记忆对照GPU会话小时：'+Number(m.status.gpu_hours||0).toFixed(3));for(let f of m.features)lines.push(f.family+' 特征 '+f.complete+'/'+f.planned);for(let t of m.training)lines.push(t.arm+' 训练 '+t.step+'/'+t.planned);if(m.review)lines.push('最终动作诊断已完成；闭环增量尚未测量');}if(d.data_status){let c=d.data_status;lines.push('移动历史重验：'+c.status+' '+(c.completed_traces||0)+'/'+(c.planned_traces||72));if(c.error)lines.push('数据错误：'+c.error);}if(s.error)lines.push('错误：'+s.error);if(d.result)lines.push('本机原生SR：'+(100*d.result.sr).toFixed(1)+'%（'+d.result.complete+'/'+d.result.planned+'；不计作我们的增量）');
document.getElementById('strong-detail').textContent=lines.join('\\n');}catch(e){document.getElementById('strong-status').textContent='状态读取失败：'+e;}}
strongUpdate();setInterval(strongUpdate,5000);</script>'''

class Handler(previous.Handler):
    def do_GET(self):
        if self.path not in ('/','/api/strong'):return super().do_GET()
        if self.path=='/api/strong':
            run=u.HERE/'runs'/(u.HERE/'LAST_RUN.txt').read_text().strip()
            method=None
            if (u.HERE/'METHOD_RUN.json').exists():
                folder=Path(u.read(u.HERE/'METHOD_RUN.json')['path'])
                method=dict(status=u.read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else {'status':'STARTING'},
                    features=[u.read(p) for p in sorted((folder/'features').glob('*/STATUS.json'))],
                    training=[u.read(p) for p in sorted((folder/'training').glob('*/PROGRESS.json'))],review=(folder/'ACTION_REVIEW.json').exists())
            body=json.dumps(dict(method=method,run=run.name,status=u.read(run/'STATUS.json'),data_status=u.read(u.HERE/'data_runs/moving_003/STATUS.json') if (u.HERE/'data_runs/moving_003/STATUS.json').exists() else None,result=u.read(run/'RESULT.json') if (run/'RESULT.json').exists() else None),ensure_ascii=False).encode()
            mime='application/json; charset=utf-8'
        else:
            page=(previous.base.OLD/'index.html').read_text()
            body=page.replace('<h1>',CARD+previous.base.CARD+'<h1>',1).encode();mime='text/html; charset=utf-8'
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
