"""Isolated loopback-only, GET-only benchmark monitor. No trainer mutations."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('summary',HERE/'summary.py');summary=importlib.util.module_from_spec(s);s.loader.exec_module(summary)

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>R2R-CE 完整验证</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#101b2c;color:#e4ecf8;margin:30px auto;max-width:1100px;padding:0 18px}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{padding:18px;background:#1b2d45;border-radius:10px;min-width:180px}.number{font-size:28px;color:#79d9ca}table{border-collapse:collapse;width:100%;background:#18283d}th,td{padding:8px 14px;border-bottom:1px solid #34435b;text-align:left}.note{color:#f8ce80}progress{width:100%;height:24px}pre{white-space:pre-wrap}a{color:#9bc9ff}</style>
<h1>普通导航基座 · R2R-CE 完整验证</h1><p id="state">正在读取</p><p class="note" id="warning"></p><progress id="bar" max="1839" value="0"></progress><div class="cards" id="cards"></div>
<p>本页只读、5秒刷新；训练仍用原来的18766页面。没有自动停止或最短路回退。</p><h2>房屋覆盖</h2><table><thead><tr><th>房屋</th><th>完成 / 计划</th><th>成功条数</th><th>当前 SR</th><th>当前 SPL</th></tr></thead><tbody id="houses"></tbody></table>
<h2>完整结果</h2><pre id="final">尚未完成全部验证</pre><a href="/api/status">JSON 状态接口</a>
<script>const percent=x=>x==null?'—':(100*x).toFixed(1)+'%';const f=x=>x==null?'—':x.toFixed(2);async function refresh(){try{let r=await fetch('/api/status',{cache:'no-store'});let d=await r.json();document.getElementById('state').textContent=`${d.status} · 检查点 ${d.checkpoint_updates} · ${d.completed}/${d.total} 条`;document.getElementById('warning').textContent=d.warning;document.getElementById('bar').value=d.completed;document.getElementById('bar').max=d.total;let values=[['已完成路线成功率',percent(d.partial_sr)],['已完成路线SPL',percent(d.partial_spl)],['平均终点距离',f(d.partial_ne)+' m'],['粗略剩余时间',d.eta_seconds==null?'待实测':(d.eta_seconds/3600).toFixed(1)+' 小时']];document.getElementById('cards').innerHTML=values.map(([a,b])=>`<div class="card">${a}<div class="number">${b}</div></div>`).join('');document.getElementById('houses').innerHTML=d.by_house.map(x=>`<tr><td>${x.house}</td><td>${x.completed} / ${x.total}</td><td>${x.successes}</td><td>${percent(x.sr)}</td><td>${percent(x.spl)}</td></tr>`).join('');document.getElementById('final').textContent=d.final.status?JSON.stringify(d.final,null,2):'尚未完成全部验证；请勿引用局部均值作为完整成绩。';}catch(e){document.getElementById('state').textContent='读取失败：'+e;}}refresh();setInterval(refresh,5000);</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/':payload=PAGE.encode();kind='text/html; charset=utf-8'
        elif self.path=='/api/status':payload=json.dumps(summary.collect(),ensure_ascii=False,allow_nan=False).encode();kind='application/json; charset=utf-8'
        elif self.path=='/healthz':payload=b'{"ok":true,"read_only":true}';kind='application/json'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
    def do_POST(self):self.send_error(405)
    do_PUT=do_POST;do_DELETE=do_POST;do_PATCH=do_POST
    def log_message(self,*args):pass


if __name__=='__main__':
    with ThreadingHTTPServer(('127.0.0.1',18768),Handler) as server:server.serve_forever(poll_interval=.5)
