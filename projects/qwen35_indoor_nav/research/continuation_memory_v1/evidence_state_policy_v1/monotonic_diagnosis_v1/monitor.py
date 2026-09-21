"""Keep the existing completed-run monitor and add the CPU diagnosis report."""
from html import escape
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import HERE,HOLDOUT,load,read

def main():
    old=load('diagnosis_original_monitor',HOLDOUT/'monitor.py')
    reader=old.Reader(HOLDOUT/'runs/holdout_001');out=HERE/'runs/diagnosis_001'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':
                    panel='<section><h2>最新：只读退化诊断已完成</h2><p>1488条轨迹 + 98,792次自主决策复核；六模型2832次冻结FIT接管前向。新增训练/GPU=0。</p><p>MONOTONIC 1210：FIT主任务状态236/236正确，仍漏停7/64；控制任务另漏停15次。优先定位状态到STOP的动作读出，尚无新增方法收益。</p><p><a style="color:#8bd5ff" href="/diagnosis">查看完整诊断、数据覆盖缺口和下一修复条件</a></p></section>'
                    page=(HOLDOUT/'index.html').read_text();pos=page.index('<section>');body=(page[:pos]+panel+page[pos:]).encode();mime='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(reader.snapshot(),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
                elif self.path=='/api/diagnosis':body=json.dumps(read(out/'RESULT.json'),ensure_ascii=False).encode();mime='application/json; charset=utf-8'
                elif self.path=='/diagnosis':
                    body=('<!doctype html><meta charset="utf-8"><title>MONOTONIC退化诊断</title><style>body{max-width:1050px;margin:35px auto;padding:20px;background:#101725;color:#dde7f5;font:16px system-ui}pre{white-space:pre-wrap;line-height:1.7}a{color:#8bd5ff}</style><a href="/">返回测试监控</a><pre>'+escape((out/'REPORT_ZH.md').read_text())+'</pre>').encode();mime='text/html; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
            except (OSError,ValueError,KeyError) as exc:self.send_error(503);print(repr(exc),flush=True)
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',18770),Handler).serve_forever()

if __name__=='__main__':main()
