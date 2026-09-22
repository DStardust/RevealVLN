"""Read-only progress for compilation, real Qwen features, six training jobs and paired evaluation."""
import argparse,json,sys,time
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
reader_module=load('scale_monitor_reader',PARENT/'monotonic_holdout_v1/monitor.py')
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();reader=reader_module.Reader(a.run)
 def snapshot():
  v=reader.snapshot();cfg=read(a.run/'PROTOCOL.json');v['training']=[]
  for seed in cfg['seeds']:
   for arm in cfg['arms']:
    d=a.run/'train'/f'{arm}_{seed}';step=read(d/'PROGRESS.json')['step'] if (d/'PROGRESS.json').exists() else 0
    v['training'].append(dict(model=d.name,step=step,target=1200,complete=(d/'RESULT.json').exists()))
  v['prepare']=read(a.run/'PREPARE_PROGRESS.json') if (a.run/'PREPARE_PROGRESS.json').exists() else None
  v['feature_workers']=[read(x) for x in (a.run/'features').glob('PROGRESS_*.json')]
  parts=list((a.run/'features').glob('session_*/FEATURES_PART_*.json'))
  v['feature_windows_written']=sum(read(p)['end']-read(p)['start'] for p in parts)
  v['feature_total']=read(a.run/'PREPARED.json')['feature_windows'] if (a.run/'PREPARED.json').exists() else None
  v['model_sessions_loaded']=len(list((a.run/'features').glob('session_*/RUNTIME_IDENTITY.json')))
  v['new_training_updates']=sum(t['step'] for t in v['training']);v['planned_training_updates']=7200
  v['note']='旧数据 vs 扩充数据；同一MONOTONIC模型。部分分数不作最终结论。四屋为已暴露开发留出集。'
  return v
 class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
   try:
    if self.path=='/api/status':body=json.dumps(snapshot(),ensure_ascii=False).encode();mime='application/json'
    elif self.path=='/':body=(HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
    else:self.send_error(404);return
    self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
   except Exception as e:self.send_error(503);print(repr(e),flush=True)
  def log_message(self,*a):pass
 HTTPServer(('127.0.0.1',18770),Handler).serve_forever()
if __name__=='__main__':main()
