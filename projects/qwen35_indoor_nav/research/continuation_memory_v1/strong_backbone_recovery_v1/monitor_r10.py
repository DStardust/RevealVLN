"""Expose safe scheduler draining without changing the existing result display."""
import importlib.util
import json
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer

spec=importlib.util.spec_from_file_location('async_previous_monitor',Path(__file__).with_name('monitor_r9.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);u=old.u


class Handler(old.Handler):
    def do_GET(self):
        if self.path!='/api/intervention_v2':return super().do_GET()
        root=Path(u.read(u.HERE/'INTERVENTION_V2_RUN.json')['path'])
        def read(path):return u.read(path) if path.exists() else None
        status=u.read(root/'STATUS.json');scheduler=read(root/'ACCELERATION_STATUS.json')
        if scheduler and scheduler['status']=='DRAINING':
            status=dict(status,phase='当前批次自然收尾，随后切换动态派单',
                recorded_groups=scheduler['recorded_groups'],sealed_groups=scheduler['sealed_groups'],workers=scheduler['workers'],
                gpu_hours=scheduler['gpu_hours_upper_bound'],gpu_hours_scope='Conservative controller-drain upper bound')
        elif status.get('scheduler')=='ASYNC_V1' and status['phase']=='COLLECT_TRAIN':
            status=dict(status,phase='采集 TRAIN（动态派单已启用）')
        payload=dict(status=status,scheduler=scheduler,training=[u.read(p) for p in root.glob('gates/*/PROGRESS.json')],
            counts=read(root/'gate_data/COUNTS.json'),gates=read(root/'GATE_REVIEW.json'),
            live=read(root/'unseen/LIVE_RESULT.json') or read(root/'train/LIVE_RESULT.json'),result=read(root/'RESULT.json'))
        body=json.dumps(payload,ensure_ascii=False).encode()
        self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
