"""Original 18766 interface: low-LR branch, with the failed high-LR result retained."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'monitor_charts_continue_v2'
assert hashlib.sha256((OLD/'server.py').read_bytes()).hexdigest()=='a6899d8c31f8dd7651057d53933d6252e4fe88cc5dfc38572ca149b5c0baafec'
s=importlib.util.spec_from_file_location('low_lr_monitor_parent',OLD/'server.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.TRAIN=HERE.parent/'ordinary_expanded_low_lr_v3';m.REVIEW=LINE/'reviews/Q35N_ORDINARY_LOW_LR_V3'
m.r.RECOVERY=m.TRAIN;m.r.FORMAL=m.TRAIN/'formal'


def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu)
    d['monitor_version']='ordinary_low_lr_v3'
    d['navigation']['high_lr']=d['navigation']['continued']
    d['navigation']['continued']=m.prior.eval_state('ordinary_expanded_low_lr_dev_v3')
    d['branch_note']='From 4000, same next data as failed V2, peak LR 5e-6; failed high-LR compute counted separately.'
    return d


def html():
    text=m.html().decode()
    start=text.index('<section class="panel full" style="margin-top:20px">');end=text.index('</main>',start)
    text=text[:start]+(HERE/'panel.html').read_text()+text[end:]
    text=text.replace('当前继续至累计8,000','低LR分支重新从4,000继续至8,000（高LR失败另计）')
    return text.encode()


m.r.b.collect=collect
class Handler(m.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        else:super().do_GET()
    do_HEAD=do_GET


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);p.add_argument('--check',action='store_true');a=p.parse_args()
    html()
    if a.check:
        d=collect(False);print(json.dumps({k:d[k] for k in ('monitor_version','display_state','training_completion')},ensure_ascii=False))
    else:
        with m.r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
            server.daemon_threads=True;server.serve_forever(poll_interval=.5)
