"""Bind base chart data cards to the new snapshot too, not just stage progress."""
import functools
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('expanded_monitor_base',HERE/'server.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
snapshot=m.LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/RESULT.json'
m.r.v3.original_collect=functools.partial(m.r.v3.original_collect,snapshot=snapshot)


if __name__=='__main__':
    import argparse
    import json
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);p.add_argument('--check',action='store_true');a=p.parse_args()
    m.html()
    if a.check:
        d=m.collect(False);assert d['snapshot_counts']['instruction_conditioned_decisions']==2650347
        print(json.dumps(dict(version=d['monitor_version'],snapshot=d['snapshot_counts'],state=d['display_state'])))
    else:
        with m.r.b.ThreadingHTTPServer(('127.0.0.1',a.port),m.Handler) as server:
            server.daemon_threads=True;server.serve_forever(poll_interval=.5)
