"""One shard producer; no unfinished-route overwrite or hidden retry."""
import argparse
import fcntl
from pathlib import Path
import signal
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c


def execute(shard,attempt):
    lock=c.approved(c.lane_for(shard));root=c.shard_root(shard)
    assert c.read(root/'INPUT_LOCK.json')==lock,'SHARD_INPUT_LOCK_MISMATCH'
    assert attempt.resolve().is_relative_to(HERE/'lanes') and attempt.is_dir()
    producer=(root/'PRODUCER.lock').open('a');fcntl.flock(producer,fcntl.LOCK_EX|fcntl.LOCK_NB)
    initial=c.state(shard);assert not initial['complete'],'CLOSED_SHARD_NO_RERUN'
    c.save(attempt/f'SHARD_{shard}_INITIAL_STATE.json',initial)
    module=c.worker_module(shard)
    c.save(attempt/f'SHARD_{shard}_TRANSPORT.json',dict(shard=shard,gpu=c.lane_for(shard),
        source=str(c.BASE/'worker.py'),source_sha256=c.sha(c.BASE/'worker.py'),
        exact_transport_only=True,strict_audit_source=str(c.BASE/'recovery_v1/audit.py'),
        output_root=str(root),training_started=False))
    def interrupt(signum,frame):raise InterruptedError('SUPERVISOR_STOP:'+str(signum))
    signal.signal(signal.SIGTERM,interrupt)
    module.main()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--attempt',type=Path,required=True)
    args=p.parse_args();execute(args.shard,args.attempt)
