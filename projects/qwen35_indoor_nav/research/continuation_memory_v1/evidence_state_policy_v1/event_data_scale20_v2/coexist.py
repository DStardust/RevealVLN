"""Share the authorized holder lease without touching the still-running first batch."""
import os
from pathlib import Path
from shared import *

def verify_prior_lease_pid(pid,cfg,monitor):
    identity=monitor.process_identity(int(pid))
    path=Path('/proc')/str(pid)
    args=path.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    group=path.joinpath('cgroup').read_text()
    if identity['uid']!=os.getuid() or cfg['prior_pipeline'] not in args or '/system.slice/'+cfg['prior_unit'] not in group:
        raise ValueError('UNRECOGNIZED_LEASE_OWNER')
    if sha(Path(cfg['prior_pipeline']))!=cfg['prior_pipeline_sha256']:raise ValueError('PRIOR_PIPELINE_CHANGED')
    return dict(identity=identity,argv=args,cgroup=group)

def acquire_preflight(cfg,resource):
    state=resource.lease('status')
    if state['manual_paused'] or state['external_pids']:raise RuntimeError('RESOURCE_NOT_AVAILABLE')
    if not state['leases']:return resource.verify_placeholders()
    owners=[verify_prior_lease_pid(pid,cfg,resource.monitor) for pid in state['leases']]
    return dict(state=state,verified_prior_owners=owners,sharing='Independent leases; previous batch is never signalled.')

def protected_devices(cfg,monitor):
    run=Path(cfg['prior_run'])
    protected=set()
    # Include reported workers to cover the launch/exit journal boundary.
    status=read(run/'STATUS.json')
    if status.get('status') in ('RUNNING','PREFLIGHT'):
        protected.update(int(w['gpu']) for w in status.get('workers',[]))
        if status.get('pending_houses'):return {d['gpu'] for d in cfg['devices']}
    for folder in (run/'attempts').iterdir():
        op=folder/'OWNER.json';cp=folder/'COMMAND.json'
        if not op.exists() or not cp.exists():continue
        owner=read(op)
        try:current=monitor.process_identity(owner['pid'])
        except (FileNotFoundError,ProcessLookupError):continue
        if current!=owner:continue
        if current['uid']!=os.getuid():raise ValueError('PRIOR_WORKER_UID')
        group=(Path('/proc')/str(current['pid'])/'cgroup').read_text()
        if '/system.slice/'+cfg['prior_unit'] not in group:raise ValueError('PRIOR_WORKER_CGROUP')
        protected.add(read(cp)['assignment']['gpu'])
    return protected

def available(devices,protected,active):
    return [d for d in devices if d['gpu'] not in protected|active]

