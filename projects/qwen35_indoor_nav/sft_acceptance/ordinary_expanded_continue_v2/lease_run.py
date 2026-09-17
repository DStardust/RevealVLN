"""Same exact-identity restore protocol, upgraded preflight telemetry includes graphics."""
import importlib.util as _iu
from pathlib import Path as _P
_s = _iu.spec_from_file_location('expanded_reuse_lease', _P(__file__).with_name('reuse.py'))
_r = _iu.module_from_spec(_s); _s.loader.exec_module(_r)
_code = _r.source('lease_run.py')
_old = "if __name__ == '__main__':\n    main()"
assert _code.count(_old) == 1
exec(compile(_code.replace(_old, ''), __file__+':parent', 'exec'), globals())


def gpu_snapshot(uuid):
    import xml.etree.ElementTree as ET
    raw = subprocess.check_output(['nvidia-smi','-q','-x','-i',uuid], text=True, timeout=15)
    gpu = ET.fromstring(raw).find('gpu'); require(gpu.findtext('uuid') == uuid, 'GPU_UUID')
    processes = {}
    for row in gpu.findall('processes/process_info'):
        pid=int(row.findtext('pid')); value=float(row.findtext('used_memory').split()[0])
        processes[pid]=processes.get(pid,0)+value
    return dict(uuid=uuid, processes=processes, memory_mib=float(gpu.findtext('fb_memory_usage/used').split()[0]),
                graphics_contexts_included=True)


if __name__ == '__main__':
    # Freeze the runbook itself and every local/dependency source before holder signals.
    admission=json.loads((HERE/'MAIN_AGENT_APPROVAL.json').read_text())
    require(sha256(HERE/'RUNBOOK.json') == admission['runbook_sha256'], 'RUNBOOK_CHANGED')
    require(sha256(HERE/'PROTOCOL_FILESTORE.json') == admission['protocol_sha256'], 'PROTOCOL_CHANGED')
    main()
