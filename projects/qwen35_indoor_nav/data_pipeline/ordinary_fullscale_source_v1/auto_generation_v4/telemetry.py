"""Explicit active-worker-only conservative accounting amendment, no raw mutation."""
import math

def active_assessment(snapshot,own,original_guard,record):
    """Caller must first preserve the original sample. No idle/restore caller uses this.

    All values are validated. Consistent observations execute the original guard.
    Only its precise GPU_MEMORY_ACCOUNTING failure permits conservative max-total
    accounting, with a stricter entire-device/process-total <4096 MiB cap.
    The derived receipt is separate from original data; persistence failure stops.
    """
    assert type(own) is int and own>0,'ACTIVE_OWN_PID_REQUIRED'
    values=[snapshot['memory_mib'],*snapshot['processes'].values()]
    assert all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values),'INVALID_GPU_MEMORY_VALUE'
    assert all(type(pid) is int and pid>0 for pid in snapshot['processes']),'INVALID_GPU_PID'
    total=snapshot['memory_mib'];process_total=sum(snapshot['processes'].values())
    assert math.isfinite(process_total),'INVALID_GPU_MEMORY_SUM'
    external={p:m for p,m in snapshot['processes'].items() if p!=own}
    external_total=sum(external.values());conservative=max(total,process_total)
    inconsistent=process_total>total
    receipt=dict(amendment='ACTIVE_CONSERVATIVE_ACCOUNTING_V1',own_pid=own,
        device_total_mib=total,process_total_mib=process_total,external_total_mib=external_total,
        conservative_total_mib=conservative,conservative_upper_mib=conservative-external_total,
        accounting_inconsistent=inconsistent,original_guard_passed=False,amended_acceptance=False)
    record(dict(receipt,status='PENDING'))
    try:
        if not inconsistent or own not in snapshot['processes']:
            upper=original_guard(snapshot,own)
            receipt.update(original_guard_passed=True,guard_upper_mib=upper)
        else:
            # Enforce original external checks first; any other error remains fatal.
            try:original_guard(snapshot,own)
            except AssertionError as exc:
                if exc.args!=('GPU_MEMORY_ACCOUNTING',):raise
            else:raise AssertionError('UNEXPECTED_ACCOUNTING_GUARD_BEHAVIOR')
            assert all(0<=m<=768 for m in external.values()) and external_total<=2048,'EXTERNAL_GPU_RESOURCE'
            assert conservative<4096,'INCONSISTENT_TOTAL_GPU_CAP'
            upper=conservative-external_total
            assert 0<=upper<4096,'OWN_GPU_UPPER_BOUND'
            receipt.update(amended_acceptance=True,guard_upper_mib=upper,
                           original_guard_error='GPU_MEMORY_ACCOUNTING')
    except BaseException as exc:
        record(dict(receipt,status='REJECTED',error=repr(exc)))
        raise
    record(dict(receipt,status='ACCEPTED'))
    return receipt
