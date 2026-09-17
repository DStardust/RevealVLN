"""CPU-testable bounded sampling amendment, NOT connected to any supervisor.

Caller supplies the original XML parser/check_gpu, the exact worker PID, and
the ORIGINAL supervision wall deadline. Query must honor its timeout argument.
Never wrap a lease/holder query with this worker-only function.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time


class DurableLog:
    """New append-only receipt. Constructor creates no GPU processes/queries."""
    def __init__(self,path,output_root):
        path=Path(path);root=Path(output_root).resolve(strict=True)
        project=Path(__file__).resolve().parents[7]
        assert root.is_relative_to(project) and path.parent.resolve(strict=True)==root
        self.stream=path.open('x')
    def __call__(self,event):
        self.stream.write(json.dumps(event,allow_nan=False)+'\n')
        self.stream.flush();os.fsync(self.stream.fileno())
    def close(self):self.stream.close()


def retry_eligible(snapshot,worker):
    """All old quantitative guards plus a STRICTER gross-memory prerequisite."""
    total=snapshot['memory_mib'];processes=snapshot['processes']
    assert type(total) is int and total>=0,'INVALID_MEMORY_VALUE'
    assert type(worker) is int and worker>0,'EXACT_WORKER_PID_REQUIRED'
    vals=[v['mib'] for v in processes.values()]
    assert all(type(m) is int and m>=0 for m in vals),'INVALID_PROCESS_MEMORY'
    external=[v['mib'] for p,v in processes.items() if p!=worker]
    assert all(m<=768 for m in external) and sum(external)<=2048,'EXTERNAL_RESOURCE_LOAD'
    assert 0<=total-sum(external)<4096,'OWN_GPU_MEMORY_UPPER_BOUND'
    assert max(total,sum(vals))<4096,'INCONSISTENT_SAMPLE_GROSS_MEMORY_NOT_SAFE'
    return dict(total_mib=total,process_sum_mib=sum(vals),gross_upper_mib=max(total,sum(vals)))


def sample(query,parse,check,record,worker,*,wall_deadline,clock=time.monotonic):
    """At most one first query + two retries; retry window <=1s including I/O.

Only exact original AssertionError('MEMORY_ACCOUNTING') can enter retry.
Every query's original XML is durably logged BEFORE parse and guard execution;
the parsed object and guard failure/success are separate chronological receipts.
The callback must be durable (DurableLog implements that). Any callback error
propagates without further GPU queries. Failed raw values are never rewritten.
"""
    assert math.isfinite(wall_deadline) and type(worker) is int and worker>0
    first_error=None;retry_deadline=None
    for attempt in range(3):
        before=clock();deadline=min(wall_deadline,retry_deadline) if retry_deadline is not None else wall_deadline
        remaining=deadline-before
        if remaining<=0:
            record(dict(event='sampling_stopped',attempt=attempt,monotonic=before,reason='DEADLINE',first_error=repr(first_error)))
            if first_error is not None:raise first_error
            raise TimeoutError('ORIGINAL_SUPERVISION_WALL_EXHAUSTED')
        timeout=min(15.0,remaining)
        try:raw=query(timeout)
        except BaseException as error:
            record(dict(event='query_error',attempt=attempt,query_started=before,monotonic=clock(),timeout=timeout,error=repr(error),first_error=repr(first_error)))
            raise
        received=clock()
        assert type(raw) is str,'QUERY_MUST_RETURN_UNMODIFIED_XML_TEXT'
        record(dict(event='raw_xml',attempt=attempt,query_started=before,monotonic=received,timeout=timeout,
                    raw_xml=raw,utf8_sha256=hashlib.sha256(raw.encode()).hexdigest()))
        try:parsed=parse(raw)
        except BaseException as error:
            record(dict(event='parse_error',attempt=attempt,monotonic=clock(),error=repr(error),first_error=repr(first_error)))
            raise
        record(dict(event='parsed_sample',attempt=attempt,monotonic=clock(),snapshot=parsed))
        if clock()>=deadline:
            record(dict(event='sampling_stopped',attempt=attempt,monotonic=clock(),reason='LATE_QUERY_OR_PERSISTENCE',first_error=repr(first_error)))
            if first_error is not None:raise first_error
            raise TimeoutError('ORIGINAL_SUPERVISION_WALL_EXHAUSTED')
        try:upper=check(parsed,worker)
        except BaseException as error:
            observed_error=clock()
            if type(error) is AssertionError and error.args==('MEMORY_ACCOUNTING',) and first_error is None:
                first_error=error;retry_deadline=observed_error+1.0
            record(dict(event='guard_error',attempt=attempt,monotonic=observed_error,error=repr(error),
                        first_error=repr(first_error),retry_deadline=retry_deadline))
            if type(error) is not AssertionError or error.args!=('MEMORY_ACCOUNTING',):raise
            try:gross=retry_eligible(parsed,worker)
            except BaseException as unsafe:
                record(dict(event='sampling_stopped',attempt=attempt,monotonic=clock(),reason='UNSAFE_INCONSISTENT_SAMPLE',error=repr(unsafe),first_error=repr(first_error)))
                raise
            record(dict(event='bounded_resample_eligible',attempt=attempt,monotonic=clock(),**gross))
            if attempt==2 or clock()>=min(wall_deadline,retry_deadline):raise first_error
            continue
        record(dict(event='guard_pass',attempt=attempt,monotonic=clock(),own_memory_upper_mib=upper,
                    retries=attempt,first_error=repr(first_error)))
        if clock()>=deadline:
            record(dict(event='sampling_stopped',attempt=attempt,monotonic=clock(),reason='LATE_PASS_PERSISTENCE_NOT_ACCEPTED',first_error=repr(first_error)))
            if first_error is not None:raise first_error
            raise TimeoutError('ORIGINAL_SUPERVISION_WALL_EXHAUSTED')
        return parsed,upper
    raise AssertionError('UNREACHABLE')
