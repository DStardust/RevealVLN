"""Aggregate all fixed seeds after their independent metric/log readbacks."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import review
c=review.c


def recoverable_sessions():
    allowed=dict(c.read(HERE/'INFRA_REPAIR_001.json')['recoverable_sessions'])
    pause=c.read(HERE/'EXECUTION_PAUSE_RESULT_001.json')
    assert pause['status']=='PLANNED_PAUSE_SEALED_RESUMABLE'
    assert not pause['score_based_selection'] and not pause['foreign_processes_signaled']
    assert pause['request_sha256']==c.sha(HERE/'EXECUTION_PAUSE_REQUEST_001.json')
    session=c.LINE/pause['session']
    seal_path=c.LINE/pause['state_seal_path']
    assert seal_path.parent==session and c.sha(seal_path)==pause['state_seal_sha256']
    seal=c.read(seal_path)
    assert seal['unchanged'] and len(seal['group_ranks'])==pause['complete_groups']
    assert c.sha(session/'LAUNCH_RESULT.json')==pause['launch_result_sha256']
    assert c.sha(session/'FAILURE.json')==pause['failure_sha256']
    assert c.read(session/'FAILURE.json')['error']=="InterruptedError('GRACEFUL_RESOURCE_STOP')"
    allowed[str(session.relative_to(HERE))]=dict(
        launch_result_sha256=pause['launch_result_sha256'],failure_sha256=pause['failure_sha256'],
        preserved_group_ranks=seal['group_ranks'])
    return allowed


def main():
    result=review.aggregate(verify=False)
    assert result['status']=='VALID_COMPLETE'
    resumable=recoverable_sessions()
    evidence={}
    for seed in result['seeds']:
        path=HERE/f'SEED_REVIEW_{seed}.json'
        verified=c.read(path)
        assert verified['metric_and_log_recomputed']
        current=dict(result['per_seed'][str(seed)])
        current['metric_and_log_recomputed']=True
        assert current==verified,'POST_AUDIT_SUMMARY_CHANGED'
        for row in review.committed(seed).values():
            folder=Path(row['path']).parent
            for arm in review.ARMS:
                assert c.read(folder/arm/f"episode_{row['index']:02d}.json")==row['episodes'][arm]
                for name,digest in row['logs'][arm].items():
                    assert c.sha(folder/arm/name)==digest,'POST_AUDIT_TRACE_CHANGED'
            session=folder.parents[1]
            launch=session/'LAUNCH_RESULT.json'
            if c.read(launch)['status']!='COMPLETE':
                allowed=resumable[str(session.relative_to(HERE))]
                assert c.sha(launch)==allowed['launch_result_sha256']
                assert c.sha(session/'FAILURE.json')==allowed['failure_sha256']
                assert row['rank'] in allowed['preserved_group_ranks']
        result['per_seed'][str(seed)]=verified
        evidence[str(seed)]=dict(path=str(path.relative_to(HERE)),sha256=c.sha(path))
    result.update(metric_and_log_recomputed=True,independent_per_seed_readbacks=evidence,
        verification_scope='Metrics recomputed once per seed before aggregation; final summaries, episode records and trace SHA rechecked. No simulator or inference replay.',
        finalizer_source_sha256=c.sha(Path(__file__)))
    result['infrastructure_repair_sha256']=c.sha(HERE/'INFRA_REPAIR_001.json')
    result['planned_pause_receipt_sha256']=c.sha(HERE/'EXECUTION_PAUSE_RESULT_001.json')
    c.write(HERE/'RESULT.json',result,True)
    print({k:v for k,v in result.items() if k not in ('per_seed','comparisons')})


if __name__=='__main__':main()
