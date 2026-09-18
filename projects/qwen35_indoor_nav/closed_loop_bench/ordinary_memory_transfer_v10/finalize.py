"""Aggregate all fixed seeds after their independent metric/log readbacks."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import review
c=review.c


def main():
    result=review.aggregate(verify=False)
    assert result['status']=='VALID_COMPLETE'
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
            assert c.read(folder.parents[1]/'LAUNCH_RESULT.json')['status']=='COMPLETE'
        result['per_seed'][str(seed)]=verified
        evidence[str(seed)]=dict(path=str(path.relative_to(HERE)),sha256=c.sha(path))
    result.update(metric_and_log_recomputed=True,independent_per_seed_readbacks=evidence,
        verification_scope='Metrics recomputed once per seed before aggregation; final summaries, episode records and trace SHA rechecked. No simulator or inference replay.',
        finalizer_source_sha256=c.sha(Path(__file__)))
    c.write(HERE/'RESULT.json',result,True)
    print({k:v for k,v in result.items() if k not in ('per_seed','comparisons')})


if __name__=='__main__':main()
