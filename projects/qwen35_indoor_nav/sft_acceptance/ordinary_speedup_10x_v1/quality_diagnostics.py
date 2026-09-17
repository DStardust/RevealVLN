"""Read-only recent-window diagnostics; no evaluation or training authority."""
import math

def valid(row):
    if not isinstance(row,dict):return False
    c,m=row.get('cursor'),row.get('metrics')
    if not isinstance(c,dict) or not isinstance(m,dict):return False
    n=c.get('decisions');matrix=m.get('confusion');ce=m.get('mean_ce')
    return (type(n) is int and n>0 and type(ce) in (int,float) and math.isfinite(ce)
        and isinstance(matrix,list) and len(matrix)==4 and all(isinstance(r,list) and len(r)==4
        and all(type(x) is int and x>=0 for x in r) for r in matrix) and sum(map(sum,matrix))==n)

def summarize(records,window=1000):
    rows=[r for r in records if valid(r)]
    if len(rows)<2:return {'available':False,'reason':'INSUFFICIENT_VALID_CUMULATIVE_RECORDS'}
    last=rows[-1];end=last['cursor']['decisions']
    # Reject restart/counter resets within a window, rather than subtracting epochs.
    start_index=0
    for i in range(1,len(rows)):
        if rows[i]['cursor']['decisions']<=rows[i-1]['cursor']['decisions']:start_index=i
    rows=rows[start_index:]
    if len(rows)<2:return {'available':False,'reason':'COUNTER_RESET'}
    first=min(rows[:-1],key=lambda r:abs(r['cursor']['decisions']-(end-window)))
    begin=first['cursor']['decisions'];n=end-begin
    matrix=[[last['metrics']['confusion'][i][j]-first['metrics']['confusion'][i][j] for j in range(4)] for i in range(4)]
    if n<=0 or any(v<0 for r in matrix for v in r):return {'available':False,'reason':'INVALID_COUNTER_DIFFERENCE'}
    support=[sum(r) for r in matrix];prediction=[sum(matrix[i][j] for i in range(4)) for j in range(4)]
    recall=[matrix[i][i]/support[i] if support[i] else None for i in range(4)]
    accuracy=sum(matrix[i][i] for i in range(4))/n
    baseline=support[0]/n
    ce=(last['metrics']['mean_ce']*end-first['metrics']['mean_ce']*begin)/n
    return dict(available=True,requested_window=window,decisions=n,start_decisions=begin,end_decisions=end,
        mean_ce=ce,accuracy=accuracy,always_forward_accuracy=baseline,accuracy_gain_over_always_forward=accuracy-baseline,
        target_counts=support,prediction_counts=prediction,recall=recall,confusion=matrix,
        macro_recall=sum(x for x in recall if x is not None)/sum(x is not None for x in recall),
        forward_prediction_fraction=prediction[0]/n,stop_predictions=prediction[3],stop_targets=support[3],
        scope='RECENT_ONLINE_FIT_TRAINING_NOT_FIXED_PANEL_NOT_DEV_NOT_NAVIGATION_SUCCESS')
