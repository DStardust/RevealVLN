"""Teacher-forced tokens are labels, never inputs to the current feature."""
def replay_token(trace, step, offset, token_ids, eos):
    queries=trace['query_steps']
    assert step in queries, 'UNREGISTERED_QUERY_STEP'
    q=queries.index(step)
    end=queries[q+1] if q+1<len(queries) else len(trace['actions'])
    return token_ids[trace['actions'][step+offset]] if step+offset<end else eos


def forced_scores(scores, token):
    import torch
    result=torch.full_like(scores,torch.finfo(scores.dtype).min)
    result[:,token]=0
    return result
