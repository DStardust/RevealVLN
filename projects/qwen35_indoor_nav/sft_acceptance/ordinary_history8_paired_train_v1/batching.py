"""Same history8-based microbatch partitions in both arms; exact global sample order."""
def history8_est(sample):
    return sample['est'] + (459 if sample['t']==0 else 396)
def partition(group,base_index):
    if len(group)!=32:raise ValueError('RANK_GROUP_SIZE')
    batches=[];offset=0
    while offset<32:
        size=min(8,32-offset)
        if size*max(history8_est(x) for x in group[offset:offset+size])>6144:size=min(4,32-offset)
        if size*max(history8_est(x) for x in group[offset:offset+size])>6144:raise ValueError('TOKEN_BOUND_UNSUPPORTED')
        batches.append(list(range(base_index+offset,base_index+offset+size)))
        offset+=size
    return batches

