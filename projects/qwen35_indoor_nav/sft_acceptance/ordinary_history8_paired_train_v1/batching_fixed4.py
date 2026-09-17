"""One fixed microbatch shape policy in both arms; no B8/B4 equivalence claim."""
def history8_est(sample):
    return sample['est']+(459 if sample['t']==0 else 396)
def partition(group,base_index):
    if len(group)!=32:raise ValueError('RANK_GROUP_SIZE')
    result=[list(range(base_index+i,base_index+i+4)) for i in range(0,32,4)]
    if any(4*max(history8_est(group[i+j]) for j in range(4))>6144 for i in range(0,32,4)):
        raise ValueError('TOKEN_BOUND_UNSUPPORTED')
    return result

