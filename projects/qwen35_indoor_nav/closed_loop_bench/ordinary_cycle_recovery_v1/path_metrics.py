"""Exact Euclidean DTW, following official VLN-CE FDTW=False semantics."""
import math
import numpy as np


def ndtw(positions,reference):
    assert positions and reference
    unique=[]
    for position in positions:
        p=list(position)
        if not unique or p!=unique[-1]:unique.append(p)
    points=np.asarray(unique,dtype=np.float64);ref=np.asarray(reference,dtype=np.float64)
    assert points.ndim==ref.ndim==2 and points.shape[1]==ref.shape[1]==3
    assert np.isfinite(points).all() and np.isfinite(ref).all()
    previous=[0.]+[math.inf]*len(ref)
    for position in points:
        cost=np.linalg.norm(ref-position,axis=1);current=[math.inf]
        for j,distance in enumerate(cost,1):
            current.append(float(distance)+min(previous[j],current[-1],previous[j-1]))
        previous=current
    return math.exp(-previous[-1]/(len(ref)*3.))

