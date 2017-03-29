from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
import numpy as np

def gower(xi,xj, cat_idx, ranges):
    cat = np.sum([xi[cat_idx]!=xj[cat_idx]])
    con = np.sum(np.abs(xi[~cat_idx]-xj[~cat_idx])/ranges[~cat_idx])
    return (cat+con)/len(ranges)

def compute_sim(X, cat_idx):
    Xsd = (X - X.min(axis=0))/X.ptp(axis=0)
    d = pairwise_distances(Xsd, None,
                           partial(gower,
                                   cat_idx=cat_idx),
                           n_jobs=-1)
    return d

