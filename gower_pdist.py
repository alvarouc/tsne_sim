from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
import numpy as np

def gower(xi,xj, cat_idx, ranges):
    cat = np.sum([xi[cat_idx]!=xj[cat_idx]])
    con = np.sum(np.abs(xi[~cat_idx]-xj[~cat_idx])/ranges[~cat_idx])
    return (cat+con)/len(ranges)

def compute_sim(X, cat_idx):
    X[:,~cat_idx] = X[:,~cat_idx]/X[:,~cat_idx].max(axis=0)
    ranges = np.ptp(X, axis=0)
    d = pairwise_distances(X,
                           partial(gower,
                                   cat_idx=cat_idx,
                                   ranges=ranges),
                           n_jobs=-1)
    return d

