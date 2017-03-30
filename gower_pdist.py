from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
import numpy as np

def gower(xi,xj, cat_bool):

    cat = np.sum([xi[cat_bool]!=xj[cat_bool]])
    
    con = np.sum(np.abs(xi[~cat_bool]-xj[~cat_bool]))
    return (cat+con)/len(xi)

def compute_sim(X, cat_bool):
    Xsd = (X - X.min(axis=0))/X.ptp(axis=0)
    d = pairwise_distances(Xsd, None,
                           partial(gower,
                                   cat_bool=cat_bool),
                           n_jobs=-1)
    return d

