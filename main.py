from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from gower_pdist import compute_sim

# Convert continuous vector to 3 level categorical
def cat3(vector):
    """
    quantize a vector into 3 levels
    """
    return pd.qcut(vector, [0,.33,.66,1]).codes - 2

# Simulation parameters
N_SAMPLES = 5000
N_CLUSTERS = 3
N_FEATURES_CONT = 5
N_FEATURES_CAT = 5
N_FEATURES_NOISE = 2

N_FEATURES = N_FEATURES_CONT + N_FEATURES_CAT + N_FEATURES_NOISE

# Make dataset with 3 clusters
X, y = make_classification(n_samples=N_SAMPLES,
                           n_features=N_FEATURES,
                           n_informative=N_FEATURES_CONT + N_FEATURES_CAT,
                           n_classes=3,
                           n_clusters_per_class = 1,
                           shuffle=False,
                           random_state=1988)

# quantize features
X[:,:N_FEATURES_CAT] = np.apply_along_axis(
    cat3, axis=0, arr=X[:,:N_FEATURES_CAT])

dist = compute_sim(X, cat_idx=np.arange(0,N_FEATURES_CAT))




