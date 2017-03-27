import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from gower_pdist import compute_sim
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Convert continuous vector to 3 level categorical
def cat3(vector):
    """
    quantize a vector into 3 levels
    """
    return pd.qcut(vector, [0,.33,.66,1]).codes - 2

# Simulation parameters
N_SAMPLES = 500
N_FEATURES_CONT = 5
N_FEATURES_CAT = 1
N_FEATURES_NOISE = 2
N_FEATURES = N_FEATURES_CONT + N_FEATURES_CAT + N_FEATURES_NOISE
N_CLUSTERS = 2


# Make dataset
X, y = make_classification(n_samples=N_SAMPLES,
                           n_features=N_FEATURES,
                           n_informative=N_FEATURES_CONT + N_FEATURES_CAT,
                           n_redundant=N_FEATURES_NOISE,
                           n_classes=N_CLUSTERS,
                           class_sep=2,
                           n_clusters_per_class = 1,
                           shuffle=False,
                           random_state=1988)
logger.info('X.shape  = {}'.format(X.shape))

logger.info('Prediction score : {}'.format(
    np.mean(cross_val_score(RFC(), X, y))))
# quantize features
X[:,:N_FEATURES_CAT] = np.apply_along_axis(
    cat3, axis=0, arr=X[:,:N_FEATURES_CAT])
logger.info('Prediction score q : {}'.format(
    np.mean(cross_val_score(RFC(), X, y))))

            
dist = compute_sim(X, cat_idx=np.arange(0,N_FEATURES_CAT))
#from sklearn.metrics.pairwise import pairwise_distances
#dist = pairwise_distances(X, None, 'euclidean')
ts = TSNE(perplexity=10, metric='precomputed', early_exaggeration=10)
X2 = ts.fit_transform(dist)
logger.info('Prediction score tsne : {}'.format(
    np.mean(cross_val_score(RFC(), X2, y))))

logger.info('KL divergence: {}'.format(ts.kl_divergence_))
plt.scatter(X2[:,0], X2[:,1], c=y)
plt.show()
