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
    return pd.qcut(vector, [0,.33,.66,1]).codes

def prep_data(n_samples=500, n_real=5, n_categorical=3, n_noisy=2, n_cluters=3):
    """
    Prepares dataset for TSNE experiment

    Input parameters: 
    - n_samples= number of total samples
    - n_real = number of real variables
    - n_categorical = number of categorical variables
    - n_noisy = number of noisy variables
    - n_clusters = number of clusters to create

    Output: 
    - X = dataset
    - y = cluster labels
    - cat_bool = boolean index of categorical variables in X 
    """

    n_features = n_features_cont + n_features_cat + n_features_noise

    # Make dataset
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_features_cont + n_features_cat,
                               n_redundant=n_features_noise,
                               n_classes=n_clusters,
                               class_sep=10,
                               n_clusters_per_class = 1,
                               shuffle=False,
                               random_state=1988)
    logger.info('Generated dataset with shape {}'.format(X.shape))
    logger.info('Prediction score : {}'.format(
        np.mean(cross_val_score(RFC(), X, y))))

    logger.info('Quantizing {} variables'.format(n_features_cat))
    X[:,:n_features_cat] = np.apply_along_axis(
        cat3, axis=0, arr=X[:,:n_features_cat])
    logger.info('Prediction score after quantization : {}'.format(
        np.mean(cross_val_score(RFC(), X, y))))

    cat_bool = np.arange(n_features)<n_features_cat
    return (X, y, cat_bool)


def compute_tsne(X, cat_bool):

    dist = compute_sim(X, cat_bool)

    ts = TSNE(perplexity=30, metric='precomputed')
    X2 = ts.fit_transform(dist)

    logger.info('Prediction score tsne : {}'.format(
        np.mean(cross_val_score(RFC(), X2, y))))

    logger.info('KL divergence: {}'.format(ts.kl_divergence_))
    plt.scatter(X2[:,0], X2[:,1], c=y, alpha=0.8, marker='.')
    plt.savefig('tsne_result.png')
