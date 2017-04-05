import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import starmap
from collections import namedtuple
from sklearn.datasets import make_classification
from gower_pdist import compute_sim
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

import logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TSNE_sim')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('sim.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# Convert continuous vector to 3 level categorical
def cat3(vector):
    """
    quantize a vector into 3 levels
    """
    return pd.qcut(vector, [0,.33,.66,1]).codes

def prep_data(n_samples=500, n_real=5, n_categorical=3, n_noisy=2, n_clusters=3):
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
    logger.info('N_SAMPLES: {}, N_REAL: {}, N_CATEGORICAL: {}, N_NOISY: {}'\
                .format(n_samples, n_real, n_categorical, n_noisy))

    n_features = n_real + n_categorical

    # Make dataset
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_real + n_categorical,
                               n_classes=n_clusters,
                               n_redundant=0,
                               n_repeated=0,
                               class_sep=2,
                               n_clusters_per_class = 1,
                               shuffle=False,
                               random_state=1988)

    X = np.concatenate([ X, np.random.random((n_samples, n_noisy))], axis=1)
    
    logger.info('Generated dataset with shape {}'.format(X.shape))
    logger.info('Prediction score : {:.2}'.format(
        np.mean(cross_val_score(RFC(), X, y))))

    if n_categorical>0:
        logger.info('Quantizing {} variables'.format(n_categorical))
        X[:,:n_categorical] = np.apply_along_axis(
            cat3, axis=0, arr=X[:,:n_categorical])
        logger.info('Prediction score after quantization : {:.2}'.format(
            np.mean(cross_val_score(RFC(), X, y))))

    cat_bool = np.arange(n_features+n_noisy)<n_categorical
    return (X, y, cat_bool)


def compute_tsne(X, cat_bool):

    logger.info('Computing Gower pair-wise distance')
    dist = compute_sim(X, cat_bool)
    logger.info('Computing TSNE')
    ts = TSNE(perplexity=30, metric='precomputed')
    X2 = ts.fit_transform(dist)
    logger.info('Done: KL divergence = {:.2}'.format(ts.kl_divergence_))
    return X2, ts.kl_divergence_

def compute_ae(X):

    logger.info('Computing autoencoder')

def compute_params():

    #1. All variables are predictive
    #2. Most variables are predictive (n=8); some noise (n=2)
    #3. Equal number of predictive variables (n=5), noisy variables (n=5)
    #4. Most variables are noisy (n=8); small number of predictive variables (n=2)

    #-Cluster label predicted by quantitative variables only
    #-Cluster label predicted by categorical variables only
    #-Cluster label predicted by both categorical and quantitative variables

    n_samples = 5000
    n_clusters = 3

    param = namedtuple('params',
                       ['n_samples','n_real',
                        'n_categorical', 'n_noisy',
                        'n_clusters'])
    params = []
    for n_noisy in [0,2,5,8]:
        n_predictive = 10 - n_noisy
        for p in [0, .5, 1]:            
            n_categorical = round(n_predictive * p)
            n_real = n_predictive-n_categorical
            params.append( param(n_samples=n_samples,
                                 n_real=n_real,
                                 n_categorical=n_categorical,
                                 n_noisy=n_noisy,
                                 n_clusters=n_clusters))   
                        
    return params

def run_sim(n_samples, n_real, n_categorical, n_noisy, n_clusters):
    
    X, y, cat_bool = prep_data(n_samples= n_samples,
                               n_real= n_real,
                               n_categorical= n_categorical,
                               n_noisy= n_noisy,
                               n_clusters= n_clusters)
    # np.savetxt('X_{}_{}_{}_{}.npy'.format(n_samples, n_real,
    #                                       n_categorical, n_noisy), X)
    # np.savetxt('Y_{}_{}_{}_{}.npy'.format(n_samples, n_real,
    #                                       n_categorical, n_noisy), y)
    # np.savetxt('cat_{}_{}_{}_{}.npy'.format(n_samples, n_real,
    #                                         n_categorical, n_noisy), cat_bool)
    
    X2 , kl = compute_tsne(X, cat_bool)
    scores = cross_val_score(RFC(), X2, y)
    logger.info('Prediction score tsne : {:.2}'.format(np.mean(scores)))
    return (np.mean(scores), kl)


if __name__== "__main__":

    params = compute_params()
    results = list(starmap(run_sim, params))
    
    df = pd.concat( (pd.DataFrame(params),
                     pd.DataFrame(results,
                                  columns=['TSNE_score', 'TSNE_kl'])),
                    axis=1)

    df.to_csv('TSNE_simulation.csv')
