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
from sklearn.decomposition import PCA
from autoencoder import build_autoencoder
import seaborn as sns
import matplotlib.pyplot as plt


import logging
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    return pd.qcut(vector, [0, .33, .66, 1]).codes


def prep_data(n_samples=500, n_real=5, n_categorical=3, n_noisy=2,
              n_clusters=3):
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
    logger.info('N_SAMPLES: {}, N_REAL: {}, N_CATEGORICAL: {}, N_NOISY: {}'
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
                               n_clusters_per_class=1,
                               shuffle=False,
                               random_state=1988)

    X = np.concatenate([X, np.random.random((n_samples, n_noisy))], axis=1)

    logger.info('Generated dataset with shape {}'.format(X.shape))
    logger.info('Prediction score : {:.2}'.format(
        np.mean(cross_val_score(RFC(), X, y))))

    if n_categorical > 0:
        logger.info('Quantizing {} variables'.format(n_categorical))
        X[:, :n_categorical] = np.apply_along_axis(
            cat3, axis=0, arr=X[:, :n_categorical])
        logger.info('Prediction score after quantization : {:.2}'.format(
            np.mean(cross_val_score(RFC(), X, y))))

    cat_bool = np.arange(n_features + n_noisy) < n_categorical
    return (X, y, cat_bool)


def compute_pca(X):

    logger.info('Computing PCA')
    pca = PCA(n_components=2, whiten=True)
    X2 = pca.fit_transform(X - X.mean(axis=0))
    logger.info('Done: PCA divergence = {:.2}'.format())
    return X2, 1 - pca.explained_variance_ratio_.sum()


def compute_tsne(X, cat_bool, init=None):

    logger.info('Computing Gower pair-wise distance')
    dist = compute_sim(X, cat_bool)
    logger.info('Computing TSNE')
    if init == 'pca':
        ts = TSNE(perplexity=30, metric='precomputed', init='pca')
    else:
        ts = TSNE(perplexity=30, metric='precomputed')
    X2 = ts.fit_transform(dist)
    logger.info('Done: KL divergence = {:.2}'.format(ts.kl_divergence_))
    return X2, ts.kl_divergence_


def compute_ae(X):

    logger.info('Computing autoencoder')

    ae, enc = build_autoencoder(X.shape[1], layers_dim=[10, 2],
                                activations=['relu', 'sigmoid'],
                                inits=['glorot_uniform', 'glorot_normal'],
                                optimizer='adadelta',
                                loss='mse')
    Xs = (X - X.min(axis=0)) / (X.max() - X.min())
    h = ae.fit(Xs, Xs, batch_size=64, epochs=50, verbose=0)
    X2 = enc.predict(Xs)
    return X2, h.history['loss'][-1]


def compute_ae_tsne(X, cat_bool):

    ae, enc = build_autoencoder(X.shape[1], layers_dim=[10, 5],
                                activations=['relu', 'sigmoid'],
                                inits=['glorot_uniform', 'glorot_normal'],
                                optimizer='adadelta',
                                loss='mse')
    Xs = (X - X.min(axis=0)) / (X.max() - X.min())
    h = ae.fit(Xs, Xs, batch_size=64, epochs=50, verbose=0)
    X5 = enc.predict(Xs)
    ts = TSNE(perplexity=30, metric='euclidean')
    X2 = ts.fit_transform(X5)
    return X2, ts.kl_divergence_


def compute_params():

    # 1. All variables are predictive
    # 2. Most variables are predictive (n=8); some noise (n=2)
    # 3. Equal number of predictive variables (n=5), noisy variables (n=5)
    # 4. Most variables are noisy (n=8); small number of predictive variables
    # (n=2)

    # -Cluster label predicted by quantitative variables only
    # -Cluster label predicted by categorical variables only
    # -Cluster label predicted by both categorical and quantitative variables

    n_samples = 5000
    n_clusters = 3

    param = namedtuple('params',
                       ['n_samples', 'n_real',
                        'n_categorical', 'n_noisy',
                        'n_clusters'])
    params = []
    for n_noisy in [0, 2, 5, 8]:
        n_predictive = 10 - n_noisy
        for p in [0, .5, 1]:
            n_categorical = round(n_predictive * p)
            n_real = n_predictive - n_categorical
            params.append(param(n_samples=n_samples,
                                n_real=n_real,
                                n_categorical=n_categorical,
                                n_noisy=n_noisy,
                                n_clusters=n_clusters))

    return params


def run_sim(n_samples, n_real, n_categorical, n_noisy, n_clusters):

    X, y, cat_bool = prep_data(n_samples=n_samples,
                               n_real=n_real,
                               n_categorical=n_categorical,
                               n_noisy=n_noisy,
                               n_clusters=n_clusters)
    # np.savetxt('X_{}_{}_{}_{}.npy'.format(n_samples, n_real,
    #                                       n_categorical, n_noisy), X)
    # np.savetxt('Y_{}_{}_{}_{}.npy'.format(n_samples, n_real,
    #                                       n_categorical, n_noisy), y)
    # np.savetxt('cat_{}_{}_{}_{}.npy'.format(n_samples, n_real,
    # n_categorical, n_noisy), cat_bool)

    logger.info('Running PCA')
    X2, pca_loss = compute_pca(X)
    pca_score = np.mean(cross_val_score(RFC(), X2, y))
    logger.info('PCA prediction score: {:.2}'.format(pca_score))

    logger.info('Running TSNE simulation')
    X2, tsne_loss = compute_tsne(X, cat_bool)
    tsne_score = np.mean(cross_val_score(RFC(), X2, y))
    logger.info('TSNE prediction score: {:.2}'.format(tsne_score))

    logger.info('Running PCA+TSNE simulation')
    X2, pca_tsne_loss = compute_tsne(X, cat_bool, init='pca')
    pca_tsne_score = np.mean(cross_val_score(RFC(), X2, y))
    logger.info('PCA+TSNE prediction score: {:.2}'.format(pca_tsne_score))

    logger.info('Running AE simulation')
    X2, ae_loss = compute_ae(X)
    ae_score = np.mean(cross_val_score(RFC(), X2, y))
    logger.info('AE prediction score: {:.2}'.format(ae_score))

    logger.info('Running AE+tsne simulation')
    X2, ae_tsne_loss = compute_ae_tsne(X, cat_bool)
    ae_tsne_score = np.mean(cross_val_score(RFC(), X2, y))
    logger.info('AE+TSNE prediction score: {:.2}'.format(ae_tsne_score))

    return {'pca': pca_score, 'pca_loss': pca_loss,
            'pca+tsne': pca_tsne_score, 'pca+tsne_loss': pca_tsne_loss,
            'tsne': tsne_score, 'tsne_loss': tsne_loss,
            'ae': ae_score, 'ae_loss': ae_loss,
            'ae+tsne': ae_tsne_score, 'ae+tsne_loss': ae_tsne_loss}


def plot(df):
    plt.figure()
    dfm = pd.melt(df,
                  value_vars=['ae', 'tsne', 'pca', 'pca+tsne', 'ae+tsne'],
                  id_vars=['n_noisy', 'n_real'])
    sns.lmplot(data=dfm, x='n_noisy', y='value', hue='variable',
               hue_order=['tsne', 'ae+tsne', 'ae'])
    plt.ylabel('RF score')
    plt.savefig('score_noisy.png')

    # plt.figure()
    # dfm = pd.melt(df, value_vars=[
    #               'ae_loss', 'tsne_loss', 'ae+tsne_loss', 'pca_loss', ],
    #               id_vars=['n_noisy', 'n_real'])
    # sns.lmplot(data=dfm, x='n_noisy', y='value', hue='variable',
    #            hue_order=['tsne_loss', 'ae+tsne_loss', 'ae_loss'])
    # plt.ylabel('Loss')
    # plt.savefig('loss_noisy.png')


if __name__ == "__main__":

    params = compute_params()
    results = list(starmap(run_sim, params))

    df = pd.concat((pd.DataFrame(params),
                    pd.DataFrame(results)),
                   axis=1)

    df.to_csv('TSNE_simulation.csv', index=False)
    plot(df)
