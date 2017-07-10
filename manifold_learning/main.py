from time import time
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import silhouette_score
import pandas as pd
from autoencoder import run_ae
from tsne import run_tsne
from logger import make_logger

log = make_logger('manifold')


def to_csv(Y, label, name):

    df = pd.DataFrame({'Y1': Y[:, 0],
                       'Y2': Y[:, 1],
                       'label': label})
    df.to_csv(name, header=False, index=False)


def plotx(Y, label, name):
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s=3)
    sil = silhouette_score(Y, color.ravel())
    plt.title("%s (score: %.2f)" % (name, sil))
    return sil


names = ['5000_20_20_10', '5000_25_25_0', '5000_5_5_40']
X_paths = ['data/{0:}/X_{0:}.csv'.format(name) for name in names]
Y_paths = ['data/{0:}/Y_{0:}.csv'.format(name) for name in names]

for X_path, Y_path, name in zip(X_paths, Y_paths, names):
    X = pd.read_csv(X_path, header=None, sep=' ').values
    color = pd.read_csv(Y_path, header=None, sep=' ').values.ravel()

    n_points = X.shape[0]

    n_neighbors = 10
    n_components = 2

    fig = plt.figure(figsize=(15, 8))
    plt.suptitle("Manifold Learning (%s) %i points, %i neighbors"
                 % (name, n_points, n_neighbors), fontsize=14)

    methods = [
        'standard',
        'ltsa',
        'hessian',
        'modified']
    labels = [
        'LLE',
        'LTSA',
        'Hessian LLE',
        'Modified LLE']

    for i, method in enumerate(methods):
        t0 = time()
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='dense',
                                            method=method).fit_transform(X)
        t1 = time()
        log.debug("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(251 + i)
        sil = plotx(Y, color, labels[i])
        to_csv(Y, color, 'projections/{}_{}.csv'.format(labels[i], name))

    t0 = time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    t1 = time()
    log.debug("Isomap: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(255)
    sil = plotx(Y, color, 'Isomap')
    log.info("Isomap %s, score: %.2f", name, sil)
    to_csv(Y, color, 'projections/Isomap_{}.csv'.format(name))

    t0 = time()
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    Y = mds.fit_transform(X)
    t1 = time()
    log.debug("MDS: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(256)
    sil = plotx(Y, color, 'MDS')
    log.info("MDS %s, score: %.2f", name, sil)
    to_csv(Y, color, 'projections/MDS_{}.csv'.format(name))

    t0 = time()
    se = manifold.SpectralEmbedding(n_components=n_components,
                                    n_neighbors=n_neighbors)
    Y = se.fit_transform(X)
    t1 = time()
    log.debug("SpectralEmbedding: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(257)
    sil = plotx(Y, color, 'Spectral Embedding')
    log.info("Spectral Embedding %s, score: %.2f", name, sil)
    to_csv(Y, color, 'projections/Spectral Embedding_{}.csv'.format(name))

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components,
                         init='pca', random_state=0)
    Y = tsne.fit_transform(X)
    t1 = time()
    log.debug("t-SNE: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(258)
    sil = plotx(Y, color, 't-SNE + PCA')
    log.info("t-SNE + PCA %s, score: %.2f", name, sil)
    to_csv(Y, color, 'projections/tSNE_{}.csv'.format(name))

    t0 = time()
    Y, _, _ = run_ae(X, epochs=500, layers_dim=[100, 100, 100, n_components])
    t1 = time()
    log.debug("Autoencoder: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(2, 5, 9)
    sil = plotx(Y, color, 'Autoencoder')
    log.info("Autoencoder %s, score: %.2f", name, sil)
    to_csv(Y, color, 'projections/Autoencoder_{}.csv'.format(name))

    t0 = time()
    X2, _, _ = run_ae(X, epochs=500, layers_dim=[50, 50, 30])
    Y = run_tsne(Y, n_components=n_components, perplexity=50,
                 init='pca')
    t1 = time()
    log.debug("AE+TSNE: %.2f sec" % (t1 - t0))
    ax = fig.add_subplot(2, 5, 10)
    sil = plotx(Y, color, 'AE + TSNE')
    log.info("Autoencoder + TSNE %s, score: %.2f", name, sil)
    to_csv(Y, color, 'projections/Autoencoder-tSNE_{}.csv'.format(name))

    plt.savefig('plots/result_{}.png'.format(name), dpi=600)
