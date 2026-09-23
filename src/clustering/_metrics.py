"""Internal clustering quality metrics: pooled distortion and gap statistic."""

import numpy as np

from scipy.spatial.distance import pdist


def pooled_distortion_score(X, labels, metric="sqeuclidean"):
    """Compute the pooled within-cluster distortion.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature array.
    labels : numpy.ndarray, shape (n_samples,)
        Predicted cluster labels.
    metric : str
        Distance metric passed to :func:`scipy.spatial.distance.pdist`.

    Returns
    -------
    float
        Sum of (pairwise-distance sum / 2n) across clusters.

    References
    ----------
    .. [1] https://github.com/DistrictDataLabs/yellowbrick/blob/develop/yellowbrick/cluster/elbow.py
    .. [2] https://web.stanford.edu/~hastie/Papers/gap.pdf
    """
    unique_labels = np.unique(labels)
    distortion = 0
    for current_label in unique_labels:
        instances = X[labels == current_label]
        distances = pdist(instances, metric=metric)
        distortion += distances.sum() / (2 * instances.shape[0])
    return distortion


def gap_score(
    clusterer,
    X,
    labels,
    n_refs=10,
    method="star",
    distribution="normal",
    random_state=None,
):
    """Compute the gap statistic of a given clusterer.

    Parameters
    ----------
    clusterer : sklearn.base.ClusterMixin
        A fitted scikit-learn clusterer.
    X : numpy.ndarray, shape (n_samples, n_features)
        Training instances.
    labels : numpy.ndarray, shape (n_samples,)
        Predicted cluster labels.
    n_refs : int, optional
        Number of random reference datasets, by default 10.
    method : {'log', 'star'}, optional
        Gap computation variant, by default 'star'.
    distribution : {'normal', 'uniform'}, optional
        Reference distribution, by default 'normal'.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    gap : float
        Gap statistic.
    sk : float
        s_k value from Tibshirani et al. (2001).
    method : str
        The method used.

    References
    ----------
    .. [1] https://statweb.stanford.edu/~gwalther/gap
    .. [2] https://arxiv.org/pdf/1103.4767.pdf
    .. [3] https://doi.org/10.1111/j.1541-0420.2007.00784.x
    """
    if random_state is not None:
        np.random.seed(random_state)

    assert method in [
        "log",
        "star",
    ], f'Method {method} not available. Use "log" or "star".'

    real_dispersion = pooled_distortion_score(X, labels)

    ref_dispersions = np.zeros(n_refs)
    for i in range(n_refs):
        if distribution == "normal":
            random_data = np.random.normal(loc=X.mean(0), scale=X.std(0), size=X.shape)
        elif distribution == "uniform":
            x_min, x_max = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)
            random_data = (
                np.random.random_sample(size=X.shape) * (x_max - x_min) + x_min
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        ref_labels = clusterer.fit_predict(random_data)
        ref_dispersions[i] = pooled_distortion_score(random_data, ref_labels)

    if method == "log":
        final_ref = np.log(ref_dispersions)
        final_real = np.log(real_dispersion)
    else:
        final_ref = ref_dispersions
        final_real = real_dispersion

    gap = np.mean(final_ref) - final_real
    sdk = np.std(final_ref)
    sk = np.sqrt(1.0 + 1.0 / n_refs) * sdk
    return gap, sk, method
