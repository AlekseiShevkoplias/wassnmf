import numpy as np
from sklearn.metrics import pairwise_distances


def generate_data(scenario):
    n_features = scenario["n_features"]
    n_samples = scenario["n_samples"]
    coord = np.linspace(-12, 12, n_features)
    X = np.zeros((n_features, n_samples))
    sigma = 1.0

    if scenario["name"] == "gaussian_mixture":
        for i in range(n_samples):
            X[:, i] = (
                np.random.rand()
                * np.exp(-((coord - (sigma * np.random.randn() + 6)) ** 2))
                + np.random.rand() * np.exp(-((coord - sigma * np.random.randn()) ** 2))
                + np.random.rand()
                * np.exp(-((coord - (sigma * np.random.randn() - 6)) ** 2))
            )

    # Normalize to simplex
    X /= X.sum(axis=0, keepdims=True)

    # Generate kernel
    C = pairwise_distances(coord.reshape(-1, 1), metric="sqeuclidean")
    C /= np.mean(C)
    K = np.exp(-C / 0.025)
    cost_matrix = C

    return X, K, coord, cost_matrix
