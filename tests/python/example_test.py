import numpy as np
import pytest
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from wassnmf.wassnmf import WassersteinNMF

def test_julia_notebook_analog():
    """
    Reproduce the exact steps from the Julia notebook example:
    https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev/examples/nmf/
    """
    # Data generation parameters exactly matching Julia
    n_features = 100
    n_samples = 100
    coord = np.linspace(-12, 12, n_features)
    X = np.zeros((n_features, n_samples))
    sigma = 1.0
    
    # Set same random seed as Julia example
    np.random.seed(42)
    
    # Generate data exactly as in Julia notebook
    for i in range(n_samples):
        X[:, i] = (
            np.random.rand() * np.exp(-(coord - (sigma * np.random.randn() + 6)) ** 2) +
            np.random.rand() * np.exp(-(coord - sigma * np.random.randn()) ** 2) +
            np.random.rand() * np.exp(-(coord - (sigma * np.random.randn() - 6)) ** 2)
        )
    
    # Normalize columns to probability simplex (no epsilon added)
    X /= X.sum(axis=0, keepdims=True)
    
    # Create cost matrix and kernel exactly as in Julia
    C = pairwise_distances(coord.reshape(-1, 1), metric="sqeuclidean")
    C /= np.mean(C)  # Same normalization as Julia
    eps = 0.025      # Same epsilon as Julia notebook
    K = np.exp(-C / eps)
    
    # Initialize and run WassersteinNMF
    wnmf = WassersteinNMF(
        n_components=3,
        epsilon=eps,    # Match Julia exactly
        rho1=0.05,      # Match Julia exactly  
        rho2=0.05,      # Match Julia exactly
        n_iter=10,
        verbose=True
    )
    
    D, Lambda = wnmf.fit_transform(X, K)
    
    # Validate results
    assert D.shape == (n_features, 3)
    assert Lambda.shape == (3, n_samples)
    assert np.all(D >= 0)
    assert np.all(Lambda >= 0)
    assert np.allclose(D.sum(axis=0), 1.0, atol=1e-4)
    assert np.allclose(Lambda.sum(axis=0), 1.0, atol=1e-4)
    
    # Create plot matching Julia's style
    plt.figure(figsize=(10, 6))
    for i in range(D.shape[1]):
        plt.plot(coord, D[:, i], label=f"Component {i+1}")
    plt.title("NMF with Wasserstein loss")
    plt.xlabel("Coordinate")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig('example_test.png')
    plt.close()

if __name__ == "__main__":
    test_julia_notebook_analog()