import numpy as np
import pytest
import sys
from pathlib import Path

# Insert "src" into sys.path so we can import the local wassnmf package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from wassnmf.wassnmf import WassersteinNMF

# Helper function for data generation
def generate_test_data(n_features=100, n_samples=100):
    coord = np.linspace(-12, 12, n_features)
    def f(x, μ): return np.exp(-(x - μ) ** 2)
    
    np.random.seed(42)
    X = np.column_stack([
        np.random.rand() * f(coord, np.random.randn() + 6) +
        np.random.rand() * f(coord, np.random.randn()) +
        np.random.rand() * f(coord, np.random.randn() - 6)
        for _ in range(n_samples)
    ])
    return X / X.sum(axis=0, keepdims=True)

@pytest.mark.parametrize("dims", [(100, 100), (20, 10)])
def test_generate_test_data(dims):
    n_features, n_samples = dims
    X = generate_test_data(n_features, n_samples)
    assert X.shape == (n_features, n_samples)
    np.testing.assert_allclose(X.sum(axis=0), 1.0, atol=1e-6)
    assert np.all(X >= 0)

def test_kernel_matrix():
    n_features = 10
    coord = np.arange(n_features)
    cost_matrix = np.array([[np.linalg.norm(i - j)**2 for j in coord] for i in coord])
    K = np.exp(-cost_matrix / 0.025)
    assert K.shape == (n_features, n_features)
    assert np.allclose(K, K.T, atol=1e-6)

def test_wasserstein_nmf_utility():
    n_features, n_samples, n_components = 10, 5, 2
    X = generate_test_data(n_features, n_samples)
    
    coord = np.arange(n_features)
    cost_matrix = np.array([[np.linalg.norm(i - j)**2 for j in coord] for i in coord])
    K = np.exp(-cost_matrix / 0.025)
    
    wnmf = WassersteinNMF(n_components=n_components, verbose=False)
    D, Lambda = wnmf.fit_transform(X, K)
    
    assert D.shape == (n_features, n_components)
    assert Lambda.shape == (n_components, n_samples)
    np.testing.assert_allclose(D.sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(Lambda.sum(axis=0), 1.0, atol=1e-6)
    assert np.all(D >= 0)
    assert np.all(Lambda >= 0)

def test_wasserstein_nmf_custom_params():
    n_features, n_samples, n_components = 15, 8, 3
    X = generate_test_data(n_features, n_samples)
    
    coord = np.arange(n_features)
    cost_matrix = np.array([[np.linalg.norm(i - j)**2 for j in coord] for i in coord])
    K = np.exp(-cost_matrix / 0.025)
    
    wnmf = WassersteinNMF(
        n_components=n_components,
        epsilon=0.1,
        rho1=0.1,
        rho2=0.1,
        n_iter=5,
        verbose=False
    )
    D, Lambda = wnmf.fit_transform(X, K)
    
    assert D.shape == (n_features, n_components)
    assert Lambda.shape == (n_components, n_samples)
    np.testing.assert_allclose(D.sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(Lambda.sum(axis=0), 1.0, atol=1e-6)
    assert np.all(D >= 0)
    assert np.all(Lambda >= 0)

def test_edge_cases():
    # Minimal size
    X_tiny = generate_test_data(4, 3)
    coord = np.arange(4)
    cost_matrix = np.array([[np.linalg.norm(i - j)**2 for j in coord] for i in coord])
    K = np.exp(-cost_matrix / 0.025)
    wnmf = WassersteinNMF(n_components=2, verbose=False)
    D, Lambda = wnmf.fit_transform(X_tiny, K)
    assert D.shape == (4, 2)
    assert Lambda.shape == (2, 3)

    # Large regularization
    X_small = generate_test_data(10, 5)
    coord = np.arange(10)
    cost_matrix = np.array([[np.linalg.norm(i - j)**2 for j in coord] for i in coord])
    K = np.exp(-cost_matrix / 1.0)
    wnmf = WassersteinNMF(n_components=2, verbose=False)
    D, Lambda = wnmf.fit_transform(X_small, K)
    assert D.shape == (10, 2)
    assert Lambda.shape == (2, 5)
