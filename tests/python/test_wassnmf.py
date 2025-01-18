import pytest
import numpy as np
import sys
from pathlib import Path

# Insert "src" into sys.path so we can import the local wassnmf package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


from wassnmf.wassnmf import WassersteinNMF

def generate_test_data(n_features=100, n_samples=20):
    """Generate synthetic data similar to Julia example."""
    x = np.linspace(-12, 12, n_features)
    X = np.zeros((n_features, n_samples))
    
    np.random.seed(42)
    for i in range(n_samples):
        # Three Gaussian components
        X[:, i] = (
            np.random.rand() * np.exp(-(x - (np.random.randn() + 6))**2)
            + np.random.rand() * np.exp(-(x - np.random.randn())**2)
            + np.random.rand() * np.exp(-(x - (np.random.randn() - 6))**2)
        )
    
    # Normalize columns to lie on a probability simplex
    return X / X.sum(axis=0)

@pytest.fixture
def wnmf():
    """Fixture for WassersteinNMF instance."""
    return WassersteinNMF(n_components=3)

@pytest.fixture
def test_data():
    """Fixture for test data."""
    return generate_test_data()

def test_initialization():
    """Test WassersteinNMF initialization."""
    wnmf = WassersteinNMF(n_components=3)
    assert wnmf.n_components == 3
    assert wnmf.epsilon == 0.025  # default value
    assert wnmf.n_iter == 10      # default value

def test_input_validation(wnmf):
    """Test input validation."""
    # Test negative values
    X_neg = np.array([[1, -1], [1, 1]])
    with pytest.raises(ValueError, match="negative values"):
        wnmf.fit_transform(X_neg)
    
    # Test wrong dimensions
    X_1d = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="2D array"):
        wnmf.fit_transform(X_1d)

def test_basic_functionality(wnmf, test_data):
    """Test basic functionality."""
    n_features, n_samples = test_data.shape
    
    # Run decomposition
    D, Lambda = wnmf.fit_transform(test_data)
    
    # Check shapes
    assert D.shape == (n_features, 3)
    assert Lambda.shape == (3, n_samples)
    
    # Check non-negativity
    assert np.all(D >= 0)
    assert np.all(Lambda >= 0)
    
    # Check simplex constraints (columns should sum to 1)
    np.testing.assert_array_almost_equal(D.sum(axis=0), 1.0)
    np.testing.assert_array_almost_equal(Lambda.sum(axis=0), 1.0)

def test_different_sizes():
    """Test with different matrix sizes."""
    small_data = generate_test_data(n_features=20, n_samples=10)
    wnmf = WassersteinNMF(n_components=2)
    D, Lambda = wnmf.fit_transform(small_data)
    
    assert D.shape == (20, 2)
    assert Lambda.shape == (2, 10)

def test_different_parameters():
    """Test with different parameter values."""
    data = generate_test_data(n_features=30, n_samples=15)
    wnmf = WassersteinNMF(
        n_components=2,
        epsilon=0.1,
        rho1=0.1,
        rho2=0.1,
        n_iter=5,
        verbose=True
    )
    D, Lambda = wnmf.fit_transform(data)
    
    assert D.shape == (30, 2)
    assert Lambda.shape == (2, 15)

def test_reproducibility():
    """
    Test reproducibility with the same random seed. 
    Results aren't exactly equal due to the optimizer's nature, 
    but they should be close.
    """
    data = generate_test_data()
    
    # Run twice with the same parameters
    wnmf1 = WassersteinNMF(n_components=3, n_iter=5)
    wnmf2 = WassersteinNMF(n_components=3, n_iter=5)
    
    D1, Lambda1 = wnmf1.fit_transform(data)
    D2, Lambda2 = wnmf2.fit_transform(data)
    
    # Check closeness
    assert np.allclose(D1, D2, rtol=1e-5, atol=1e-5)
    assert np.allclose(Lambda1, Lambda2, rtol=1e-5, atol=1e-5)
