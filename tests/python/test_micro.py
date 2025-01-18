import numpy as np
import sys
from pathlib import Path
import pytest
from sklearn.metrics import pairwise_distances
# Add the src directory to sys.path for importing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from wassnmf.wassnmf import WassersteinNMF


def test_micro():
    """
    Minimal test for WassersteinNMF with user-provided kernel K.
    Ensures basic functionality: shapes, non-negativity, etc.
    """

    # Create a small (n_features=5, n_samples=3) dataset
    np.random.seed(42)
    X = np.random.rand(5, 3)
    
    # Generate a simple cost matrix based on row coordinates
    # Then convert to a kernel K = exp(-C / epsilon)
    coords = np.arange(X.shape[0]).reshape(-1, 1)
    C = pairwise_distances(coords, metric="sqeuclidean")
    C /= np.mean(C)
    epsilon = 0.025
    K = np.exp(-C / epsilon)
    
    # Initialize WassersteinNMF with minimal parameters
    wnmf = WassersteinNMF(n_components=2, epsilon=epsilon, rho1=0.05, rho2=0.05, n_iter=1, verbose=False)

    # Decompose X using the precomputed kernel K
    D, Lambda = wnmf.fit_transform(X, K)

    # Verify the basic shape constraints
    assert D.shape == (5, 2), f"Expected D shape (5,2), got {D.shape}"
    assert Lambda.shape == (2, 3), f"Expected Lambda shape (2,3), got {Lambda.shape}"
    
    # Check non-negativity
    assert np.all(D >= 0), "D has negative entries"
    assert np.all(Lambda >= 0), "Lambda has negative entries"

    print("Micro test passed with user-provided kernel K!")

if __name__ == "__main__":
    test_micro()
