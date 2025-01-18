import numpy as np
from wassnmf.wassnmf import WassersteinNMF
from julia.api import Julia


def test_micro():
    """
    A minimal test to verify the basic functionality of WassersteinNMF.
    Ensures the decomposition runs successfully and produces expected shapes.
    """
    # Generate a small random non-negative matrix
    np.random.seed(0)
    X = np.random.rand(10, 5)  # 10 features, 5 samples

    # Initialize WassersteinNMF with minimal parameters
    jl = Julia(compiled_modules=False)
    
    wnmf = WassersteinNMF(n_components=2, n_iter=1, verbose=False)

    # Run decomposition
    D, Lambda = wnmf.fit_transform(X)

    # Validate results
    assert D.shape == (10, 2), f"Unexpected D shape: {D.shape}"
    assert Lambda.shape == (2, 5), f"Unexpected Lambda shape: {Lambda.shape}"
    assert np.all(D >= 0), "D contains negative values"
    assert np.all(Lambda >= 0), "Lambda contains negative values"

    print("Micro test passed!")

# Run the micro test
if __name__ == "__main__":
    test_micro()
