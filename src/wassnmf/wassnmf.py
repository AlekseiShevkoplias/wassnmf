# from julia.api import Julia
# jl = Julia(compiled_modules=False)

from julia import Main
import numpy as np
from typing import Tuple
from pathlib import Path


class WassersteinNMF:
    """
    Wasserstein Non-negative Matrix Factorization using Julia implementation.

    This is a Python wrapper around the JuWassNMF.jl implementation, which performs
    NMF using Wasserstein distance as the objective.

    Parameters
    ----------
    n_components : int
        Number of components to decompose into
    epsilon : float, optional (default=0.025)
        Entropic regularization parameter
    rho1 : float, optional (default=0.05)
        First optimization parameter
    rho2 : float, optional (default=0.05)
        Second optimization parameter
    n_iter : int, optional (default=10)
        Number of iterations
    verbose : bool, optional (default=False)
        Whether to print progress information
    """

    def __init__(
        self,
        n_components: int,
        epsilon: float = 0.025,
        rho1: float = 0.05,
        rho2: float = 0.05,
        n_iter: int = 10,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.epsilon = epsilon
        self.rho1 = rho1
        self.rho2 = rho2
        self.n_iter = n_iter
        self.verbose = verbose
        print("Initializing WassersteinNMF...")

        # Get path to Julia project
        julia_pkg_path = Path(__file__).parent.parent.parent / "JuWassNMF"
        if not julia_pkg_path.exists():
            raise ImportError(f"Julia package directory not found at {julia_pkg_path}")

        # Initialize Julia with correct project and dependencies
        try:
            
            # Activate Julia project and import required packages
            Main.eval(f'using Pkg; Pkg.activate("{julia_pkg_path}")')
            Main.eval("""
                using JuWassNMF
                using LinearAlgebra
                using Distances
            """)
        except Exception as e:
            raise ImportError(
                f"Failed to initialize Julia environment at {julia_pkg_path}. "
                f"Error: {str(e)}"
            ) from e

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose the input matrix into dictionary and weights matrices.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)
            The data matrix to decompose. Should be non-negative.
            Will be normalized to lie on probability simplex.

        Returns
        -------
        D : ndarray of shape (n_features, n_components)
            Dictionary matrix
        Lambda : ndarray of shape (n_components, n_samples)
            Weights matrix

        Raises
        ------
        ValueError
            If X contains negative values or incorrect dimensions
        RuntimeError
            If Julia computation fails
        """
        # Input validation
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if np.any(X < 0):
            raise ValueError("Input matrix contains negative values")

        try:
            # Assign the Python matrix to Julia
            Main.X = X

            # Run Wasserstein NMF in Julia
            Main.eval(
                f"""
                D, Λ = wasserstein_nmf(
                    X,
                    {self.n_components},
                    eps={self.epsilon},
                    rho1={self.rho1},
                    rho2={self.rho2},
                    n_iter={self.n_iter},
                    verbose={str(self.verbose).lower()}
                )
                """
            )

            # Retrieve results from Julia
            D = np.array(Main.eval("D"))
            Lambda = np.array(Main.eval("Λ"))

            return D, Lambda

        except Exception as e:
            raise RuntimeError(
                "Julia computation failed. Make sure JuWassNMF.jl is properly installed "
                f"and all dependencies are available. Error: {str(e)}"
            ) from e
