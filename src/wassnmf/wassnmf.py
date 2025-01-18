from juliacall import Main as jl
import numpy as np
from typing import Tuple
from pathlib import Path


class WassersteinNMF:
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

        # Initialize Julia environment
        julia_pkg_path = Path(__file__).parent.parent.parent / "JuWassNMF"
        if not julia_pkg_path.exists():
            raise ImportError(f"Julia package directory not found at {julia_pkg_path}")

        try:
            # Activate the Julia project and import packages
            jl.seval(f'using Pkg; Pkg.activate("{julia_pkg_path}")')
            jl.seval("""
                using JuWassNMF
                using LinearAlgebra
                using Distances
                using PythonCall
                
                # Initialize global variables
                global py_input_X = nothing
                global py_input_K = nothing
                global X_proc = nothing
                global K_proc = nothing
                global D = nothing
                global Λ = nothing
            """)
            
        except Exception as e:
            raise ImportError(f"Failed to initialize Julia environment: {str(e)}") from e

    def _validate_input(self, X: np.ndarray, K: np.ndarray) -> None:
        """Basic input validation"""
        if X.ndim != 2:
            raise ValueError(f"Expected X to be 2D, got {X.ndim}D")
        if np.any(X < 0):
            raise ValueError("Input matrix X contains negative values")
        if K.shape != (X.shape[0], X.shape[0]):
            raise ValueError(f"Kernel K shape {K.shape} must match ({X.shape[0]}, {X.shape[0]})")

    def fit_transform(self, X: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose input matrix X into dictionary (D) and weight (Lambda) matrices.
        """
        # Basic input validation
        X = np.asarray(X, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        self._validate_input(X, K)

        try:
            # First, assign the numpy arrays to Julia global variables
            jl.py_input_X = X
            jl.py_input_K = K

            # Now process the arrays and run optimization
            jl.seval("""
                # Convert Python arrays to Julia matrices
                X = convert(Matrix{Float64}, py_input_X)
                K = convert(Matrix{Float64}, py_input_K)
                
                # Normalize X
                X ./= sum(X, dims=1)
                
                # Run optimization with provided parameters
                global D, Λ = wasserstein_nmf(
                    X, K, $n_components;
                    eps=$epsilon,
                    rho1=$rho1,
                    rho2=$rho2,
                    n_iter=$n_iter,
                    verbose=$verbose
                )
            """.replace("$n_components", str(self.n_components))
               .replace("$epsilon", str(self.epsilon))
               .replace("$rho1", str(self.rho1))
               .replace("$rho2", str(self.rho2))
               .replace("$n_iter", str(self.n_iter))
               .replace("$verbose", str(self.verbose).lower()))

            # Convert results back to numpy
            D = np.array(jl.D)
            Lambda = np.array(jl.Λ)

            # Final validation
            if not np.all(np.isfinite(D)) or not np.all(np.isfinite(Lambda)):
                raise RuntimeError("Optimization produced non-finite values")

            return D, Lambda

        except Exception as e:
            raise RuntimeError(f"Julia computation failed: {str(e)}") from e