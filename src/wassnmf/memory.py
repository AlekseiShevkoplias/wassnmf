from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform


class WassDictionaryLearning:
    """
    Python interface for Wasserstein NMF algorithm based on the paper
    "Fast Dictionary Learning with a Smoothed Wasserstein Loss"
    by Antoine Rolet, Marco Cuturi, and Gabriel Peyré.
    """

    def __init__(
        self,
        n_components: int,
        epsilon: float = 0.1,
        rho1: float = 0.1,
        rho2: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-4,
        memory_efficient: bool = False,
        verbose: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize WassersteinNMF.

        Parameters:
        -----------
        n_components: Number of dictionary elements
        epsilon: Entropic regularization parameter
        rho1: Regularization parameter for weights
        rho2: Regularization parameter for dictionary
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        memory_efficient: Whether to use memory-efficient projections
        verbose: Whether to print progress information
        random_state: Random seed for reproducible results
        """
        self.n_components = n_components
        self.epsilon = epsilon
        self.rho1 = rho1
        self.rho2 = rho2
        self.max_iter = max_iter
        self.tol = tol
        self.memory_efficient = memory_efficient
        self.verbose = verbose
        self.random_state = random_state

        # Initialize Julia environment
        try:
            from juliacall import Main as jl

            jl.seval("""
                global X, C, k, D, Λ
                include("/home/user920/Documents/SilkNest/pyandju/wassnmf/JuWassNMF/src/WassDiL.jl")
                using .WassDiL
            """)

            # Set up Julia imports
            self.jl = jl

        except ImportError:
            raise ImportError(
                "juliacall is required. Install it with: pip install juliacall"
            )

    def _validate_input(self, X: Union[np.ndarray, torch.Tensor]):
        """Validate input data matrix."""
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X)

        if X_np.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_np.ndim}D")

        # Check non-negativity
        if np.any(X_np < 0):
            raise ValueError("Input matrix must be non-negative")

        # Normalize columns to be probability distributions
        X_norm = X_np.copy()
        col_sums = X_norm.sum(axis=0)
        nonzero_cols = col_sums > 0
        X_norm[:, nonzero_cols] = X_norm[:, nonzero_cols] / col_sums[nonzero_cols]

        return X_norm

    def _create_cost_matrix(self, X: np.ndarray, metric: str = "euclidean"):
        """Create cost matrix based on provided metric."""
        n_features = X.shape[0]

        if metric == "euclidean":
            # Create a grid of coordinates for 1D data
            coords = np.arange(n_features).reshape(-1, 1)
            # Compute pairwise distances
            C = squareform(pdist(coords, metric="euclidean"))
            # Normalize
            C = C / C.mean()

        elif metric == "cosine":
            # For text data, often cosine distance is used
            # This assumes the rows of X represent different words
            # and we use the column averages as word embeddings
            embeddings = X.mean(axis=1).reshape(-1, 1)
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-10)
            # Compute cosine distance
            similarity = normalized @ normalized.T
            C = 1 - similarity

        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return C

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        C: Optional[Union[np.ndarray, torch.Tensor]] = None,
        metric: str = "euclidean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit Wasserstein NMF model to data and return the dictionary and weights.

        Parameters:
        -----------
        X: Input data matrix (features × samples)
        C: Cost matrix (optional). If None, will be computed based on metric
        metric: Metric to use for computing cost matrix if C is None
            Options: 'euclidean' (default), 'cosine'

        Returns:
        --------
        D: Dictionary matrix (features × components)
        Lambda: Weight matrix (components × samples)
        """
        # Validate and normalize input
        X_norm = self._validate_input(X)

        # Create cost matrix if not provided
        if C is None:
            C_matrix = self._create_cost_matrix(X_norm, metric=metric)
        else:
            if isinstance(C, torch.Tensor):
                C_matrix = C.cpu().numpy()
            else:
                C_matrix = np.asarray(C)

        # Validate cost matrix dimensions
        if C_matrix.shape != (X_norm.shape[0], X_norm.shape[0]):
            raise ValueError(
                f"Cost matrix shape {C_matrix.shape} does not match input features {X_norm.shape[0]}"
            )

        # Convert to Julia arrays
        jl = self.jl
        # jl.seval("using WassDiL")  # Just in case

        # Convert Python arrays to native Julia arrays
        X_jl = jl.Matrix(X_norm)
        C_jl = jl.Matrix(C_matrix)

        # Call the function directly
        D_jl, Lambda_jl = jl.WassDiL.wasserstein_dil(
            X_jl,
            C_jl,
            self.n_components,
            ε=self.epsilon,
            ρ1=self.rho1,
            ρ2=self.rho2,
            max_iter=self.max_iter,
            tol=self.tol,
            memory_efficient=self.memory_efficient,
            seed=self.random_state if self.random_state is not None else None,
        )
        D = np.array(D_jl)
        Lambda = np.array(Lambda_jl)
        print("D shape", D.shape)
        print("Lambda shape", Lambda.shape)

        # Get results
        D = np.array(jl.D)
        Lambda = np.array(jl.Λ)

        return D, Lambda

    def fit(self, X: Union[np.ndarray, torch.Tensor], C=None, metric="euclidean"):
        """
        Fit Wasserstein NMF model to data.

        Parameters:
        -----------
        X: Input data matrix (features × samples)
        C: Cost matrix (optional). If None, will be computed based on metric
        metric: Metric to use for computing cost matrix if C is None

        Returns:
        --------
        self: The fitted model
        """
        self.components_, self.weights_ = self.fit_transform(X, C, metric)
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform data X to weights using the fitted dictionary.

        Parameters:
        -----------
        X: Input data matrix (features × samples)

        Returns:
        --------
        Lambda: Weight matrix (components × samples)
        """
        if not hasattr(self, "components_"):
            raise ValueError("Model not fitted yet. Call fit first.")

        X_norm = self._validate_input(X)

        # TODO: Implement proper transform using Wasserstein distance
        # For now, use a simple approximation
        # In a real implementation, this would solve for weights using the
        # fixed dictionary and Wasserstein distance

        # Solve for weights using non-negative least squares
        weights = np.zeros((self.n_components, X_norm.shape[1]))

        for i in range(X_norm.shape[1]):
            # Simple approximation using least squares
            # In practice, this should be Wasserstein projection
            w = np.linalg.lstsq(self.components_, X_norm[:, i], rcond=None)[0]
            weights[:, i] = np.maximum(w, 0)
            weights[:, i] /= weights[:, i].sum() + 1e-10

        return weights

    def reconstruct(
        self, X: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> np.ndarray:
        """
        Reconstruct data using the fitted model.

        Parameters:
        -----------
        X: Input data (optional). If provided, transforms X first.

        Returns:
        --------
        X_reconstructed: Reconstructed data matrix
        """
        if not hasattr(self, "components_"):
            raise ValueError("Model not fitted yet. Call fit first.")

        if X is not None:
            weights = self.transform(X)
        else:
            weights = self.weights_

        return self.components_ @ weights
