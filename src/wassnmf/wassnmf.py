import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from juliacall import Main as jl


class WassersteinNMF:
    def __init__(
        self,
        n_components: int,
        epsilon: float = 0.025,
        rho1: float = 0.05,
        rho2: float = 0.05,
        n_iter: int = 10,
        verbose: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        use_gpu: bool = False
    ):
        """
        Initialize WassersteinNMF with optional GPU support.

        Args:
            n_components: Number of components for the decomposition
            epsilon: Entropic regularization parameter
            rho1: First proximal term weight
            rho2: Second proximal term weight
            n_iter: Number of iterations
            verbose: Whether to print progress
            device: PyTorch device to use ('cpu', 'cuda', or torch.device)
        """
        self.n_components = n_components
        self.epsilon = epsilon
        self.rho1 = rho1
        self.rho2 = rho2
        self.n_iter = n_iter
        self.verbose = verbose
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.use_gpu = False if device is None else use_gpu

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

        self.logger.info(f"Using device: {repr(self.device)}")
        self.logger.info("Initializing Julia environment...")

        # Initialize Julia environment
        julia_pkg_path = Path(__file__).parent.parent.parent / "JuWassNMF"
        if not julia_pkg_path.exists():
            self.logger.error(
                f"Julia package directory not found at {julia_pkg_path}"
            )
            raise ImportError(f"Julia package directory not found at {julia_pkg_path}")

        try:
            # Activate the Julia project and import packages
            jl.seval(f'using Pkg; Pkg.activate("{julia_pkg_path}")')
            jl.seval("""
                using JuWassNMF
                using LinearAlgebra
                using Distances
                using PythonCall
                # using CUDA
                
                # Initialize global variables
                global py_input_X = nothing
                global py_input_K = nothing
                global X_proc = nothing
                global K_proc = nothing
                global D = nothing
                global Λ = nothing
            """)

        except Exception as e:
            self.logger.error(f"Failed to initialize Julia environment: {str(e)}")
            raise ImportError(
                f"Failed to initialize Julia environment: {str(e)}"
            ) from e

    def _validate_input(
        self, X: Union[np.ndarray, torch.Tensor], K: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Validate input tensors/arrays"""
        if isinstance(X, torch.Tensor):
            if X.dim() != 2:
                self.logger.error(f"Expected X to be 2D, got {X.dim()}D")
                raise ValueError(f"Expected X to be 2D, got {X.dim()}D")
            if torch.any(X < 0):
                self.logger.error("Input tensor X contains negative values")
                raise ValueError("Input tensor X contains negative values")
        else:
            if X.ndim != 2:
                self.logger.error(f"Expected X to be 2D, got {X.ndim}D")
                raise ValueError(f"Expected X to be 2D, got {X.ndim}D")
            if np.any(X < 0):
                self.logger.error("Input matrix X contains negative values")
                raise ValueError("Input matrix X contains negative values")

        # Check K dimensions
        K_shape = K.shape if isinstance(K, np.ndarray) else tuple(K.size())
        X_shape = X.shape if isinstance(X, np.ndarray) else tuple(X.size())
        if K_shape != (X_shape[0], X_shape[0]):
            self.logger.error(
                f"Kernel K shape {K_shape} must match ({X_shape[0]}, {X_shape[0]})"
            )
            raise ValueError(
                f"Kernel K shape {K_shape} must match ({X_shape[0]}, {X_shape[0]})"
            )

    def _to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array"""
        if isinstance(x, torch.Tensor):
            self.logger.info("Converting torch tensor to numpy array")
            return x.cpu().numpy()
        self.logger.info("Input is already a numpy array")
        return np.asarray(x)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on the correct device"""
        self.logger.info("Converting numpy array to torch tensor")
        return torch.from_numpy(x).to(self.device)

    def fit_transform(
        self, X_np: Union[np.ndarray], K_np: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose input matrix X into dictionary (D) and weight (Lambda) matrices.

        Args:
            X_np: Input matrix (numpy array)
            K_np: Kernel matrix (numpy array)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Dictionary matrix D and weight matrix Lambda
        """
        # Input validation
        self.logger.info("Validating input...")
        self._validate_input(X_np, X_np)

        try:
            # Assign numpy arrays to Julia global variables
            self.logger.info("Moving X and K to Julia...")
            jl.py_input_X = X_np
            jl.py_input_K = K_np

            # Process arrays and run optimization
            if self.use_gpu:
                self.logger.info("Starting the Julia GPU execution...")
                jl.seval(
                    """
                    # Convert Python arrays to Julia matrices
                    X = convert(Matrix{Float64}, py_input_X)
                    K = convert(Matrix{Float64}, py_input_K)
                    
                    # Normalize X
                    X ./= sum(X, dims=1)
                    
                    # Run GPU optimization
                    global D, Λ = wasserstein_nmf_gpu(
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
                    .replace("$verbose", str(self.verbose).lower())
                )
            else:
                self.logger.info("Starting the Julia CPU execution...")
                jl.seval(
                    """
                    # Convert Python arrays to Julia matrices
                    X = convert(Matrix{Float64}, py_input_X)
                    K = convert(Matrix{Float64}, py_input_K)

                    # Moved normalization to preprocessing!

                    # Run CPU optimization
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
                    .replace("$verbose", str(self.verbose).lower())
                )

            # Convert results to PyTorch tensors
            D = self._to_torch(np.array(jl.D))
            Lambda = self._to_torch(np.array(jl.Λ))

            # Final validation
            if not torch.all(torch.isfinite(D)) or not torch.all(
                torch.isfinite(Lambda)
            ):
                self.logger.error("Optimization produced non-finite values")
                raise RuntimeError("Optimization produced non-finite values")

            return D, Lambda

        except Exception as e:
            self.logger.error(f"Julia computation failed: {str(e)}")
            raise RuntimeError(f"Julia computation failed: {str(e)}") from e
