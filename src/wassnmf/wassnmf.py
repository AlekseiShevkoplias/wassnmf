from juliacall import Main as jl
import numpy as np
import torch
from typing import Tuple, Union, Optional
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
        device: Optional[Union[str, torch.device]] = None,
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
        
        # Handle device setup
        self.device = torch.device('cpu') if device is None else torch.device(device)
        self.use_gpu = self.device.type == 'cuda'

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
                using CUDA
                
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

    def _validate_input(self, X: Union[np.ndarray, torch.Tensor], 
                       K: Union[np.ndarray, torch.Tensor]) -> None:
        """Validate input tensors/arrays"""
        if isinstance(X, torch.Tensor):
            if X.dim() != 2:
                raise ValueError(f"Expected X to be 2D, got {X.dim()}D")
            if torch.any(X < 0):
                raise ValueError("Input tensor X contains negative values")
        else:
            if X.ndim != 2:
                raise ValueError(f"Expected X to be 2D, got {X.ndim}D")
            if np.any(X < 0):
                raise ValueError("Input matrix X contains negative values")
            
        # Check K dimensions
        K_shape = K.shape if isinstance(K, np.ndarray) else tuple(K.size())
        X_shape = X.shape if isinstance(X, np.ndarray) else tuple(X.size())
        if K_shape != (X_shape[0], X_shape[0]):
            raise ValueError(f"Kernel K shape {K_shape} must match ({X_shape[0]}, {X_shape[0]})")

    def _to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array"""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on the correct device"""
        return torch.from_numpy(x).to(self.device)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], 
                     K: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose input matrix X into dictionary (D) and weight (Lambda) matrices.
        
        Args:
            X: Input matrix (numpy array or PyTorch tensor)
            K: Kernel matrix (numpy array or PyTorch tensor)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Dictionary matrix D and weight matrix Lambda
        """
        # Input validation
        self._validate_input(X, K)
        
        # Convert to numpy arrays
        X_np = self._to_numpy(X)
        K_np = self._to_numpy(K)
        
        try:
            # Assign numpy arrays to Julia global variables
            jl.py_input_X = X_np
            jl.py_input_K = K_np

            # Process arrays and run optimization
            if self.use_gpu:
                jl.seval("""
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
                   .replace("$verbose", str(self.verbose).lower()))
            else:
                jl.seval("""
                    # Convert Python arrays to Julia matrices
                    X = convert(Matrix{Float64}, py_input_X)
                    K = convert(Matrix{Float64}, py_input_K)
                    
                    # Normalize X
                    X ./= sum(X, dims=1)
                    
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
                   .replace("$verbose", str(self.verbose).lower()))

            # Convert results to PyTorch tensors
            D = self._to_torch(np.array(jl.D))
            Lambda = self._to_torch(np.array(jl.Λ))

            # Final validation
            if not torch.all(torch.isfinite(D)) or not torch.all(torch.isfinite(Lambda)):
                raise RuntimeError("Optimization produced non-finite values")

            return D, Lambda

        except Exception as e:
            raise RuntimeError(f"Julia computation failed: {str(e)}") from e