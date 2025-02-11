import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

class WassersteinDiL:
    def __init__(self, dtype=torch.float32):
        self.logger = logging.getLogger(__name__)
        self.dtype = dtype
        logging.basicConfig(level=logging.INFO)

    def _ensure_tensor(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Ensure tensor is on correct device and dtype."""
        return tensor.to(device=device, dtype=self.dtype)

    def simplex_norm(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Normalize tensor along specified dimension to lie on probability simplex."""
        return tensor / torch.sum(tensor, dim=dim, keepdim=True)

    def E_star(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Compute log-sum-exp along specified dimension."""
        return torch.logsumexp(tensor, dim=dim)

    def ot_entropic_semidual(self, X: torch.Tensor, G: torch.Tensor, eps: float, 
                            K: torch.Tensor) -> torch.Tensor:
        """Compute entropic optimal transport semi-dual."""
        log_term = torch.log(X + 1e-10)  # Add small constant for numerical stability
        exp_term = torch.exp((G + log_term) / eps)
        return eps * (exp_term @ K)

    def dual_obj_weights(self, X: torch.Tensor, K: torch.Tensor, eps: float, 
                        D: torch.Tensor, G: torch.Tensor, rho1: float) -> torch.Tensor:
        """Compute dual objective for weights."""
        ot_term = self.ot_entropic_semidual(X, G, eps, K).sum()
        reg_term = rho1 * self.E_star(-D.T @ G / rho1).sum()
        return ot_term + reg_term

    def dual_obj_dict(self, X: torch.Tensor, K: torch.Tensor, eps: float,
                     Lambda: torch.Tensor, G: torch.Tensor, rho2: float) -> torch.Tensor:
        """Compute dual objective for dictionary."""
        ot_term = self.ot_entropic_semidual(X, G, eps, K).sum()
        reg_term = rho2 * self.E_star(-G @ Lambda.T / rho2).sum()
        return ot_term + reg_term

    def get_primal_weights(self, D: torch.Tensor, G: torch.Tensor, 
                          rho1: float) -> torch.Tensor:
        """Compute primal weights from dual variables."""
        return F.softmax(-D.T @ G / rho1, dim=0)

    def get_primal_dict(self, Lambda: torch.Tensor, G: torch.Tensor, 
                        rho2: float) -> torch.Tensor:
        """Compute primal dictionary from dual variables."""
        return F.softmax(-G @ Lambda.T / rho2, dim=0)

    def solve_weights(self, X: torch.Tensor, K: torch.Tensor, eps: float, 
                     D: torch.Tensor, rho1: float, device: torch.device,
                     max_iter: int = 250, g_tol: float = 1e-4) -> torch.Tensor:
        """Solve for optimal weights using gradient descent."""
        G = torch.zeros_like(X, device=device, dtype=self.dtype, requires_grad=True)
        optimizer = torch.optim.LBFGS([G], max_iter=max_iter, tolerance_grad=g_tol)

        def closure():
            optimizer.zero_grad()
            loss = self.dual_obj_weights(X, K, eps, D, G, rho1)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.get_primal_weights(D, G.detach(), rho1)

    def solve_dict(self, X: torch.Tensor, K: torch.Tensor, eps: float, 
                  Lambda: torch.Tensor, rho2: float, device: torch.device,
                  max_iter: int = 250, g_tol: float = 1e-4) -> torch.Tensor:
        """Solve for optimal dictionary using gradient descent."""
        G = torch.zeros_like(X, device=device, dtype=self.dtype, requires_grad=True)
        optimizer = torch.optim.LBFGS([G], max_iter=max_iter, tolerance_grad=g_tol)

        def closure():
            optimizer.zero_grad()
            loss = self.dual_obj_dict(X, K, eps, Lambda, G, rho2)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.get_primal_dict(Lambda, G.detach(), rho2)

    def fit(self, X: torch.Tensor, K: torch.Tensor, k: int, eps: float = 0.025,
            rho1: float = 0.05, rho2: float = 0.05, n_iter: int = 10,
            device: Optional[torch.device] = None, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit Wasserstein DiL model.
        
        Args:
            X: Input data matrix
            K: Cost matrix
            k: Number of components
            eps: Entropic regularization parameter
            rho1: First regularization parameter
            rho2: Second regularization parameter
            n_iter: Number of alternating minimization iterations
            device: torch.device to use ('cpu' or 'cuda')
            verbose: Whether to print progress
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move data to device and ensure correct dtype
        X = self._ensure_tensor(X, device)
        K = self._ensure_tensor(K, device)

        if verbose:
            self.logger.info(f"Training on device: {device}")
            self.logger.info(f"X shape: {X.shape}, K shape: {K.shape}")

        # Initialize D and Lambda randomly on probability simplex
        init_D = torch.rand(X.shape[0], k, device=device, dtype=self.dtype)
        init_Lambda = torch.rand(k, X.shape[1], device=device, dtype=self.dtype)
        
        D = self.simplex_norm(tensor=init_D)
        Lambda = self.simplex_norm(tensor=init_Lambda)

        for iter in range(n_iter):
            if verbose:
                self.logger.info(f"Wasserstein-DiL: iteration {iter+1}/{n_iter}")

            # Update dictionary
            D = self.solve_dict(X, K, eps, Lambda, rho2, device)
            
            # Update weights
            Lambda = self.solve_weights(X, K, eps, D, rho1, device)

        return D, Lambda