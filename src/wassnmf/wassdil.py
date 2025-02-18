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
    
#------------------------------------------------------------
#|  Batched Computations                                 |
    #------------------------------------------------------------
    def ot_entropic_semidual_batch(self, X_batch: torch.Tensor, G_batch: torch.Tensor, eps: float, K: torch.Tensor) -> torch.Tensor:
        """Batched computation of entropic optimal transport semi-dual.
        
        Args:
            X_batch: Input batch tensor of shape (batch_size, n_features)
            G_batch: Dual variable batch tensor of shape (batch_size, n_features)
            eps: Entropic regularization parameter
            K: Cost matrix of shape (n_features, n_features)
        
        Returns:
            Tensor of shape (batch_size, n_features)
        """
        log_term = torch.log(X_batch + 1e-10)
        exp_term = torch.exp((G_batch + log_term) / eps)
        
        # Reshape exp_term to (batch_size, n_features, 1)
        exp_term = exp_term.unsqueeze(-1)
        
        # Reshape K to (n_features, n_features) if it has extra dimensions
        if K.dim() > 2:
            K = K.squeeze(0)
        
        # Reshape K to match batch dimension (1, n_features, n_features)
        K = K.unsqueeze(0)
        
        # Expand K to match batch size if needed
        if exp_term.size(0) > 1:
            K = K.expand(exp_term.size(0), -1, -1)
        
        # Perform batch matrix multiplication
        result = torch.bmm(exp_term.transpose(1, 2), K)
        
        # Remove the extra dimension and return
        return eps * result.squeeze(1)


    def dual_obj_weights_batch(
        self,
        X_batch: torch.Tensor,   # (bs, n_features)
        K: torch.Tensor,         # (n_features, n_features)
        eps: float,
        D_batch: torch.Tensor,   # (bs, k)
        G_batch: torch.Tensor,   # (bs, n_features)
        rho1: float
    ) -> torch.Tensor:
        # Entropic OT term, shape = (bs, n_features) => sum => scalar
        ot_term = self.ot_entropic_semidual_batch(X_batch, G_batch, eps, K).sum()

        # Now the regularization piece
        # D_batch.T is (k, bs), G_batch is (bs, n_features) => (k, n_features)
        reg_input = -(D_batch.T @ G_batch) / rho1  # => (k, n_features)

        # E_star(...) along dim=0 => (n_features,) => sum => scalar
        reg_term = rho1 * self.E_star(reg_input, dim=0).sum()

        return ot_term + reg_term

    def dual_obj_dict_batch(self, X_batch: torch.Tensor, K: torch.Tensor, eps: float,
                        Lambda: torch.Tensor, G_batch: torch.Tensor, rho2: float) -> torch.Tensor:
        """Batched dual objective for dictionary."""
        ot_term = self.ot_entropic_semidual_batch(X_batch, G_batch, eps, K).sum()
        # Ensure G_batch and Lambda have correct dimensions for matrix multiplication
        reg_term = rho2 * self.E_star(-G_batch @ Lambda.T / rho2).sum()
        return ot_term + reg_term


    def solve_weights_batch(
        self,
        X: torch.Tensor,        # (N, n_features)
        K: torch.Tensor,        # (n_features, n_features)
        eps: float,
        D: torch.Tensor,        # (N, k)  <--- dictionary across all samples
        rho1: float,
        device: torch.device,
        batch_size: int,
        max_iter: int = 250,
        g_tol: float = 1e-4
    ) -> torch.Tensor:
        """
        Returns: final weights of shape (k, n_features).
        """
        num_batches = (X.shape[0] + batch_size - 1) // batch_size
        all_Lambda = []

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, X.shape[0])

            # Slice the batch
            X_batch = X[start:end, :]       # (bs, n_features)
            D_batch = D[start:end, :]       # (bs, k)

            # Initialize G_batch
            G_batch = torch.zeros_like(X_batch, device=device, dtype=self.dtype, requires_grad=True)
            optimizer = torch.optim.LBFGS([G_batch], max_iter=max_iter, tolerance_grad=g_tol)

            def closure():
                optimizer.zero_grad()
                loss = self.dual_obj_weights_batch(X_batch, K, eps, D_batch, G_batch, rho1)
                loss.backward()
                return loss

            optimizer.step(closure)

            # Compute the weights for this batch => shape (k, n_features)
            Lambda_batch = self.get_primal_weights(D_batch, G_batch.detach(), rho1)
            all_Lambda.append(Lambda_batch)

        # You could average or sum across batches, depending on your modeling
        Lambda = torch.stack(all_Lambda, dim=0).mean(dim=0)
        return self.simplex_norm(Lambda)   # shape (k, n_features)



    def solve_dict_batch(self, X: torch.Tensor, K: torch.Tensor, eps: float, 
                        Lambda: torch.Tensor, rho2: float, device: torch.device,
                        batch_size: int, max_iter: int = 250, g_tol: float = 1e-4) -> torch.Tensor:
        """Solve for optimal dictionary using gradient descent with batching."""
        num_batches = (X.shape[0] + batch_size - 1) // batch_size
        all_D = []
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, X.shape[0])
            X_batch = X[start:end]
            G_batch = torch.zeros_like(X_batch, device=device, dtype=self.dtype, requires_grad=True)
            optimizer = torch.optim.LBFGS([G_batch], max_iter=max_iter, tolerance_grad=g_tol)

            def closure():
                optimizer.zero_grad()
                loss = self.dual_obj_dict_batch(X_batch, K, eps, Lambda, G_batch, rho2)
                loss.backward()
                return loss

            optimizer.step(closure)
            D_batch = self.get_primal_dict(Lambda, G_batch.detach(), rho2)
            all_D.append(D_batch)
        
        return torch.cat(all_D, dim=0)

    def fit(self, X: torch.Tensor, K: torch.Tensor, k: int, eps: float = 0.025,
            rho1: float = 0.05, rho2: float = 0.05, n_iter: int = 10,
            device: Optional[torch.device] = None, verbose: bool = True,
            batch_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
            self.logger.info(f"Batch size: {batch_size}")

        # Initialize D and Lambda randomly on probability simplex
        init_D = torch.rand(X.shape[0], k, device=device, dtype=self.dtype)
        init_Lambda = torch.rand(k, X.shape[1], device=device, dtype=self.dtype)
        
        D = self.simplex_norm(tensor=init_D)
        Lambda = self.simplex_norm(tensor=init_Lambda)

        for iter in range(n_iter):
            if verbose:
                self.logger.info(f"Wasserstein-DiL: iteration {iter+1}/{n_iter}")

            if batch_size is None:
                D = self.solve_dict(X, K, eps, Lambda, rho2, device)
                Lambda = self.solve_weights(X, K, eps, D, rho1, device)
            else:
                D = self.solve_dict_batch(X, K, eps, Lambda, rho2, device, batch_size)
                Lambda = self.solve_weights_batch(X, K, eps, D, rho1, device, batch_size)

        return D, Lambda