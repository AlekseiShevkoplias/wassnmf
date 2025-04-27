from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim import LBFGS


# Julia Equivalent: OptimalTransport.add_singleton
def add_singleton(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.unsqueeze(dim)


# Julia Equivalent: OptimalTransport.dot_vecwise
def dot_vecwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y)


# Julia Equivalent: Dual.ot_entropic_semidual
def ot_entropic_semidual(
    X: torch.Tensor, G: torch.Tensor, eps: float, K: torch.Tensor
) -> torch.Tensor:
    xlogx = X * torch.log(X.clamp(min=1e-20))  # Avoid log(0)
    term1 = -dot_vecwise(xlogx - X, torch.ones_like(X))
    term2 = dot_vecwise(X, torch.log((K @ torch.exp(G / eps)).clamp(min=1e-20)))
    return eps * (term1 + term2)


# Julia Equivalent: Dual.ot_entropic_semidual_grad
def ot_entropic_semidual_grad(
    X: torch.Tensor, G: torch.Tensor, eps: float, K: torch.Tensor
) -> torch.Tensor:
    return K.T @ (X / (K @ torch.exp(G / eps))) * torch.exp(G / eps)


# Julia Equivalent: Dual.getprimal_ot_entropic_semidual
def getprimal_ot_entropic_semidual(
    mu: torch.Tensor, v: torch.Tensor, eps: float, K: torch.Tensor
) -> torch.Tensor:
    return torch.exp(v / eps) * (K.T @ (mu / (K @ torch.exp(v / eps))))


# Julia Equivalent: Dual.ot_entropic_dual
def ot_entropic_dual(
    u: torch.Tensor, v: torch.Tensor, eps: float, K: torch.Tensor
) -> torch.Tensor:
    return eps * torch.log(dot_vecwise(torch.exp(u / eps), K @ torch.exp(v / eps)))


# Julia Equivalent: Dual.ot_entropic_dual_grad
def ot_entropic_dual_grad(
    u: torch.Tensor, v: torch.Tensor, eps: float, K: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    U = torch.exp(u / eps)
    V = torch.exp(v / eps)
    KV = K @ V
    grad_u = (U * KV) / dot_vecwise(U, KV)
    K_t_U = K.T @ U
    grad_v = (V * K_t_U) / dot_vecwise(V, K_t_U)
    return grad_u, grad_v


# Julia Equivalent: Dual.getprimal_ot_entropic_dual
def getprimal_ot_entropic_dual(
    u: torch.Tensor, v: torch.Tensor, eps: float, K: torch.Tensor
) -> torch.Tensor:
    return (
        K
        * add_singleton(torch.exp(-u / eps), 1)
        * add_singleton(torch.exp(-v / eps), 0)
    )


def simplex_norm_(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    x.div_(x.sum(dim=dim, keepdim=True))
    return x


def e_star(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim, keepdim=True)


def e_star_grad(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return F.softmax(x, dim=dim)


def dual_obj_weights(
    X: torch.Tensor,
    K: torch.Tensor,
    eps: float,
    D: torch.Tensor,
    G: torch.Tensor,
    rho1: float,
) -> torch.Tensor:
    return (
        ot_entropic_semidual(X, G, eps, K).sum()
        + rho1 * e_star(-torch.matmul(D.T, G) / rho1).sum()
    )


def dual_obj_weights_grad(
    X: torch.Tensor,
    K: torch.Tensor,
    eps: float,
    D: torch.Tensor,
    G: torch.Tensor,
    rho1: float,
) -> torch.Tensor:
    return ot_entropic_semidual_grad(X, G, eps, K) - torch.matmul(
        D, e_star_grad(-torch.matmul(D.T, G) / rho1)
    )


def dual_obj_dict(
    X: torch.Tensor,
    K: torch.Tensor,
    eps: float,
    Lambda: torch.Tensor,
    G: torch.Tensor,
    rho2: float,
) -> torch.Tensor:
    return (
        ot_entropic_semidual(X, G, eps, K).sum()
        + rho2 * e_star(-torch.matmul(G, Lambda.T) / rho2).sum()
    )


def dual_obj_dict_grad(
    X: torch.Tensor,
    K: torch.Tensor,
    eps: float,
    Lambda: torch.Tensor,
    G: torch.Tensor,
    rho2: float,
) -> torch.Tensor:
    return ot_entropic_semidual_grad(X, G, eps, K) - torch.matmul(
        e_star_grad(-torch.matmul(G, Lambda.T) / rho2), Lambda
    )


def getprimal_weights(D: torch.Tensor, G: torch.Tensor, rho1: float) -> torch.Tensor:
    return F.softmax(-torch.matmul(D.T, G) / rho1, dim=0)


def getprimal_dict(Lambda: torch.Tensor, G: torch.Tensor, rho2: float) -> torch.Tensor:
    return F.softmax(-torch.matmul(G, Lambda.T) / rho2, dim=0)


def solve_weights(
    X: torch.Tensor,
    K: torch.Tensor,
    eps: float,
    D: torch.Tensor,
    rho1: float,
    use_gpu: bool = False,
) -> torch.Tensor:
    device = torch.device("cuda" if use_gpu else "cpu")
    X = X.to(device)
    K = K.to(device)
    D = D.to(device)

    G = torch.zeros_like(X, device=device)  # Initialize dual variable
    G.requires_grad_(True)  # Ensure G has gradient tracking enabled

    def closure():
        optimizer.zero_grad()
        loss = dual_obj_weights(X, K, eps, D, G, rho1)
        loss.backward(
            retain_graph=True
        )  # Keep the graph for subsequent backward passes
        return loss

    optimizer = LBFGS(
        [G], max_iter=250, tolerance_grad=1e-4, line_search_fn="strong_wolfe"
    )
    optimizer.step(closure)

    return getprimal_weights(D, G, rho1)


def solve_dict(
    X: torch.Tensor,
    K: torch.Tensor,
    eps: float,
    Lambda: torch.Tensor,
    rho2: float,
    use_gpu: bool = False,
) -> torch.Tensor:
    device = torch.device("cuda" if use_gpu else "cpu")
    X = X.to(device)
    K = K.to(device)
    Lambda = Lambda.to(device)

    G = torch.zeros_like(
        X, device=device
    )  # Initialize dual variable. Note: This G is *different* from the G in solve_weights
    G.requires_grad_(True)  # Ensure G has gradient tracking enabled

    def closure():
        optimizer.zero_grad()
        loss = dual_obj_dict(X, K, eps, Lambda, G, rho2)
        loss.backward(
            retain_graph=True
        )  # Keep the graph for subsequent backward passes
        return loss

    optimizer = LBFGS(
        [G], max_iter=250, tolerance_grad=1e-4, line_search_fn="strong_wolfe"
    )
    optimizer.step(closure)

    return getprimal_dict(Lambda, G, rho2)


def wasserstein_nmf(
    X: torch.Tensor,
    K: torch.Tensor,
    k: int,
    eps: float = 0.025,
    rho1: float = 0.05,
    rho2: float = 0.05,
    n_iter: int = 10,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = X.shape
    D = torch.rand(m, k)
    simplex_norm_(D, dim=0)
    Lambda = torch.rand(k, n)
    simplex_norm_(Lambda, dim=0)

    for iter_num in range(n_iter):
        if verbose:
            print(f"Wasserstein-NMF: iteration {iter_num + 1}")

        D = solve_dict(X, K, eps, Lambda, rho2)
        Lambda = solve_weights(X, K, eps, D, rho1)

    return D, Lambda


def wasserstein_nmf_gpu(
    X: torch.Tensor,
    K: torch.Tensor,
    k: int,
    eps: float = 0.025,
    rho1: float = 0.05,
    rho2: float = 0.05,
    n_iter: int = 10,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot run wasserstein_nmf_gpu.")

    device = torch.device("cuda")
    X = X.to(device)
    K = K.to(device)

    m, n = X.shape
    D = torch.rand(m, k, device=device)
    simplex_norm_(D, dim=0)
    Lambda = torch.rand(k, n, device=device)
    simplex_norm_(Lambda, dim=0)

    for iter_num in range(n_iter):
        if verbose:
            print(f"Wasserstein-NMF (GPU): iteration {iter_num + 1}")
        D = solve_dict(X, K, eps, Lambda, rho2, use_gpu=True)
        Lambda = solve_weights(X, K, eps, D, rho1, use_gpu=True)

    return D.cpu(), Lambda.cpu()


# Example Usage (assuming you've saved the above code as wnmf.py):
if __name__ == "__main__":
    import torch

    # Example data
    m = 50  # Number of rows in X
    n = 100  # Number of columns in X
    k = 3  # Number of components
    X = torch.rand(m, n)
    X = X / X.sum(dim=0, keepdim=True)  # ensure columns of X sum to 1

    # Create a cost matrix (example: squared Euclidean distance)
    coords = torch.randn(m, 2)  # Example: 2D coordinates for each row of X
    K = torch.cdist(coords, coords) ** 2
    K = torch.exp(-K / 0.1)  # Gibbs Kernel

    # Move data to GPU
    X_cuda = X.cuda()
    K_cuda = K.cuda()

    # Run GPU version
    D_torch, Lambda_torch = wasserstein_nmf_gpu(X_cuda, K_cuda, k)

    # Print Results (they are already on the CPU)
    print("D:", D_torch)
    print("Lambda:", Lambda_torch)

    # Verify reconstruction (should be close to the original X)
    X_reconstructed = torch.matmul(D_torch, Lambda_torch)
    print(
        "\nReconstruction Error (GPU):", torch.norm(X.cpu() - X_reconstructed)
    )  # compare with X on the CPU
