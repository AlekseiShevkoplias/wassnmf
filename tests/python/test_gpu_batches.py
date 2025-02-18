import torch
import pytest
# from WassersteinDiL import WassersteinDiL  # Assuming your code is in WassersteinDiL.py
from wassnmf.wassdil import WassersteinDiL

@pytest.fixture
def model():
    return WassersteinDiL()

@pytest.fixture
def data():
    X = torch.rand(20, 10)
    K = torch.ones(10, 10)
    k = 3
    eps = 0.025
    rho1 = 0.05
    rho2 = 0.05
    n_iter = 3  # Reduced iterations for faster testing
    batch_size = 4
    return X, K, k, eps, rho1, rho2, n_iter, batch_size

@pytest.mark.parametrize("device", [torch.device('cpu'), torch.device('cuda')])
def test_no_batch_vs_batch(model, data, device):
    if device == torch.device('cuda') and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    X, K, k, eps, rho1, rho2, n_iter, batch_size = data

    D_no_batch, Lambda_no_batch = model.fit(
        X, K, k, eps, rho1, rho2, n_iter, device=device, batch_size=None
    )
    D_batch, Lambda_batch = model.fit(
        X, K, k, eps, rho1, rho2, n_iter, device=device, batch_size=batch_size
    )

    assert torch.allclose(D_no_batch, D_batch, atol=1e-5)
    assert torch.allclose(Lambda_no_batch, Lambda_batch, atol=1e-5)

@pytest.mark.parametrize("device", [torch.device('cpu'), torch.device('cuda')])
def test_batch_size_1(model, data, device):
    if device == torch.device('cuda') and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    X, K, k, eps, rho1, rho2, n_iter, _ = data  # Don't need batch_size here

    D_batch1, Lambda_batch1 = model.fit(
        X, K, k, eps, rho1, rho2, n_iter, device=device, batch_size=1
    )
    D_no_batch, Lambda_no_batch = model.fit(
        X, K, k, eps, rho1, rho2, n_iter, device=device, batch_size=None
    )

    assert torch.allclose(D_batch1, D_no_batch, atol=1e-5)
    assert torch.allclose(Lambda_batch1, Lambda_no_batch, atol=1e-5)