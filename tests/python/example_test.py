import numpy as np
import pytest
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from wassnmf.wassnmf import WassersteinNMF

def test_julia_notebook_analog():
    """
    Reproduce the exact steps from the Julia notebook example with additional
    comparison to standard NMF and timing measurements.
    """
    # Data generation parameters exactly matching Julia
    n_features = 100
    n_samples = 100
    coord = np.linspace(-12, 12, n_features)
    X = np.zeros((n_features, n_samples))
    sigma = 1.0
    
    # Set same random seed as Julia example
    np.random.seed(42)
    
    # Generate data exactly as in Julia notebook
    for i in range(n_samples):
        X[:, i] = (
            np.random.rand() * np.exp(-(coord - (sigma * np.random.randn() + 6)) ** 2) +
            np.random.rand() * np.exp(-(coord - sigma * np.random.randn()) ** 2) +
            np.random.rand() * np.exp(-(coord - (sigma * np.random.randn() - 6)) ** 2)
        )
    
    # Normalize columns to probability simplex (no epsilon added)
    X /= X.sum(axis=0, keepdims=True)
    
    # Create cost matrix and kernel exactly as in Julia
    C = pairwise_distances(coord.reshape(-1, 1), metric="sqeuclidean")
    C /= np.mean(C)  # Same normalization as Julia
    eps = 0.025      # Same epsilon as Julia notebook
    K = np.exp(-C / eps)
    
    print("\nRunning WassersteinNMF...")
    start_time = time.time()
    # Initialize and run WassersteinNMF
    wnmf = WassersteinNMF(
        n_components=3,
        epsilon=eps,    # Match Julia exactly
        rho1=0.05,      # Match Julia exactly  
        rho2=0.05,      # Match Julia exactly
        n_iter=10,
        verbose=True
    )
    
    D_wass, Lambda_wass = wnmf.fit_transform(X, K)
    wass_time = time.time() - start_time
    print(f"WassersteinNMF completed in {wass_time:.2f} seconds")
    
    print("\nRunning standard NMF...")
    start_time = time.time()
    # Run standard NMF for comparison
    nmf = NMF(
        n_components=3, 
        init='random',
        random_state=42,
        max_iter=200
    )
    W_standard = nmf.fit_transform(X)
    H_standard = nmf.components_
    standard_time = time.time() - start_time
    print(f"Standard NMF completed in {standard_time:.2f} seconds")
    
    # Validate WassNMF results
    assert D_wass.shape == (n_features, 3)
    assert Lambda_wass.shape == (3, n_samples)
    assert np.all(D_wass >= 0)
    assert np.all(Lambda_wass >= 0)
    assert np.allclose(D_wass.sum(axis=0), 1.0, atol=1e-4)
    assert np.allclose(Lambda_wass.sum(axis=0), 1.0, atol=1e-4)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Wasserstein NMF components
    for i in range(D_wass.shape[1]):
        ax1.plot(coord, D_wass[:, i], label=f"Component {i+1}")
    ax1.set_title(f"Wasserstein NMF Components\nTime: {wass_time:.2f}s")
    ax1.set_xlabel("Coordinate")
    ax1.set_ylabel("Magnitude")
    ax1.legend()
    ax1.grid(True)
    
    # Plot standard NMF components (normalize for comparison)
    W_normalized = W_standard / W_standard.sum(axis=0)
    for i in range(W_normalized.shape[1]):
        ax2.plot(coord, W_normalized[:, i], label=f"Component {i+1}")
    ax2.set_title(f"Standard NMF Components\nTime: {standard_time:.2f}s")
    ax2.set_xlabel("Coordinate")
    ax2.set_ylabel("Magnitude")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('nmf_comparison.png')
    plt.close()
    
    # Create matrix visualizations
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    
    # Original Data Matrix
    im0 = axs[0, 0].imshow(X, aspect='auto', cmap='viridis')
    axs[0, 0].set_title("Original Data Matrix X")
    axs[0, 0].set_xlabel("Samples")
    axs[0, 0].set_ylabel("Features")
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Empty plot for symmetry
    axs[0, 1].remove()
    
    # Wasserstein NMF results
    im1 = axs[1, 0].imshow(D_wass, aspect='auto', cmap='viridis')
    axs[1, 0].set_title(f"Wasserstein Dict Matrix D\nTime: {wass_time:.2f}s")
    axs[1, 0].set_xlabel("Components")
    axs[1, 0].set_ylabel("Features")
    plt.colorbar(im1, ax=axs[1, 0])
    
    im2 = axs[1, 1].imshow(Lambda_wass, aspect='auto', cmap='viridis')
    axs[1, 1].set_title("Wasserstein Weight Matrix Λ")
    axs[1, 1].set_xlabel("Samples")
    axs[1, 1].set_ylabel("Components")
    plt.colorbar(im2, ax=axs[1, 1])
    
    # Standard NMF results
    im3 = axs[2, 0].imshow(W_standard, aspect='auto', cmap='viridis')
    axs[2, 0].set_title(f"Standard NMF W Matrix\nTime: {standard_time:.2f}s")
    axs[2, 0].set_xlabel("Components")
    axs[2, 0].set_ylabel("Features")
    plt.colorbar(im3, ax=axs[2, 0])
    
    im4 = axs[2, 1].imshow(H_standard, aspect='auto', cmap='viridis')
    axs[2, 1].set_title("Standard NMF H Matrix")
    axs[2, 1].set_xlabel("Samples")
    axs[2, 1].set_ylabel("Components")
    plt.colorbar(im4, ax=axs[2, 1])
    
    plt.tight_layout()
    plt.savefig('matrix_comparison.png')
    plt.close()
    
    # Print reconstruction errors
    wass_recon = D_wass @ Lambda_wass
    standard_recon = W_standard @ H_standard
    
    wass_error = np.mean((X - wass_recon) ** 2)
    standard_error = np.mean((X - standard_recon) ** 2)
    
    print("\nReconstruction Errors:")
    print(f"Wasserstein NMF: {wass_error:.6f}")
    print(f"Standard NMF: {standard_error:.6f}")

if __name__ == "__main__":
    test_julia_notebook_analog()