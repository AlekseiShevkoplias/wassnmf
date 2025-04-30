import numpy as np
import ot
import torch

def tensor_to_npy(tensor, save_path):
    numpy_array = tensor.detach().cpu().numpy()

    np.save(save_path, numpy_array)

    print(f"Tensor saved as NumPy array to {save_path}")
    return numpy_array

def downsample_grid(X, original_size=50, target_size=10, method="mean"):
    """
    Downsample a grid of size original_size x original_size to target_size x target_size
    using either mean or max pooling.

    Parameters
    ----------
    X : ndarray of shape (original_size * original_size, n_samples)
        Input data to downsample. Each column is a sample represented as a flattened grid.
    original_size : int
        Original size of the grid (assumed to be square).
    target_size : int
        Target size of the downsampled grid (assumed to be square).
    method : str
        Pooling method, either 'mean' or 'max'.

    Returns
    -------
    X_downsampled : ndarray of shape (target_size * target_size, n_samples)
        Downsampled data.
    """
    assert original_size % target_size == 0, "Target size must evenly divide original size"
    factor = original_size // target_size

    n_samples = X.shape[1]
    X_downsampled = np.zeros((target_size * target_size, n_samples))

    for idx in range(n_samples):
        grid = X[:, idx].reshape(original_size, original_size)
        pooled = np.zeros((target_size, target_size))
        for i in range(target_size):
            for j in range(target_size):
                patch = grid[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
                pooled[i, j] = patch.mean() if method == "mean" else patch.max()
        X_downsampled[:, idx] = pooled.flatten()

    return X_downsampled


def compute_entropic_wasserstein_distance(X, X_hat, cost_matrix, reg=0.025):
    """
    Compute average Sinkhorn OT distance between columns of X and X_hat
    using the cost_matrix and entropic regularization 'reg'.

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)
    X_hat : ndarray of shape (n_features, n_samples)
    cost_matrix : ndarray of shape (n_features, n_features)
    reg : float
        Entropic regularization strength (same as epsilon in sinkhorn).

    Returns
    -------
    float
        Average OT distance across the columns.
    """
    n_samples = X.shape[1]
    total_dist = 0.0

    for j in range(n_samples):
        # column j
        col_orig = X[:, j].astype(np.float64)
        col_reco = X_hat[:, j].astype(np.float64)

        # Make sure each column sums to 1 (POT expects distributions)
        # If your data is already on the simplex, you can skip normalization
        sum_orig = col_orig.sum()
        sum_reco = col_reco.sum()

        if sum_orig > 0:
            col_orig /= sum_orig
        if sum_reco > 0:
            col_reco /= sum_reco

        # ot.sinkhorn2 returns the regularized OT cost
        # (the "squared" cost by default, but with cost_matrix that is just cost*g).
        dist_j = ot.sinkhorn2(
            col_orig,  # histogram 1
            col_reco,  # histogram 2
            cost_matrix,  # ground-cost matrix
            reg,  # entropic regularization
            method="sinkhorn",  # or 'sinkhorn_log' etc.
        )

        total_dist += dist_j

    return total_dist / n_samples
