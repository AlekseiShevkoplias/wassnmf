import numpy as np
import ot  # from the "pot" library

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
            col_orig,          # histogram 1
            col_reco,          # histogram 2
            cost_matrix,       # ground-cost matrix
            reg,               # entropic regularization
            method='sinkhorn', # or 'sinkhorn_log' etc.
        )
        
        total_dist += dist_j
    
    return total_dist / n_samples
