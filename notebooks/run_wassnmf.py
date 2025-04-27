import sys

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

sys.path.insert(0, "../src")
from wassnmf.wassnmf import WassersteinNMF


def tensor_to_npy(tensor, save_path):
    numpy_array = tensor.detach().cpu().numpy()

    np.save(save_path, numpy_array)

    print(f"Tensor saved as NumPy array to {save_path}")
    return numpy_array


X = pd.read_csv("../data/activity_for_wassDL_recent.csv", header=None).values

grid_size = 50
x = np.arange(grid_size)
y = np.arange(grid_size)
_X, _Y = np.meshgrid(x, y)
locations = np.column_stack([_X.flatten(), _Y.flatten()])

C = pairwise_distances(locations, metric="sqeuclidean")

eps = 0.025
K = np.exp(-C / eps)

print("Locations shape:", locations.shape)
print("C shape:", C.shape)
print("K shape:", K.shape)


wnmf = WassersteinNMF(
    n_components=16, epsilon=eps, rho1=0.1, rho2=0.1, n_iter=20, verbose=True
)
print("wnmf initialized, proceeding to fit transform")

D, Lambda = wnmf.fit_transform(X, K)
