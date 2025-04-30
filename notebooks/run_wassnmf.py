import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform, pdist

sys.path.insert(0, "../src")
from wassnmf.wassnmf import WassersteinNMF
from wassnmf.utils import downsample_grid, tensor_to_npy
import matplotlib.pyplot as plt

TARGET_SIZE = 5

os.makedirs("wassnmf_results", exist_ok=True)

X = pd.read_csv("../data/activity_for_wassDL_recent.csv", header=None).values

X = downsample_grid(
    X, original_size=50, target_size=TARGET_SIZE, method="mean"
)

X = X / np.sum(X, axis=0, keepdims=True)

grid_size = TARGET_SIZE

C = pairwise_distances(X, metric="correlation")


eps = 0.025
K = np.exp(-C / eps)
K = np.clip(K, 0, 1)
print("C shape:", C.shape)
print("K shape:", K.shape)
print("C min:", C.min(), "C max:", C.max(), "C median:", np.median(C))
print("K min:", K.min(), "K max:", K.max(), "K median:", np.median(K))


print(np.isnan(X).any(), np.isinf(X).any(), np.max(np.abs(X)))
print(np.isnan(K).any(), np.isinf(K).any(), np.max(np.abs(K)))

# Visualize X, K, and C
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(X, cmap="gray")
axes[0].set_title("X")
axes[2].imshow(K, cmap="gray")
axes[2].set_title("K")
axes[1].imshow(C, cmap="gray")
axes[1].set_title("C")
plt.show()


wnmf = WassersteinNMF(
    n_components=16, epsilon=eps, rho1=0.1, rho2=0.1, n_iter=10, verbose=True
)
print("wnmf initialized, proceeding to fit transform")

D, Lambda = wnmf.fit_transform(X, K)

tensor_to_npy(D, f"wassnmf_results/D_{TARGET_SIZE}.npy")
tensor_to_npy(Lambda, f"wassnmf_results/Lambda_{TARGET_SIZE}.npy")

# plot D and Lambda



# subplots D and Lambda
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(D, cmap="inferno")
axes[0].set_title("D")
axes[1].imshow(Lambda, cmap="inferno")
axes[1].set_title("Lambda")
# plt.colorbar(axes[0].imshow(D[:, 0].reshape(grid_size, grid_size), cmap="gray"), ax=axes[0])
# plt.colorbar(axes[1].imshow(Lambda[:, 0].reshape(grid_size, grid_size), cmap="gray"), ax=axes[1])
plt.show()
