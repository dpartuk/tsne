
import time

import matplotlib.pyplot as plt
import numpy as np
# from scipy.spatial.distance import pdist, squareform
# from sklearn.datasets import fetch_openml
# from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
# from tsne import tsne_local




def binary_search_beta(distances, target_perplexity, tol=1e-5, max_iter=50):
    """Find the optimal beta value (precision) for a desired perplexity.

    This function implements binary search to find the precision parameter (beta)
    that achieves the desired perplexity. The perplexity can be interpreted as a
    smooth measure of the effective number of neighbors. The beta value determines
    how quickly the probability distribution falls off with distance.

    Parameters:
    - distances: Array of squared distances to convert to probabilities
    - target_perplexity: Desired perplexity value (typically between 5 and 50)
    - tol: Tolerance for the binary search convergence
    - max_iter: Maximum number of binary search iterations

    Returns:
    - beta: Optimal precision value that achieves the target perplexity
    """
    beta_min = -np.inf
    beta_max = np.inf
    beta = 1.0

    for _ in range(max_iter):
        p = np.exp(-distances * beta)
        p[p < 1e-12] = 1e-12
        sum_p = np.sum(p)
        if sum_p == 0:
            H = 0
        else:
            H = np.log(sum_p) + beta * np.sum(distances * p) / sum_p

        Hdiff = H - np.log(target_perplexity)
        if np.abs(Hdiff) < tol:
            break

        if Hdiff > 0:
            beta_min = beta
            beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
        else:
            beta_max = beta
            beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

    return beta



# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

# def apply_tsne(X, perplexity=30.0, exaggeration=12.0):
#     """
#     Returns:
#         X_embedded: Low-dimensional representation of the data
#     """
#     print(
#         f"Applying t-SNE with perplexity={perplexity}, exaggeration={exaggeration}..."
#     )
#
#     # Initialize t-SNE with parameters
#     # Perplexity: Balance between local and global aspects of the data
#     # Exaggeration: Controls how tight natural clusters are in the embedding space
#     tsne = TSNE(
#         n_components=2,  # 2D embedding
#         perplexity=perplexity,
#         early_exaggeration=exaggeration,
#         learning_rate="auto",
#         n_iter=1000,
#         random_state=42,
#     )
#
#     # Apply dimensionality reduction
#     # t-SNE algorithm steps:
#     # 1. Compute pairwise similarities in high-dimensional space using Gaussian kernel
#     # 2. Define target similarities in low-dimensional space using t-distribution
#     # 3. Minimize KL divergence between these distributions through gradient descent
#     X_embedded = tsne.fit_transform(X)
#
#     return X_embedded

