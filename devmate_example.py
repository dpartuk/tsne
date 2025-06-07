#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST t-SNE Visualization
This script uses t-SNE to visualize the MNIST dataset in 2D space.
It includes both scikit-learn's implementation and a custom implementation.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_mnist(n_samples=None):
    """
    Load and preprocess the MNIST dataset.

    Args:
        n_samples: Number of samples to load (None for all)

    Returns:
        X: Image data (samples x features)
        y: Labels
    """
    print("Loading MNIST dataset...")
    # Fetch data from OpenML
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    # Limit samples if specified
    if n_samples is not None:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    # Normalize features
    X = StandardScaler().fit_transform(X)

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y


class LocalTSNE:
    """
    Custom t-SNE implementation from scratch.

    t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction
    technique that visualizes high-dimensional data by giving each datapoint a location
    in a 2D or 3D map.
    """

    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        learning_rate=200.0,
        early_exaggeration=12.0,
        n_iter=1000,
        random_state=None,
    ):
        """
        Initialize the t-SNE parameters.

        Args:
            n_components: Dimension of the embedded space (typically 2 or 3)
            perplexity: Related to the number of nearest neighbors used in manifold learning
            learning_rate: The learning rate for gradient descent
            early_exaggeration: Coefficient to exaggerate the distances in early iterations
            n_iter: Maximum number of iterations for optimization
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.early_exaggeration = early_exaggeration
        self.n_iter = n_iter
        self.random_state = random_state

    def _compute_pairwise_distances(self, X):
        """
        Compute pairwise Euclidean distances between points.

        Args:
            X: Input data matrix (n_samples, n_features)

        Returns:
            distances: Pairwise distance matrix (n_samples, n_samples)
        """
        # Calculate pairwise squared Euclidean distances
        distances = squareform(pdist(X, "sqeuclidean"))
        return distances

    def _compute_joint_probabilities(self, distances):
        """
        Compute joint probabilities p_ij from distances using Gaussian kernel.

        Args:
            distances: Pairwise distance matrix

        Returns:
            P: Joint probability matrix
        """
        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))

        # Target entropy of the conditional distribution
        target_entropy = np.log(self.perplexity)

        # For each point, find the optimal sigma (precision) to achieve target perplexity
        for i in range(n_samples):
            # Set diagonal to infinity to exclude self-similarity
            distances_i = distances[i].copy()
            distances_i[i] = np.inf

            # Binary search for the precision (sigma) that gives desired perplexity
            sigma_min = 1e-10
            sigma_max = 1e10
            sigma = 1.0

            for _ in range(50):  # Maximum binary search iterations
                # Compute conditional probabilities with current sigma
                exp_distances = np.exp(-distances_i / (2 * sigma**2))
                sum_exp_distances = np.sum(exp_distances)

                if sum_exp_distances == 0:
                    # Avoid division by zero
                    p_i = np.zeros_like(distances_i)
                else:
                    p_i = exp_distances / sum_exp_distances

                # Calculate entropy of the distribution
                entropy = -np.sum(p_i * np.log2(p_i + 1e-10))

                # Adjust sigma based on entropy comparison
                if np.abs(entropy - target_entropy) < 1e-5:
                    break

                if entropy < target_entropy:
                    sigma_min = sigma
                    sigma = (sigma + sigma_max) / 2.0
                else:
                    sigma_max = sigma
                    sigma = (sigma + sigma_min) / 2.0

            # Store the conditional probabilities for point i
            P[i] = p_i

        # Symmetrize the probability matrix and normalize
        P = (P + P.T) / (2 * n_samples)

        # Ensure minimum probability to avoid numerical issues
        P = np.maximum(P, 1e-12)

        return P

    def _compute_q_matrix(self, Y):
        """
        Compute the Q matrix (joint probabilities in low-dimensional space)
        using t-distribution.

        Args:
            Y: Low-dimensional embedding

        Returns:
            Q: Joint probability matrix in low-dimensional space
        """
        # Compute squared Euclidean distances
        distances = self._compute_pairwise_distances(Y)

        # Convert distances to joint probabilities using t-distribution
        # The t-distribution with one degree of freedom (Student's t-distribution)
        # has heavier tails than the Gaussian, which helps alleviate the crowding problem
        inv_distances = 1.0 / (1.0 + distances)

        # Set diagonal to zero (no self-interactions)
        np.fill_diagonal(inv_distances, 0.0)

        # Normalize to get probabilities
        Q = inv_distances / np.sum(inv_distances)

        # Ensure minimum probability to avoid numerical issues
        Q = np.maximum(Q, 1e-12)

        return Q

    def fit_transform(self, X):
        """
        Apply t-SNE to the data.

        Args:
            X: Input data matrix (n_samples, n_features)

        Returns:
            Y: Embedded data in low-dimensional space (n_samples, n_components)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]

        # Compute pairwise distances and joint probabilities
        distances = self._compute_pairwise_distances(X)
        P = self._compute_joint_probabilities(distances)

        # Apply early exaggeration
        P *= self.early_exaggeration

        # Initialize the embedding randomly
        Y = np.random.normal(0, 1e-4, (n_samples, self.n_components))

        # Initialize optimization variables
        Y_incs = np.zeros_like(Y)

        # Gradient descent loop
        for iteration in range(self.n_iter):
            # Compute Q matrix
            Q = self._compute_q_matrix(Y)

            # Compute gradients
            grad = np.zeros_like(Y)

            for i in range(n_samples):
                # Difference between P and Q
                diff = (P[i] - Q[i]).reshape(-1, 1)

                # Calculate repulsive forces
                y_diff = Y[i] - Y
                inv_dist = 1.0 / (1.0 + np.sum(y_diff**2, axis=1)).reshape(-1, 1)

                # Update gradients
                grad[i] = 4 * np.sum(diff * inv_dist * y_diff, axis=0)

            # Update embedding with momentum and learning rate
            Y_incs = 0.9 * Y_incs - self.learning_rate * grad
            Y += Y_incs

            # Center the embedding to avoid drifting
            Y -= np.mean(Y, axis=0)

            # Remove early exaggeration after 250 iterations
            if iteration == 250:
                P /= self.early_exaggeration

            # Print progress
            if iteration % 50 == 0:
                cost = np.sum(P * np.log(P / Q))
                print(f"Iteration {iteration}: error = {cost:.5f}")

        return Y


def visualize_tsne(X_embedded, y, title, filename=None):
    """
    Visualize the t-SNE embedding.

    Args:
        X_embedded: t-SNE embedding (n_samples, 2)
        y: Labels
        title: Plot title
        filename: Output filename for the saved plot (None to not save)
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="tab10", alpha=0.8
    )
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()


def main():
    """
    Main function to run the t-SNE algorithm on MNIST.
    """
    # Load data (using a subset for speed)
    X, y = load_mnist(n_samples=5000)

    # Configure t-SNE parameters
    perplexity = 30.0
    learning_rate = 200.0
    exaggeration = 12.0

    # Initialize and run t-SNE
    print("Running t-SNE...")
    start_time = time.time()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=exaggeration,
        n_iter=1000,
        random_state=42,
    )

    X_embedded_sklearn = tsne.fit_transform(X)
    print(f"Scikit-learn t-SNE completed in {time.time() - start_time:.2f} seconds")

    # Visualize scikit-learn results
    visualize_tsne(
        X_embedded_sklearn,
        y,
        f"Scikit-learn t-SNE visualization of MNIST (perplexity={perplexity}, "
        f"learning_rate={learning_rate})",
        "tsne_sklearn_mnist.png",
    )

    # Run custom t-SNE implementation
    print("Running custom t-SNE implementation...")
    start_time = time.time()

    local_tsne = LocalTSNE(
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=exaggeration,
        n_iter=1000,
        random_state=42,
    )
    X_embedded_custom = local_tsne.fit_transform(X)
    print(f"Custom t-SNE completed in {time.time() - start_time:.2f} seconds")

    # Visualize custom implementation results
    visualize_tsne(
        X_embedded_custom,
        y,
        f"Custom t-SNE visualization of MNIST (perplexity={perplexity}, "
        f"learning_rate={learning_rate}, early_exaggeration={exaggeration})",
        "tsne_custom_mnist.png",
    )


if __name__ == "__main__":
    main()