"""
t-SNE Visualization of MNIST Dataset

This script demonstrates the use of t-SNE (t-Distributed Stochastic Neighbor Embedding)
for visualizing high-dimensional MNIST digits data in 2D space.

Usage:
    python tsne_mnist.py [options]

Options:
    --n-samples INT         Number of samples to use (default: 10000)
    --perplexity FLOAT     Perplexity value for main visualization (default: 30)
    --compare-perplexity   Enable perplexity comparison visualization
    --perplexity-values    List of perplexity values for comparison (default: 5,30,50,100)
    --output PATH          Save visualizations to specified path
    --use-local           Use local t-SNE implementation (warning: significantly slower)
    --learning-rate       Learning rate for local t-SNE implementation (default: 200.0)

Examples:
    python tsne_mnist.py --n-samples 5000 --perplexity 40
    python tsne_mnist.py --compare-perplexity --perplexity-values 10,20,30,40
    python tsne_mnist.py --output tsne_plot.png
    python tsne_mnist.py --use-local --learning-rate 150.0 --n-samples 1000

t-SNE is a dimensionality reduction technique that is particularly well suited for
visualizing high-dimensional data. It works by converting similarities between data
points to joint probabilities and tries to minimize the Kullback-Leibler divergence
between the joint probabilities of the low-dimensional embedding and the high-dimensional data.

Key t-SNE Parameters:
- implementation: Can use sklearn's optimized version or a local implementation for
  educational purposes. The local implementation is significantly slower, especially
  for larger datasets, but helps in understanding the algorithm.
- perplexity: Balance between local and global structure (typical values: 5-50)
  Lower values focus on local structure, higher values on global structure
- n_iter: Maximum number of iterations for optimization (default: 1000)
- learning_rate: Usually in range [10-1000], if too high, data may look like 'ball'
- early_exaggeration: How tight natural clusters are in the embedding space
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# Parse command-line arguments
parser = argparse.ArgumentParser(description="t-SNE visualization of MNIST dataset")
parser.add_argument(
    "--n-samples", type=int, default=10000, help="Number of samples to use"
)
parser.add_argument(
    "--perplexity",
    type=float,
    default=30,
    help="Perplexity value for main visualization",
)
parser.add_argument(
    "--compare-perplexity",
    action="store_true",
    help="Enable perplexity comparison visualization",
)
parser.add_argument(
    "--perplexity-values",
    type=str,
    default="5,30,50,100",
    help="Comma-separated list of perplexity values for comparison",
)
parser.add_argument("--output", type=str, help="Path to save visualization")
parser.add_argument(
    "--use-local", action="store_true", help="Use local t-SNE implementation"
)
parser.add_argument(
    "--learning-rate", type=float, default=200.0, help="Learning rate for local t-SNE"
)

args = parser.parse_args()


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


def tsne_local(X, n_components=2, perplexity=30.0, n_iter=1000, learning_rate=200.0):
    """
    Simplified t-SNE implementation for educational purposes.

    Warning: This implementation stores the full N×N pairwise distance matrix in memory.
    Memory usage grows quadratically with the number of samples. For datasets with
    N > 10000 samples, this may require significant RAM (>1GB). For large datasets,
    consider using sklearn.manifold.TSNE instead.

    Parameters:
    - X: Input data matrix (n_samples, n_features)
    - n_components: Dimension of the embedded space (default: 2)
    - perplexity: Balance between local and global structure (default: 30.0)
    - n_iter: Number of iterations (default: 1000)
    - learning_rate: Learning rate for gradient descent (default: 200.0)

    Returns:
    - Y: Embedded coordinates (n_samples, n_components)
    """
    n_samples = X.shape[0]

    # Step 1: Compute pairwise distances
    distances = squareform(pdist(X, metric="euclidean"))
    distances_squared = distances**2

    # Step 2: Convert distances to probabilities (P matrix)
    P = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        beta = binary_search_beta(distances_squared[i], perplexity)
        P[i] = np.exp(-distances_squared[i] * beta)
        P[i, i] = 0
        P[i] /= np.sum(P[i])

    # Symmetrize P matrix
    P = (P + P.T) / (2 * n_samples)
    P = np.maximum(P, 1e-12)

    # Step 3: Initialize low-dimensional representation
    rng = np.random.RandomState(42)
    Y = rng.normal(0, 1e-4, (n_samples, n_components))

    # Step 4: Gradient descent
    for iteration in range(n_iter):
        # Compute Q matrix (low-dimensional affinities)
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n_samples), range(n_samples)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradients
        PQ = P - Q
        grad = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            grad[i] = 4 * np.sum(
                np.tile(PQ[:, i] * num[:, i], (n_components, 1)).T * (Y[i] - Y), 0
            )

        # Update Y
        Y = Y - learning_rate * grad

        # Show progress every 10 iterations
        if iteration % 10 == 0:
            progress = (iteration + 1) / n_iter * 100
            print(f"\rProgress: [{iteration + 1}/{n_iter}] {progress:.1f}%", end="")
    print()  # New line after progress bar

    return Y


# ------------------------------------------------------------------------------
# Main execution code
# ------------------------------------------------------------------------------

# Load MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Use subset of data for faster computation
if args.n_samples < X.shape[0]:
    rng = np.random.RandomState(42)
    indices = rng.choice(X.shape[0], args.n_samples, replace=False)
    X = X[indices]
    y = y[indices]

# Preprocess the data by scaling to [0,1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction
print("Applying t-SNE transformation...")
start_time = time.time()
if args.use_local:
    print("Using local t-SNE implementation...")
    X_tsne = tsne_local(
        X_scaled,
        n_components=2,
        perplexity=args.perplexity,
        n_iter=1000,
        learning_rate=args.learning_rate,
    )
else:
    tsne = TSNE(
        n_components=2, perplexity=args.perplexity, random_state=42, n_iter=1000
    )
    X_tsne = tsne.fit_transform(X_scaled)
end_time = time.time()
print(f"t-SNE completed in {end_time - start_time:.2f} seconds")

# Create the visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10")
plt.colorbar(scatter)
plt.title("t-SNE visualization of MNIST digits")
plt.xlabel("First t-SNE dimension")
plt.ylabel("Second t-SNE dimension")
plt.legend(*scatter.legend_elements(), title="Digits")

if args.output:
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
plt.show()


def plot_tsne_perplexity_comparison(
    X, y, perplexities=[5, 30, 50, 100], save_path=None
):
    """
    Creates multiple t-SNE visualizations with different perplexity values.

    Lower perplexity values (5-10):   Focus on local structure, tight clusters
    Medium perplexity values (30-50): Balance between local and global structure
    Higher perplexity values (100+):  Emphasis on global data patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    for idx, perp in enumerate(perplexities):
        print(f"Computing t-SNE with perplexity {perp}...")
        start_time = time.time()

        if args.use_local and idx == 0:
            print("Warning: Local t-SNE implementation may be significantly slower!")

        if not args.use_local:
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
            X_tsne = tsne.fit_transform(X_scaled)
        else:
            X_tsne = tsne_local(
                X_scaled, perplexity=perp, learning_rate=args.learning_rate
            )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.2f} seconds")

        scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=30)
        axes[idx].legend(*scatter.legend_elements(), title="Digits", loc="best")
        axes[idx].set_title(f"Perplexity: {perp}")
        axes[idx].set_xlabel("First t-SNE dimension")
        axes[idx].set_ylabel("Second t-SNE dimension")

    plt.suptitle("Effect of Perplexity on t-SNE Visualization", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Compare different perplexity values if requested
if args.compare_perplexity:
    perplexity_values = [float(p) for p in args.perplexity_values.split(",")]
    plot_tsne_perplexity_comparison(
        X_scaled, y, perplexities=perplexity_values, save_path=args.output
    )

    # Conclusions about perplexity effects:
    print("\nConclusions from perplexity comparison:")
    print(
        "- Low perplexity (5): Creates tighter clusters but may miss global structure"
    )
    print(
        "- Medium perplexity (30-50): Provides good balance between local and global patterns"
    )
    print(
        "- High perplexity (100): Emphasizes global relationships but may blur local details"
    )
    print("- Higher perplexity values generally result in more dispersed clusters")

print("\nScript completed successfully!")