import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from tsne_local import tsne_local


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

def plot_tsne_perplexity_comparison(
    args, X_scaled, y, perplexities=[5, 30, 50, 100], save_path=None):
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
                X_scaled,
                perplexity=perp,
                n_iter=args.n_iterations,
                learning_rate=args.learning_rate
            )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.2f} seconds")

        encoded_y = encode_colors(y)
        scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=encoded_y, cmap="tab10", s=30)
        axes[idx].legend(*scatter.legend_elements(), title="Digits", loc="best")
        axes[idx].set_title(f"Perplexity: {perp}")
        axes[idx].set_xlabel("First t-SNE dimension")
        axes[idx].set_ylabel("Second t-SNE dimension")

    plt.suptitle("Effect of Perplexity on t-SNE Visualization", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def read_args():
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
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="Perplexity value for main visualization",
    )
    parser.add_argument("--output", type=str, help="Path to save visualization")
    parser.add_argument(
        "--use-local", action="store_true", help="Use local t-SNE implementation"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=200.0, help="Learning rate for local t-SNE"
    )

    args = parser.parse_args()
    print("Args type: ", type(args))
    print("Args: ", args)
    return args

def compare_perplexity(args, X_scaled, y):
    # Compare different perplexity values if requested
    if args.compare_perplexity:
        perplexity_values = [float(p) for p in args.perplexity_values.split(",")]
        plot_tsne_perplexity_comparison(
            args, X_scaled, y, perplexities=perplexity_values, save_path=args.output
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

def encode_colors(y):
    # Create a LabelEncoder object
    le = LabelEncoder()
    # Fit and transform your categorical labels
    y_encoded = le.fit_transform(y)
    # Now you can pass y_encoded to plt.scatter
    return y_encoded
