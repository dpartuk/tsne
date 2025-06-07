from scipy.spatial.distance import pdist, squareform
import numpy as np
import tools
import time
from sklearn.manifold import TSNE

def run_tsne(X, args):
    print("Applying t-SNE transformation...")
    start_time = time.time()

    if args.use_local:
        print("Using local t-SNE implementation...")
        X_tsne = tsne_local(X, args)
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            early_exaggeration=args.exaggeration,
            random_state=42, n_iter=args.n_iterations
        )
        X_tsne = tsne.fit_transform(X)

    end_time = time.time()
    print(f"t-SNE completed in {end_time - start_time:.2f} seconds")

    return X_tsne


def tsne_local(X, args):
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
    n_components = 2
    random_state = 42
    perplexity = args.perplexity
    n_iter = args.n_iterations
    learning_rate = args.learning_rate

    n_samples = X.shape[0]

    # Step 1: Compute pairwise distances
    distances = squareform(pdist(X, metric="euclidean"))
    distances_squared = distances**2

    # Step 2: Convert distances to probabilities (P matrix)
    P = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        beta = tools.binary_search_beta(distances_squared[i], perplexity)
        P[i] = np.exp(-distances_squared[i] * beta)
        P[i, i] = 0
        P[i] /= np.sum(P[i])

    # Symmetrize P matrix
    P = (P + P.T) / (2 * n_samples)
    P = np.maximum(P, 1e-12)

    # Step 3: Initialize low-dimensional representation
    rng = np.random.RandomState(random_state)
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