from scipy.spatial.distance import pdist, squareform
import numpy as np
import tools
import time
from sklearn.manifold import TSNE
from evaluation import TSNEEvaluator

def run_tsne(X, args,
             perplexity=30,
             learning_rate=200,
             exaggeration=12,
             n_iterations=1000,
             random_state=42,):
    print("Applying t-SNE transformation...")
    start_time = time.time()

    if args.use_local:
        print("Using local t-SNE implementation...")
        X_tsne = tsne_local(X,
                            perplexity=perplexity,
                            learning_rate=learning_rate,
                            n_iter=n_iterations,
                            random_state=random_state)
    elif args.use_custom:
        print("Using custom t-SNE implementation...")
        local_tsne = LocalTSNE(
            perplexity=perplexity,
            learning_rate=learning_rate,
            early_exaggeration=exaggeration,
            n_iter=n_iterations,
            random_state=random_state,
        )
        X_tsne = local_tsne.fit_transform(X)
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            early_exaggeration=exaggeration,
            learning_rate=learning_rate,
            random_state=random_state,
            n_iter=n_iterations
        )
        X_tsne = tsne.fit_transform(X)

    end_time = time.time()
    print(f"t-SNE completed in {end_time - start_time:.2f} seconds")

    # Evaluate t-SNE quality if requested
    if hasattr(args, 'evaluate') and args.evaluate:
        print("\nEvaluating t-SNE quality...")
        evaluator = TSNEEvaluator(k_neighbors=10, n_samples=min(1000, X.shape[0]))
        evaluation_results = evaluator.evaluate_tsne(X, X_tsne, verbose=True)

        # Create Shepard diagram if requested
        if hasattr(args, 'plot_shepard') and args.plot_shepard:
            evaluator.plot_shepard_diagram(X, X_tsne, save_path=args.output.replace('.png',
                                                                                    '_shepard.png') if args.output else None)
    return X_tsne


def tsne_local(X,
               perplexity=30,
               learning_rate=200,
               n_iter=1000,
               random_state=42, ):
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
    # random_state = 42
    # perplexity = args.perplexity
    # n_iter = args.n_iterations
    # learning_rate = args.learning_rate

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

def compare_tsne_implementations(X, args, perplexity=30, learning_rate=200, exaggeration=12, n_iterations=1000, random_state=42):
    """
    Compare sklearn and local t-SNE implementations with evaluation metrics.

    Args:
        X: Input data matrix
        args: Arguments object
        perplexity: Perplexity value
        learning_rate: Learning rate for local implementation
        exaggeration: Exaggeration factor
        n_iterations: Number of iterations
        random_state: Random seed

    Returns:
        dict: Comparison results
    """
    print("Comparing sklearn vs local t-SNE implementations...")

    # Run sklearn implementation
    print("\nRunning sklearn t-SNE...")
    sklearn_start = time.time()
    tsne_sklearn = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=exaggeration,
        learning_rate=learning_rate,
        random_state=random_state,
        n_iter=n_iterations
    )
    X_sklearn = tsne_sklearn.fit_transform(X)
    sklearn_time = time.time() - sklearn_start
    print(f"sklearn t-SNE completed in {sklearn_time:.2f} seconds")

    # Run local implementation
    print("\nRunning local t-SNE...")
    local_start = time.time()
    X_local = tsne_local(X, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iterations, random_state=random_state)
    local_time = time.time() - local_start
    print(f"Local t-SNE completed in {local_time:.2f} seconds")

    # Evaluate both implementations
    if hasattr(args, 'evaluate') and args.evaluate:
        print("\nEvaluating both implementations...")
        evaluator = TSNEEvaluator(k_neighbors=getattr(args, 'k_neighbors', 10), n_samples=min(1000, X.shape[0]))
        comparison_results = evaluator.compare_implementations(X, X_sklearn, X_local)

        # Add timing information
        comparison_results['timing'] = {
            'sklearn_time': sklearn_time,
            'local_time': local_time,
            'speedup': local_time / sklearn_time if sklearn_time > 0 else float('inf')
        }

        print(f"\nTiming Comparison:")
        print(f"sklearn time: {sklearn_time:.2f} seconds")
        print(f"local time: {local_time:.2f} seconds")
        print(f"speedup: {comparison_results['timing']['speedup']:.2f}x")

        return comparison_results

    return {
        'sklearn_embedding': X_sklearn,
        'local_embedding': X_local,
        'timing': {
            'sklearn_time': sklearn_time,
            'local_time': local_time,
            'speedup': local_time / sklearn_time if sklearn_time > 0 else float('inf')
        }
    }

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
        random_state=42,
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