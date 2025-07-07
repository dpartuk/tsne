import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time


class TSNEEvaluator:
    """
    Evaluation metrics for t-SNE dimensionality reduction quality.
    
    This class implements three key metrics to assess how well t-SNE preserves
    the structure of the original high-dimensional space in the low-dimensional embedding:
    
    1. Continuity: Measures how well local neighborhoods are preserved
    2. Mean Local Error: Average error in preserving local structure
    3. Shepard Correlations: Correlation between original and embedded distances
    """
    
    def __init__(self, k_neighbors=10, n_samples=None):
        """
        Initialize the evaluator.
        
        Args:
            k_neighbors: Number of nearest neighbors to consider for local metrics
            n_samples: Number of samples to use for evaluation (None for all)
        """
        self.k_neighbors = k_neighbors
        self.n_samples = n_samples
        
    def evaluate_tsne(self, X_high, X_low, labels=None, verbose=True):
        """
        Evaluate t-SNE quality using multiple metrics.
        
        Args:
            X_high: High-dimensional data (n_samples, n_features)
            X_low: Low-dimensional embedding (n_samples, n_components)
            labels: Optional labels for visualization
            verbose: Whether to print results
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        if verbose:
            print("Computing t-SNE evaluation metrics...")
            start_time = time.time()
        
        # Subsample if specified
        if self.n_samples and self.n_samples < len(X_high):
            indices = np.random.choice(len(X_high), self.n_samples, replace=False)
            X_high = X_high[indices]
            X_low = X_low[indices]
            if labels is not None:
                labels = labels[indices]
        
        # Compute all metrics
        results = {}
        
        # 1. Continuity
        results['continuity'] = self.compute_continuity(X_high, X_low)
        
        # 2. Mean Local Error
        results['mean_local_error'] = self.compute_mean_local_error(X_high, X_low)
        
        # 3. Shepard Correlations
        results['shepard_correlations'] = self.compute_shepard_correlations(X_high, X_low)
        
        # 4. Additional metrics
        results['kl_divergence'] = self.compute_kl_divergence(X_high, X_low)
        results['stress'] = self.compute_stress(X_high, X_low)
        
        if verbose:
            end_time = time.time()
            print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
            self.print_results(results)
        
        return results
    
    def compute_continuity(self, X_high, X_low):
        """
        Compute continuity metric.
        
        Continuity measures how well local neighborhoods are preserved.
        Higher values (closer to 1) indicate better preservation of local structure.
        
        Args:
            X_high: High-dimensional data
            X_low: Low-dimensional embedding
            
        Returns:
            float: Continuity score between 0 and 1
        """
        n_samples = X_high.shape[0]
        
        # Find k-nearest neighbors in high-dimensional space
        nbrs_high = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X_high)
        distances_high, indices_high = nbrs_high.kneighbors(X_high)
        
        # Find k-nearest neighbors in low-dimensional space
        nbrs_low = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X_low)
        distances_low, indices_low = nbrs_low.kneighbors(X_low)
        
        continuity_scores = []
        
        for i in range(n_samples):
            # Get neighbors in high-dimensional space (excluding self)
            high_neighbors = set(indices_high[i][1:])  # Skip first (self)
            
            # Get neighbors in low-dimensional space (excluding self)
            low_neighbors = set(indices_low[i][1:])    # Skip first (self)
            
            # Compute intersection
            intersection = len(high_neighbors.intersection(low_neighbors))
            
            # Continuity is the fraction of preserved neighbors
            continuity = intersection / self.k_neighbors
            continuity_scores.append(continuity)
        
        return np.mean(continuity_scores)
    
    def compute_mean_local_error(self, X_high, X_low):
        """
        Compute mean local error.
        
        This metric measures the average error in preserving local structure.
        Lower values indicate better preservation of local relationships.
        
        Args:
            X_high: High-dimensional data
            X_low: Low-dimensional embedding
            
        Returns:
            float: Mean local error
        """
        n_samples = X_high.shape[0]
        
        # Compute pairwise distances
        distances_high = squareform(pdist(X_high, metric='euclidean'))
        distances_low = squareform(pdist(X_low, metric='euclidean'))
        
        # Normalize distances to [0, 1] for fair comparison
        distances_high_norm = distances_high / np.max(distances_high)
        distances_low_norm = distances_low / np.max(distances_low)
        
        # Find k-nearest neighbors in high-dimensional space
        nbrs_high = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X_high)
        distances_knn, indices_knn = nbrs_high.kneighbors(X_high)
        
        local_errors = []
        
        for i in range(n_samples):
            # Get k-nearest neighbors (excluding self)
            neighbors = indices_knn[i][1:]
            
            # Compute error for local neighborhood
            high_distances = distances_high_norm[i, neighbors]
            low_distances = distances_low_norm[i, neighbors]
            
            # Mean squared error for local neighborhood
            mse = np.mean((high_distances - low_distances) ** 2)
            local_errors.append(mse)
        
        return np.mean(local_errors)
    
    def compute_shepard_correlations(self, X_high, X_low):
        """
        Compute Shepard correlations.
        
        Shepard correlations measure the correlation between distances in the
        original high-dimensional space and the low-dimensional embedding.
        Higher correlations indicate better preservation of distance relationships.
        
        Args:
            X_high: High-dimensional data
            X_low: Low-dimensional embedding
            
        Returns:
            dict: Dictionary containing Pearson and Spearman correlations
        """
        # Compute pairwise distances
        distances_high = squareform(pdist(X_high, metric='euclidean'))
        distances_low = squareform(pdist(X_low, metric='euclidean'))
        
        # Flatten distance matrices (excluding diagonal)
        mask = ~np.eye(distances_high.shape[0], dtype=bool)
        high_distances_flat = distances_high[mask]
        low_distances_flat = distances_low[mask]
        
        # Compute correlations
        pearson_corr, pearson_p = pearsonr(high_distances_flat, low_distances_flat)
        spearman_corr, spearman_p = spearmanr(high_distances_flat, low_distances_flat)
        
        return {
            'pearson': pearson_corr,
            'pearson_p': pearson_p,
            'spearman': spearman_corr,
            'spearman_p': spearman_p
        }
    
    def compute_kl_divergence(self, X_high, X_low):
        """
        Compute KL divergence between high and low-dimensional probability distributions.
        
        This is similar to the objective function that t-SNE minimizes.
        Lower values indicate better preservation of probability structure.
        
        Args:
            X_high: High-dimensional data
            X_low: Low-dimensional embedding
            
        Returns:
            float: KL divergence
        """
        n_samples = X_high.shape[0]
        
        # Compute pairwise distances
        distances_high = squareform(pdist(X_high, metric='euclidean'))
        distances_low = squareform(pdist(X_low, metric='euclidean'))
        
        # Compute probability distributions (similar to t-SNE)
        # High-dimensional space (Gaussian)
        sigma = np.median(distances_high)
        P = np.exp(-distances_high**2 / (2 * sigma**2))
        np.fill_diagonal(P, 0)
        P = P / np.sum(P)
        P = np.maximum(P, 1e-12)
        
        # Low-dimensional space (t-distribution)
        Q = 1 / (1 + distances_low**2)
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q)
        Q = np.maximum(Q, 1e-12)
        
        # Compute KL divergence
        kl_div = np.sum(P * np.log(P / Q))
        
        return kl_div
    
    def compute_stress(self, X_high, X_low):
        """
        Compute stress metric (similar to MDS stress).
        
        Stress measures the discrepancy between original and embedded distances.
        Lower values indicate better preservation of distance relationships.
        
        Args:
            X_high: High-dimensional data
            X_low: Low-dimensional embedding
            
        Returns:
            float: Stress value
        """
        # Compute pairwise distances
        distances_high = squareform(pdist(X_high, metric='euclidean'))
        distances_low = squareform(pdist(X_low, metric='euclidean'))
        
        # Flatten distance matrices (excluding diagonal)
        mask = ~np.eye(distances_high.shape[0], dtype=bool)
        high_distances_flat = distances_high[mask]
        low_distances_flat = distances_low[mask]
        
        # Compute stress
        numerator = np.sum((high_distances_flat - low_distances_flat) ** 2)
        denominator = np.sum(high_distances_flat ** 2)
        
        stress = np.sqrt(numerator / denominator)
        
        return stress
    
    def print_results(self, results):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Dictionary containing evaluation metrics
        """
        print("\n" + "="*60)
        print("t-SNE EVALUATION RESULTS")
        print("="*60)
        
        print(f"Continuity:                    {results['continuity']:.4f}")
        print(f"Mean Local Error:              {results['mean_local_error']:.4f}")
        print(f"KL Divergence:                 {results['kl_divergence']:.4f}")
        print(f"Stress:                        {results['stress']:.4f}")
        
        shepard = results['shepard_correlations']
        print(f"Shepard Pearson Correlation:   {shepard['pearson']:.4f}")
        print(f"Shepard Spearman Correlation:  {shepard['spearman']:.4f}")
        
        print("="*60)
        
        # Interpretation
        print("\nINTERPRETATION:")
        print(f"• Continuity: {'Good' if results['continuity'] > 0.7 else 'Fair' if results['continuity'] > 0.5 else 'Poor'} local structure preservation")
        print(f"• Mean Local Error: {'Good' if results['mean_local_error'] < 0.1 else 'Fair' if results['mean_local_error'] < 0.3 else 'Poor'} local distance preservation")
        print(f"• Shepard Correlation: {'Good' if shepard['pearson'] > 0.7 else 'Fair' if shepard['pearson'] > 0.5 else 'Poor'} distance relationship preservation")
    
    def plot_shepard_diagram(self, X_high, X_low, save_path=None):
        """
        Create a Shepard diagram showing the relationship between
        original and embedded distances.
        
        Args:
            X_high: High-dimensional data
            X_low: Low-dimensional embedding
            save_path: Optional path to save the plot
        """
        # Compute pairwise distances
        distances_high = squareform(pdist(X_high, metric='euclidean'))
        distances_low = squareform(pdist(X_low, metric='euclidean'))
        
        # Flatten distance matrices (excluding diagonal)
        mask = ~np.eye(distances_high.shape[0], dtype=bool)
        high_distances_flat = distances_high[mask]
        low_distances_flat = distances_low[mask]
        
        # Subsample for visualization if too many points
        if len(high_distances_flat) > 10000:
            indices = np.random.choice(len(high_distances_flat), 10000, replace=False)
            high_distances_flat = high_distances_flat[indices]
            low_distances_flat = low_distances_flat[indices]
        
        # Create Shepard diagram
        plt.figure(figsize=(10, 8))
        plt.scatter(high_distances_flat, low_distances_flat, alpha=0.5, s=1)
        
        # Add perfect preservation line
        max_dist = max(np.max(high_distances_flat), np.max(low_distances_flat))
        plt.plot([0, max_dist], [0, max_dist], 'r--', linewidth=2, label='Perfect preservation')
        
        # Compute and display correlations
        pearson_corr, _ = pearsonr(high_distances_flat, low_distances_flat)
        spearman_corr, _ = spearmanr(high_distances_flat, low_distances_flat)
        
        plt.title(f'Shepard Diagram\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
        plt.xlabel('Original Distances')
        plt.ylabel('Embedded Distances')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_implementations(self, X_high, sklearn_embedding, local_embedding, labels=None):
        """
        Compare the quality of sklearn vs local t-SNE implementations.
        
        Args:
            X_high: High-dimensional data
            sklearn_embedding: Embedding from sklearn TSNE
            local_embedding: Embedding from local implementation
            labels: Optional labels for visualization
            
        Returns:
            dict: Comparison results
        """
        print("Comparing sklearn vs local t-SNE implementations...")
        
        # Evaluate sklearn implementation
        print("\nEvaluating sklearn implementation:")
        sklearn_results = self.evaluate_tsne(X_high, sklearn_embedding, labels, verbose=False)
        
        # Evaluate local implementation
        print("\nEvaluating local implementation:")
        local_results = self.evaluate_tsne(X_high, local_embedding, labels, verbose=False)
        
        # Print comparison
        print("\n" + "="*80)
        print("IMPLEMENTATION COMPARISON")
        print("="*80)
        
        metrics = ['continuity', 'mean_local_error', 'kl_divergence', 'stress']
        for metric in metrics:
            sklearn_val = sklearn_results[metric]
            local_val = local_results[metric]
            
            if metric in ['continuity']:
                better = "sklearn" if sklearn_val > local_val else "local"
                print(f"{metric:20s}: sklearn={sklearn_val:.4f}, local={local_val:.4f} ({better} better)")
            else:
                better = "sklearn" if sklearn_val < local_val else "local"
                print(f"{metric:20s}: sklearn={sklearn_val:.4f}, local={local_val:.4f} ({better} better)")
        
        # Shepard correlations comparison
        sklearn_pearson = sklearn_results['shepard_correlations']['pearson']
        local_pearson = local_results['shepard_correlations']['pearson']
        better = "sklearn" if sklearn_pearson > local_pearson else "local"
        print(f"shepard_pearson:      sklearn={sklearn_pearson:.4f}, local={local_pearson:.4f} ({better} better)")
        
        print("="*80)
        
        return {
            'sklearn': sklearn_results,
            'local': local_results
        }


def evaluate_tsne_quality(X_high, X_low, k_neighbors=10, n_samples=None, verbose=True):
    """
    Convenience function to evaluate t-SNE quality.
    
    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional embedding
        k_neighbors: Number of nearest neighbors for local metrics
        n_samples: Number of samples to use for evaluation
        verbose: Whether to print results
        
    Returns:
        dict: Evaluation results
    """
    evaluator = TSNEEvaluator(k_neighbors=k_neighbors, n_samples=n_samples)
    return evaluator.evaluate_tsne(X_high, X_low, verbose=verbose)


def compare_tsne_implementations(X_high, sklearn_embedding, local_embedding, k_neighbors=10, n_samples=None):
    """
    Convenience function to compare t-SNE implementations.
    
    Args:
        X_high: High-dimensional data
        sklearn_embedding: Embedding from sklearn TSNE
        local_embedding: Embedding from local implementation
        k_neighbors: Number of nearest neighbors for local metrics
        n_samples: Number of samples to use for evaluation
        
    Returns:
        dict: Comparison results
    """
    evaluator = TSNEEvaluator(k_neighbors=k_neighbors, n_samples=n_samples)
    return evaluator.compare_implementations(X_high, sklearn_embedding, local_embedding) 