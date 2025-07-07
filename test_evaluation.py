#!/usr/bin/env python3
"""
Test script for t-SNE evaluation metrics.

This script demonstrates how to use the evaluation metrics to assess
the quality of t-SNE dimensionality reduction.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from evaluation import TSNEEvaluator, evaluate_tsne_quality, compare_tsne_implementations
from tsne import tsne_local
import matplotlib.pyplot as plt


def test_evaluation_metrics():
    """Test the evaluation metrics with synthetic data."""
    print("Testing t-SNE evaluation metrics...")
    
    # Generate synthetic data with clear clusters
    n_samples = 500
    n_features = 20
    n_clusters = 3
    
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                     centers=n_clusters, random_state=42, cluster_std=1.0)
    
    print(f"Generated {n_samples} samples with {n_features} features in {n_clusters} clusters")
    
    # Apply t-SNE
    print("\nApplying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_embedded = tsne.fit_transform(X)
    
    # Evaluate t-SNE quality
    print("\nEvaluating t-SNE quality...")
    results = evaluate_tsne_quality(X, X_embedded, k_neighbors=10, verbose=True)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original data (first 2 dimensions)
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('Original Data (First 2 Dimensions)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # t-SNE embedding
    plt.subplot(1, 3, 2)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('t-SNE Embedding')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Shepard diagram
    plt.subplot(1, 3, 3)
    evaluator = TSNEEvaluator()
    evaluator.plot_shepard_diagram(X, X_embedded)
    plt.title('Shepard Diagram')
    
    plt.tight_layout()
    plt.show()
    
    return results


def test_implementation_comparison():
    """Test comparison between sklearn and local implementations."""
    print("\n" + "="*60)
    print("Testing implementation comparison...")
    print("="*60)
    
    # Generate smaller dataset for faster comparison
    n_samples = 200
    n_features = 10
    n_clusters = 2
    
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                     centers=n_clusters, random_state=42, cluster_std=1.0)
    
    print(f"Generated {n_samples} samples with {n_features} features in {n_clusters} clusters")
    
    # Run sklearn implementation
    print("\nRunning sklearn t-SNE...")
    tsne_sklearn = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    X_sklearn = tsne_sklearn.fit_transform(X)
    
    # Run local implementation
    print("Running local t-SNE...")
    X_local = tsne_local(X, perplexity=30, n_iter=500, random_state=42)
    
    # Compare implementations
    print("\nComparing implementations...")
    evaluator = TSNEEvaluator(k_neighbors=10, n_samples=min(100, X.shape[0]))
    comparison = evaluator.compare_implementations(X, X_sklearn, X_local)
    
    # Visualize both embeddings
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_sklearn[:, 0], X_sklearn[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('sklearn t-SNE')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_local[:, 0], X_local[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('Local t-SNE')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()
    
    return comparison


def test_fake_news_evaluation():
    """Test evaluation with fake news dataset (if available)."""
    print("\n" + "="*60)
    print("Testing fake news evaluation...")
    print("="*60)
    
    try:
        from dataset import load_dataset
        import args
        
        # Create args object
        arg_parser = args.read_args()
        arg_parser.dataset = 'FAKE'
        arg_parser.n_samples = 500  # Use smaller sample for testing
        
        # Load fake news dataset
        print("Loading fake news dataset...")
        X, y = load_dataset(arg_parser, 'bert-base-uncased')
        
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        
        # Apply t-SNE
        print("\nApplying t-SNE to fake news embeddings...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        X_embedded = tsne.fit_transform(X)
        
        # Evaluate
        print("\nEvaluating fake news t-SNE...")
        results = evaluate_tsne_quality(X, X_embedded, k_neighbors=10, verbose=True)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.title('t-SNE Visualization of Fake News Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(label='News Category')
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"Could not test fake news evaluation: {e}")
        print("Make sure the fake news dataset is available and dependencies are installed.")
        return None


if __name__ == "__main__":
    print("t-SNE Evaluation Test Suite")
    print("="*60)
    
    # Test 1: Basic evaluation metrics
    results1 = test_evaluation_metrics()
    
    # Test 2: Implementation comparison
    results2 = test_implementation_comparison()
    
    # Test 3: Fake news evaluation (if available)
    results3 = test_fake_news_evaluation()
    
    print("\n" + "="*60)
    print("Test suite completed!")
    print("="*60)
    
    if results1:
        print("\nSynthetic data evaluation results:")
        print(f"Continuity: {results1['continuity']:.4f}")
        print(f"Mean Local Error: {results1['mean_local_error']:.4f}")
        print(f"Shepard Pearson: {results1['shepard_correlations']['pearson']:.4f}")
    
    if results2:
        print("\nImplementation comparison completed successfully!")
    
    if results3:
        print("\nFake news evaluation completed successfully!") 