# tsne

"""
t-SNE Visualization of MNIST Dataset

This script demonstrates the use of t-SNE (t-Distributed Stochastic Neighbor Embedding)
for visualizing high-dimensional MNIST digits data in 2D space.

t-SNE is a nonlinear dimensionality reduction technique that preserves local similarities
between data points, making it excellent for visualizing high-dimensional data clusters.
Unlike PCA, t-SNE focuses on preserving the local structure of the data.

The t-SNE algorithm works through the following steps:
1. Computes pairwise similarities in high-dimensional space using Gaussian distributions
2. Defines target similarities in low-dimensional space using t-distributions (heavy-tailed)
3. Minimizes the Kullback-Leibler divergence between these distributions via gradient descent
4. The t-distribution in the low-dimensional space helps address the "crowding problem"
   by allowing dissimilar objects to be modeled far apart
   
Usage:
    python main.py [options]

Options:
    --n-samples INT        Number of samples to use (default: 10000)
    --perplexity FLOAT     Perplexity value for main visualization (default: 30)
    --exaggeration FLOAT   Controls the tightness of clusters (default: 12)
    --compare-perplexity   Enable perplexity comparison visualization
    --perplexity-values    List of perplexity values for comparison (default: 5,30,50,100)
    --output PATH          Save visualizations to specified path
    --use-local            Use local t-SNE implementation (warning: significantly slower)
    --learning-rate        Learning rate for local t-SNE implementation (default: 200.0)

Examples:
    python main.py --n-samples 5000 --perplexity 40
    python main.py --compare-perplexity --perplexity-values 10,20,30,40
    python main.py --output tsne_plot.png
    python main.py --use-local --learning-rate 150.0 --n-samples 1000

Additional Info:
    --perplexity FLOAT    Controls the balance between preserving local and global structure
                          Lower values (5-10) focus on very local structure, creating tighter clusters
                          Higher values (30-50) preserve more global structure and relationships
                          Too high values can cause oversmoothing of the data
                          Too low values may create fragmented clusters
                          Default: 30.0 
    --exaggeration FLOAT  Controls the tightness of clusters in the visualization
                          Higher values (15-20) create more separated, distinct clusters
                          Lower values (4-8) produce more uniformly distributed points
                          Very high values may create artificial clustering
                          Default: 12.0

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
