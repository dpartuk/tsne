# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import tools
from tsne_local import tsne_local
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

import warnings
warnings.filterwarnings("ignore")

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

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.




def main():

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
    n_iter = args.n_iterations
    print("Number of iterations: ", n_iter)
    if args.use_local:
        print("Using local t-SNE implementation...")
        X_tsne = tsne_local(
            X_scaled,
            n_components=2,
            perplexity=args.perplexity,
            random_state=42,
            n_iter=args.n_iterations,
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
    encoded_y = tools.encode_colors(y)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=encoded_y, cmap="tab10")
    plt.colorbar(scatter)
    plt.title("t-SNE visualization of MNIST digits")
    plt.xlabel("First t-SNE dimension")
    plt.ylabel("Second t-SNE dimension")
    plt.legend(*scatter.legend_elements(), title="Digits")

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.show()

    tools.compare_perplexity(args, X_scaled, y)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    args = tools.read_args()
    main()

    print("\nScript completed successfully!")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
