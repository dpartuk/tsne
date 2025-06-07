# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# import tools
import args
from dataset import load_dataset
from tsne import run_tsne
from visualize import visualize_tsne, compare_hyperparameters
# from sklearn.datasets import fetch_openml
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import time


import warnings
warnings.filterwarnings("ignore")

"""
t-SNE Visualization of MNIST Dataset

Options:
    --n-samples INT        Number of samples to use (default: 10000)
    --perplexity FLOAT     Perplexity value for main visualization (default: 30)
    --exaggeration FLOAT   Controls the tightness of clusters (default: 12)
    --compare-perplexity   Enable perplexity comparison visualization
    --perplexity-values    List of perplexity values for comparison (default: 5,30,50,100)
    --output PATH          Save visualizations to specified path
    --use-local            Use local t-SNE implementation (warning: significantly slower)
    --learning-rate        Learning rate for local t-SNE implementation (default: 200.0)

"""


def main(args):

    # ------------------------------------------------------------------------------
    # Main execution code
    # ------------------------------------------------------------------------------

    print("Loading dataset...", args.dataset)
    X, y = load_dataset(args)

    X_tsne = run_tsne(X, args)

    visualize_tsne(X_tsne, y)

    compare_hyperparameters(args, X, y)


    # X, y = load_mnist(n_samples=args.n_samples)  # Using fewer samples for faster execution


    # # Apply t-SNE for dimensionality reduction
    # print("Applying t-SNE transformation...")
    # start_time = time.time()
    # n_iter = args.n_iterations
    # print("Number of iterations: ", n_iter)
    # if args.use_local:
    #     print("Using local t-SNE implementation...")
    #     X_tsne = tsne_local(
    #         X_scaled,
    #         n_components=2,
    #         perplexity=args.perplexity,
    #         random_state=42,
    #         n_iter=args.n_iterations,
    #         learning_rate=args.learning_rate,
    #     )
    # else:
    #     tsne = TSNE(
    #         n_components=2, perplexity=args.perplexity, early_exaggeration=args.exaggeration, random_state=42, n_iter=1000
    #     )
    #     X_tsne = tsne.fit_transform(X_scaled)
    # end_time = time.time()
    # print(f"t-SNE completed in {end_time - start_time:.2f} seconds")

    # Create the visualization
    # encoded_y = tools.encode_colors(y)
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=encoded_y, cmap="tab10")
    # plt.colorbar(scatter)
    # plt.title("t-SNE visualization of MNIST digits")
    # plt.xlabel("First t-SNE dimension")
    # plt.ylabel("Second t-SNE dimension")
    # plt.legend(*scatter.legend_elements(), title="Digits")
    #
    # if args.output:
    #     plt.savefig(args.output, dpi=300, bbox_inches="tight")
    # plt.show()
    #
    # tools.compare_perplexity(args, X_scaled, y)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tools.print_hi('PyCharm')
    print("\nScript is starting...\n!")

    args = args.read_args()
    main(args)

    print("\nScript completed successfully!")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
