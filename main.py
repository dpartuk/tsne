# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# import tools
import args
from dataset import load_dataset
from tsne import run_tsne
from visualize import visualize_tsne, compare_hyperparameters, visualize_tsne2
# from sklearn.datasets import fetch_openml
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import time


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", Warning)  # Ignore all warnings

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
    X, y = load_dataset(args, 'distilbert-base-uncased')

    X_tsne = run_tsne(X, args,
                      perplexity=args.perplexity,
                      learning_rate=args.learning_rate,
                      exaggeration=args.exaggeration,
                      n_iterations=args.n_iterations,
                      random_state=42)

    visualize_tsne(args, X_tsne, y,
                   perplexity=args.perplexity,
                   exaggeration=args.exaggeration)

    visualize_tsne2(args, X_tsne, y,
                   perplexity=args.perplexity,
                   exaggeration=args.exaggeration)

    if args.compare_perplexity:
        if args.dataset == 'FAKE':
            # 128, 768, 2048, 4096
            embedding_size = ['distilbert-base-uncased', 'bert-base-uncased', 'xlnet-large-cased', 'albert-xxlarge-v2']
            # embedding_size = ['bert-base-uncased']
            for value in embedding_size:
                X, y = load_dataset(args, value)
                compare_hyperparameters(args, X, y)
        else:
            compare_hyperparameters(args, X, y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tools.print_hi('PyCharm')
    print("\nScript is starting...\n!")

    args = args.read_args()
    main(args)

    print("\nScript completed successfully!")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
