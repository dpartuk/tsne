import time

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from tsne import tsne_local

import args


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

def visualize_tsne(X_embedded, labels):
    """
    Visualize t-SNE results with colors representing different classes.

    The visualization uses:
    - Color mapping: Each digit class (0-9) is assigned a distinct color using the tab10 colormap
    - Small point size (s=5): Reduces visual clutter when visualizing many data points
    - Alpha transparency (0.7): Allows visibility of overlapping points to show density
    - Colorbar: Provides a legend mapping colors to digit classes for easy interpretation

    Args:
        X_embedded: Low-dimensional embedding from t-SNE
        labels: Class labels for each point
    """
    print("Creating visualization...")

    # Create a scatter plot
    plt.figure(figsize=(10, 8))

    # Convert labels to integers if they're strings
    if isinstance(labels[0], str):
        # labels = labels.astype(int)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

    # Create a scatter plot with points colored by digit class
    scatter = plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="tab10", alpha=0.7, s=5
    )

    # Add a color bar to show the mapping of colors to digits
    plt.colorbar(scatter, ticks=range(10), label="Digit")
    plt.title("t-SNE visualization of MNIST digits")
    plt.show()

def compare_hyperparameters(args, X, y):
