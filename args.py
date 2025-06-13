import argparse

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
        "--exaggeration",
        type=float,
        default=12.0,
        help="Exaggeration factor for t-SNE (default: 12.0)",
    )
    parser.add_argument(
        "--exaggeration-values",
        type=str,
        default="4,8,12",
        help="Comma-separated list of exaggeration values for comparison",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="Number of iterations for t-SNE (default: 1000, min: 250)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist_784",
        choices=['CT', 'mnist_784', 'CIFAR_10', 'iris', 'Fashion-MNIST', 'SVHN', 'GSE45827'],
        help="Which dataset to use between CT, mnist_784, CIFAR_10, Fashion-MNIST, GSE45827, SVHN and iris(default: mnist_784)",
    )
    parser.add_argument("--output", type=str, help="Path to save visualization")
    parser.add_argument(
        "--use-local", action="store_true", help="Use local t-SNE implementation"
    )
    parser.add_argument(
        "--use-custom", action="store_true", help="Use local custom t-SNE implementation"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=200.0, help="Learning rate for local t-SNE"
    )

    args = parser.parse_args()
    print("Args type: ", type(args))
    print("Args: ", args)
    return args