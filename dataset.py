from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import args
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_dataset(args):

    X, y = fetch_openml(args.dataset, version=1,return_X_y=True, as_frame=False)

    # X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    # Use subset of data for faster computation
    if args.n_samples < X.shape[0]:
        rng = np.random.RandomState(42)
        indices = rng.choice(X.shape[0], args.n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    # Preprocess the data by scaling to [0,1] range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def load_mnist_data(n_samples=5000):
    """
    Load MNIST dataset and return a subset of samples.

    Args:
        n_samples: Number of samples to return

    Returns:
        X: Image data
        y: Labels
    """
    print("Loading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    # Normalize the data to [0,1] range, this approach is good for images
    X = X / 255.0

    # Take a subset of the data to speed up computation
    if n_samples:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y


def original_load_mnist_data(n_samples=5000):
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