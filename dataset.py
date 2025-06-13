from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import args
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ct_image_processing import ImageProcessing
from ct_visualizer import CTVisualizer


def load_dataset(args):

    if args.dataset == 'CT':
        X, y = load_ct_dataset()
    else:

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


def load_ct_dataset():
    root = 'ct_images/'  # '/Users/yigal/CT-Datasets/'

    liver_images_path = f"{root}Task03_Liver/imagesTr/"
    liver_labels_path = f"{root}Task03_Liver/labelsTr/"

    number_of_ct_patients = 131

    imgProcessor = ImageProcessing()

    ctVisualizer = CTVisualizer()

    X_all = []
    Y_all = []

    # Create Dataset
    print(f"Start Building CT Dataset with {number_of_ct_patients} patients")

    X_all, Y_all, patient_ids, total = imgProcessor.create_dataset(liver_images_path,
                                                                   liver_labels_path,
                                                                   binary=True,
                                                                   target_size=(256, 256),
                                                                   hu_window=(30, 180),
                                                                   number_of_ct_patients=number_of_ct_patients,
                                                                   labeled_only=True)

    print(total)
    print('Len (X, Y, Patients):', len(X_all), len(Y_all), len(patient_ids))
    print(f'Sample Patient Shapes ({patient_ids[2]}): X[2] Y[2]:', X_all[2].shape, Y_all[2].shape)

    return X_scaled, Y_all



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