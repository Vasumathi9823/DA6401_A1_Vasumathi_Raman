import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_and_prep_data(dataset_name='mnist'):
    """
    Loads, normalizes, and prepares the specified dataset using scikit-learn.
    Returns: X_train, y_train_oh, X_val, y_val, X_test, y_test
    """
    
    # 1. Load the data using scikit-learn instead of Keras
    if dataset_name.lower() == 'mnist':
        print("Downloading MNIST from OpenML (this may take a few seconds)...")
        # parser='auto' prevents a deprecation warning in newer sklearn versions
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    elif dataset_name.lower() == 'fashion_mnist':
        print("Downloading Fashion-MNIST from OpenML (this may take a few seconds)...")
        X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'mnist' or 'fashion_mnist'.")

    # 2. Normalize to [0, 1] (Data is already flattened to 784 by OpenML)
    X = X / 255.0
    y = y.astype(int)

    # 3. Create splits
    # OpenML gives us the full 70k dataset. Standard split is 60k train, 10k test.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )

    # Validation Set (10% of the 60k training data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    # 4. One-hot encode ONLY the training labels for the Loss function
    def one_hot(labels, num_classes=10):
        encoded = np.zeros((labels.size, num_classes))
        encoded[np.arange(labels.size), labels] = 1.0
        return encoded

    y_train_oh = one_hot(y_train)

    return X_train, y_train_oh, X_val, y_val, X_test, y_test
