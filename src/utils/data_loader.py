"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

def load_and_prep_data(dataset_name='mnist'):
    """
    Loads, flattens, normalizes, and prepares the specified dataset.
    Returns: X_train, y_train_oh, X_val, y_val, X_test, y_test
    """
    
    # 1. Load the data
    if dataset_name.lower() == 'mnist':
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    elif dataset_name.lower() == 'fashion_mnist':
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'mnist' or 'fashion_mnist'.")

    # 2. Flatten from 28x28 to 784 and normalize to [0, 1]
    X_train_full = X_train_full.reshape(X_train_full.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # 3. Create Validation Set (10% of the original training data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    # 4. One-hot encode ONLY the training labels for the Loss function
    def one_hot(y, num_classes=10):
        encoded = np.zeros((y.size, num_classes))
        encoded[np.arange(y.size), y] = 1.0
        return encoded

    y_train_oh = one_hot(y_train)

    # Note: y_val and y_test are kept as integer labels because our evaluation 
    # metrics and the model.evaluate() method handle the conversion internally.
    return X_train, y_train_oh, X_val, y_val, X_test, y_test