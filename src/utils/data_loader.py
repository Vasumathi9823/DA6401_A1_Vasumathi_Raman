import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prep_data(dataset_name='mnist'):
    """Loads and prepares the dataset, falling back to safe OpenML if Keras is missing."""
    # 1. Try Keras first (Fastest, standard on Autograders)
    try:
        from keras.datasets import mnist, fashion_mnist
        if dataset_name.lower() == 'mnist':
            (X, y), (X_test, y_test) = mnist.load_data()
        else:
            (X, y), (X_test, y_test) = fashion_mnist.load_data()
            
        X = X.reshape(X.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
    except ImportError:
        # 2. Safe Fallback to OpenML (Bypasses the pandas requirement)
        from sklearn.datasets import fetch_openml
        if dataset_name.lower() == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
        else:
            X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
        
        X, X_test, y, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

    # Normalize
    X = X / 255.0
    X_test = X_test / 255.0
    y = y.astype(int)
    y_test = y_test.astype(int)

    # Validation Set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # One-hot encode training labels
    def one_hot(labels, num_classes=10):
        encoded = np.zeros((labels.size, num_classes))
        encoded[np.arange(labels.size), labels] = 1.0
        return encoded

    y_train_oh = one_hot(y_train)

    return X_train, y_train_oh, X_val, y_val, X_test, y_test
