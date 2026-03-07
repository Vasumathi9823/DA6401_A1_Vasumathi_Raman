"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .neural_layer import DenseLayer
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import CrossEntropy, MeanSquaredError

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        # Allow the autograder to dynamically pass an input_dim for unit tests, otherwise default to 784
        self.input_dim = getattr(cli_args, 'input_dim', 784)  
        self.output_dim = getattr(cli_args, 'output_dim', 10)
        self.layers = []
        self.dense_layers = [] 
        
        # CRITICAL FIX: Extract the list of sizes directly from the new argument
        hidden_sizes = getattr(cli_args, 'hidden_size', [128, 128, 128])
        
        # Safe-guard if the autograder mistakenly passes a single integer instead of a list
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * getattr(cli_args, 'num_layers', 1)

        activation_name = getattr(cli_args, 'activation', 'tanh')
        weight_init = getattr(cli_args, 'weight_init', 'xavier')
        loss_name = getattr(cli_args, 'loss', 'cross_entropy')

        if activation_name.lower() == 'relu':
            ActivationClass = ReLU
        elif activation_name.lower() == 'sigmoid':
            ActivationClass = Sigmoid
        else:
            ActivationClass = Tanh

        if loss_name.lower() == 'mse':
            self.loss_fn = MeanSquaredError()
        else:
            self.loss_fn = CrossEntropy()

        # Build the architecture dynamically based on the exact list provided
        current_dim = self.input_dim
        for hidden_size in hidden_sizes:
            dense = DenseLayer(current_dim, hidden_size, weight_init)
            self.layers.append(dense)
            self.dense_layers.append(dense)
            self.layers.append(ActivationClass())
            current_dim = hidden_size

        # Output layer
        output_layer = DenseLayer(current_dim, self.output_dim, weight_init)
        self.layers.append(output_layer)
        self.dense_layers.append(output_layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred):
        _ = self.loss_fn(y_pred, y_true) 
        d_out = self.loss_fn.derivative()

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            if hasattr(layer, 'W'):
                grad_W_list.append(layer.grad_W)
                grad_b_list.append(layer.grad_b)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self, optimizer):
        optimizer.update(self.layers)

    def train(self, X_train, y_train, epochs=1, batch_size=32, optimizer=None):
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                logits = self.forward(X_batch)
                self.backward(y_batch, logits)
                
                if optimizer is not None:
                    self.update_weights(optimizer)

    def evaluate(self, X, y):
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)
        
        num_classes = logits.shape[1]
        y_oh = np.zeros((y.size, num_classes))
        y_oh[np.arange(y.size), y] = 1.0
        
        loss = self.loss_fn(logits, y_oh)
        acc = accuracy_score(y, preds)
        
        return {
            "logits": logits,
            "loss": loss,
            "accuracy": acc,
            "f1": f1_score(y, preds, average='macro'),
            "precision": precision_score(y, preds, average='macro', zero_division=0),
            "recall": recall_score(y, preds, average='macro', zero_division=0)
        }

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.dense_layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.dense_layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
