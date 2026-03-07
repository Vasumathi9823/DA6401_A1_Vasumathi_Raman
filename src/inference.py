"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import os

# Import custom modules based on the required directory structure
from utils.data_loader import load_and_prep_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    # TA's required arguments, defaulted to your winning configuration
    parser.add_argument('--model_path', type=str, default='best_model.npy', help='Relative path to saved model weights')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--hidden_layers', type=int, default=3, help='List of hidden layer sizes (count)')
    parser.add_argument('--num_neurons', type=int, default=128, help='Number of neurons in hidden layers')
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'sigmoid', 'tanh'])
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Ensure it is in the correct directory.")
    
    # allow_pickle=True and .item() extract the saved weight dictionary
    return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
    Returns Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    # We already built the heavy lifting into the NeuralNetwork's evaluate method!
    return model.evaluate(X_test, y_test)


def main():
    """
    Main inference function.
    Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    
    print(f"Loading {args.dataset} test data...")
    # We only need X_test and y_test for the final evaluation
    _, _, _, _, X_test, y_test = load_and_prep_data(args.dataset)
    
    print("Reconstructing Neural Network architecture...")
    # Initialize the model structure so we have a place to inject the weights
    model = NeuralNetwork(args)
    
    print(f"Loading weights from {args.model_path}...")
    weights = load_model(args.model_path)
    model.set_weights(weights)
    
    print("Evaluating model on test data...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "="*35)
    print("FINAL INFERENCE RESULTS: ")
    print("="*35)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Loss:      {metrics['loss']:.4f}")
    print("="*35 + "\n")
    
    print("Evaluation complete!")
    
    # CRITICAL: The TA skeleton requires main() to return the dictionary
    return metrics


if __name__ == '__main__':
    main()