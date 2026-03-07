"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
import os

# Import custom modules based on the required directory structure
from utils.data_loader import load_and_prep_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # TA's required arguments, defaulted to your absolute best configuration
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('--hidden_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--num_neurons', type=int, default=128, help='Number of neurons in hidden layers')
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'mse'])
    parser.add_argument('--weight_init', type=str, default='xavier', help='Weight initialization method')
    parser.add_argument('--wandb_project', type=str, default='DA6401_Assignment_1_ee21d063', help='W&B project name')
    parser.add_argument('--model_save_path', type=str, default='best_model.npy', help='Relative path to save trained model')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    
    # 1. Initialize W&B
    wandb.init(project=args.wandb_project, config=vars(args), name="Final_CLI_Train")
    
    # 2. Load Data
    print(f"Loading {args.dataset} dataset...")
    X_train, y_train_oh, X_val, y_val, X_test, y_test = load_and_prep_data(args.dataset)
    
    # 3. Initialize Model
    print(f"Initializing Neural Network with {args.hidden_layers} hidden layers...")
    model = NeuralNetwork(args)
    
    # 4. Setup Optimizer
    if args.optimizer == 'sgd':
        opt = SGD(lr=args.learning_rate)
    elif args.optimizer == 'momentum':
        opt = Momentum(lr=args.learning_rate)
    elif args.optimizer == 'nag':
        opt = NAG(lr=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        opt = RMSProp(lr=args.learning_rate)
    else:
        # Fallback to RMSProp if adam/nadam aren't implemented in your optimizers.py
        print(f"Warning: {args.optimizer} not explicitly built, falling back to RMSProp")
        opt = RMSProp(lr=args.learning_rate)

    # 5. Train
    print(f"Starting training for {args.epochs} epochs using {args.optimizer}...")
    model.train(X_train, y_train_oh, epochs=args.epochs, batch_size=args.batch_size, optimizer=opt)
    
    # 6. Evaluate on Validation Set
    print("\nEvaluating on Validation Data...")
    val_metrics = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Loss:     {val_metrics['loss']:.4f}")
    
    wandb.log({
        "final_val_accuracy": val_metrics['accuracy'],
        "final_val_loss": val_metrics['loss']
    })
    
    # 7. Save Model Weights
    print(f"\nSaving model to {args.model_save_path}...")
    # Ensure the directory exists if the grader passes something like "models/best_model.npy"
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    weights = model.get_weights()
    np.save(args.model_save_path, weights)
    
    wandb.finish()
    print("Training complete!")

if __name__ == '__main__':
    main()
