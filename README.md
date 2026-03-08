# DA6401_A1_Vasumathi_Raman
This is a part of DA6401 - Introduction to Deep Learning course Assignment 

# Assignment 1: Multi-Layer Perceptron for Image Classification

# 2.1 Data Exploration & Class Profiling
- W&B Table Integration: Leveraged wandb.Table to organize and visualize five distinct image samples for each of the ten categories, using wandb.Image for rendering.
- Balanced Sampling: Utilized a dictionary-based counter to ensure exactly 50 samples (5 per class) were logged, providing a high-level overview of class diversity.
- Structural Overlap Analysis: Conducted a visual audit to pinpoint naturally similar pairs, such as digits 4 and 9.
- Model Impact Assessment: Determined that high visual ambiguity populates the off-diagonal cells of the confusion matrix, necessitating robust normalization or deeper architectures.

The MNIST sample visualization is available in 

# 2.2 Hyperparameter Optimization Strategy
- Automated Sweep Setup: Configured a comprehensive wandb.sweep dictionary to iterate through variations in learning rates, batch sizes, optimizers, and activations.
- Random Search Methodology: Employed a randomized search strategy across 100+ individual runs to maximize coverage of the high-dimensional parameter space efficiently.
- Parallel Coordinates Analysis: Used W&B interactive plots to isolate high-performing "hyperparameter paths," identifying learning_rate as the most sensitive variable.
- Metadata Export: Identified the configuration with peak validation accuracy and exported its specific weights and parameters for the final test-set benchmark.

# 2.3 Optimizer Performance Comparison
- Controlled Benchmarking: Fixed the neural architecture at three layers (128 units each) to isolate the performance of SGD, Momentum, NAG, and RMSProp.
- Convergence Dynamics: Logged training loss at every step to visualize the trajectory and steepness of the descent during the critical initial training phase.
- Superiority Mapping: Identified which adaptive or momentum-based methods achieved the lowest loss threshold within the first five epochs.
- RMSProp Justification: Highlighted how RMSProp’s adaptive scaling of the learning rate prevents vertical oscillations, allowing for more aggressive horizontal progress.

# 2.4 Gradient Flow & Activation Functions
- ReLU vs. Sigmoid Contrast: Executed parallel training runs with identical hyperparameters to compare the gradient stability of traditional vs. modern activations.
- Gradient Norm Tracking: Monitored the L_2 norm of the first hidden layer's weights using np.linalg.norm to detect the onset of signal decay.
- Vanishing Gradient Proof: Documented the rapid decay of Sigmoid gradients toward zero, contrasting it with the sustained signal flow of the ReLU model.
- Saturation Resistance: Analyzed how ReLU avoids the "saturation zones" of Sigmoid, ensuring that deep networks continue to learn across all layers.


# 2.5 ReLU "Dead Neuron" Analysis
- Activation Distribution: Utilized wandb.Histogram to visualize the output distribution of the ReLU hidden layers during the training process.
- Symptom Identification: Defined "Dead Neurons" by identifying high-density spikes at exactly 0.0, indicating neurons that have permanently ceased firing.
- Tanh Comparison: Contrasted this behavior with Tanh’s zero-centered, non-terminating activations, showing that Tanh neurons slow down but rarely "die."
- Learning Rate Impact: Demonstrated how excessive learning rates (e.g., 0.1) force ReLU biases into a permanent negative state, leading to accuracy plateaus.

# 2.6 Loss Function Evaluation
- Loss Pairing: Compared the training efficiency of Mean Squared Error (MSE) versus Cross-Entropy using an identical architecture and learning rate.
- Convergence Trajectory: Overlaid the training curves of both models to visualize which loss function minimized error more effectively.
- Gradient Slope Analysis: Mathematically justified Cross-Entropy’s speed, as its derivative with Softmax provides a steeper gradient for incorrect classifications.
- Optimization Deadlocks: Documented how MSE experiences "learning stalls" when predictions are significantly distant from the target labels.

# 2.7 Generalization & Global Performance
- Overlay Visualization: Combined train_accuracy and test_accuracy metrics for all sweep runs into a single overlay plot for holistic analysis.
- Overfitting Diagnostics: Flagged specific configurations that yielded near-perfect training scores but failed to generalize on the test data.
- Gap Interpretation: Defined the delta between training and test performance as the "Generalization Gap," a clear indicator of data memorization.
- Refinement Strategy: Concluded that high-gap models require regularization techniques like weight_decay or dropout to bridge the performance divide.

# 2.8 Comprehensive Error Analysis
- Confusion Matrix Profiling: Generated an interactive W&B confusion matrix on the test set to identify specific class-wise weaknesses.
- Failure Gallery Log: Filtered and logged misclassified samples to provide a visual "Failure Gallery," highlighting specific images that confused the model.
- Confusion Clustering: Observed common error patterns, such as the model's inability to distinguish between different types of shirts in the Fashion-MNIST set.
- Failure Heatmaps: Created visual averages of "incorrect" images to detect whether background noise or stroke thickness influenced misclassification.

# 2.9 Initialization & Symmetry Breaking
- Strategy Comparison: Analyzed the fundamental difference between Zeros Initialization and Xavier Initialization over 50 iterations.
- Neuron Trajectory Tracking: Logged the gradients of five specific neurons to check if their update paths diverged or remained identical.
- Symmetry Evidence: Documented how zero-initialization causes perfectly overlapping gradient lines, effectively reducing layer capacity to a single neuron.
- Symmetry Breaking Necessity: Argued that random initialization is mathematically required to allow neurons to "specialize" in different geometric features.

# 2.10 Fashion-MNIST Adaptability
- Feature Processing: Implemented pixel scaling (0 to 1) and Mini-Batch Gradient Descent (batch size 64) to stabilize training on more complex images.
- Transfer Testing: Evaluated the best-performing MNIST configurations (NAG/Momentum) on the Fashion-MNIST dataset to test for generalizability.
- Complexity Findings: Reported that accuracy typically drops on clothing data due to the higher entropy and texture variance compared to simple digits.
- Hyperparameter Scaling: Justified increasing hidden_neurons (128 to 256) to provide the model with the necessary capacity to learn intricate clothing details.

WandB report: https://wandb.ai/vasumathi1998-indian-institute-of-technology-madras/DA6401_Assignment_1_ee21d063/reports/DA6401_EE21D063_ASSIGNMENT1--VmlldzoxNjEzNjEzOQ/edit?draftId=VmlldzoxNjEzNjUxNQ==  






























































































































































































































































































































