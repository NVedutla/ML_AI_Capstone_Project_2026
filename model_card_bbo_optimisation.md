# Model Card – Hybrid Black-Box Optimisation System

## 1. Model Overview

This model is a hybrid black-box optimisation system developed as part of an ML/AI capstone project. It is designed to optimise unknown mathematical functions (F1–F8) using iterative learning from input-output observations.

The system evolves over time from simple local search methods into a multi-strategy optimisation framework combining surrogate modelling, Bayesian optimisation, clustering, PCA, and reinforcement learning-inspired exploration.

---

## 2. Model Inputs and Outputs

### Inputs
- Continuous numeric vectors in range [0, 1]
- Variable dimensionality depending on function:
  - F1–F8: 2D to 8D input spaces

### Outputs
- Single continuous scalar value per input
- Represents black-box function evaluation (unknown internal function)

---

## 3. Model Architecture

The final model is a **hybrid optimisation pipeline** composed of:

### 3.1 Gaussian Process Surrogate Model
- Approximates unknown objective functions
- Provides mean prediction and uncertainty estimation

### 3.2 Bayesian Optimisation (Expected Improvement)
- Selects candidates based on predicted improvement over best observed value
- Balances exploration and exploitation

### 3.3 Clustering-Based Search
- Identifies high-performing regions in the search space
- Uses centroid-based candidate generation

### 3.4 PCA-Guided Directional Search
- Extracts principal directions from top-performing samples
- Guides search along dominant variance directions

### 3.5 Attention-Inspired Feature Weighting
- Assigns importance to input dimensions based on variance
- Reduces noise from low-impact features

### 3.6 Reinforcement Learning-Inspired Exploration
- Adds stochastic exploration using epsilon noise
- Supports adaptive exploration of unknown regions

---

## 4. Performance Summary

The model was evaluated over 12 iterative rounds across 8 benchmark functions (F1–F8).

### Key Performance Improvements:
- Stable convergence across all functions
- Improved performance in higher-dimensional functions (F6–F8)
- Reduced randomness in candidate selection over time
- Better exploitation of high-performing regions

### Observed Behaviour:
- Early rounds: high exploration, unstable outputs
- Mid rounds: improved stability with GP + EI
- Final rounds: strong convergence using hybrid methods

---

## 5. Training Strategy

The model is not trained in a traditional supervised learning manner. Instead, it uses:

- Sequential optimisation
- Iterative feedback from function evaluations
- Adaptive candidate generation

Each iteration builds upon previous observations.

---

## 6. Limitations

### 6.1 Black-Box Dependency
The model cannot interpret or access internal function structure.

### 6.2 Limited Evaluation Budget
Only a small number of observations per function are available.

### 6.3 Sensitivity to Noise
Gaussian Process and PCA components may be sensitive to outliers.

### 6.4 No Guaranteed Global Optimum
The system may converge to local optima depending on exploration balance.

---

## 7. Trade-offs

### Exploration vs Exploitation
- Exploration ensures discovery of new regions
- Exploitation improves known good regions
- Balanced using EI + epsilon noise

### Complexity vs Performance
- More complex models improved accuracy
- But increased computational cost

### Interpretability vs Accuracy
- PCA and clustering improved interpretability
- Surrogate models improved performance but reduced transparency

---

## 8. Ethical Considerations

While the model is applied to synthetic benchmark functions, similar optimisation techniques in real-world applications may impact:
- Pricing systems
- Resource allocation
- Automated decision systems

Care must be taken to ensure fairness, transparency, and robustness in real-world deployments.

---

## 9. Real-World Applications

This optimisation framework can be applied to:

- Hyperparameter tuning in machine learning models
- Financial portfolio optimisation
- Supply chain optimisation
- Engineering design problems
- Reinforcement learning policy search

---

## 10. Summary

This project demonstrates how combining multiple optimisation strategies leads to stronger performance than relying on a single method. The final hybrid system integrates probabilistic modelling, directional search, clustering, and exploration strategies to effectively solve complex black-box optimisation problems.

The key insight is that **robust optimisation requires adaptability, not just accuracy**.
