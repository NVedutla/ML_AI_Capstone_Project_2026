# BBO - Model Card – Optimisation

---

## 1. Model Overview

This model is a **Hybrid Black-Box Optimisation System** designed to optimise unknown mathematical functions (F1–F8) using only input-output observations.

The model does not have access to the underlying function definitions. Instead, it learns patterns from observed evaluations and progressively improves its search strategy over time.

The final system combines multiple machine learning and optimisation techniques into a unified framework.

---

## 2. Input and Output

### Inputs
- Continuous numerical vectors
- Dimensionality varies by function:
  - F1–F2: 2D
  - F3: 3D
  - F4–F5: 4D
  - F6: 5D
  - F7: 6D
  - F8: 8D
- All values are constrained to the range **[0, 1]**

### Outputs
- Scalar real-valued function score
- Represents the performance or quality of a given input vector

---

## 3. Model Architecture

The model is a **multi-stage hybrid optimisation pipeline** consisting of:

### 3.1 Gaussian Process Regression (Surrogate Model)
- Used to approximate unknown black-box functions
- Kernel: Matern (nu = 2.5)
- Provides mean prediction and uncertainty estimation

---

### 3.2 Bayesian Optimisation (Expected Improvement)

- Uses Expected Improvement (EI) acquisition function
- Balances exploration and exploitation
- Selects next best candidate based on predicted gain

---

### 3.3 Support Vector Machine (SVM) Filtering

- Used in early modules to classify “high-quality regions”
- RBF kernel with probability estimates
- Filters candidate space before expensive evaluation

---

### 3.4 PCA-Based Directional Search

- Principal Component Analysis identifies dominant search directions
- Reduces random exploration in structured landscapes
- Improves convergence speed in higher-dimensional functions

---

### 3.5 Clustering-Based Optimisation

- Groups high-performing samples
- Focuses search around cluster centroids
- Helps exploit promising regions efficiently

---

### 3.6 Reinforcement Learning-Inspired Exploration

- Uses epsilon-greedy exploration strategy
- Introduces controlled randomness
- Prevents premature convergence to local optima

---

### 3.7 Adaptive Candidate Generation

- Dynamically adjusts sampling strategy based on performance history
- Combines:
  - Random sampling
  - Surrogate-guided sampling
  - Local search refinement

---

## 4. Training / Optimisation Process

The model is not trained in a traditional supervised learning sense.

Instead, it follows a **sequential optimisation loop**:

1. Generate candidate input points
2. Evaluate using black-box function
3. Store input-output pair
4. Fit surrogate model (GP)
5. Select next candidates using EI / SVM / RL logic
6. Repeat iteratively over 13+ rounds

This creates a **self-improving feedback loop system**.

---

## 5. Performance Summary

The model was evaluated across all functions (F1–F8) over multiple rounds.

### Key Observations:

- Performance improved steadily across iterations (Modules 13 → 24)
- Strongest improvements observed in higher-dimensional functions (F6–F8)
- Bayesian Optimisation significantly improved convergence speed
- PCA and clustering improved stability in later stages
- RL-inspired exploration improved robustness and prevented stagnation

### Overall Behaviour:

- Early stages: high exploration, unstable convergence
- Mid stages: surrogate-guided improvement
- Final stages: hybrid stable optimisation system

---

## 6. Limitations

### 6.1 Black-Box Dependency
The model relies entirely on observed outputs and cannot access true function structure.

---

### 6.2 Limited Data Rounds
Only a finite number of optimisation rounds are available, limiting long-term convergence evaluation.

---

### 6.3 Computational Cost
Gaussian Processes scale poorly with very large datasets.

---

### 6.4 Sensitivity to Hyperparameters
Performance depends on:
- Kernel selection
- Exploration rate (epsilon)
- Candidate sampling size
- Threshold values for filtering methods

---

## 7. Trade-offs

### Exploration vs Exploitation
- High exploration improves discovery of new regions
- High exploitation improves refinement of known good regions
- The model balances both using EI + epsilon-greedy logic

### Model Complexity vs Stability
- Adding multiple modules improves performance
- But increases tuning complexity and computation cost

---

## 8. Ethical Considerations

This model does not use personal or sensitive data.

However, the optimisation methods used (e.g. surrogate modelling, automated decision systems) may be applied in real-world scenarios such as:

- Pricing optimisation
- Resource allocation
- Recommendation systems

In such contexts, fairness, transparency, and bias mitigation would be important considerations.

---

## 9. Intended Use

This model is intended for:

- Educational black-box optimisation problems
- Research into surrogate-based optimisation
- Benchmarking optimisation strategies
- Studying exploration vs exploitation trade-offs

It is not intended for real-world deployment without adaptation.

---

## 10. Summary

This hybrid optimisation model demonstrates how combining multiple machine learning techniques can significantly improve performance on black-box optimisation problems.

The system evolves from simple surrogate modelling into a structured hybrid framework that integrates Bayesian optimisation, SVM filtering, clustering, PCA, and reinforcement learning-inspired exploration.

The key insight is that **no single algorithm is sufficient**, but a carefully designed combination of methods can produce robust and scalable optimisation behaviour.
