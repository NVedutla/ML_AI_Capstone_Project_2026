# BBO Capstone Project – Hybrid Black-Box Optimisation

## NON-TECHNICAL EXPLANATION OF MY PROJECT

This project explores how to solve complex optimisation problems where the underlying function is unknown. The goal is to find the best possible input values that maximise or improve an unknown output using only feedback from previous attempts.

To achieve this, I built a hybrid machine learning system that learns from past evaluations and uses this information to suggest better future guesses. The system combines techniques such as Gaussian Processes, clustering, PCA, and reinforcement learning-style exploration.

Over time, the model improves its understanding of the search space and becomes more efficient at finding high-performing solutions across multiple functions with different dimensions.

---

## DATA

The dataset consists of multiple rounds of black-box optimisation results for functions F1 to F8. Each function has a different input dimension ranging from 2D to 8D.

Each record includes:
- Input vectors (candidate solutions)
- Output values (function evaluations)

The data was generated during the capstone simulation process, where each new round builds on previous results. No external datasets were used.

---

## MODEL

The final system is a hybrid optimisation pipeline combining:

- Gaussian Process Regression (surrogate modelling)
- Expected Improvement (Bayesian optimisation acquisition function)
- PCA (directional structure discovery)
- Clustering (identifying high-performing regions)
- Local search (exploitation around best solutions)
- Random exploration (to avoid local optima)

This combination allows the system to balance exploration and exploitation while adapting to different function landscapes.

---

## HYPERPARAMETER OPTIMSATION

Hyperparameters were tuned manually and iteratively across rounds:

- Gaussian Process kernel (Matern kernel with different length scales)
- Exploration constant (xi in Expected Improvement)
- Step size for local search
- Noise levels for exploration
- Number of candidate samples per iteration
- Clustering thresholds (top percentile selection)

These were adjusted based on performance improvements observed across rounds rather than a fixed grid search.

---

## RESULTS

The system showed consistent improvement across all benchmark functions (F1–F8). Early approaches relied heavily on random sampling, but later versions became more structured and data-driven.

Key improvements included:
- Faster convergence to high-performing regions
- More stable results across different function dimensions
- Better balance between exploration and exploitation

The final hybrid approach performed significantly better than early baseline strategies.

---

## CONTACT DETAILS

GitHub: https://github.com/NVedutla/ML_AI_Capstone_Project_2026
