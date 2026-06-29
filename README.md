# BBO Capstone Project – Hybrid Black-Box Optimisation System (Dec 2025 to June 2026) - Imperial University

## Project Overview

This project is a Black-Box Optimisation (BBO) capstone developed over a 24-module learning journey. The goal of the project is to optimise a set of unknown mathematical functions (F1–F8) using iterative machine learning and optimisation strategies.

Rather than relying on a single algorithm, the system evolves into a **hybrid optimisation framework** that combines multiple techniques including Gaussian Processes, Bayesian Optimisation, clustering, PCA, reinforcement learning-inspired exploration, and surrogate modelling.

The key idea behind the project is:

> **Optimisation performance improves when multiple learning strategies are combined to balance exploration, exploitation, and uncertainty modelling.**

---

## Repository Structure

The repository is organised to reflect the progression of the capstone:

```
ML_AI_Capstone_Project_2026/
│
├── Code/
│   ├── Module_13/ → Gaussian Process Surrogates
│   ├── Module_14/ → Hyperparameter Optimisation
│   ├── Module_15/ → Expected Improvement (Bayesian Optimisation)
│   ├── Module_16/ → Local Search Enhancements
│   ├── Module_17/ → Candidate Generation Strategies
│   ├── Module_18/ → Hybrid Optimisation Systems
│   ├── Module_19/ → Feature Weighting (Attention-inspired)
│   ├── Module_20/ → Bayesian Optimisation Refinement
│   ├── Module_21/ → Transparency & Interpretability Methods
│   ├── Module_22/ → Clustering-Based Optimisation
│   ├── Module_23/ → PCA-Guided Search
│   ├── Module_24/ → Reinforcement Learning-Inspired Search
│   └── Final_Model/
│
├── Data/
│   ├── Inputs/
│   ├── Outputs/
│   └── Full_History/
│
├── Documentation/
│   ├── model_card_optimisation.md
│   ├── data_sheet_bbo_capstone.md
│   └── reflections/
│
├── Presentation/
└── Images/
```

---

## Objective

The objective of this project is to approximate and optimise hidden black-box functions using only input-output observations.

The system must:
- Explore unknown function landscapes efficiently
- Balance exploration vs exploitation
- Adapt strategies over time
- Improve performance across multiple function types (F1–F8)
- Generalise across different input dimensions (2D–8D)

---

## Core Optimisation Approach

The final system is built around a **hybrid optimisation pipeline**:

1. Surrogate modelling using Gaussian Processes
2. Bayesian optimisation using Expected Improvement
3. Local search around high-performing regions
4. Feature weighting to prioritise important dimensions
5. Clustering to identify promising regions
6. PCA to learn directional structure of the search space
7. Reinforcement learning-inspired exploration
8. Adaptive candidate generation
## Evolution of the Model (Modules 13 → 24)

The system was not built in a single step. Instead, it evolved gradually as each module introduced new ideas that were directly tested in the optimisation pipeline.

### Module 13–14: Gaussian Processes & Hyperparameter Optimisation
The initial improvement came from introducing Gaussian Process (GP) regression as a surrogate model. This allowed the system to approximate expensive black-box functions based on previous observations.

Hyperparameter tuning (especially kernel length scale) improved model stability and prediction quality.

---

### Module 15–16: Bayesian Optimisation & Expected Improvement
Expected Improvement (EI) was introduced to guide candidate selection.

This marked a key shift:
- From random exploration
- To probability-guided exploitation of promising regions

Local search was also added to refine solutions around known good points.

---

### Module 17–18: Hybrid Candidate Generation
At this stage, the system began combining multiple strategies:
- Random exploration
- Surrogate predictions
- Local exploitation

This hybrid structure improved robustness across all function types (F1–F8), especially higher-dimensional cases.

---

### Module 19–20: Feature Weighting & Refinement
An attention-inspired weighting mechanism was introduced to prioritise more influential input dimensions.

This improved performance stability by:
- Reducing noise from less important features
- Increasing focus on high-impact dimensions

Bayesian optimisation was also refined for better candidate selection.

---

### Module 21: Transparency & Interpretability
This module focused on making the optimisation process explainable.

Key additions:
- Gaussian Process interpretability analysis
- Kernel sensitivity tuning
- Logging and structured evaluation of candidate decisions

This improved understanding of why certain points were selected.

---

### Module 22: Clustering-Based Optimisation
Clustering was introduced to group high-performing samples.

Key idea:
> Good solutions often exist in regions, not isolated points.

The system:
- Identified top-performing quartile
- Computed cluster centroids
- Generated new candidates around these regions

---

### Module 23: PCA-Guided Search
Principal Component Analysis (PCA) was used to identify dominant directions in the search space.

This allowed:
- Directional optimisation instead of purely random sampling
- Faster convergence in structured landscapes

---

### Module 24: Reinforcement Learning-Inspired Exploration
The final module introduced adaptive exploration using:
- Weighted reward sampling
- Directional learning
- Epsilon-based exploration noise

This allowed the system to dynamically balance:
- Exploitation of known good regions
- Exploration of unknown areas

---

## Results Summary

Across the full capstone project, performance improved steadily as more advanced optimisation strategies were introduced.

Key outcomes:
- Better stability across all 8 benchmark functions (F1–F8)
- Improved convergence in higher-dimensional functions (F6–F8)
- More efficient exploration using surrogate-guided search
- Reduced randomness in candidate selection

The final model successfully combined multiple optimisation paradigms into a single unified framework.

---

## Technologies Used

- Python 3
- NumPy
- SciPy
- Scikit-learn
- Gaussian Process Regression
- PCA (Principal Component Analysis)

---

## How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/NVedutla/ML_AI_Capstone_Project_2026.git
```

2. Navigate to code directory:
```bash
cd ML_AI_Capstone_Project_2026/Code
```

3. Run final model:
```bash
python final_model.py
```

---

## Key Learnings

- Optimisation improves through **layered strategy design**, not single algorithms
- Surrogate models reduce unnecessary evaluations
- Exploration vs exploitation balance is critical
- Structural patterns (PCA/clustering) improve convergence
- Interpretability improves debugging and model refinement

---

## Future Improvements

If more time were available, the system could be improved by:
- Ensemble surrogate models (GP + NN + SVR)
- Adaptive kernel learning for Gaussian Processes
- Meta-learning for automatic strategy selection
- More advanced reinforcement learning policies

---

## Author

**BBO Capstone Project – ML/AI 2026**

GitHub Repository:
https://github.com/NVedutla/ML_AI_Capstone_Project_2026

---

## License

For academic and educational use only.

Each module added in the course contributed a new component to this pipeline, progressively improving robustness and performance.
