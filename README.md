# BBO Capstone Project – Hybrid Black-Box Optimisation (Dec 2025 – June 2026)
Imperial University

---

## PROJECT TITLE
Hybrid Black-Box Optimisation System using Gaussian Processes, Bayesian Optimisation, PCA and Reinforcement Learning-Inspired Search

---

## NON-TECHNICAL EXPLANATION (≈100 words)

This project explores how to optimise unknown mathematical functions using only input and output examples, without knowing the underlying formula. The system learns from past attempts and gradually improves its guesses over time. It uses machine learning methods such as Gaussian Processes to predict good solutions, Bayesian Optimisation to guide search decisions, and clustering and PCA to identify patterns in successful regions. Over time, the model becomes more efficient at finding high-performing inputs across multiple complex functions. The goal is to simulate real-world decision-making problems where the best solution must be discovered through experimentation and learning rather than direct calculation.

---

## DATA

The dataset consists of black-box function evaluations (F1–F8) provided over 13 weekly modules.

Each function contains:
- Inputs: vectors of varying dimensionality (2D to 8D depending on function)
- Outputs: real-valued scores representing function performance

Data structure:
Data/
├── Week_1/
│ ├── inputs.txt
│ └── outputs.txt
├── Week_2/
...
├── Week_13/


Each `inputs.txt` contains a list of numpy arrays representing sampled points.
Each `outputs.txt` contains corresponding evaluation scores.

The data is generated as part of the BBO capstone environment and is not externally sourced.

---

## MODEL

The final model is a **hybrid optimisation system** combining multiple techniques:

- Gaussian Process Regression (surrogate modelling)
- Bayesian Optimisation (Expected Improvement acquisition function)
- Support Vector Machines (region filtering in early stages)
- PCA-based directional search
- Clustering-based region identification
- Reinforcement Learning-inspired exploration (epsilon-greedy sampling)
- Adaptive candidate generation across multiple strategies

The model evolves over time by combining exploration (random search) and exploitation (guided search from learned models).

---

## HYPERPARAMETER OPTIMISATION

Key hyperparameters used:

- Gaussian Process kernel: Matern kernel (nu=2.5)
- Alpha (noise level): 1e-6
- Expected Improvement exploration factor (xi): 0.01
- SVM RBF kernel with probability estimates
- Candidate pool sizes: 1000–5000 samples per iteration
- Exploration threshold (SVM filtering): probability > 0.6
- RL epsilon (exploration rate): 0.1
- PCA components: dynamic based on function dimension

These were tuned iteratively across modules based on performance trends across F1–F8.

---

## CODE STRUCTURE

The repository is organised by module progression:
Code/
├── week_1.py
├── week_2.py
├── ...
├── week_13.py
├── Module_13/
├── Module_14/
├── ...
├── Module_24/


Each file represents incremental improvements to the optimisation strategy.

---

## RESULTS

The model shows steady improvement across all benchmark functions (F1–F8) over time.

### Key Observations:
- Early modules relied on simple local search and Gaussian noise exploration.
- Mid-stage modules (15–18) introduced Bayesian Optimisation and surrogate modelling, improving convergence speed.
- Later modules (19–24) added PCA, clustering, and reinforcement learning-inspired exploration, improving robustness.

### Final Performance Summary:
- Strongest stability achieved in higher-dimensional functions (F6–F8)
- Improved convergence consistency across noisy landscapes
- Reduced randomness in candidate selection
- Better balance between exploration and exploitation

Overall, the hybrid system significantly outperformed early baseline approaches.

---

## MODEL CARD

See `model_card_optimisation.md` for detailed model architecture, performance metrics, limitations, and trade-offs.

---

## DATA SHEET

See `data_sheet_bbo_capstone.md` for dataset motivation, structure, collection process, and limitations.

---

## KEY LEARNINGS

- Combining multiple optimisation strategies is more effective than relying on one algorithm
- Surrogate models significantly reduce expensive evaluations
- Exploration vs exploitation balance is critical for success
- Dimensionality reduction (PCA) improves search efficiency
- Iterative experimentation leads to better system design than static modelling

---

## FUTURE IMPROVEMENTS

- Ensemble surrogate models (GP + Neural Networks + SVR)
- Adaptive kernel learning for Gaussian Processes
- Meta-learning for automatic strategy selection
- More advanced reinforcement learning-based exploration policies

---

## AUTHOR

BBO Capstone Project – ML/AI 2026  
Imperial University

GitHub Repository:  
https://github.com/NVedutla/ML_AI_Capstone_Project_2026

---

## LICENSE

For academic and educational use only.
