# BBO Capstone Project – Black-Box Optimisation

## Overview

This project implements a **Black-Box Optimisation (BBO)** framework to optimise eight unknown mathematical functions (F1–F8). The objective is to find optimal input vectors using limited function evaluations without access to gradient or functional form information.

The approach combines:
- Bayesian Optimisation (Gaussian Processes)
- Expected Improvement (EI) acquisition function
- Random exploration strategies
- SVM-based filtering (early rounds)
- Local search refinement
- Hybrid exploration–exploitation strategies

The project is structured across **13 optimisation rounds**, where each round improves query selection based on newly observed data.

---

## Problem Description

We are given 8 unknown black-box functions:

- Input space: continuous values in range [0, 1]
- Dimensionality: 2D to 8D depending on function
- Output: scalar real-valued function output
- No access to gradients or internal function structure

### Objective:
Maximise each function using efficient query selection.

---

## Methodology

### 1. Exploration Phase (Rounds 1–3)
- Random uniform sampling
- Initial data collection
- Broad search space coverage

### 2. Bayesian Optimisation Phase (Rounds 4–6)
- Gaussian Process Regression (GPR)
- Matern kernel
- Expected Improvement (EI) acquisition function

### 3. Hybrid Optimisation Phase (Rounds 7–8)
- SVM-based classification of promising regions
- Batch Bayesian Optimisation
- Constant liar strategy
- Local refinement around best points

### 4. Advanced & Interpretability Phase (Rounds 9–10)
- Hybrid surrogate ranking
- Transformer-inspired scoring (attention-style selection)
- Improved local exploitation
- Increased focus on transparency and interpretability

---

## Key Features

- Gaussian Process surrogate modelling
- Uncertainty-aware decision making (EI)
- Log transformation of outputs for numerical stability
- Hybrid exploration + exploitation strategy
- Local search refinement
- Transparent decision tracking

---

## Performance Summary

Performance is evaluated using:
- Best observed function value
- Improvement across rounds
- Stability of optimisation
- Efficiency of sampling strategy

### Observations:
- Strong convergence observed in higher-dimensional functions (F5–F8)
- Slower convergence in noisy or low-dimensional functions (F1–F3)
- Hybrid strategies improved performance over pure random sampling
- Diminishing returns observed in later rounds

---

## Assumptions

- Functions are smooth enough for Gaussian Process modelling
- Past evaluations are informative for future predictions
- Global optimum lies within bounded domain [0,1]
- Observations are noise-free or low-noise

---

## Limitations

- Gaussian Processes scale poorly with large datasets
- Risk of convergence to local optima
- Sampling bias toward high-performing regions
- Limited exploration in later rounds
- Sensitive to kernel and hyperparameter selection

---

## Ethical Considerations

Although synthetic, this project demonstrates key principles of responsible AI:

- Transparency in optimisation decisions
- Reproducibility of results
- Clear documentation of assumptions and limitations
- Awareness of bias introduced by sampling strategies

---

## Reproducibility

To reproduce results:

1. Run rounds sequentially (Round 1 → Round 10)
2. Use fixed random seeds
3. Maintain identical GP kernel settings
4. Store all input-output query pairs

---

## Files in This Repository

### 📊 Dataset Documentation
- `datasheet_bbo_capstone.md`  
  → Full dataset description including collection, structure, and usage

### 🤖 Model Documentation
- `model_card_bbo_optimisation.md`  
  → Description of optimisation strategy, performance, and limitations

### 💻 Code
- `round_01.py` → `round_10.py` (or main optimisation script)
- Supporting utility scripts (if applicable)

---

## Future Improvements

- Deep kernel Gaussian Processes
- Transformer-based surrogate models
- Reinforcement learning for adaptive exploration
- Multi-objective optimisation
- Improved uncertainty estimation techniques

---

## Author

BBO Capstone Project Submission  
Module: Advanced Machine Learning / Optimisation  
Focus: Bayesian Optimisation, Interpretability, and Scaling Behaviour

---

## Documentation Links

- 📄 Datasheet: `datasheet_bbo_capstone.md`
- 📘 Model Card: `model_card_bbo_optimisation.md`

---

## License

Educational use only. Not intended for production deployment.
