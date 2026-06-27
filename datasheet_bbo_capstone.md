# Dataset Datasheet – BBO Capstone Project

---

## MOTIVATION

The dataset was created as part of a black-box optimisation (BBO) learning environment. The purpose was to evaluate different optimisation strategies and understand how machine learning methods can improve search efficiency in unknown function spaces.

The dataset was generated within the capstone project environment. No external organisation or funding was involved.

---

## COMPOSITION

The dataset contains:
- Input vectors (candidate solutions)
- Output values (function evaluations)

Each function (F1–F8) has a different input dimension:
- F1–F2: 2D
- F3: 3D
- F4–F5: 4D
- F6: 5D
- F7: 6D
- F8: 8D

There is no missing or confidential data.

---

## COLLECTION PROCESS

Data was collected iteratively across multiple rounds of optimisation. Each round involved:
1. Generating candidate input vectors
2. Evaluating them on hidden functions
3. Recording outputs
4. Using results to guide future search

The dataset grew sequentially over time as new optimisation rounds were completed.

---

## PREPROCESSING / CLEANING

Several preprocessing steps were applied:
- Padding or truncation of vectors to match required dimensions
- Normalisation of inputs to the range [0, 1]
- Log transformation applied to outputs in some models for stability

No sensitive data processing was required.

---

## USES

The dataset can be used for:
- Black-box optimisation research
- Machine learning model benchmarking
- Surrogate modelling experiments
- Bayesian optimisation studies

### Limitations for use:
- Not suitable for real-world prediction tasks
- Functions are synthetic and not interpretable
- Results may not generalise outside this environment

---

## DISTRIBUTION

The dataset is included within the GitHub repository for educational use only. It is not licensed for commercial use or external redistribution.

---

## MAINTENANCE

This dataset is static and will not be updated after the completion of the capstone project. It is maintained only for reproducibility and academic review.
