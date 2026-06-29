# BBO Capstone Dataset Sheet

## 1. Dataset Overview

This dataset was generated as part of a Black-Box Optimisation (BBO) capstone project. The dataset consists of multiple rounds of input-output observations for eight unknown mathematical functions (F1–F8).

Each function represents a different black-box optimisation problem with varying dimensionality and complexity.

The dataset is sequential and iterative, meaning each round builds upon previous evaluations.

---

## 2. Motivation

The dataset was created for educational and research purposes within a machine learning optimisation course.

The main objectives were:
- To simulate real-world black-box optimisation problems
- To evaluate different optimisation strategies over time
- To support iterative improvement of surrogate-based models
- To study exploration vs exploitation trade-offs

The dataset was provided as part of the ML/AI capstone learning environment.

---

## 3. Composition

The dataset consists of:

- 8 functions (F1–F8)
- 12 optimisation rounds
- Each function has different input dimensions:

| Function | Dimensions |
|----------|------------|
| F1       | 2D         |
| F2       | 2D         |
| F3       | 3D         |
| F4       | 4D         |
| F5       | 4D         |
| F6       | 5D         |
| F7       | 6D         |
| F8       | 8D         |

### Data Structure

Each record contains:
- Input vector (continuous values in range [0, 1])
- Output scalar value (black-box function evaluation)

---

## 4. Data Collection Process

The dataset was generated through an iterative optimisation environment:

1. At each round, candidate solutions were submitted
2. The environment evaluated each candidate using hidden functions
3. Outputs were returned as scalar values
4. Results were stored and reused in future optimisation steps

This created a sequential decision-making dataset rather than a static dataset.

---

## 5. Preprocessing

Several preprocessing steps were applied during modelling:

### 5.1 Dimensional Alignment
Because each function had different input sizes:
- Inputs were padded with zeros if too short
- Inputs were truncated if too long

### 5.2 Normalisation
All inputs were constrained to:
- Range: [0, 1]

### 5.3 Log Transformation (used in some modules)
To stabilise extreme output values:
- Log(|y| + ε) was used for Gaussian Process training

### 5.4 Outlier Handling
No explicit removal of outliers was performed, but robust models (GP, clustering, PCA) were used to reduce their impact.

---

## 6. Uses of the Dataset

This dataset can be used for:

- Black-box optimisation benchmarking
- Surrogate modelling experiments
- Bayesian optimisation research
- Reinforcement learning in continuous spaces
- Exploration vs exploitation studies

---

## 7. Limitations

The dataset has several limitations:

### 7.1 Synthetic Nature
The functions are artificially generated and may not reflect real-world noise patterns.

### 7.2 Limited Observations
Only 12 rounds of observations are available, limiting long-term convergence analysis.

### 7.3 Dimensional Constraints
Each function has fixed dimensionality, limiting transfer learning across functions.

### 7.4 Sequential Bias
Later samples depend on earlier optimisation strategies, introducing sampling bias.

---

## 8. Ethical Considerations

There are no direct ethical risks associated with this dataset as it does not contain:
- Personal data
- Sensitive information
- Real-world individuals

However, the optimisation techniques developed using this dataset may be applied in real-world domains such as pricing or resource allocation, where ethical considerations would become important.

---

## 9. Maintenance

This dataset is maintained as part of an academic capstone project.

It is not updated after completion of the course.

---

## 10. Summary

The BBO dataset provides a structured environment for studying iterative optimisation strategies across multiple function types and dimensions. It is particularly useful for evaluating surrogate models, Bayesian optimisation techniques, and hybrid machine learning approaches in controlled settings.
