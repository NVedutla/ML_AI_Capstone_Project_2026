# BBO Capstone Dataset Sheet

---

## 1. Dataset Overview

This dataset was generated as part of a Black-Box Optimisation (BBO) capstone project within a machine learning and AI course.

It contains sequential input-output observations for eight unknown mathematical functions (F1–F8). Each function represents a separate optimisation problem with different input dimensionalities and varying complexity.

The dataset is not static. Instead, it is **iterative and sequential**, meaning each new round of data depends on previously evaluated candidate solutions and the optimisation strategies used at that stage.

---

## 2. Motivation

The dataset was created for educational purposes to simulate real-world black-box optimisation problems where:

- The underlying function is unknown
- Only input-output feedback is available
- Evaluation is expensive and limited
- Learning must occur iteratively over time

The dataset supports the study of:

- Surrogate modelling approaches
- Bayesian optimisation techniques
- Exploration vs exploitation trade-offs
- Sequential decision-making under uncertainty
- Hybrid machine learning optimisation systems

It was provided as part of the ML/AI capstone learning environment.

---

## 3. Dataset Composition

The dataset consists of:

- **8 black-box functions (F1–F8)**
- **13 optimisation rounds (weekly iterations)**
- Continuous input spaces in the range **[0, 1]**

### Function Dimensionality

| Function | Input Dimension |
|----------|----------------|
| F1       | 2D             |
| F2       | 2D             |
| F3       | 3D             |
| F4       | 4D             |
| F5       | 4D             |
| F6       | 5D             |
| F7       | 6D             |
| F8       | 8D             |

### Data Format

Each record contains:
- Input vector: NumPy array of floats
- Output value: scalar float representing function evaluation score

---

## 4. Data Collection Process

The dataset was generated through an iterative optimisation loop:

1. Candidate input points were generated using different optimisation strategies
2. Each candidate was evaluated using hidden black-box functions
3. Output values were returned as scalar performance scores
4. Results were stored and reused in future rounds to improve sampling decisions

This creates a **closed-loop optimisation dataset**, where the data distribution evolves over time based on the algorithm’s behaviour.

---

## 5. Preprocessing and Feature Engineering

Several preprocessing steps were applied during modelling:

### 5.1 Dimensional Alignment

Because each function has different input sizes:

- Inputs shorter than the required dimension were **zero-padded**
- Inputs longer than required were **truncated**

This ensured compatibility across optimisation modules.

---

### 5.2 Normalisation

All input values were constrained to:

- Range: **[0, 1]**

This was enforced by the environment and maintained throughout modelling.

---

### 5.3 Output Transformation

For stability in Gaussian Process training:

- A log transformation was applied in some modules:
  \[
  y' = \log(|y| + \epsilon)
  \]

This helped reduce the impact of extreme output values.

---

### 5.4 Noise Handling

No explicit outlier removal was applied. Instead, robustness was achieved through:

- Gaussian Process regression (probabilistic modelling)
- Clustering-based filtering of high-value regions
- PCA-based dimensional structure analysis

---

## 6. Uses of the Dataset

This dataset supports experimentation in:

- Black-box optimisation benchmarking
- Bayesian optimisation
- Surrogate modelling (GP, SVM, etc.)
- Reinforcement learning in continuous spaces
- Exploration vs exploitation research
- High-dimensional optimisation analysis

---

## 7. Limitations

### 7.1 Synthetic Environment

The dataset is generated in a controlled simulation environment and does not reflect real-world noise distributions perfectly.

---

### 7.2 Limited Evaluation Rounds

Only 13 optimisation rounds are available, which limits long-term convergence analysis.

---

### 7.3 Fixed Dimensionality

Each function has fixed input dimensionality, limiting transfer learning between functions.

---

### 7.4 Sequential Dependency Bias

Each round depends on previous optimisation strategies, meaning the dataset is **policy-dependent** and not independently sampled.

---

## 8. Ethical Considerations

This dataset contains no personal or sensitive data.

It does not involve:

- Personal information
- Private datasets
- Real-world individuals

However, techniques developed using this dataset (e.g. optimisation, pricing, resource allocation) may be applicable in real-world domains where ethical considerations such as fairness, bias, and transparency become important.

---

## 9. Maintenance

This dataset is maintained solely for academic purposes as part of the BBO capstone project.

It is not updated after course completion and is intended as a static educational resource.

---

## 10. Summary

This dataset provides a structured environment for studying iterative black-box optimisation across multiple functions and dimensionalities.

It is particularly useful for evaluating:

- Gaussian Process surrogate models
- Bayesian optimisation strategies
- Hybrid machine learning optimisation pipelines

The sequential nature of the dataset allows analysis of how optimisation strategies evolve over time and improve performance under uncertainty.
