# Model Card – BBO Capstone Hybrid Optimisation System

## Model Description

**Input:**  
Continuous numerical vectors representing candidate solutions. Dimensions vary from 2D to 8D depending on the function (F1–F8).

**Output:**  
A scalar value representing the performance of the input vector on an unknown black-box function.

**Model Architecture:**  
The system is a hybrid optimisation framework combining:
- Gaussian Process Regression (surrogate model)
- Bayesian Optimisation (Expected Improvement)
- PCA-based direction learning
- Clustering-based region identification
- Local search and noise-based exploration

This is not a single predictive model but a multi-method optimisation pipeline.

---

## Performance

Performance was evaluated over multiple sequential optimisation rounds rather than a fixed dataset.

Key indicators of success:
- Improvement in function outputs over time
- Stability of selected query points
- Better performance consistency across different function dimensions

The hybrid model outperformed early random and single-method approaches by converging faster and exploring more effectively.

---

## LIMITATIONS

- Gaussian Processes become slower with larger datasets
- Performance depends heavily on sampling quality
- PCA and clustering may be unstable in low-data scenarios
- No guarantee of global optimum due to black-box nature of functions
- Sensitive to hyperparameter choices

---

## TRADE-OFFS

- Exploration vs exploitation: balancing new search vs refining known good regions
- Accuracy vs computational cost: more complex models improved results but required more computation
- Global vs local search: combining both improved robustness but increased system complexity

---

## ETHICAL CONSIDERATIONS

This system is designed for optimisation research and educational purposes. It does not involve personal or sensitive data and is not intended for real-world decision-making without further validation.

---

## INTENDED USE

- Black-box optimisation problems
- Machine learning research
- Algorithm benchmarking
- Educational use in optimisation and AI courses
