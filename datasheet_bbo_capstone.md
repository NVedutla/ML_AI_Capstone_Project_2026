# Datasheet for BBO Capstone Project Dataset

## 1. Motivation

### What task does this dataset help solve?

This dataset supports a Black-Box Optimisation (BBO) task. The goal is to identify input values that maximise or minimise unknown objective functions without access to their mathematical form. The dataset records the history of optimisation queries and function evaluations collected throughout the capstone project.

### Who created it and why?

The dataset was created by me as part of the BBO Capstone Project. It was developed to support experimentation with optimisation strategies, including Bayesian Optimisation, Gaussian Process Regression, exploration-exploitation trade-offs and transformer-inspired methods.

### Was it funded or supported by an organisation?

The dataset was generated as part of my AI / ML coursework project with Imperial University and was not directly funded by an external organisation.

---

## 2. Composition

### What does the dataset contain?

The dataset contains:

* Input query vectors submitted to the BBO system.
* Function evaluations returned by the system.
* Historical optimisation data collected over ten rounds.
* Eight separate unknown objective functions (F1–F8).

### Dataset size

* 10 optimisation rounds.
* 8 objective functions.
* Input dimensions ranging from 2 to 8 variables.
* 80 total function evaluations (10 rounds × 8 functions).

### Data format

Inputs are stored as floating-point arrays:

```text
[0.758593, 0.726086]
[0.522584, 0.641593, 0.359176]
[0.081283, 0.059042, ..., 0.931456]
```

Outputs are floating-point numerical values:

```text
0.3043538085798325
-3.434147297215279
1694.8396824139172
```

### Missing data

No missing values were observed in the collected evaluations.

### Relationships between instances

Each query vector is linked directly to one corresponding function evaluation. Query histories across rounds are sequentially related because later queries are generated using information from earlier evaluations.

### Privacy and sensitive information

The dataset contains only synthetic numerical values. No personal, sensitive or identifiable information is included.

---

## 3. Collection Process

### How was the data collected?

Data was collected through iterative interaction with a black-box optimisation platform. For each round:

1. Query points were submitted.
2. Function outputs were returned.
3. Previous results informed future query selection.

### Sampling strategy

The sampling strategy evolved over time:

**Rounds 1–3**

* Broad exploration.
* Random and heuristic sampling.

**Rounds 4–6**

* Bayesian Optimisation.
* Gaussian Process surrogate models.

**Rounds 7–8**

* Expected Improvement acquisition functions.
* Local exploitation around promising regions.

**Rounds 9–10**

* Transformer-inspired candidate ranking.
* Attention-based scoring concepts.
* Increased emphasis on transparency and interpretability.

### Time frame

Data was collected sequentially across ten optimisation rounds during the capstone project.

### Ethical considerations

No human participants were involved. No consent, privacy or institutional review requirements applied because all data consisted of synthetic numerical evaluations.

---

## 4. Preprocessing, Cleaning and Labelling

### Preprocessing applied

Several preprocessing steps were used:

* Input validation.
* Normalisation of values to the range [0,1].
* Log transformation of outputs when numerical ranges became extremely large.
* Removal of invalid candidate points.
* Formatting data into NumPy arrays for modelling.

### Cleaning

Data consistency checks were performed to ensure:

* Correct dimensionality.
* No empty arrays.
* No invalid numerical values.

### Labelling

No manual labels were added. Function outputs acted as optimisation targets.

### Preservation of raw data

Original query inputs and outputs were preserved alongside processed versions.

---

## 5. Uses

### Intended uses

This dataset is intended for:

* Black-box optimisation research.
* Bayesian Optimisation experiments.
* Surrogate model evaluation.
* Exploration versus exploitation analysis.
* Interpretability and transparency studies.

### Inappropriate uses

This dataset should not be used for:

* Medical decision-making.
* Financial decision-making.
* Safety-critical optimisation.
* Benchmarking real-world industrial systems.

### Risks and biases

Potential biases include:

* Concentration of samples near high-performing regions.
* Reduced exploration in later rounds.
* Small sample size relative to the search space.
* Assumptions that local improvements indicate globally promising regions.

---

## 6. Distribution

### Availability

The dataset is distributed through the project's GitHub repository.

### Access

The dataset is available to instructors, reviewers and researchers interested in optimisation methods.

### Licensing

The dataset follows the same licence as the associated capstone repository.

### Cost

No fee is required for access.

---

## 7. Maintenance

### Who maintains the dataset?

The student who created the capstone project maintains the dataset.

### Version control

Version history is managed through GitHub commits.

### Updates

Additional rounds may be added if optimisation continues beyond the current project stage.

### Long-term storage

The GitHub repository serves as the primary archive and record of dataset versions.

---

## Additional Notes

Transparency and reproducibility were important design goals throughout the project. Query generation strategies, optimisation assumptions and modelling decisions were documented to allow future researchers to understand and reproduce the optimisation process.
