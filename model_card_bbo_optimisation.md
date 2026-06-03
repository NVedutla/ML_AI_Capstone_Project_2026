# Model Card – BBO Optimisation Approach

## Model Overview

**Model Name:** BBO Capstone Optimisation Strategy

**Version:** v1.0 (Rounds 1–10)

**Model Type:** Black-Box Optimisation (BBO)

**Developer:** Neeraja Vedutla

**Project:** BBO Capstone Project

**Description:**

This optimisation approach was developed to identify high-performing query points for eight unknown black-box functions. The objective was to maximise function outputs while operating under limited information and a restricted evaluation budget.

The strategy evolved over ten rounds by combining exploration and exploitation techniques informed by previous observations and function evaluations.

---

## Intended Use

### Primary Use

* Optimising unknown objective functions.
* Sequential decision-making problems.
* Bayesian optimisation experiments.
* Research and educational demonstrations of black-box optimisation.

### Target Users

* Students studying optimisation.
* Machine learning practitioners.
* Researchers evaluating optimisation strategies.

### Out-of-Scope Uses

This approach should not be used as the sole decision-making mechanism in:

* Healthcare diagnosis.
* Financial investment decisions.
* Hiring or recruitment systems.
* Safety-critical engineering systems.

Additional validation would be required before deployment in high-risk environments.

---

## Inputs and Outputs

### Inputs

For each function:

* Historical query points.
* Corresponding function evaluations.
* Search space constrained to values between 0 and 1.

Example:

Input:

x = [0.758593, 0.726086]

Output:

f(x) = 0.304354

### Outputs

The model produces:

* One candidate query point for the next optimisation round.
* Query values formatted to six decimal places.

---

## Optimisation Strategy

### Rounds 1–3

Initial exploration phase.

Methods:

* Random sampling.
* Broad coverage of search space.
* Identification of promising regions.

Goal:

* Gather information about function behaviour.

---

### Rounds 4–6

Bayesian optimisation phase.

Methods:

* Gaussian Process (GP) surrogate models.
* Expected Improvement (EI) acquisition function.
* Exploration–exploitation balancing.

Goal:

* Improve search efficiency.

---

### Rounds 7–9

Advanced optimisation phase.

Methods:

* GP surrogate modelling.
* Local search around best solutions.
* Candidate ranking.
* Hybrid exploration and exploitation.

Goal:

* Refine promising regions while maintaining diversity.

---

### Round 10

Transparency and interpretability phase.

Methods:

* Explicit documentation of decision rules.
* Feature importance through GP length scales.
* Uncertainty estimation.
* Query justification reporting.

Goal:

* Improve reproducibility and trustworthiness.

---

## Performance Summary

### Evaluation Metric

Primary metric:

* Function value returned by the black-box system.

Secondary metrics:

* Improvement over previous rounds.
* Consistency of optimisation.
* Search efficiency.

### Observed Results

The strategy successfully identified:

* High-performing regions for several functions.
* Significant improvements compared with early random exploration.
* Stable convergence behaviour on multiple functions.

Examples:

| Function | Best Observed Output |
| -------- | -------------------- |
| F1       | Near zero optimum    |
| F2       | 0.6770               |
| F5       | 2398.3640            |
| F7       | 2.4426               |
| F8       | 9.9367               |

Performance varied because each function exhibited different landscapes and levels of complexity.

---

## Assumptions

The optimisation strategy assumes:

1. Similar inputs produce similar outputs.
2. Function behaviour is sufficiently smooth for surrogate modelling.
3. Previous observations contain useful information about future regions.
4. The optimum lies within the bounded search space [0,1].

These assumptions improve efficiency but may not hold for highly discontinuous functions.

---

## Limitations

### Data Scarcity

Only a limited number of evaluations were available.

### Sampling Bias

Queries become concentrated around promising regions.

### Local Optima Risk

The optimisation process may converge prematurely.

### Surrogate Model Error

Gaussian Processes may poorly approximate highly irregular functions.

### Generalisability

Results are specific to the eight benchmark functions used in the capstone.

---

## Ethical Considerations

### Transparency

All optimisation decisions are documented.

### Reproducibility

The query generation process can be recreated using:

* Historical query data.
* Recorded outputs.
* Optimisation code.
* Hyperparameter settings.

### Accountability

Decision-making steps are traceable through:

* Query histories.
* Acquisition functions.
* Model configurations.

### Risk Mitigation

To reduce overconfidence:

* Exploration was retained throughout optimisation.
* Multiple candidate regions were evaluated.
* Uncertainty estimates were incorporated into decision making.

---

## Interpretability

The optimisation approach improves interpretability through:

* Explicit query selection criteria.
* Gaussian Process uncertainty estimates.
* Feature relevance inferred from kernel parameters.
* Documented reasoning for each optimisation round.

These measures support understanding of why specific query points were selected.

---

## Future Improvements

Potential enhancements include:

* Transformer-based surrogate models.
* Attention-based feature analysis.
* Ensemble Bayesian optimisation.
* Multi-objective optimisation.
* Automated hyperparameter tuning.
* Larger evaluation budgets.

---

## Version History

| Version | Description                                         |
| ------- | --------------------------------------------------- |
| v1.0    | Initial model card covering rounds 1–10             |
| Future  | Additional updates as optimisation strategy evolves |

---

## Contact

Project: BBO Capstone Project

Maintainer: Neeraja Vedutla

Repository: [GitHub Repository Link]
