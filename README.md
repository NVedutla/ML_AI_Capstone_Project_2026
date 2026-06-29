# BBO Capstone Project – Hybrid Black-Box Optimisation (2026)
Imperial University


## PROJECT TITLE

Hybrid Black-Box Optimisation using Gaussian Processes, Bayesian Optimisation, and Learning-Based Search Strategies

## NON-TECHNICAL EXPLANATION (≈100 words)

This project is about solving optimisation problems where the underlying function is unknown. In simple terms, we are trying to find the best possible input that produces the highest output without knowing how the function works internally.

To do this, the system learns from past attempts and gradually improves its guesses. It uses machine learning techniques such as Gaussian Processes to predict good regions, Bayesian Optimisation to guide decisions, and clustering and PCA to identify patterns in successful areas.

Over time, the system becomes better at exploring and focusing its search, improving performance across multiple complex functions.

## DATA

The dataset is made up of black-box optimisation samples collected over 13 iterative rounds.

Each function (F1–F8) contains:
- Input vectors (continuous values between 0 and 1)
- Output values (performance score from hidden functions)

### Structure:
Data/
├── Week_1/
│   ├── inputs.txt
│   ├── outputs.txt
├── Week_2/
...
├── Week_13/


Each input is a NumPy array of different dimensionality depending on the function, and each output is a single numeric score.

## MODEL

The final system is a hybrid optimisation model built by combining multiple techniques learned throughout the module.

It includes:

- Gaussian Process Regression (to estimate unknown functions)
- Bayesian Optimisation using Expected Improvement
- Support Vector Machines (to filter promising regions early on)
- PCA-based directional search
- Clustering of high-performing regions
- Reinforcement learning-style exploration (epsilon-greedy strategy)
- Adaptive candidate generation

The key idea is simple:  
instead of relying on one method, we combine several strategies so the system can both explore new areas and refine good ones.

## HYPERPARAMETER OPTIMISATION

Key settings used across the project:

- Gaussian Process kernel: Matern (nu = 2.5)
- Noise level (alpha): 1e-6
- Expected Improvement exploration factor (xi): 0.01
- SVM kernel: RBF with probability estimates
- Candidate pool size: 1000–5000 points per iteration
- SVM filtering threshold: 0.6 probability
- RL exploration rate (epsilon): 0.1
- PCA components: adjusted based on function dimension

These values were refined gradually during experimentation across modules.

## CODE STRUCTURE

The repository follows the learning progression of the course:
Code/
├── week_1.py
├── week_2.py
├── ...
├── week_13.py
├── Module_13/
├── Module_14/
├── ...
├── Module_24/
└── 


Each file represents a step forward in improving the optimisation strategy.

## RESULTS SUMMARY

Across all functions (F1–F8), performance improved steadily as more advanced methods were introduced.

### What worked best:
- Bayesian Optimisation improved efficiency early on
- SVM filtering helped focus search regions
- PCA improved performance in higher dimensions
- Clustering improved stability in later stages
- RL-style exploration prevented getting stuck in local optima

### Overall outcome:
The final system performs consistently across all functions, especially in higher-dimensional cases where simple methods struggled.

## KEY LEARNINGS
No single algorithm is enough for black-box optimisation
Combining multiple methods gives better stability and performance
Surrogate models reduce unnecessary expensive evaluations
Exploration vs exploitation balance is critical
Dimensionality-aware methods (like PCA) improve efficiency

## FUTURE IMPROVEMENTS

If the project continued, improvements could include:

Combining multiple surrogate models (GP + Neural Networks)
Adaptive kernel selection for Gaussian Processes
More advanced reinforcement learning policies

## DATA SHEET & MODEL CARD

Please refer to:

bbo_capstone_dataset_sheet.md
bbo_model_card_optimisation.md

for full technical and ethical documentation.

## AUTHOR

BBO Capstone Project – Machine Learning & AI (2026)
Imperial University

GitHub
https://github.com/NVedutla/ML_AI_Capstone_Project_2026

## LICENSE

For academic use only.
Automated strategy selection depending on function type

## REPRODUCIBILITY

To run the project:

```bash
pip install -r requirements.txt
python Code/final_model.py
FINAL QUERY: x1-x2-x3-...
Best value: <float>

