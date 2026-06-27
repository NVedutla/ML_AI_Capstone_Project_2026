#-----------------------------------------------
# CODE BASE FOR MODULE-13 - LOGISTIC REGRESSION
#-----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor #module-13
from sklearn.gaussian_process.kernels import Matern #module-13
from scipy.stats import norm  # for CDF and PDF in EI #module-13
import warnings #module-13
from sklearn.exceptions import ConvergenceWarning #module-13
warnings.filterwarnings("ignore", category=ConvergenceWarning) #module-13
from sklearn.svm import SVC #module-13


# =========================
# LOAD ORIGINAL DATA
# =========================
data_input_f1 = np.load('initial_inputs_f1.npy')
data_output_f1 = np.load('initial_outputs_f1.npy')

print("Original inputs:\n", data_input_f1)
print("Original outputs:\n", data_output_f1)

x = data_input_f1[:, 0]
y = data_input_f1[:, 1]
z = data_output_f1

# PLOT INITIAL DATA
plt.scatter(x, y, c=z, cmap='viridis')
plt.colorbar(label='Output (score)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Function 1 Initial Data')
plt.show()

# BEST POINT FROM ORIGINAL DATA
best_idx = np.argmax(data_output_f1)
x_best = data_input_f1[best_idx]
best_value = data_output_f1[best_idx]

print("Best output:", best_value)
print("Best input x_best:", x_best)

# SMALL GAUSSIAN NOISE SAMPLE
np.random.seed(42)  # reproducibility
x_new = x_best + np.random.normal(0, 0.05, size=2)
x_new = np.clip(x_new, 0, 1)
print("New query point x_new:", x_new)

# =================================================
# BAYESIAN OPTIMISATION PART (APPENDED) - module-13
# =================================================

# Copy original data (DO NOT overwrite originals)
X_bo = data_input_f1.copy()
y_bo = data_output_f1.copy().reshape(-1, 1)

# Stabilize very small values
y_bo = np.log(np.abs(y_bo) + 1e-12)

# GP model
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

# Expected Improvement function using scipy.stats.norm (fixed shape issue)
def expected_improvement(X_cand, model, y_best, xi=0.01):
    mu, sigma = model.predict(X_cand, return_std=True)
    mu = mu.ravel()
    sigma = sigma.ravel()

    with np.errstate(divide='warn'):
        improvement = mu - y_best - xi
        Z = improvement / sigma

        pdf = norm.pdf(Z)
        cdf = norm.cdf(Z)

        ei = improvement * cdf + sigma * pdf
        ei[sigma == 0.0] = 0.0  # works because shapes match

    return ei

# Function to optimize (replace with real function if needed)
def f(x):
    return np.exp(-((x[0]-0.7)**2 + (x[1]-0.7)**2)*50)

# Bayesian Optimization loop
for iteration in range(5):

    # Fit GP
    gp.fit(X_bo, y_bo)

    # Candidate points (randomly sampled)
    X_candidates = np.random.uniform(0, 1, (1000, 2))

    # Current best value
    y_best_bo = np.max(y_bo)

    # Compute Expected Improvement
    ei = expected_improvement(X_candidates, gp, y_best_bo)

    # Select next point to sample
    x_next = X_candidates[np.argmax(ei)]

    # Evaluate function
    y_next = f(x_next)
    y_next = np.log(np.abs(y_next) + 1e-12)

    # Append new point
    X_bo = np.vstack((X_bo, x_next))
    y_bo = np.vstack((y_bo, [[y_next]]))

    print(f"\n[BO] Iteration {iteration+1}")
    print("[BO] Next point:", x_next)
    print("[BO] Best value so far:", np.max(y_bo))

# =========================
# FINAL RESULTS
# =========================
print("\nFinal Bayesian Optimization best value:", np.max(y_bo))
best_idx_final = np.argmax(y_bo)
print("Best input after BO:", X_bo[best_idx_final])

#MOdule - 14. 
#Step-1 -  Use SVM to focus on the good regions only 
#Step-2 - Use Bayesian Optimisation to fine tune within that region (uses uncertainty and prediction)


# =========================
# LOAD DATA
# =========================
X = np.load('initial_inputs_f1.npy')
y = np.load('initial_outputs_f1.npy')

# =========================
# STEP 1: SVM (CLASSIFICATION)
# =========================

# Convert to binary labels (top 30% = good)
threshold = np.percentile(y, 70)
labels = (y >= threshold).astype(int)

svm = SVC(kernel='rbf', probability=True)
svm.fit(X, labels)

# =========================
# STEP 2: GENERATE CANDIDATES
# =========================
np.random.seed(42)
X_candidates = np.random.uniform(0, 1, (5000, 2))

# Get probability of being "good"
probs = svm.predict_proba(X_candidates)[:, 1]

# Keep only promising region
mask = probs > 0.6   # threshold (tune if needed)
X_promising = X_candidates[mask]

# If too few points, fallback to all
if len(X_promising) < 50:
    X_promising = X_candidates

print("Number of promising candidates:", len(X_promising))

# =========================
# STEP 3: BAYESIAN OPTIMISATION
# =========================

# Fit GP on real data
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
gp.fit(X, y)

# Expected Improvement
def expected_improvement(X_cand, model, y_best, xi=0.01):
    mu, sigma = model.predict(X_cand, return_std=True)
    mu = mu.ravel()
    sigma = sigma.ravel()

    with np.errstate(divide='warn'):
        improvement = mu - y_best - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# Compute EI only on promising region
y_best = np.max(y)
ei = expected_improvement(X_promising, gp, y_best)

# Select best next point
x_next = X_promising[np.argmax(ei)]

print("\nFinal suggested x_next:", x_next)

# =========================
# FORMAT FOR SUBMISSION
# =========================
submission = "-".join([f"{val:.6f}" for val in x_next])
print("Submit this:", submission)
