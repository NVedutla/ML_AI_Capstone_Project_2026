#---------------------------------------------------------
# CODE BASE FOR MODULE-13 - LOGISTIC REGRESSION - Week-1
#----------------------------------------------------------
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

#===========
# OUTPUTS
#===========
# Original inputs:
 # [[0.31940389 0.76295937]
 # [0.57432921 0.8798981 ]
 # [0.73102363 0.73299988]
 # [0.84035342 0.26473161]
 # [0.65011406 0.68152635]
 # [0.41043714 0.1475543 ]
 # [0.31269116 0.07872278]
 # [0.68341817 0.86105746]
 # [0.08250725 0.40348751]
 # [0.88388983 0.58225397]]
# Original outputs:
 # [ 1.32267704e-079  1.03307824e-046  7.71087511e-016  3.34177101e-124
 # -3.60606264e-003 -2.15924904e-054 -2.08909327e-091  2.53500115e-040
 # 3.60677119e-081  6.22985647e-048]

#=====================================
# FURTHER REFINEMENTS FOR OBSERVATIONS
#=====================================
#---------------------------------------------
# Module-14 - Support Vector Machine - Week-1
#---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning

# ---------------------------
# Ignore convergence warnings
# ---------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =========================
# LOAD FUNCTION 2 DATA
# =========================
data_input_f2 = np.load('initial_inputs_f2.npy')
data_output_f2 = np.load('initial_outputs_f2.npy')

print("Function 2 inputs:\n", data_input_f2)
print("Function 2 outputs:\n", data_output_f2)

x = data_input_f2[:, 0]
y = data_input_f2[:, 1]
z = data_output_f2

# Plot initial data
plt.figure(figsize=(6,5))
plt.scatter(x, y, c=z, cmap='viridis', s=80)
plt.colorbar(label='Output (score)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Function 2 Initial Data')
plt.show()

# Find best point in initial dataset
best_idx = np.argmax(data_output_f2)
best_point = data_input_f2[best_idx]
print("Best initial point:", best_point)
print("Best initial output:", data_output_f2[best_idx])

# Small Gaussian noise sample
np.random.seed(42)
x_new = best_point + np.random.normal(0, 0.05, size=2)
x_new = np.clip(x_new, 0, 1)
print("Random noise sample point:", x_new)

# =========================
# BAYESIAN OPTIMIZATION PART
# =========================

# Copy data to avoid overwriting
X_bo = data_input_f2.copy()
y_bo = data_output_f2.copy().reshape(-1,1)

# Stabilize very small or negative values for log-transform
y_bo = np.log(np.abs(y_bo) + 1e-12)

# GP model
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

# Expected Improvement function
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
        ei[sigma == 0.0] = 0.0
    return ei

# Function to optimize (replace with real function if known)
def f2(x):
    # Example: placeholder function
    # Replace this with the actual black-box function if available
    return np.sin(5*x[0]) * np.cos(5*x[1]) * np.exp(-x[0]**2 - x[1]**2)

# Bayesian Optimization loop
for iteration in range(5):

    # Fit GP
    gp.fit(X_bo, y_bo)

    # Sample candidate points
    X_candidates = np.random.uniform(0,1,(1000,2))
    y_best_bo = np.max(y_bo)

    # Compute Expected Improvement
    ei = expected_improvement(X_candidates, gp, y_best_bo)

    # Select next point to sample
    x_next = X_candidates[np.argmax(ei)]

    # Evaluate function (use f2 or real function)
    y_next = f2(x_next)
    y_next = np.log(np.abs(y_next) + 1e-12)

    # Append new point
    X_bo = np.vstack((X_bo, x_next))
    y_bo = np.vstack((y_bo, [[y_next]]))

    print(f"\n[BO] Iteration {iteration+1}")
    print("[BO] Next suggested point:", x_next)
    print("[BO] Best value so far:", np.max(y_bo))

# Final best point
best_idx_final = np.argmax(y_bo)
print("\nFinal Bayesian Optimization best value:", np.max(y_bo))
print("Best input after BO:", X_bo[best_idx_final])
#===========
# OUTPUTS
#==========
# Function 2 inputs:
 # [[0.66579958 0.12396913]
 # [0.87779099 0.7786275 ]
 # [0.14269907 0.34900513]
 # [0.84527543 0.71112027]
 # [0.45464714 0.29045518]
 # [0.57771284 0.77197318]
 # [0.43816606 0.68501826]
 # [0.34174959 0.02869772]
 # [0.33864816 0.21386725]
 # [0.70263656 0.9265642 ]]
# Function 2 outputs:
 # [ 0.53899612  0.42058624 -0.06562362  0.29399291  0.21496451  0.02310555
  # 0.24461934  0.03874902 -0.01385762  0.61120522]
