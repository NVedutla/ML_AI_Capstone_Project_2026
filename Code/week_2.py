#--------------------------------------------------------
# Module - 15 - Neural Networks Deep Learning - Week -1
#---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning

# ---------------------------
# Ignore GP convergence warnings
# ---------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =========================
# LOAD FUNCTION 3 DATA
# =========================
data_input_f3 = np.load('initial_inputs_f3.npy')
data_output_f3 = np.load('initial_outputs_f3.npy')

print("Function 3 inputs:\n", data_input_f3)
print("Function 3 outputs:\n", data_output_f3)

# Find best point
best_idx = np.argmax(data_output_f3)  # less negative is better
best_point = data_input_f3[best_idx]
print("Best input:", best_point)
print("Best output:", data_output_f3[best_idx])

# Pairplot-like visualization (x1 vs x2)
x1 = data_input_f3[:, 0]
x2 = data_input_f3[:, 1]
x3 = data_input_f3[:, 2]
z = data_output_f3

plt.figure(figsize=(6,5))
plt.scatter(x1, x2, c=z, cmap='viridis', s=80)
plt.colorbar(label='Output (score)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Function 3: x1 vs x2')
plt.show()

# Small Gaussian noise sample
np.random.seed(42)
x_new = best_point + np.random.normal(0, 0.05, size=3)
x_new = np.clip(x_new, 0, 1)
print("Random noise sample point:", x_new)

# =========================
# BAYESIAN OPTIMIZATION PART
# =========================

# Copy data to avoid overwriting
X_bo = data_input_f3.copy()
y_bo = data_output_f3.copy().reshape(-1,1)

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

# Function to optimize (replace with actual black-box function if available)
def f3(x):
    # Example placeholder function: smooth 3D function
    return np.sin(5*x[0]) * np.cos(5*x[1]) * np.exp(-x[2])  

# Bayesian Optimization loop
for iteration in range(5):

    # Fit GP
    gp.fit(X_bo, y_bo)

    # Sample candidate points (1000 random 3D points)
    X_candidates = np.random.uniform(0,1,(1000,3))
    y_best_bo = np.max(y_bo)

    # Compute Expected Improvement
    ei = expected_improvement(X_candidates, gp, y_best_bo)

    # Select next point to sample
    x_next = X_candidates[np.argmax(ei)]

    # Evaluate function
    y_next = f3(x_next)
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
#===========
# [BO] Iteration 1
# [BO] Next suggested point: [0.15601864 0.15599452 0.05808361 0.86617615]
# [BO] Best value so far: 3.4850991019427293

# [BO] Iteration 2
# [BO] Next suggested point: [0.14924947 0.26817437 0.36107473 0.40845558]
# [BO] Best value so far: 3.4850991019427293

# [BO] Iteration 3
# [BO] Next suggested point: [0.5684722  0.36372552 0.75653858 0.25736546]
# [BO] Best value so far: 3.4850991019427293

# [BO] Iteration 4
# [BO] Next suggested point: [0.52868585 0.92796906 0.42875141 0.86981237]
# [BO] Best value so far: 3.4850991019427293

# [BO] Iteration 5
# [BO] Next suggested point: [0.97453255 0.22206517 0.47266473 0.06113763]
# [BO] Best value so far: 3.4850991019427293

# Final Bayesian Optimization best value: 3.4850991019427293
# Best input after BO: [0.94838936 0.89451301 0.85163782 0.55219629]
#=============
# REFINED CODE
#=============
#---------
# Week-3a
#---------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings from GP
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =========================
# LOAD FUNCTION 5 DATA
# =========================
data_input_f5 = np.load('initial_inputs_f5.npy')
data_output_f5 = np.load('initial_outputs_f5.npy')

# Plot x1 vs x2
plt.scatter(data_input_f5[:,0], data_input_f5[:,1], c=data_output_f5)
plt.colorbar(label='Output')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Function 5: x1 vs x2')
plt.show()

# =========================
# FIND CURRENT BEST
# =========================
best_idx = np.argmax(data_output_f5)
x_best = data_input_f5[best_idx]
print('Best initial input:', x_best)
print('Best initial output:', data_output_f5[best_idx])

# Small perturbation sample (exploit)
np.random.seed(42)
x_new = x_best + np.random.normal(0, 0.05, size=4)
x_new = np.clip(x_new, 0, 1)
print("Random small perturbation x_new:", x_new)

# =========================
# PREPARE DATA FOR BAYESIAN OPTIMIZATION
# =========================
X_bo = data_input_f5.copy()
y_bo = data_output_f5.copy().reshape(-1,1)

# Optional: log transform if values vary widely
y_bo = np.log(np.abs(y_bo) + 1e-12)

# =========================
# DEFINE EXPECTED IMPROVEMENT
# =========================
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

# =========================
# PLACEHOLDER FUNCTION F5
# Replace this with your actual black-box function
# =========================
def f5(x):
    # Example: smooth 4D function
    return np.sin(5*x[0]) * np.cos(5*x[1]) + np.exp(-x[2]) * np.sin(5*x[3])

# =========================
# BAYESIAN OPTIMIZATION LOOP
# =========================
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

for iteration in range(5):
    # Fit GP to current data
    gp.fit(X_bo, y_bo)
    
    # Sample candidate points randomly
    X_candidates = np.random.uniform(0,1,(1000,4))
    
    # Compute EI
    y_best_bo = np.max(y_bo)
    ei = expected_improvement(X_candidates, gp, y_best_bo)
    
    # Pick next point
    x_next = X_candidates[np.argmax(ei)]
    
    # Evaluate function
    y_next = f5(x_next)
    y_next = np.log(np.abs(y_next) + 1e-12)
    
    # Append to dataset
    X_bo = np.vstack((X_bo, x_next))
    y_bo = np.vstack((y_bo, [[y_next]]))
    
    print(f"\n[BO] Iteration {iteration+1}")
    print("[BO] Next suggested point:", x_next)
    print("[BO] Best value so far:", np.max(y_bo))

# =========================
# FINAL BEST
# =========================
best_idx_final = np.argmax(y_bo)
print("\nFinal BO best value:", np.max(y_bo))
print("Best input after BO:", X_bo[best_idx_final])

#==========
# OUTPUT
#==========
# Best initial input: [0.22418902 0.84648049 0.87948418 0.87851568]
# Best initial output: 1088.8596181962705
# Random small perturbation x_new: [0.24902473 0.83956728 0.91186861 0.95466718]

# [BO] Iteration 1
# [BO] Next suggested point: [0.15601864 0.15599452 0.05808361 0.86617615]
# [BO] Best value so far: 6.992886205712814

# [BO] Iteration 2
# [BO] Next suggested point: [0.14924947 0.26817437 0.36107473 0.40845558]
# [BO] Best value so far: 6.992886205712814

# [BO] Iteration 3
# [BO] Next suggested point: [0.5684722  0.36372552 0.75653858 0.25736546]
# [BO] Best value so far: 6.992886205712814

# [BO] Iteration 4
# [BO] Next suggested point: [0.52868585 0.92796906 0.42875141 0.86981237]
# [BO] Best value so far: 6.992886205712814

# [BO] Iteration 5
# [BO] Next suggested point: [0.97453255 0.22206517 0.47266473 0.06113763]
# [BO] Best value so far: 6.992886205712814

# Final BO best value: 6.992886205712814
# Best input after BO: [0.22418902 0.84648049 0.87948418 0.87851568]
