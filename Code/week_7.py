#Module-18 - Hyperparameters
#---------
# Week-7
#---------

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

np.random.seed(42)

# =========================
# EXPECTED IMPROVEMENT
# =========================
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

# =========================
# FORMAT OUTPUT
# =========================
def format_query(x):
    return "-".join([f"{v:.6f}" for v in x])

# =========================
# CORE BO FUNCTION
# =========================
def train_gp(X, y):
    y_transformed = np.log(np.abs(y) + 1e-12)
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                  alpha=1e-6,
                                  normalize_y=True)
    gp.fit(X, y_transformed)
    return gp, y_transformed

# =========================
# STRATEGY 1: SEQUENTIAL BO
# =========================
def sequential_bo(X, y):
    gp, y_t = train_gp(X, y)

    X_candidates = np.random.uniform(0,1,(3000,X.shape[1]))
    ei = expected_improvement(X_candidates, gp, np.max(y_t))

    return X_candidates[np.argmax(ei)]

# =========================
# STRATEGY 2: BATCH BO
# =========================
def batch_bo(X, y, batch_size=3):
    gp, y_t = train_gp(X, y)

    X_candidates = np.random.uniform(0,1,(5000,X.shape[1]))
    ei = expected_improvement(X_candidates, gp, np.max(y_t))

    idx = np.argsort(ei)[-batch_size:]
    return X_candidates[idx]

# =========================
# STRATEGY 3: CONSTANT LIAR
# =========================
def constant_liar_bo(X, y, batch_size=3):
    X_temp = X.copy()
    y_temp = y.copy()

    new_points = []

    for _ in range(batch_size):
        gp, y_t = train_gp(X_temp, y_temp)

        X_candidates = np.random.uniform(0,1,(3000,X.shape[1]))
        ei = expected_improvement(X_candidates, gp, np.max(y_t))

        x_next = X_candidates[np.argmax(ei)]
        new_points.append(x_next)

        # Lie = current best
        lie = np.max(y_temp)

        X_temp = np.vstack([X_temp, x_next])
        y_temp = np.append(y_temp, lie)

    return np.array(new_points)

# =========================
# STRATEGY 4: LOCAL PENALISATION
# =========================
def local_penalisation_bo(X, y, radius=0.1):
    gp, y_t = train_gp(X, y)

    X_candidates = np.random.uniform(0,1,(3000,X.shape[1]))
    ei = expected_improvement(X_candidates, gp, np.max(y_t))

    # Penalise near existing points
    for i, x in enumerate(X_candidates):
        distances = np.linalg.norm(X - x, axis=1)
        if np.min(distances) < radius:
            ei[i] *= 0.1  # reduce score

    return X_candidates[np.argmax(ei)]

# =========================
# FINAL MIX STRATEGY (YOUR STYLE)
# =========================
def generate_query_hyper(X, y):

    # best known point
    best_idx = np.argmax(y)
    x_best = X[best_idx]

    # choose strategies
    x_seq = sequential_bo(X, y)
    x_local = local_penalisation_bo(X, y)

    # combine (your idea)
    noise = np.random.normal(0, 0.03, size=X.shape[1])
    x_new = 0.5 * x_seq + 0.3 * x_local + 0.2 * (x_best + noise)

    return np.clip(x_new, 0, 1)

# =========================
# RUN ALL FUNCTIONS
# =========================
data_files = [
    ("initial_inputs_f1.npy", "initial_outputs_f1.npy"),
    ("initial_inputs_f2.npy", "initial_outputs_f2.npy"),
    ("initial_inputs_f3.npy", "initial_outputs_f3.npy"),
    ("initial_inputs_f4.npy", "initial_outputs_f4.npy"),
    ("initial_inputs_f5.npy", "initial_outputs_f5.npy"),
    ("initial_inputs_f6.npy", "initial_outputs_f6.npy"),
    ("initial_inputs_f7.npy", "initial_outputs_f7.npy"),
    ("initial_inputs_f8.npy", "initial_outputs_f8.npy"),
]

print("\n===== HYPERPARAMETER STRATEGY COMPARISON =====\n")

for i, (inp_file, out_file) in enumerate(data_files):

    X = np.load(inp_file)
    y = np.load(out_file)

    print(f"\nFunction F{i+1}")

    # Sequential
    seq = sequential_bo(X, y)
    print("Sequential:", format_query(seq))

    # Batch
    batch = batch_bo(X, y)
    print("Batch (top 3):")
    for b in batch:
        print("  ", format_query(b))

    # Constant liar
    liar = constant_liar_bo(X, y)
    print("Constant liar:")
    for l in liar:
        print("  ", format_query(l))

    # Local penalisation
    local = local_penalisation_bo(X, y)
    print("Local penalisation:", format_query(local))

    # Final combined query (submit this)
    final = generate_query_hyper(X, y)
    print("FINAL SUBMISSION:", format_query(final))
#===========  
# Output
#===========  
# Function F1
# FINAL SUBMISSION: 0.372657-0.154031
# Function F2
# FINAL SUBMISSION: 0.584971-0.842874
# Function F3
# FINAL SUBMISSION: 0.653747-0.264248-0.524267
# Function F4
# FINAL SUBMISSION: 0.401125-0.451065-0.215110-0.737013
# Function F5
# FINAL SUBMISSION: 0.534087-0.394835-0.594121-0.746574
# Function F6
# FINAL SUBMISSION: 0.424047-0.269102-0.300890-0.482014-0.557884
# Function F7
# FINAL SUBMISSION: 0.470986-0.462369-0.322170-0.219087-0.494695-0.444355
# Function F8
# FINAL SUBMISSION: 0.122314-0.220395-0.095280-0.107088-0.510297-0.515939-0.297397-0.637243
  
