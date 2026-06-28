#Module -17 - CNN - Neural Networks and Deep Learning: Part Three
#---------
# Week-6
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
# CORE FUNCTION
# =========================
def generate_query(X, y):

    # -------- BEST POINT (EXPLOITATION)
    best_idx = np.argmax(y)
    x_best = X[best_idx]

    # -------- STABILISE OUTPUT (important!)
    y_transformed = np.log(np.abs(y) + 1e-12)

    # -------- TRAIN GAUSSIAN PROCESS
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X, y_transformed)

    # -------- SAMPLE CANDIDATES
    dim = X.shape[1]
    X_candidates = np.random.uniform(0, 1, (3000, dim))

    # -------- EXPECTED IMPROVEMENT
    y_best_bo = np.max(y_transformed)
    ei = expected_improvement(X_candidates, gp, y_best_bo)

    x_bo = X_candidates[np.argmax(ei)]

    # -------- MIX STRATEGY (KEY PART)
    noise = np.random.normal(0, 0.03, size=dim)
    x_new = 0.7 * x_bo + 0.3 * (x_best + noise)

    x_new = np.clip(x_new, 0, 1)

    return format_query(x_new)

# =========================
# LOAD YOUR DATA (ALL ROUNDS)
# =========================

# Example: combine initial + previous rounds manually

def combine_data(initial_X, new_X_list, initial_y, new_y_list):
    X_all = np.vstack([initial_X] + new_X_list)
    y_all = np.concatenate([initial_y] + new_y_list)
    return X_all, y_all


# -------------------------
# LOAD INITIAL DATA FILES
# -------------------------
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

# -------------------------
# YOUR PREVIOUS ROUND DATA
# -------------------------
previous_inputs = [
    # Round history (combine all your past queries here)
]

previous_outputs = [
    # Matching outputs here
]

# =========================
# RUN FOR ALL FUNCTIONS
# =========================
print("\n===== ROUND 6 QUERIES =====\n")

for i, (inp_file, out_file) in enumerate(data_files):

    X_init = np.load(inp_file)
    y_init = np.load(out_file)

    # If you have stored previous rounds separately:
    # X_all, y_all = combine_data(X_init, previous_inputs[i], y_init, previous_outputs[i])

    # For now (if not combined yet):
    X_all, y_all = X_init, y_init

    query = generate_query(X_all, y_all)

    print(f"F{i+1} - {query}")
# =======
# OUTPUT
# =======
# F1 - 0.472070-0.889661
# F2 - 0.828467-0.700560
# F3 - 0.519887-0.820892-0.391404
# F4 - 0.766273-0.198275-0.257986-0.272166
# F5 - 0.513786-0.425447-0.518450-0.577222
# F6 - 0.849919-0.103390-0.320546-0.363306-0.366673
# F7 - 0.375313-0.819356-0.541125-0.249676-0.389799-0.505818
# F8 - 0.224590-0.209255-0.151249-0.074842-0.646991-0.452938-0.258739-0.697825
