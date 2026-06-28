#Module-14 - combine SVM with Bayesian Optimisation
#---------
# Week-3
#---------
import numpy as np
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
# MAIN LOOP
# =========================
for f in range(1, 9):

    print(f"\n===== FUNCTION {f} =====")

    # Load data
    X = np.load(f'initial_inputs_f{f}.npy')
    y = np.load(f'initial_outputs_f{f}.npy')

    dim = X.shape[1]

    # =========================
    # STEP 1: FIND BEST (argmax)
    # =========================
    best_idx = np.argmax(y)
    x_best = X[best_idx]
    y_best = y[best_idx]

    print("Best value:", y_best)
    print("Best x:", x_best)

    # =========================
    # STEP 2: SVM (find good region)
    # =========================
    threshold = np.percentile(y, 70)
    labels = (y >= threshold).astype(int)

    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X, labels)

    # =========================
    # STEP 3: SAMPLE CANDIDATES
    # =========================
    np.random.seed(42 + f)
    X_candidates = np.random.uniform(0, 1, (5000, dim))

    probs = svm.predict_proba(X_candidates)[:, 1]

    # Keep promising points
    mask = probs > 0.6
    X_promising = X_candidates[mask]

    if len(X_promising) < 100:
        X_promising = X_candidates  # fallback

    # =========================
    # STEP 4: BAYESIAN OPTIMISATION
    # =========================
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    gp.fit(X, y)

    ei = expected_improvement(X_promising, gp, np.max(y))

    x_bo = X_promising[np.argmax(ei)]

    # =========================
    # STEP 5: COMBINE WITH LOCAL SEARCH
    # =========================
    x_local = x_best + np.random.normal(0, 0.05, size=dim)

    # Mix BO + local (balanced strategy)
    x_new = 0.7 * x_bo + 0.3 * x_local

    # Clip to valid range
    x_new = np.clip(x_new, 0, 1)

    # =========================
    # FORMAT OUTPUT
    # =========================
    submission = "-".join([f"{val:.6f}" for val in x_new])

    print("Final query:", submission)
  # ========
  # OUTPUT
  # ========
  # ===== FUNCTION 1 =====
  # Best value: 7.710875114502849e-16
  # Best x: [0.73102363 0.73299988]
  # Final query: 0.292952-0.646862
  
  # ===== FUNCTION 2 =====
  # Best value: 0.6112052157614438
  # Best x: [0.70263656 0.9265642 ]
  # Final query: 0.801933-0.361611
  
  # ===== FUNCTION 3 =====
  # Best value: -0.034835313350078584
  # Best x: [0.49258141 0.61159319 0.34017639]
  # Final query: 0.825538-0.573767-0.293159

  # ===== FUNCTION 4 =====
  # Best value: -4.025542281908162
  # Best x: [0.57776561 0.42877174 0.42582587 0.24900741]
  # Final query: 0.510489-0.404398-0.437596-0.344825
  
  # ===== FUNCTION 5 =====
  # Best value: 1088.8596181962705
  # Best x: [0.22418902 0.84648049 0.87948418 0.87851568]
  # Final query: 0.128898-0.952902-0.771331-0.503220
  
  # ===== FUNCTION 6 =====
  # Best value: -0.7142649478202404
  # Best x: [0.7281861  0.15469257 0.73255167 0.69399651 0.05640131]
  # Final query: 0.224208-0.684896-0.422543-0.418996-0.579892
  
  # ===== FUNCTION 7 =====
  # Best value: 1.3649683044991994
  # Best x: [0.05789554 0.49167222 0.24742222 0.21811844 0.42042833 0.73096984]
  # Final query: 0.080460-0.348773-0.532607-0.648519-0.129003-0.684683

  # ===== FUNCTION 8 =====
  # Best value: 9.598482002566342
  # Best x: [0.05644741 0.06595555 0.02292868 0.03878647 0.40393544 0.80105533
   # 0.48830701 0.89308498]
  # Final query: 0.087357-0.342602-0.061864-0.140867-0.692759-0.541137-0.254324-0.651230
