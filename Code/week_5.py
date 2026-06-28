# Module-16
#---------
# Week-5
#---------
# =====================================================
# ROUND 5 BBO QUERY GENERATOR
# Neural Network + Best Previous Point + Exploration
# Uses current data with 14+ points
# Generates F1 to F8 queries in 6 decimal format
# =====================================================

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# -----------------------------------------------------
# FUNCTION DIMENSIONS
# -----------------------------------------------------
functions = {
    "F1": 2,
    "F2": 2,
    "F3": 3,
    "F4": 4,
    "F5": 4,
    "F6": 5,
    "F7": 6,
    "F8": 8
}

# -----------------------------------------------------
# LOOP THROUGH ALL FUNCTIONS
# -----------------------------------------------------
for fname, dim in functions.items():

    print("=" * 55)
    print(fname)

    # -----------------------------------------
    # LOAD DATA
    # -----------------------------------------
    X = np.load(f"initial_inputs_{fname.lower()}.npy")
    y = np.load(f"initial_outputs_{fname.lower()}.npy")

    # -----------------------------------------
    # BEST PREVIOUS POINT
    # -----------------------------------------
    best_idx = np.argmax(y)
    x_best = X[best_idx]

    # -----------------------------------------
    # TRAIN NEURAL NETWORK
    # -----------------------------------------
    model = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=4000,
            random_state=42
        ))
    ])

    model.fit(X, y)

    # -----------------------------------------
    # RANDOM CANDIDATES
    # -----------------------------------------
    candidates = np.random.uniform(0, 1, size=(5000, dim))

    # Predict outputs
    pred = model.predict(candidates)

    # Best predicted point
    best_pred_idx = np.argmax(pred)
    x_pred = candidates[best_pred_idx]

    # -----------------------------------------
    # COMBINE EXPLOITATION + MODEL PREDICTION
    # -----------------------------------------
    x_new = 0.60 * x_best + 0.40 * x_pred

    # Small exploration noise
    x_new += np.random.normal(0, 0.03, size=dim)

    # Keep inside bounds
    x_new = np.clip(x_new, 0, 1)

    # -----------------------------------------
    # FORMAT OUTPUT
    # -----------------------------------------
    query = "-".join([f"{v:.6f}" for v in x_new])

    print("Best previous output:", y[best_idx])
    print("Suggested Query:")
    print(query)
    # ----------
    #  OUTPUT
    # ----------
    =======================================================
    # F1
    # Best previous output: 7.710875114502849e-16
    # Suggested Query:
    # 0.395546-0.805829
    # =======================================================
    # F2
    # Best previous output: 0.6112052157614438
    # Suggested Query:
    # 0.858572-0.523935
    # =======================================================
    # F3
    # Best previous output: -0.034835313350078584
    # Suggested Query:
    # 0.565325-0.812752-0.332891
    # =======================================================
    # F4
    # Best previous output: -4.025542281908162
    # Suggested Query:
    # 0.478065-0.452968-0.474360-0.306532
    # =======================================================
    # F5
    # Best previous output: 1088.8596181962705
    # Suggested Query:
    # 0.355176-0.947655-0.932016-0.928354
    # =======================================================
    # F6
    # Best previous output: -0.7142649478202404
    # Suggested Query:
    # 0.532328-0.245278-0.684577-0.685785-0.128909
    # =======================================================
    # F7
    # Best previous output: 1.3649683044991994
    # Suggested Query:
    # 0.028619-0.349476-0.211105-0.138869-0.315087-0.765714
    # =======================================================
    # F8
    # Best previous output: 9.598482002566342
    # Suggested Query:
    # 0.097378-0.044393-0.415530-0.251160-0.333159-0.675900-0.352720-0.943765
