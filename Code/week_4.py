#Module-15 - Neural networks and deep learning - Part one
#---------
# Week-4
#---------

# ==========================================
# ROUND 4 BBO CODE
# Neural Network + Random Search
# Works for F1 to F8
# Outputs query points in 6 decimal format
# ==========================================

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# SETTINGS
# ==========================================
np.random.seed(42)

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

# ==========================================
# LOOP THROUGH F1 TO F8
# ==========================================
for fname, dim in functions.items():

    print("=" * 50)
    print(fname)

    # Load files
    X = np.load(f"initial_inputs_{fname.lower()}.npy")
    y = np.load(f"initial_outputs_{fname.lower()}.npy")

    # ==========================================
    # TRAIN NEURAL NETWORK
    # ==========================================
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=3000,
            random_state=42
        ))
    ])

    model.fit(X, y)

    # ==========================================
    # RANDOM CANDIDATES
    # ==========================================
    candidates = np.random.uniform(0, 1, size=(5000, dim))

    # Predict scores
    pred = model.predict(candidates)

    # Best candidate
    best_idx = np.argmax(pred)
    x_best = candidates[best_idx]

    # ==========================================
    # FORMAT TO 6 DECIMALS
    # ==========================================
    query = "-".join([f"{v:.6f}" for v in x_best])

    print("Suggested Query:")

    # =======
    # OUTPUT
    # =======
    # F1 : 0.532360-0.303364
    # F2 : 0.860795-0.009583
    # F3 : 0.667528-0.974808-0.606426
    # F4 : 0.508871-0.418475-0.410796-0.296962
    # F5 : 0.595561-0.993735-0.987345-0.981682
    # F6 : 0.525638-0.435024-0.487157-0.554809-0.107831
    # F7 : 0.003925-0.136792-0.261528-0.042609-0.211394-0.872950
    # F8 : 0.068141-0.007975-0.023657-0.393794-0.906214-0.252996-0.133158-0.580778
    print(query)

