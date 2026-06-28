#Module-23 - Submission - 12
#---------
# Week-12
#---------
import numpy as np
from sklearn.decomposition import PCA

# =========================================================
# FORMAT OUTPUT (required submission format)
# =========================================================
def format_query(x):
    return "-".join([f"{v:.6f}" for v in x])


# =========================================================
# DIMENSION MAP (YOUR TRUE SETUP)
# =========================================================
DIM_MAP = {
    0: 2,  # F1
    1: 2,  # F2
    2: 3,  # F3
    3: 4,  # F4
    4: 4,  # F5
    5: 5,  # F6
    6: 6,  # F7
    7: 8   # F8
}


# =========================================================
# SAFE PCA QUERY GENERATOR
# =========================================================
def pca_query(X, y, dim):

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # align lengths safely
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]

    # fallback if too little data
    if n < 3:
        return np.random.uniform(0, 1, dim)

    # enforce correct dimensionality
    X = X[:, :dim]

    # -----------------------------------------
    # select top-performing region (cluster proxy)
    # -----------------------------------------
    threshold = np.percentile(y, 70)
    mask = y >= threshold

    X_top = X[mask]

    if len(X_top) < 2:
        X_top = X

    # centroid (cluster center)
    centroid = np.mean(X_top, axis=0)

    # PCA direction (principal structure)
    k = min(dim, len(X_top))
    pca = PCA(n_components=k)
    pca.fit(X_top)

    direction = pca.components_[0]

    # align direction with best point
    best_idx = np.argmax(y)
    best_point = X[best_idx]

    if np.linalg.norm(direction) > 0:
        sign = np.sign(np.dot(direction, best_point - centroid))
        direction = direction * sign

    # controlled step (prevents divergence)
    step = 0.05
    x_new = centroid + step * direction

    return np.clip(x_new, 0, 1)


# =========================================================
# ROUND 12 GENERATION (FIXED DIMENSIONS)
# =========================================================
def run_round_12(inputs, outputs):

    print("\n========================")
    print("ROUND 12 SUBMISSION")
    print("========================\n")

    for f in range(8):

        dim = DIM_MAP[f]

        X, y = [], []

        for r in range(len(inputs)):
            try:
                x_r = np.array(inputs[r][f], dtype=float)

                # enforce correct dimension explicitly
                if len(x_r) < dim:
                    x_r = np.pad(x_r, (0, dim - len(x_r)))
                else:
                    x_r = x_r[:dim]

                X.append(x_r)
                y.append(outputs[r][f])

            except:
                continue

        query = pca_query(X, y, dim)

        print(f"F{f+1}: {format_query(query)}\n")


# =========================================================
# RUN
# =========================================================
run_round_12(inputs, outputs)
# =========================================================
# OUTPUT
# =========================================================
# F1: 0.614893-0.760362
# F2: 0.819990-0.652364
# F3: 0.498069-0.602852-0.401076
# F4: 0.514819-0.397911-0.432686-0.349939
# F5: 0.354688-0.946238-0.928092-0.929987
# F6: 0.603385-0.213886-0.710450-0.712947-0.101794
# F7: 0.138582-0.337136-0.310484-0.183402-0.342869-0.749043
# F8: 0.183749-0.248939-0.124648-0.094491-0.660612-0.479186-0.257425-0.683958
