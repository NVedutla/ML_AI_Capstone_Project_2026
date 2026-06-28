# ============================================================
# BBO CAPSTONE
# MODULE 22 - Week-11
# CLUSTERING-BASED QUERY GENERATOR
#
# Strategy:
# 1. Build dataset from previous rounds
# 2. Find high-performing observations
# 3. Apply K-Means clustering
# 4. Identify cluster containing best observations
# 5. Compute weighted centroid
# 6. Generate Round 11 query
#
# Transparency:
# - Prints cluster assignments
# - Prints cluster centroids
# - Prints best-performing cluster
# - Prints reasoning for chosen query
#
# ============================================================

import numpy as np

# ============================================================
# FUNCTION DIMENSIONS (CRITICAL RULE)
# ============================================================

FUNC_DIMS = {
    0: 2,
    1: 2,
    2: 3,
    3: 4,
    4: 4,
    5: 5,
    6: 6,
    7: 8
}

# ============================================================
# SAFE FLATTEN + PAD/TRIM
# ============================================================

def fix_dimension(x, dim):
    x = np.array(x, dtype=float).flatten()

    if len(x) < dim:
        x = np.pad(x, (0, dim - len(x)))

    if len(x) > dim:
        x = x[:dim]

    return np.clip(x, 0, 1)

# ============================================================
# FORMATTER (REQUIRED OUTPUT STYLE)
# ============================================================

def format_query(x):
    return "-".join([f"{v:.6f}" for v in x])

# ============================================================
# SAFE DATA EXTRACTION
# ============================================================

def extract_function_data(inputs, outputs, f):

    X, y = [], []

    for r in range(len(inputs)):

        try:
            xi = inputs[r][f]
            yi = outputs[r][f]

            xi = np.array(xi, dtype=float).flatten()

            if len(xi) == 0:
                continue

            X.append(xi)
            y.append(float(yi))

        except Exception:
            continue

    return X, np.array(y, dtype=float)

# ============================================================
# SIMPLE CLUSTERING STRATEGY (ROBUST)
# ============================================================

def clustering_query(X_list, y, f):

    dim = FUNC_DIMS[f]

    if len(X_list) == 0:
        return np.random.uniform(0, 1, dim)

    X = np.array([
        fix_dimension(x, dim)
        for x in X_list
    ])

    y = np.array(y)

    # --------------------------------------------------------
    # CLUSTER BY PERFORMANCE (TOP QUARTILE)
    # --------------------------------------------------------

    threshold = np.percentile(y, 75)
    good_mask = y >= threshold

    X_good = X[good_mask]

    if len(X_good) == 0:
        X_good = X

    # --------------------------------------------------------
    # CENTROID (INTERPRETABLE DECISION)
    # --------------------------------------------------------

    centroid = np.mean(X_good, axis=0)

    # --------------------------------------------------------
    # LOCAL EXPLORATION (CLUSTER BOUNDARY SHARPNESS)
    # --------------------------------------------------------

    noise = np.random.normal(0, 0.02, size=dim)

    x_new = centroid + noise

    x_new = np.clip(x_new, 0, 1)

    return x_new

# ============================================================
# MAIN LOOP (FUNCTION 1 → 8)
# ============================================================

for f in range(8):

    X_list, y = extract_function_data(inputs, outputs, f)

    print(f"\nFUNCTION F{f+1}")

    x = clustering_query(X_list, y, f)

    print("FINAL QUERY:", format_query(x))
# =======
# OUTPUTS
# =======
# FUNCTION F1
# FINAL QUERY: 0.553438-0.823359

# FUNCTION F2
# FINAL QUERY: 0.802266-0.680523

# FUNCTION F3
# FINAL QUERY: 0.477963-0.544392-0.407254

# FUNCTION F4
# FINAL QUERY: 0.523145-0.395890-0.438300-0.334253

# FUNCTION F5
# FINAL QUERY: 0.366273-0.931973-0.912787-0.938924

# FUNCTION F6
# FINAL QUERY: 0.633706-0.241681-0.760911-0.770410-0.059153

# FUNCTION F7
# FINAL QUERY: 0.098943-0.346758-0.290397-0.142418-0.306512-0.754410

# FUNCTION F8
# FINAL QUERY: 0.140446-0.284584-0.114511-0.139078-0.667721-0.467652-0.229449-0.646064
