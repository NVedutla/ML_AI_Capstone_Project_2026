# ============================================
# MODULE 19 - ROUND 8 BBO QUERIES - week-8
# Prompt + Decoding Inspired BO Strategy
# ============================================

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

np.random.seed(42)

# ============================================
# EXPECTED IMPROVEMENT
# ============================================
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


# ============================================
# FORMAT QUERY TO 6DP
# ============================================
def format_query(x):
    return "-".join([f"{v:.6f}" for v in x])


# ============================================
# ROUND 8 QUERY GENERATOR
# ============================================
def generate_round8_query(X, y):

    dim = X.shape[1]

    # ----------------------------------------
    # BEST EXISTING POINT
    # ----------------------------------------
    best_idx = np.argmax(y)
    x_best = X[best_idx]

    # ----------------------------------------
    # LOG TRANSFORM OUTPUTS
    # ----------------------------------------
    y_transformed = np.log(np.abs(y) + 1e-12)

    # ----------------------------------------
    # DECODING SETTINGS (Hyperparameters)
    # ----------------------------------------

    # exploration vs exploitation balance
    xi = 0.02

    # candidate search size
    n_candidates = 5000

    # perturbation strength
    noise_scale = 0.04

    # ----------------------------------------
    # GAUSSIAN PROCESS MODEL
    # ----------------------------------------
    kernel = Matern(
        length_scale=0.3,
        nu=2.5
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5
    )

    gp.fit(X, y_transformed)

    # ----------------------------------------
    # CANDIDATE SEARCH SPACE
    # ----------------------------------------
    X_candidates = np.random.uniform(
        0,
        1,
        (n_candidates, dim)
    )

    # ----------------------------------------
    # EXPECTED IMPROVEMENT
    # ----------------------------------------
    y_best_bo = np.max(y_transformed)

    ei = expected_improvement(
        X_candidates,
        gp,
        y_best_bo,
        xi=xi
    )

    # ----------------------------------------
    # BEST BO CANDIDATE
    # ----------------------------------------
    x_bo = X_candidates[np.argmax(ei)]

    # ----------------------------------------
    # TEMPERATURE STYLE MIXING
    # (like decoding temperature)
    # ----------------------------------------
    temperature = 0.7

    noise = np.random.normal(
        0,
        noise_scale,
        size=dim
    )

    x_new = (
        temperature * x_bo
        +
        (1 - temperature) * (x_best + noise)
    )

    # keep values valid
    x_new = np.clip(x_new, 0, 1)

    return x_new


# ============================================
# LOAD INITIAL FILES
# ============================================
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

# ============================================
# PREVIOUS ROUND INPUTS
# ============================================
previous_inputs = [

    # ROUND 1
    [
        np.array([0.758593, 0.726086]),
        np.array([0.292952, 0.646862]),
        np.array([0.395546, 0.805829]),
        np.array([0.472070, 0.889661]),
        np.array([0.410611, 0.328084]),
    ],

    [
        np.array([0.752636, 0.876564]),
        np.array([0.801933, 0.361611]),
        np.array([0.858572, 0.523935]),
        np.array([0.828467, 0.700560]),
        np.array([0.395937, 0.789559]),
    ],

    [
        np.array([0.522584, 0.641593, 0.359176]),
        np.array([0.825538, 0.573767, 0.293159]),
        np.array([0.565325, 0.812752, 0.332891]),
        np.array([0.519887, 0.820892, 0.391404]),
        np.array([0.466303, 0.285405, 0.415157]),
    ],

    [
        np.array([0.530234, 0.478202, 0.370938, 0.275756]),
        np.array([0.510489, 0.404398, 0.437596, 0.344825]),
        np.array([0.478065, 0.452968, 0.474360, 0.306532]),
        np.array([0.766273, 0.198275, 0.257986, 0.272166]),
        np.array([0.769522, 0.491948, 0.375364, 0.773170]),
    ],

    [
        np.array([0.249024, 0.839567, 0.911868, 0.954667]),
        np.array([0.128898, 0.952902, 0.771331, 0.503220]),
        np.array([0.355176, 0.947655, 0.932016, 0.928354]),
        np.array([0.513786, 0.425447, 0.518450, 0.577222]),
        np.array([0.270116, 0.512029, 0.483304, 0.786213]),
    ],

    [
        np.array([0.753021, 0.147779, 0.764936, 0.770148, 0.044693]),
        np.array([0.224208, 0.684896, 0.422543, 0.418996, 0.579892]),
        np.array([0.532328, 0.245278, 0.684577, 0.685785, 0.128909]),
        np.array([0.849919, 0.103390, 0.320546, 0.363306, 0.366673]),
        np.array([0.704237, 0.576957, 0.757938, 0.565834, 0.305903]),
    ],

    [
        np.array([0.082731, 0.484759, 0.279806, 0.294269, 0.408720, 0.719263]),
        np.array([0.080460, 0.348773, 0.532607, 0.648519, 0.129003, 0.684683]),
        np.array([0.028619, 0.349476, 0.211105, 0.138869, 0.315087, 0.765714]),
        np.array([0.375313, 0.819356, 0.541125, 0.249676, 0.389799, 0.505818]),
        np.array([0.514268, 0.174887, 0.651056, 0.712418, 0.492356, 0.283453]),
    ],

    [
        np.array([0.081283, 0.059042, 0.055313, 0.114937, 0.392227, 0.789348, 0.567267, 0.931456]),
        np.array([0.087357, 0.342602, 0.061864, 0.140867, 0.692759, 0.541137, 0.254324, 0.651230]),
        np.array([0.097378, 0.044393, 0.415530, 0.251160, 0.333159, 0.675900, 0.352720, 0.943765]),
        np.array([0.224590, 0.209255, 0.151249, 0.074842, 0.646991, 0.452938, 0.258739, 0.697825]),
        np.array([0.151258, 0.293419, 0.108090, 0.065546, 0.525069, 0.450335, 0.246759, 0.528766]),
    ]
]

# ============================================
# PREVIOUS OUTPUTS
# ============================================
previous_outputs = [

    [-2.9431718687294526e-19,
     -6.335142551743484e-48,
     -3.760366671200099e-60,
     -4.245976390139691e-65,
     -5.443345027077888e-07],

    [0.3043538085798325,
     0.041014636509412894,
     0.6770263084873531,
     0.1925598114156139,
     -0.25472591083116186],

    [-0.03553548459793642,
     -0.08384670059737882,
     -0.05855001190961575,
     -0.02836321436336946,
     -0.029491535131864498],

    [-3.434147297215279,
     -1.211875210974156,
     -3.106447434863195,
     -11.48132450371136,
     -15.512784331563605],

    [1694.8396824139172,
     422.43285431921794,
     2398.3639963727787,
     22.234100482067166,
     1.9171986789845648],

    [-0.709375174386896,
     -1.573697287980454,
     -0.3912494739228939,
     -1.7077063400996997,
     -1.004100890563769],

    [1.6022236150983638,
     0.5171440850320731,
     1.6746392343960215,
     0.5222444811873926,
     0.08831779458499353],

    [9.5254440507419,
     9.9349666094695,
     9.53564261952,
     9.936662001619,
     9.9204632268519]
]

# ============================================
# ROUND 8 QUERY GENERATION
# ============================================
print("\n===== ROUND 8 QUERIES =====\n")

for i, (inp_file, out_file) in enumerate(data_files):

    # initial data
    X_init = np.load(inp_file)
    y_init = np.load(out_file)

    # combine all data
    X_all = np.vstack([X_init] + previous_inputs[i])
    y_all = np.concatenate([y_init, previous_outputs[i]])

    # generate query
    query = generate_round8_query(X_all, y_all)

    print(f"F{i+1} - {format_query(query)}")
# =======
# OUTPUT
# =======
# F1 - 0.683582-0.698176
# F2 - 0.897410-0.535695
# F3 - 0.231955-0.547785-0.751627
# F4 - 0.806675-0.688524-0.720649-0.566022
# F5 - 0.343380-0.913370-0.837098-0.967851
# F6 - 0.192849-0.703630-0.292110-0.423609-0.737237
# F7 - 0.179211-0.332576-0.347202-0.199856-0.353134-0.742883
# F8 - 0.287247-0.172038-0.096901-0.259438-0.791954-0.437922-0.211810-0.402197
