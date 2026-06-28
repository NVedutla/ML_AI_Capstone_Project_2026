# =========================================================
# MODULE 20 - ROUND 9
# TRANSFORMER + SELF-ATTENTION + BO HYBRID
# =========================================================
# Uses:
# - Token embeddings
# - Self-attention ideas
# - Transformer neural network surrogate
# - Bayesian Optimisation refinement
# - Scaling concepts from LLMs / GenAI
#
# Goal:
# Learn relationships between previous query points
# and predict better future query points.
# =========================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

# =========================================================
# REPRODUCIBILITY
# =========================================================

np.random.seed(42)
torch.manual_seed(42)

# =========================================================
# INPUTS (ALL ROUNDS)
# =========================================================

inputs = [

# ROUND 1
[
np.array([0.758593,0.726086]),
np.array([0.752636,0.876564]),
np.array([0.522584,0.641593,0.359176]),
np.array([0.530234,0.478202,0.370938,0.275756]),
np.array([0.249024,0.839567,0.911868,0.954667]),
np.array([0.753021,0.147779,0.764936,0.770148,0.044693]),
np.array([0.082731,0.484759,0.279806,0.294269,0.408720,0.719263]),
np.array([0.081283,0.059042,0.055313,0.114937,0.392227,0.789348,0.567267,0.931456])
],

# ROUND 2
[
np.array([0.292952,0.646862]),
np.array([0.801933,0.361611]),
np.array([0.825538,0.573767,0.293159]),
np.array([0.510489,0.404398,0.437596,0.344825]),
np.array([0.128898,0.952902,0.771331,0.503220]),
np.array([0.224208,0.684896,0.422543,0.418996,0.579892]),
np.array([0.080460,0.348773,0.532607,0.648519,0.129003,0.684683]),
np.array([0.087357,0.342602,0.061864,0.140867,0.692759,0.541137,0.254324,0.651230])
],

# ROUND 3
[
np.array([0.395546,0.805829]),
np.array([0.858572,0.523935]),
np.array([0.565325,0.812752,0.332891]),
np.array([0.478065,0.452968,0.474360,0.306532]),
np.array([0.355176,0.947655,0.932016,0.928354]),
np.array([0.532328,0.245278,0.684577,0.685785,0.128909]),
np.array([0.028619,0.349476,0.211105,0.138869,0.315087,0.765714]),
np.array([0.097378,0.044393,0.415530,0.251160,0.333159,0.675900,0.352720,0.943765])
],

# ROUND 4
[
np.array([0.472070,0.889661]),
np.array([0.828467,0.700560]),
np.array([0.519887,0.820892,0.391404]),
np.array([0.766273,0.198275,0.257986,0.272166]),
np.array([0.513786,0.425447,0.518450,0.577222]),
np.array([0.849919,0.103390,0.320546,0.363306,0.366673]),
np.array([0.375313,0.819356,0.541125,0.249676,0.389799,0.505818]),
np.array([0.224590,0.209255,0.151249,0.074842,0.646991,0.452938,0.258739,0.697825])
],

# ROUND 5
[
np.array([0.410611,0.328084]),
np.array([0.395937,0.789559]),
np.array([0.466303,0.285405,0.415157]),
np.array([0.769522,0.491948,0.375364,0.773170]),
np.array([0.270116,0.512029,0.483304,0.786213]),
np.array([0.704237,0.576957,0.757938,0.565834,0.305903]),
np.array([0.514268,0.174887,0.651056,0.712418,0.492356,0.283453]),
np.array([0.151258,0.293419,0.108090,0.065546,0.525069,0.450335,0.246759,0.528766])
],

# ROUND 6
[
np.array([0.683582,0.698176]),
np.array([0.897410,0.535695]),
np.array([0.231955,0.547785,0.751627]),
np.array([0.806675,0.688524,0.720649,0.566022]),
np.array([0.343380,0.913370,0.837098,0.967851]),
np.array([0.192849,0.703630,0.292110,0.423609,0.737237]),
np.array([0.179211,0.332576,0.347202,0.199856,0.353134,0.742883]),
np.array([0.287247,0.172038,0.096901,0.259438,0.791954,0.437922,0.211810,0.402197])
]
]

# =========================================================
# OUTPUTS
# =========================================================

outputs = [

[-2.9431718687294526e-19,0.3043538085798325,-0.03553548459793642,-3.434147297215279,1694.8396824139172,-0.709375174386896,1.6022236150983638,9.5254440507419],

[-6.335142551743484e-48,0.041014636509412894,-0.08384670059737882,-1.211875210974156,422.43285431921794,-1.573697287980454,0.5171440850320731,9.9349666094695],

[-3.760366671200099e-60,0.6770263084873531,-0.05855001190961575,-3.106447434863195,2398.3639963727787,-0.3912494739228939,1.6746392343960215,9.53564261952],

[-4.245976390139691e-65,0.1925598114156139,-0.02836321436336946,-11.48132450371136,22.234100482067166,-1.7077063400996997,0.5222444811873926,9.936662001619],

[-5.443345027077888e-07,-0.25472591083116186,-0.029491535131864498,-15.512784331563605,1.9171986789845648,-1.004100890563769,0.08831779458499353,9.9204632268519],

[3.705594145463796e-06,0.1659484594261268,-0.10414351060433193,-18.654894097133695,1818.969791995029,-1.9383958842640416,2.4426141625613154,9.9060505392681]

]

# =========================================================
# TRANSFORMER MODEL
# =========================================================

class TransformerRegressor(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        embed_dim = 32

        # TOKEN EMBEDDING
        self.embedding = nn.Linear(1, embed_dim)

        # SELF-ATTENTION LAYER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            batch_first=True
        )

        # TRANSFORMER ENCODER
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # OUTPUT LAYER
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        # x shape = (batch, dim)

        x = x.unsqueeze(-1)

        # TOKEN EMBEDDINGS
        x = self.embedding(x)

        # SELF-ATTENTION
        x = self.transformer(x)

        # FLATTEN
        x = x.reshape(x.size(0), -1)

        # FINAL PREDICTION
        return self.fc(x)

# =========================================================
# EXPECTED IMPROVEMENT
# =========================================================

def expected_improvement(X, model, y_best, xi=0.02):

    mu, sigma = model.predict(X, return_std=True)

    mu = mu.ravel()
    sigma = sigma.ravel()

    sigma = np.maximum(sigma, 1e-9)

    improvement = mu - y_best - xi

    Z = improvement / sigma

    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    return ei

# =========================================================
# FORMAT OUTPUT
# =========================================================

def format_query(x):
    return "-".join([f"{v:.6f}" for v in x])

# =========================================================
# MAIN LOOP
# =========================================================

print("\n===== ROUND 9 TRANSFORMER + BO QUERIES =====\n")

for f in range(8):

    # ============================================
    # BUILD DATASET
    # ============================================

    X = np.array([inputs[r][f] for r in range(len(inputs))])
    y = np.array([outputs[r][f] for r in range(len(outputs))])

    dim = X.shape[1]

    # ============================================
    # SCALE TARGETS
    # ============================================

    y_train = np.log(np.abs(y) + 1e-12)

    # ============================================
    # TRANSFORMER TRAINING
    # ============================================

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

    model = TransformerRegressor(dim)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(300):

        optimizer.zero_grad()

        pred = model(X_tensor)

        loss = loss_fn(pred, y_tensor)

        loss.backward()

        optimizer.step()

    # ============================================
    # CANDIDATE SEARCH
    # ============================================

    X_candidates = np.random.uniform(0,1,(4000,dim))

    model.eval()

    with torch.no_grad():

        X_cand_tensor = torch.tensor(X_candidates,dtype=torch.float32)

        transformer_scores = model(X_cand_tensor).numpy().ravel()

    # ============================================
    # TOP CANDIDATES FROM ATTENTION MODEL
    # ============================================

    top_idx = np.argsort(transformer_scores)[-500:]

    X_top = X_candidates[top_idx]

    # ============================================
    # BAYESIAN OPTIMISATION REFINEMENT
    # ============================================

    kernel = Matern(nu=2.5)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True
    )

    gp.fit(X, y_train)

    ei = expected_improvement(
        X_top,
        gp,
        np.max(y_train),
        xi=0.02
    )

    x_bo = X_top[np.argmax(ei)]

    # ============================================
    # LOCAL EXPLOITATION
    # ============================================

    best_idx = np.argmax(y)
    x_best = X[best_idx]

    noise = np.random.normal(0,0.02,size=dim)

    x_new = 0.75 * x_bo + 0.25 * (x_best + noise)

    x_new = np.clip(x_new,0,1)

    # ============================================
    # PRINT FINAL QUERY
    # ============================================

    print(f"F{f+1} -> {format_query(x_new)}")

    # =======
    # OUTPUT
    # =======
    # F1 -> 0.192509-0.450503
    # F2 -> 0.443541-0.640398
    # F3 -> 0.831965-0.927221-0.847269
    # F4 -> 0.808804-0.370731-0.415933-0.474136
    # F5 -> 0.260671-0.927180-0.456974-0.969596
    # F6 -> 0.591418-0.183798-0.343491-0.570640-0.656004
    # F7 -> 0.090176-0.416899-0.511352-0.226735-0.662324-0.689974
    # F8 -> 0.264303-0.594860-0.096730-0.138859-0.875695-0.279401-0.293004-0.839537
