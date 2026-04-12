import numpy as np
from pathlib import Path

MODEL_PATH = Path(r"./utils/rgi_surrogate.joblib")
from utils.interpolator import OpenMCTallyGridSurrogate
interpolator_model = OpenMCTallyGridSurrogate()
interpolator_model = interpolator_model.load(MODEL_PATH)

N_G = 8
T_FPs = np.linspace(500, 1000, 10)

X = np.array([[500, T_FP, 600] for T_FP in T_FPs])
params = interpolator_model.predict_dict(X)

print(params[0].keys())

Sigma_t = np.zeros((len(params), N_G))
Sigma_f = np.zeros((len(params), N_G))
Sigma_s0 = np.zeros((len(params), N_G, N_G))
fission_number = np.zeros((len(params), N_G))
Chi = np.zeros((len(params), N_G))
kappa = np.zeros((len(params), N_G))

for i in range(len(params)):
    Sigma_t[i] = np.array(params[i]["total_xs"]) 
    Sigma_f[i] = np.array(params[i]["fission_xs"]) 
    Sigma_s0[i] = np.array(params[i]["scatter_matrix_xs"])
    fission_number[i] = np.array(params[i]["nu"])
    Chi[i] = np.array(params[i]["chi"])
    kappa[i] = np.array(params[i]["kappa"])

print(Sigma_s0)