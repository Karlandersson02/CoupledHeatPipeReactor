from visualisation.heatpipe_visualisations import heatpipe_visualisations, heatpipe_comparisons
from CoupledSystems.Heatpipe import Heatpipe

import numpy as np

def build_flat_profile(Qtot, N):
    Q = np.repeat(np.array([Qtot / N], dtype=float), N)
    return Q

def build_flat_profiles(Q_start, Q_stop, spacing, N):
    Qtots = np.arange(Q_start, Q_stop, spacing)
    Qs = []
    for Qtot in Qtots:
        Qs.append(build_flat_profile(Qtot, N))
    return Qs

data = {
    "r_outer": .007 + 0.001 + 0.0005,
    "delta_wick": 0.0005,
    "delta_wall": 0.001,
    "l_evap": 0.1,
    "l_adiabatic": 0.3,
    "l_cond": 0.1,

    "N_wick": 15,
    "N_wall": 15,
    "N_evap": 30,
    "N_adiabatic": 90,
    "N_cond": 30,

    "adiabatic_radial_flux": False,
    "Temperature_BC": False,
    "h_vap": 1e6,
    "h_cond": 62.6,
    "T_cond": 300,
    "T_op": 870,

    "k_wick": 45.0,
    "k_wall": 21.7,

    "P_C": 2476,
    "T_C": 856
}

# Q_total = 1200
# Q = np.repeat(np.array([Q_total / data["N_evap"]], dtype=float), data["N_evap"])
# data["Q"] = Q

Qs = build_flat_profiles(800, 1200, 100, data["N_evap"])

comparison_data = {"Q": Qs}

HP_visuals = heatpipe_comparisons(data, comparison_data)
HP_visuals.plot_vapour_temperature_comparison()