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
    "T_op": 850,

    "k_wick": 45.0,
    "k_wall": 21.7,

    "P_C": 2476,
    "T_C": 856
}

data_Guoju_1 = {
    "r_outer": .007 + 0.001 + 0.0005,
    "delta_wick": 0.0005,
    "delta_wall": 0.001,
    "l_evap": 0.1,
    "l_adiabatic": 0.05,
    "l_cond": 0.35,

    "N_wick": 15,
    "N_wall": 15,
    "N_evap": 30,
    "N_adiabatic": 15,
    "N_cond": 105,

    "adiabatic_radial_flux": False,
    "Temperature_BC": True,
    "h_vap": 1e6,
    "h_cond": 59.6,
    "T_cond": 300,
    "T_op": 850,

    "k_wick": 66.2,
    "k_wall": 19.0,

    "P_C": 1300,
    "T_C": 818
}

data_Guoju_2 = {
    "r_outer": .007 + 0.001 + 0.0005,
    "delta_wick": 0.0005,
    "delta_wall": 0.001,
    "l_evap": 0.1,
    "l_adiabatic": 0.05,
    "l_cond": 0.55,

    "N_wick": 15,
    "N_wall": 15,
    "N_evap": 30,
    "N_adiabatic": 15,
    "N_cond": 165,

    "adiabatic_radial_flux": False,
    "Temperature_BC": True,
    "h_vap": 1e6,
    "h_cond": 62.6,
    "T_cond": 300,
    "T_op": 850,

    "k_wick": 66.2,
    "k_wall": 19.0,

    "P_C": 2476,
    "T_C": 856
}

# Qs = build_flat_profiles(1500, 2500, 200, data["N_evap"])
Qs = [build_flat_profile(560, data["N_evap"]), ]

comparison_data = {"Q": Qs}

HP_visuals = heatpipe_comparisons(data_Guoju_1, comparison_data)
# HP_visuals.plot_vapour_mach_number_comparison()
HP_visuals.plot_vapour_temperature_comparison()
HP_visuals.plot_vapour_pressure_drop_comparison()