from visualisation.heatpipe_visualisations import heatpipe_visualisations, heatpipe_comparisons
from coupled_systems.heatpipe import Heatpipe

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def build_flat_profile(Qtot, N):
    Q = np.repeat(np.array([Qtot / N], dtype=float), N)
    return Q

def build_flat_profiles(Q_start, Q_stop, spacing, N):
    Qtots = np.arange(Q_start, Q_stop, spacing)
    Qs = []
    for Qtot in Qtots:
        Qs.append(build_flat_profile(Qtot, N))
    return Qs

# -------------- Normal

# # Qs = build_flat_profiles(1400, 1700, 100, data["N_evap"])
# Qs = [build_flat_profile(560, data["N_evap"]), ]

# comparison_data = {"Q": Qs}

# HP_visuals = heatpipe_comparisons(data_Guoju_1, comparison_data)
# # HP_visuals.plot_vapour_mach_number_comparison()
# HP_visuals.plot_vapour_temperature_comparison()
# # HP_visuals.plot_vapour_pressure_drop_comparison()

# -------------- Total pressure drop

import json
data_Guoju = json.load("./project_data/vapour_data.json")
data_Guoju1 = data_Guoju["data_Guoju_1"]

Q = build_flat_profile(560, data_Guoju_1["N_evap"])
data_Guoju_1["Q"] = Q

heatpipe = Heatpipe(data_Guoju_1)
heatpipe.setup_fluid_models()
heatpipe_vis = heatpipe_visualisations(heatpipe)
heatpipe_vis.plot_vapour_temperature()
heatpipe_vis.plot_vapour_pressure_drop()


# -------------- Iterative

# data_Guoju_1["Q"] = build_flat_profile(640, data_Guoju_1["N_evap"])
# heatpipe = Heatpipe(data_Guoju_1)

# heatpipe.setup_fluid_models()
# heatpipe.vapour_discretised.niter = 3000
# T_direct = heatpipe.get_vapour_temperature()

# heatpipe.solve_vapour_temperature_iteratively(560, 640, 20)
# T_iterative = heatpipe.get_vapour_temperature()

# mpl.rcParams["font.size"] = 26
# mpl.rcParams["font.family"] = "Computer modern"
# mpl.rcParams["text.usetex"] = True
# fig = plt.figure(figsize=(16,9))
# ax = fig.add_subplot(111)

# ax.grid(alpha=0.4)
# ax.plot(T_direct, color="blue", label="Direct")
# ax.plot(T_iterative, color="red", label="Iterative")
# ax.set_xlabel(r"$n$")
# ax.set_ylabel(r"$T [K]$")
# ax.set_title("Vapour")

# plt.show()

# ----------------