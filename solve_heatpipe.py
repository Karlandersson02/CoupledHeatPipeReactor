import sys
import numpy as np
import matplotlib.pyplot as plt

from heat_network_model import solve_heatpipe_network_model_static

def main(args):
    D_v = 0.014
    delta_wick = 0.001
    delta_wall = 0.001

    l_evap = 0.105
    l_adiabatic = 0.0525
    l_cond = 0.5425

    T_infc = 300
    Q = 770

    k = np.array([
        21.7,  # evap wall (radial)
        45.0,  # evap wick (radial)
        45.0,  # cond wick (radial)
        21.7,  # cond wall (radial)
        45.0,  # adiabatic wick (axial)
        21.7,  # adiabatic wall (axial)
        39.0,  # convection boundary (heat transfer coefficient)
    ])

    A = np.array([
        np.pi*(D_v + 2 * delta_wick + 2 * delta_wall) * l_evap,
        np.pi*(D_v + 2 * delta_wick) * l_evap,
        np.pi*(D_v + 2 * delta_wick) * l_cond,
        np.pi*(D_v + 2 * delta_wick + 2 * delta_wall) * l_cond,
        np.pi*((D_v/2 + delta_wick)**2 - (D_v/2)**2), 
        np.pi*((D_v/2 + delta_wick + delta_wall)**2 - (D_v/2 + delta_wick)**2), 
        np.pi*(D_v + 2*delta_wick + 2*delta_wall) * l_cond 
    ])

    lamb = np.array([
        delta_wall,   # radial
        delta_wick,   # radial
        delta_wick,   # radial
        delta_wall,   # radial
        l_adiabatic,  # axial
        l_adiabatic,  # axial
        -1.0,     # dummy; convection handled via k[6]*A[6]
    ])

    if len(args) > 0 and args[0] == "network_static":
        T = solve_heatpipe_network_model_static(k, A, lamb, T_infc, Q)
        print(f"Temperature:\nEvaporator wall: {T[0]:.1f} K\nEvaporator wick: {T[1]:.1f} K\nCondenser wick: {T[2]:.1f} K\nCondenser wall: {T[3]:.1f} K\nAdiabatic wick: {T[4]:.1f} K\nAdiabatic wall: {T[5]:.1f} K")
        return 0
    else:
        raise ValueError("Invalid function argument")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
