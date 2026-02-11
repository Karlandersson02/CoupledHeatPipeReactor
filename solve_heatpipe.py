import sys
import numpy as np
import matplotlib.pyplot as plt

from models.heat_network_model import solve_heatpipe_network_model_static
from models.heat_discretised_model import heatpipe_discretised


def print_heat_pipe_temperatures(T):
    # Index mapping (from your code)
    T_ev_wall = T[0]
    T_ev_wick = T[1]
    T_co_wick = T[2]
    T_co_wall = T[3]
    T_ad_wick = T[4]
    T_ad_wall = T[5]

    print("\n   EVAPORATOR     ADIABATIC      CONDENSER")
    print("┌──────────────┬──────────────┬──────────────┐")
    print(f"│   WICK       │    WICK      │    WICK      │")
    print(f"│ {T_ev_wick:6.2f} K     │ {T_ad_wick:6.2f} K     │ {T_co_wick:6.2f} K     │")
    print(f"│ {T_ev_wick - 273.15:6.2f} C     │ {T_ad_wick - 273.15:6.2f} C     │ {T_co_wick - 273.15:6.2f} C     │")
    print("├──────────────┼──────────────┼──────────────┤")
    print(f"│   WALL       │    WALL      │    WALL      │")
    print(f"│ {T_ev_wall:6.2f} K     │ {T_ad_wall:6.2f} K     │ {T_co_wall:6.2f} K     │")
    print(f"│ {T_ev_wall - 273.15:6.2f} C     │ {T_ad_wall - 273.15:6.2f} C     │ {T_co_wall - 273.15:6.2f} C     │")
    print("└──────────────┴──────────────┴──────────────┘\n")

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

    # ///
    data_discretised = {
        "r_outer": D_v/2 + delta_wall + delta_wick,
        "delta_wick": delta_wick,
        "delta_wall": delta_wall,
        "l_evap": l_evap,
        "l_adiabatic": l_adiabatic,
        "l_cond": l_cond,
        "N_wick": 1,
        "N_wall": 1,
        "N_evap": 1,
        "N_adiabatic": 1,
        "N_cond": 1,
        "h_vap": 1e10,       # högt tal bara
        "h_cond": 39,
        "T_cond": T_infc,
        "k_wall": 45.0,
        "k_wick": 21.7,
        "Q": np.repeat(np.array([Q/1]), 1)
    }
    # \\\

    if len(args) > 0 and args[0] == "network":
        T = solve_heatpipe_network_model_static(k, A, lamb, T_infc, Q)
        print_heat_pipe_temperatures(T)
        return 0
    elif len(args) > 0 and args[0] == "discretised":
        heatpipe = heatpipe_discretised(data_discretised)
        T = heatpipe.solve_heatpipe_discretised()
        print(T)
        return 0
    else:
        raise ValueError("Invalid function argument")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
