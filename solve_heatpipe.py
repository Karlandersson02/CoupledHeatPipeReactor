import sys
import numpy as np
import matplotlib.pyplot as plt

from heat_network_model import solve_heatpipe_network_model_static

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
    print(f"│ {T_ev_wick:6.0f} K     │ {T_ad_wick:6.0f} K     │ {T_co_wick:6.0f} K     │")
    print(f"│ {T_ev_wick - 273.15:6.0f} C     │ {T_ad_wick - 273.15:6.0f} C     │ {T_co_wick - 273.15:6.0f} C     │")
    print("├──────────────┼──────────────┼──────────────┤")
    print(f"│   WALL       │    WALL      │    WALL      │")
    print(f"│ {T_ev_wall:6.0f} K     │ {T_ad_wall:6.0f} K     │ {T_co_wall:6.0f} K     │")
    print(f"│ {T_ev_wall - 273.15:6.0f} C     │ {T_ad_wall - 273.15:6.0f} C     │ {T_co_wall - 273.15:6.0f} C     │")
    print("└──────────────┴──────────────┴──────────────┘\n")

def main(args):

    k = np.array([20,45,45,20,45,20,39])
    A = np.array([0.0045,0.0045,0.0239,0.0239,4e-5,4e-5,0.0239])
    lamb = np.array([0.001, 0.001, 0.001, 0.001, 0.0525, 0.0525, 1])
    T_infc = 300
    Q = 770

    if len(args) > 0 and args[0] == "network_static":
        T = solve_heatpipe_network_model_static(k, A, lamb, T_infc, Q)
        print_heat_pipe_temperatures(T)
        return 0
    else:
        raise ValueError("Invalid function argument")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
