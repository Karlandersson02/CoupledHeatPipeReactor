import sys
import numpy as np
import matplotlib.pyplot as plt

from heat_network_model import solve_heatpipe_network_model_static

def main(args):
    # k = np.array([20,17,17,20,17,20,2.34e6])
    # A = np.array([0.183,0.183,0.183,0.183,3.58e-5,3.58e-5,0.183])
    # lamb = np.array([0.36e-3, 0.36e-3, 0.36e-3, 0.36e-3, 10, 10, 1])
    # T_infc = 300
    # Q = 5e3

    k = np.array([20,45,45,20,45,20,39])
    A = np.array([0.0045,0.0045,0.0239,0.0239,4e-5,4e-5,0.0239])
    lamb = np.array([0.001, 0.001, 0.001, 0.001, 0.0525, 0.0525, 1])
    T_infc = 300
    Q = 770

    if len(args) > 0 and args[0] == "network_static":
        T = solve_heatpipe_network_model_static(k, A, lamb, T_infc, Q)
        print(f"Temperature:\nEvaporator wall: {T[0]:.1f} K\nEvaporator wick: {T[1]:.1f} K\nCondenser wick: {T[2]:.1f} K\nCondenser wall: {T[3]:.1f} K\nAdiabatic wick: {T[4]:.1f} K\nAdiabatic wall: {T[5]:.1f} K")
        return 0
    else:
        raise ValueError("Invalid function argument")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
