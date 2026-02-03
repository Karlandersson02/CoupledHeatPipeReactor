import sys
import numpy as np
import matplotlib.pyplot as plt

from Heat_network_model import solve_heatpipe_network_model_static

def main(args):
    k = np.array([1,1,1,1,1,1,1])
    A = np.array([1,1,1,1,1,1,1])
    lamb = np.array([1,1,1,1,100,100,-1e100])
    T_infc = 10 
    Q = -100

    if len(args) > 0 and args[0] == "network_static":
        T = solve_heatpipe_network_model_static(k, A, lamb, T_infc, Q)
        print(f"Temperature: {T}")
        return 0
    else:
        raise ValueError("Invalid function argument")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
