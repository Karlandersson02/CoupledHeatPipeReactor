import numpy as np
import matplotlib.pyplot as plt

def solve_heatpipe_network_model_static(
        k: np.ndarray, 
        A: np.ndarray, 
        lamb: np.ndarray, 
        T_infc, 
        Q
) -> np.ndarray:
    """ 
    [T] = n 
    [T_infc] = 1
    [k] = (n + 1) 
    [A] = (n + 1)  
    [lamb] = n
    [Q] = 1
    """
    
    n = A.shape[0]
 
    eta = (k * A / lamb) / (
        (k[0] * A[0] / lamb[0]) + 
        (k[4] * A[4] / lamb[4]) + 
        (k[5] * A[5] / lamb[5])
        )
    
    
    eta_prime = (k * A / lamb) / (
        (k[3] * A[3] / lamb[3]) + 
        (k[4] * A[4] / lamb[4]) + 
        (k[5] * A[5] / lamb[5]) + 
        (k[6] * A[6] / 2)
        )
    

    k_i, k_j = np.meshgrid(k, k)
    A_i, A_j = np.meshgrid(A, A)
    lamb_i, lamb_j = np.meshgrid(lamb, lamb)
    xi = (k_i * A_i / lamb_i)  / ((k_i * A_i / lamb_i) + (k_j * A_j / lamb_j))
    
    M = np.array(
        [[(xi[0,1] + eta[0] - 2), xi[1,0], 0, 0, eta[4], eta[5]], 
         [xi[0,1], (xi[1,0] + xi[1,2] - 2), xi[2,1], 0, 0, 0], 
         [0, xi[1,2], (xi[2,1] + xi[2,3] - 2), xi[3,2], 0, 0], 
         [0, 0, xi[2,3], (xi[3,2] + eta[3] - 2), eta_prime[4], eta_prime[5]], 
         [eta[0], 0, 0, eta_prime[3], eta[4] + eta_prime[4] - 2, eta[5] + eta_prime[5]], 
         [eta[0], 0, 0, eta_prime[3], eta[4] + eta_prime[4], eta[5] + eta_prime[5] - 2]]
    )

    C_in = (Q/2) / ((k[0] * A[0] / lamb[0]) + (k[4] * A[4] / lamb[4]) + (k[5] * A[5] / lamb[5]))
    C_out = (k[6]*A[6]*T_infc/2) / ((k[3] * A[3] / lamb[3]) + (k[4] * A[4] / lamb[4]) + (k[5] * A[5] / lamb[5]) + (k[6] * A[6] / 2))

    C = np.array([
        C_in,
        0,
        0,
        C_out,
        C_in + C_out,
        C_in + C_out        
        ])
    
    T = np.linalg.solve(M, C)
    return T
 
