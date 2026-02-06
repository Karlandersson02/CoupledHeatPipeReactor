import numpy as np
import matplotlib.pyplot as plt

def solve_heatpipe_discretised(
        k: np.ndarray, 
        A: np.ndarray, 
        lamb: np.ndarray, 
        T_infc, 
        Q,
        n_height,
        n_wall,
        n_wick
) -> np.ndarray:
    r_outer  = 0
    delta_wick  = 0
    delta_wall  = 0
    l_evap  = 0
    l_adiabatic  = 0
    l_cond  = 0
    N_R  = 0
    N_Z  = 0

    h_vap = 0
    h_cond = 0
    T_cond = 0

    R, delta_R_p, delta_R_m, Z, delta_Z = initialize_discretization(r_outer, delta_wick, delta_wall, l_evap, l_adiabatic, l_cond, N_R, N_Z)

    surface_areas = calculate_Surfaces(R, delta_R_p, delta_R_m)

    boundary_conditions = generate_boundary_conditions(Q, h_vap, h_cond, T_cond)
    alpha = calculate_alpha(surface_areas, delta_R_m, delta_R_p, delta_Z )

    M, C = generate_matrix_form(alpha, boundary_conditions) 

    T = np.linalg.solve(M, C)

    return T


def initialize_discretization(r_outer, delta_wick, delta_wall, l_evap, l_adiabatic, l_cond, N_R, N_Z):
    return np.array([0]), np.array([0]), np.array([0]), np.array([0]), 0

def generate_boundary_conditions(Q, h_vap, h_cond, T_cond):
    return np.array([0]) # ? 

def calculate_Surfaces(R, delta_R_p, delta_R_m):
    return np.array([0])

def calculate_alpha(surface_areas, delta_R_m, delta_R_p, delta_Z ):
    return np.array([0])

def generate_matrix_form(alpha, boundary_conditions):
    return np.array([0]), np.array([0])
