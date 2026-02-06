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

    R, delta_Rp, delta_Rm, Z, delta_Z = initialize_discretization(r_outer, delta_wick, delta_wall, l_evap, l_adiabatic, l_cond, N_R, N_Z)

    surface_areas = calculate_surfaces(delta_Rp, delta_Rm, delta_Z, N_Z)

    boundary_conditions = generate_boundary_conditions(Q, h_vap, h_cond, T_cond)
    alpha = calculate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z )

    M, C = generate_matrix_form(alpha, boundary_conditions) 

    T = np.linalg.solve(M, C)

    return T


def initialize_discretization(r_outer, delta_wick, delta_wall, l_evap, l_adiabatic, l_cond, N_R, N_Z):
    return np.array([0]), np.array([0]), np.array([0]), np.array([0]), 0

def generate_boundary_conditions(Q: np.ndarray, h_vap: float, h_cond: float, T_cond: float) -> np.ndarray:
    return np.array([0]) # ? 

def calculate_surfaces(delta_Rp: np.ndarray, delta_Rm: np.ndarray, delta_Z: float, N_Z: int) -> np.ndarray:

    delta_R = delta_Rp + delta_Rm
    Rp = np.cumsum(delta_R)
    Rm = Rp - delta_R

    S_rp = Rp * 2*np.pi * delta_Z
    S_rm = Rm * 2*np.pi * delta_Z
    S_z = (Rp**2 - Rm**2) * np.pi

    surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
    surface_tensor = np.repeat(surface_tensor[None], N_Z, axis=0)

    return surface_tensor

def calculate_alpha(surface_areas: np.ndarray, delta_R_m: np.ndarray, delta_R_p: np.ndarray, delta_Z: float) -> np.ndarray:
    return np.array([0])

def generate_matrix_form(alpha: np.ndarray, boundary_conditions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.array([0]), np.array([0])
