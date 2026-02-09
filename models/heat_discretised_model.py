import numpy as np
import matplotlib.pyplot as plt


class heatpipe_discretised:
    def __init__(self):
        self.r_outer  = 0
        self.delta_wick  = 0
        self.delta_wall  = 0

        self.l_evap  = 0
        self.l_adiabatic  = 0
        self.l_cond  = 0
        
        self.N_R  = 0
        self.N_Z  = 0

        self.h_vap = 0
        self.h_cond = 0
        self.T_cond = 0

    def solve_heatpipe_discretised(
            self,
            k: np.ndarray, 
            A: np.ndarray, 
            lamb: np.ndarray, 
            T_infc, 
            Q, 
            discretization
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

        R, delta_Rp, delta_Rm, Z, delta_Z = self.initialize_discretization(r_outer, delta_wick, delta_wall, l_evap, l_adiabatic, l_cond, discretization)

        surface_areas = self.calculate_surfaces(delta_Rp, delta_Rm, delta_Z, N_Z)

        boundary_conditions = self.generate_boundary_conditions(Q, h_vap, h_cond, T_cond)

        k_matrix = self.calculate_k(k)

        alpha = self.calculate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z, k_matrix)

        M, C = self.generate_matrix_form(alpha, boundary_conditions)

        T = np.linalg.solve(M, C)
        return T


    def initialize_discretization(self, r_vapour, delta_wick, delta_wall, l_evap, l_adiabatic, l_cond, discretisation):
        N_wall = discretisation["wall"]
        N_wick = discretisation["wick"]
        N_R    = N_wall + N_wick
        N_Z    = discretisation["Z"]
        
        R         = np.zeros(2 * N_R, dtype=float)
        delta_R   = np.zeros(2 * N_R, dtype=float)
        delta_R_m = np.zeros(N_R, dtype=float)
        delta_R_p = np.zeros(N_R, dtype=float)
        Z         = np.zeros(N_Z, dtype=float)
        
        R_outer = r_vapour + delta_wick + delta_wall
        Z_total = l_evap + l_adiabatic + l_cond
        
        # Calculating the radiuses of the half-elements
        R[0] = np.sqrt(R_outer**2)
        for i in range(1, 2 * N_R):
            R[i] = np.sqrt(R[i-1]**2 + R[0]**2)

        # Calculating the differences in the radius of the half-elements
        delta_R[0] = R[0]
        for i in range(1, 2 * N_R):
            delta_R = R[i] - R[i - 1]

        delta_R_m = delta_R[0::2]
        delta_R_p = delta_R[1::2]

        # Calculating the Z-position of the bulk of the elements 
        for i in range(N_Z):
            Z[i] = (i + 1/2) * Z_total/N_Z
        
        # Calculating the difference in the Z-position of the elements
        delta_Z = Z_total/N_Z

        return R, delta_R_p, delta_R_m, Z, delta_Z

    def generate_boundary_conditions(self, Q: np.ndarray, h_vap: float, h_cond: float, T_cond: float) -> dict:
        return {"Q_evap": Q, "h_vap": h_vap, "h_cond": h_cond, "T_cond": T_cond}

    def calculate_surfaces(self, delta_Rp: np.ndarray, delta_Rm: np.ndarray, delta_Z: float, N_Z: int) -> np.ndarray:

        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R)
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi * delta_Z
        S_rm = Rm * 2*np.pi * delta_Z
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], N_Z, axis=0)

        return surface_tensor

    def generate_k_matrix(self, k: np.ndarray) -> np.ndarray:
        return np.array([0])

    def calculate_alpha(self, surface_tensor: np.ndarray, delta_Rm: np.ndarray, delta_Rp: np.ndarray, delta_Z: float, k_matrix: np.ndarray) -> np.ndarray:

        alpha_tensor = np.zeros_like(surface_tensor)

        for i in range(1, alpha_tensor.shape[0] - 1):
            for j in range(1, alpha_tensor.shape[1] - 1):
                alpha_tensor[i, j, 0] = (surface_tensor[i, j, 0] * k_matrix[i+1, j]) / (k_matrix[i, j]*delta_Rm[i+1, j] - k_matrix[i+1, j]*delta_Rp[i, j])
                alpha_tensor[i, j, 1] = (surface_tensor[i, j, 1] * k_matrix[i-1, j]) / (k_matrix[i, j]*delta_Rp[i-1, j] - k_matrix[i-1, j]*delta_Rm[i, j])
                alpha_tensor[i, j, 2] = (surface_tensor[i, j, 2] * k_matrix[i, j+1]) / (k_matrix[i, j]*delta_Rm[i, j+1] - k_matrix[i, j+1]*delta_Rp[i, j])
                alpha_tensor[i, j, 3] = (surface_tensor[i, j, 3] * k_matrix[i, j-1]) / (k_matrix[i, j]*delta_Rp[i, j-1] - k_matrix[i, j-1]*delta_Rm[i, j])

        boundary_mask = np.zeros_like(alpha_tensor, dtype=bool)
        boundary_mask[0, :, :]  = True
        boundary_mask[-1, :, :] = True
        boundary_mask[:, 0, :]  = True
        boundary_mask[:, -1, :] = True

        alpha_tensor[boundary_mask] = 0

        adiabatic_mask = np.zeros_like(alpha_tensor, dtype=bool)
        adiabatic_mask[self.N_evap:(self.N_evap + self.N_adiabatic), :, 0:2] = True
        alpha_tensor[adiabatic_mask] = 0

        return alpha_tensor

    def generate_matrix_form(self, alpha: np.ndarray, boundary_conditions: dict) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0]), np.array([0])
