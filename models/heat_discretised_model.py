import numpy as np
import matplotlib.pyplot as plt


class heatpipe_discretised:
    def __init__(self, data):
        self.r_outer  = 0
        self.delta_wick  = 0
        self.delta_wall  = 0

        self.l_evap  = 0
        self.l_adiabatic  = 0
        self.l_cond  = 0
        self.l_tot = 0
        
        self.N_R  = 0
        self.N_wick = 0
        self.N_wall = 0

        self.N_Z  = 0
        self.N_evap = 0
        self.N_adiabatic = 0
        self.N_cond = 0

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
        
        R, delta_Rp, delta_Rm, Z, delta_Z = self.initialize_discretization()

        surface_areas = self.calculate_surfaces(delta_Rp, delta_Rm, delta_Z)

        alpha = self.calculate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z, k)

        M, C = self.generate_matrix_form(alpha)

        T = np.linalg.solve(M, C)
        
        return T


    def initialize_discretization(self):        
        R         = np.zeros(2 * self.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.N_R, dtype=float)
        delta_R_m = np.zeros(self.N_R, dtype=float)
        delta_R_p = np.zeros(self.N_R, dtype=float)
        Z         = np.zeros(self.N_Z, dtype=float)
        
        R_outer = self.r_vapour + self.delta_wick + self.delta_wall
        
        # Calculating the radiuses of the half-elements
        R[0] = np.sqrt(R_outer**2)
        for i in range(1, 2 * self.N_R):
            R[i] = np.sqrt(R[i-1]**2 + R[0]**2)

        # Calculating the differences in the radius of the half-elements
        delta_R[0] = R[0]
        for i in range(1, 2 * self.N_R):
            delta_R = R[i] - R[i - 1]

        delta_R_m = delta_R[0::2]
        delta_R_p = delta_R[1::2]

        # Calculating the Z-position of the bulk of the elements 
        for i in range(self.N_Z):
            Z[i] = (i + 1/2) * self.l_tot/self.N_Z
        
        # Calculating the difference in the Z-position of the elements
        delta_Z = self.l_tot/self.N_Z

        return R, delta_R_p, delta_R_m, Z, delta_Z

    def calculate_surfaces(self, delta_Rp: np.ndarray, delta_Rm: np.ndarray, delta_Z: float) -> np.ndarray:

        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R)
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi * delta_Z
        S_rm = Rm * 2*np.pi * delta_Z
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.N_Z, axis=0)

        return surface_tensor

    def calculate_k(self, k: np.ndarray) -> np.ndarray:
        return np.array([0])

    def calculate_alpha(self, surface_tensor: np.ndarray, delta_Rm: np.ndarray, delta_Rp: np.ndarray, delta_Z: float, k_matrix: np.ndarray) -> np.ndarray:

        alpha_tensor = np.zeros_like(surface_tensor)

        for i in range(1, alpha_tensor.shape[0] - 1):
            for j in range(1, alpha_tensor.shape[1] - 1):
                alpha_tensor[i, j, 0] = (surface_tensor[i, j, 0] * k_matrix[i+1, j]) / (k_matrix[i, j]*delta_Rm[i+1, j] - k_matrix[i+1, j]*delta_Rp[i, j])
                alpha_tensor[i, j, 1] = (surface_tensor[i, j, 1] * k_matrix[i-1, j]) / (k_matrix[i, j]*delta_Rp[i-1, j] - k_matrix[i-1, j]*delta_Rm[i, j])
                alpha_tensor[i, j, 2] = (surface_tensor[i, j, 2] * k_matrix[i, j+1]) / (k_matrix[i, j]*delta_Rm[i, j+1] - k_matrix[i, j+1]*delta_Rp[i, j])
                alpha_tensor[i, j, 3] = (surface_tensor[i, j, 3] * k_matrix[i, j-1]) / (k_matrix[i, j]*delta_Rp[i, j-1] - k_matrix[i, j-1]*delta_Rm[i, j])

        mask = np.zeros_like(alpha_tensor, dtype=bool)
        mask[0, :, :]  = True
        mask[-1, :, :] = True
        mask[:, 0, :]  = True
        mask[:, -1, :] = True

        alpha_tensor[mask] = 0

        i_adiabatic_start = (self.l_evap / (self.l_evap + self.l_cond + self.l_adiabatic) * self.N_Z) // 1
        i_adiabatic_end = ((self.l_evap + self.l_adiabatic) / (self.l_evap + self.l_cond + self.l_adiabatic) * self.N_Z) // 1

        mask = np.zeros_like(alpha_tensor, dtype=bool)
        mask = 0

        return np.array([0])

    def generate_matrix_form(self, alpha: np.ndarray, boundary_conditions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N_R = self.N_R
        N_Z = boundary_conditions["N_Z"]
        N = N_R * N_Z

        M = np.zeros((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # Bulk elements
        for z in range(0, N_Z):
            for r in range(0, N_R):
                T_idx = (N_Z * z) + r

                M[T_idx][T_idx      ] = alpha[z][r][0] + alpha[z][r][1] + alpha[z][r][2] + alpha[z][r][3]        
                M[T_idx][T_idx + 1  ] = alpha[z][r][0]       
                M[T_idx][T_idx - 1  ] = alpha[z][r][1]       
                M[T_idx][T_idx + N_Z] = alpha[z][r][2]      
                M[T_idx][T_idx - N_Z] = alpha[z][r][3]      

        # Evaporator outer BC elements, not corners.
        for z in range(1, N_Z - 1):
            r = N_R - 1
            T_idx = z * N_Z + r

            C[T_idx] = boundary_conditions["Q_evap"][z] 
        
        # Evaporator vapor BC elements, not corners.

        # Todo, need to get N_cond and N_evap
        for z in range(1,boundary_conditions[''] - 1):
            r = 0
            M[(N_Z * z) + r][(N_Z *  z     ) + r    ] = alpha[z][r][0] + alpha[z][r][1] + alpha[z][r][2] + alpha[z][r][3]        
            M[(N_Z * z) + r][(N_Z *  z     ) + r + 1] = alpha[z][r][0]       
            M[(N_Z * z) + r][(N_Z *  z     ) + r - 1] = alpha[z][r][1]       
            M[(N_Z * z) + r][(N_Z * (z + 1)) + r    ] = alpha[z][r][2]      
            M[(N_Z * z) + r][(N_Z * (z - 1)) + r    ] = alpha[z][r][3]      
            C = 0

        # Condensator vapor BC elements, not corners.
        for z in range(1,N_Z - 1):
            for r in range(1,N_R - 1):
                M[(N_Z * z) + r][(N_Z *  z     ) + r    ] = alpha[z][r][0] + alpha[z][r][1] + alpha[z][r][2] + alpha[z][r][3]        
                M[(N_Z * z) + r][(N_Z *  z     ) + r + 1] = alpha[z][r][0]       
                M[(N_Z * z) + r][(N_Z *  z     ) + r - 1] = alpha[z][r][1]       
                M[(N_Z * z) + r][(N_Z * (z + 1)) + r    ] = alpha[z][r][2]      
                M[(N_Z * z) + r][(N_Z * (z - 1)) + r    ] = alpha[z][r][3]      

                C = 0

        # Condensator outer BC elements, not corners.
        for z in range(1,N_Z - 1):
            for r in range(1,N_R - 1):
                M[(N_Z * z) + r][(N_Z *  z     ) + r    ] = alpha[z][r][0] + alpha[z][r][1] + alpha[z][r][2] + alpha[z][r][3]        
                M[(N_Z * z) + r][(N_Z *  z     ) + r + 1] = alpha[z][r][0]       
                M[(N_Z * z) + r][(N_Z *  z     ) + r - 1] = alpha[z][r][1]       
                M[(N_Z * z) + r][(N_Z * (z + 1)) + r    ] = alpha[z][r][2]      
                M[(N_Z * z) + r][(N_Z * (z - 1)) + r    ] = alpha[z][r][3]      

                C = 0


        # Corner cases
            M[0][0]
            M[N_Z * N_R - 1][0]
            M[0][N_Z * N_R - 1]
            M[N_Z * N_R - 1][N_Z * N_R - 1]
        return M, C
