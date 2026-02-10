import numpy as np
import matplotlib.pyplot as plt


class heatpipe_discretised:
    def __init__(self, data):

        self.r_outer  = data.get("r_outer")
        self.delta_wick  = data.get("delta_wick")
        self.delta_wall  = data.get("delta_wall")
        self.r_vapour = self.r_outer - self.delta_wick - self.delta_wall

        self.l_evap  = data.get("l_evap")
        self.l_adiabatic  = data.get("l_adiabatic")
        self.l_cond  = data.get("l_cond")
        self.l_tot = self.l_evap + self.l_adiabatic + self.l_cond
        
        self.N_wick = data.get("N_wick")
        self.N_wall = data.get("N_wall")
        self.N_R  = self.N_wick + self.N_wall

        self.N_evap = data.get("N_evap")
        self.N_adiabatic = data.get("N_adiabatic")
        self.N_cond = data.get("N_cond")
        self.N_Z = self.N_evap + self.N_adiabatic + self.N_cond
        self.delta_Z = self.l_tot / self.N_Z

        self.h_vap = data.get("h_vap")
        self.h_cond = data.get("h_cond")
        self.T_cond = data.get("T_cond")

        self.k_wall = data.get("k_wall")
        self.k_wick = data.get("k_wick")
        self.Q = data.get("Q")

    def solve_heatpipe_discretised(
            self,
            k: np.ndarray, 
            A: np.ndarray, 
            lamb: np.ndarray, 
            T_infc, 
            Q, 
            discretization
    ) -> np.ndarray:
        
        R, delta_Rp, delta_Rm, Z = self.initialize_discretization()

        surface_areas = self.calculate_surfaces(delta_Rp, delta_Rm)

        k_matrix = self.generate_k_matrix()

        alpha = self.calculate_alpha(surface_areas, delta_Rm, delta_Rp, k_matrix)

        M, C = self.generate_matrix_form(alpha, k_matrix)

        T = np.linalg.solve(M, C)
        
        return T


    def initialize_discretization(self):        
        R         = np.zeros(2 * self.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.N_R, dtype=float)
        delta_R_m = np.zeros(self.N_R, dtype=float)
        delta_R_p = np.zeros(self.N_R, dtype=float)
        Z         = np.zeros(self.N_Z, dtype=float)
        
        # Calculating the radiuses of the half-elements
        R[0] = np.sqrt(self.r_outer**2)
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

        return R, delta_R_p, delta_R_m, Z

    def calculate_surfaces(self, delta_Rp: np.ndarray, delta_Rm: np.ndarray) -> np.ndarray:
        """
        Calculates a surface tensor representing the areas in the positive and negative radial and axial directions at every discrete element. \\
        The order is (positive radial, negative radial, positive axial, negative axial).
        
        :param self:
        :param delta_Rp: shape (N_R /2 ,)
        :type delta_Rp: np.ndarray
        :param delta_Rm: shape (N_R /2 ,)
        :type delta_Rm: np.ndarray
        :return: shape (N_z, N_r, 4)
        :rtype: ndarray[Any, Any]
        """
        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R)
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi * self.delta_Z
        S_rm = Rm * 2*np.pi * self.delta_Z
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.N_Z, axis=0)

        return surface_tensor

    def generate_k_matrix(self) -> np.ndarray:
        """
        Generates a matrix representing the conduction coefficient on every discrete element in the heat pipe.
        
        :param self:
        :return: shape (N_z, N_r)
        :rtype: ndarray[Any, Any]
        """
        k_matrix = np.zeros((self.N_Z, self.N_R))
        k_matrix[:, :self.N_wick] = self.k_wick
        k_matrix[:, self.N_wick:] = self.k_wall

        return k_matrix

    def calculate_alpha(self, surface_tensor: np.ndarray, delta_Rm: np.ndarray, delta_Rp: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates a tensor representing the alpha coefficients on every discrete element in the heat pipe
        
        :param self:
        :param surface_tensor: shape (N_z, N_r, 4)
        :type surface_tensor: np.ndarray
        :param delta_Rm: shape (N_r/2,)
        :type delta_Rm: np.ndarray
        :param delta_Rp: shape (N_r/2,)
        :type delta_Rp: np.ndarray
        :param delta_Z:
        :type delta_Z: float
        :param k_matrix: shape (N_z, N_r)
        :type k_matrix: np.ndarray
        :return: shape (N_z, N_r, 4)
        :rtype: ndarray[Any, Any]
        """
        alpha_tensor = np.zeros_like(surface_tensor)

        for i in range(1, alpha_tensor.shape[0] - 1):
            for j in range(1, alpha_tensor.shape[1] - 1):
                alpha_tensor[i, j, 0] = (surface_tensor[i, j, 0] * k_matrix[i+1, j]) / (k_matrix[i, j]*delta_Rm[i+1, j] - k_matrix[i+1, j]*delta_Rp[i, j])
                alpha_tensor[i, j, 1] = (surface_tensor[i, j, 1] * k_matrix[i-1, j]) / (k_matrix[i, j]*delta_Rp[i-1, j] - k_matrix[i-1, j]*delta_Rm[i, j])
                alpha_tensor[i, j, 2] = (surface_tensor[i, j, 2] * k_matrix[i, j+1]) / (k_matrix[i, j]*delta_Rm[i, j+1] - k_matrix[i, j+1]*delta_Rp[i, j])
                alpha_tensor[i, j, 3] = (surface_tensor[i, j, 3] * k_matrix[i, j-1]) / (k_matrix[i, j]*delta_Rp[i, j-1] - k_matrix[i, j-1]*delta_Rm[i, j])

        # Init masks
        boundary_mask = np.zeros_like(alpha_tensor, dtype=bool)
        adiabatic_mask = np.zeros_like(alpha_tensor, dtype=bool)
        vapour_mask = np.zeros_like(alpha_tensor, dtype=bool)
        cooling_mask = np.zeros_like(alpha_tensor, dtype=bool)
        
        # Configure masks
        boundary_mask[0, :, :]  = True
        boundary_mask[-1, :, :] = True
        boundary_mask[:, 0, :]  = True
        boundary_mask[:, -1, :] = True

        adiabatic_mask[self.N_evap:(self.N_evap + self.N_adiabatic), :, 0:2] = True

        vapour_mask[0:self.N_evap, 0, 1] = True
        vapour_mask[(self.N_evap + self.N_adiabatic):, 0, 1] = True

        cooling_mask[(self.N_evap + self.N_adiabatic):, self.N_R, 0] = True

        # Apply masks
        alpha_tensor[boundary_mask] = 0
        alpha_tensor[adiabatic_mask] = 0
        alpha_tensor[vapour_mask] = surface_tensor[vapour_mask]
        alpha_tensor[cooling_mask] = surface_tensor[cooling_mask]

        return alpha_tensor

    def generate_matrix_form(self, alpha: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = self.N_R * self.N_Z

        M = np.zeros((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # Bulk elements
        for z in range(0, self.N_Z):
            for r in range(0, self.N_R):
                T_idx = (self.N_Z * z) + r

                M[T_idx][T_idx      ]      = k[z][r]     * (alpha[z][r][0] + 
                                                            alpha[z][r][1] + 
                                                            alpha[z][r][2] + 
                                                            alpha[z][r][3])        
                M[T_idx][T_idx + 1  ]      = k[z][r + 1] * alpha[z][r][0]       
                M[T_idx][T_idx - 1  ]      = k[z][r - 1] * alpha[z][r][1]       
                M[T_idx][T_idx + self.N_Z] = k[z + 1][r] * alpha[z][r][2]      
                M[T_idx][T_idx - self.N_Z] = k[z - 1][r] * alpha[z][r][3]      

        # Evaporator outer BC elements, not corners.
        for z in range(0, self.N_evap):
            r = self.N_R - 1
            T_idx = z * self.N_Z + r

            C[T_idx] = self.Q[z] / self.k[-1:, :self.N_evap]

    
        return M, C
