import numpy as np
import matplotlib.pyplot as plt
from time import time


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

        self.Temperature_BC = data.get("Temperature_BC")
        self.h_vap = data.get("h_vap")
        self.h_cond = data.get("h_cond")
        self.T_cond = data.get("T_cond")
        self.T_op = data.get("T_op")

        self.k_wall = data.get("k_wall")
        self.k_wick = data.get("k_wick")
        self.Q = data.get("Q")
        if not self.Temperature_BC:
            Qnew = np.zeros(self.N_Z)
            Qnew[:self.N_evap] = self.Q
            Qout = -np.ones(self.N_cond) * np.sum(self.Q) / self.N_cond
            Qnew[(self.N_evap + self.N_adiabatic):] = Qout
            self.Q = Qnew

    def solve_heatpipe_discretised(self) -> np.ndarray:
        
        R, delta_Rp, delta_Rm, Z, delta_Z = self.initialize_discretization()

        surface_areas = self.calculate_surfaces(delta_Rp, delta_Rm, delta_Z)

        k_matrix = self.generate_k_matrix()
        h_matrix = self.generate_h_matrix()

        alpha = self.calculate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z, k_matrix)
        
        if self.Temperature_BC:
            M, C = self.generate_matrix_form_temperature_bc(alpha, k_matrix, h_matrix)
        else:
            M, C = self.generate_matrix_form_heat_bc(alpha, k_matrix, h_matrix)

        T = np.linalg.solve(M, C)
        if not self.Temperature_BC:
            T = np.concatenate([T, np.array([self.T_op])])

        return T


    def initialize_discretization(self):
        """
        Discretises the 2D model based on the information given in the heatpipe initalization. // 

        The radial discretisation results in constant area of the resulting concentric circles, while //
        the axial discretisation instead results in axial elements that reflects the relationship between // 
        the number of elements and length of the evaporator, adiabatic section and condenser.  
        
        :param self:
        :param delta_Rp: shape (N_R,)
        :type delta_Rp: np.ndarray
        :param delta_Rm: shape (N_R,)
        :type delta_Rm: np.ndarray
        :param delta_Rm: shape (N_Z,)
        :type delta_Rm: np.ndarray
        :return: shape (N_z, N_r, 4)
        :rtype: ndarray[Any, Any]
        """        
        R         = np.zeros(2 * self.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.N_R, dtype=float)
        delta_R_m = np.zeros(self.N_R, dtype=float)
        delta_R_p = np.zeros(self.N_R, dtype=float)
        Z         = np.zeros(2 * self.N_Z, dtype=float)
        delta_Z   = np.zeros(self.N_Z, dtype=float)
        
        # Calculating the radii of the half-elements
        R[0] = np.sqrt((self.r_outer**2 - self.r_vapour**2) / (self.N_R * 2) + self.r_vapour**2)

        for i in range(1, 2 * self.N_R):
            R[i] = np.sqrt(R[i-1]**2 + R[0]**2 - self.r_vapour**2)

        # Calculating the differences in the radius of the half-elements
        delta_R[0] = R[0] - self.r_vapour
        for i in range(1, 2 * self.N_R):
            delta_R[i] = R[i] - R[i - 1]

        delta_R_m = delta_R[0::2]
        delta_R_p = delta_R[1::2]

        # Calculating the Z-position of the bulk and edges of the elements 
        for i in range(self.N_evap*2):
            Z[i] = (i/2 + 1/2) * self.l_evap/self.N_evap

        for i in range(self.N_adiabatic*2):
            Z[i + self.N_evap*2] = (i/2 + 1/2) * self.l_adiabatic/self.N_adiabatic + self.l_evap

        for i in range(self.N_cond*2):
            Z[i+ self.N_evap*2 + self.N_adiabatic*2] = (i/2 + 1/2) * self.l_cond/self.N_cond + (self.l_evap + self.l_adiabatic)

        for i in range(self.N_Z):
            delta_Z[i] = Z[2*i + 1] - Z[2*i]

        return R, delta_R_p, delta_R_m, Z, delta_Z

    def calculate_surfaces(self, delta_Rp: np.ndarray, delta_Rm: np.ndarray, delta_Z: np.ndarray) -> np.ndarray:
        """
        Calculates a surface tensor representing the areas in the positive and negative radial and axial directions at every discrete element. \\
        The order is (positive radial, negative radial, positive axial, negative axial).
        
        :param self:
        :param delta_Rp: shape (N_R,)
        :type delta_Rp: np.ndarray
        :param delta_Rm: shape (N_R,)
        :type delta_Rm: np.ndarray
        :param delta_Rm: shape (N_Z,)
        :type delta_Rm: np.ndarray
        :return: shape (N_z, N_r, 4)
        :rtype: ndarray[Any, Any]
        """
        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R) + self.r_vapour
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi
        S_rm = Rm * 2*np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.N_Z, axis=0)
        surface_tensor[..., 0:2] *= 2*delta_Z[:, None, None]

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
    
    def generate_h_matrix(self) -> np.ndarray:

        h_matrix = np.zeros((self.N_Z, self.N_R))
        
        # Heat transfer coefficient for the vapor section.
        h_matrix[:self.N_evap, 0]                      = self.h_vap
        h_matrix[(self.N_Z - self.N_cond):self.N_Z, 0] = self.h_vap

        # Heat transfer coefficient for the condensator section.
        h_matrix[(self.N_Z - self.N_cond):self.N_Z, self.N_R - 1] = self.h_cond

        return h_matrix

    def calculate_alpha(self, surface_tensor: np.ndarray, delta_Rm: np.ndarray, delta_Rp: np.ndarray, delta_Z: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates a tensor representing the alpha coefficients on every discrete element in the heat pipe
        
        :param self:
        :param surface_tensor: shape (N_z, N_r, 4)
        :type surface_tensor: np.ndarray
        :param delta_Rm: shape (N_r,)
        :type delta_Rm: np.ndarray
        :param delta_Rp: shape (N_r,)
        :type delta_Rp: np.ndarray
        :param delta_Z: shape (N_Z,)
        :type delta_Z: np.ndarray
        :param k_matrix: shape (N_z, N_r)
        :type k_matrix: np.ndarray
        :return: shape (N_z, N_r, 4)
        :rtype: ndarray[Any, Any]
        """
        alpha_tensor = np.zeros_like(surface_tensor)

        for i in range(alpha_tensor.shape[0]):
            for j in range(alpha_tensor.shape[1]):
                if not (j == alpha_tensor.shape[1] - 1):
                    alpha_tensor[i, j, 0] = (surface_tensor[i, j, 0] * k_matrix[i, j+1]) / (k_matrix[i, j]*delta_Rm[j+1] + k_matrix[i, j+1]*delta_Rp[j])
                if not (j == 0):
                    alpha_tensor[i, j, 1] = (surface_tensor[i, j, 1] * k_matrix[i, j-1]) / (k_matrix[i, j]*delta_Rp[j-1] + k_matrix[i, j-1]*delta_Rm[j])
                if not (i == alpha_tensor.shape[0] - 1):
                    alpha_tensor[i, j, 2] = (surface_tensor[i, j, 2] * k_matrix[i+1, j]) / (k_matrix[i, j]*delta_Z[i+1] + k_matrix[i+1, j]*delta_Z[i])
                if not (i == 0):
                    alpha_tensor[i, j, 3] = (surface_tensor[i, j, 3] * k_matrix[i-1, j]) / (k_matrix[i, j]*delta_Z[i-1] + k_matrix[i-1, j]*delta_Z[i])

        # Init masks
        adiabatic_mask = np.zeros_like(alpha_tensor, dtype=bool)
        vapour_mask = np.zeros_like(alpha_tensor, dtype=bool)
        cooling_mask = np.zeros_like(alpha_tensor, dtype=bool)
        
        # Configure masks
        adiabatic_mask[self.N_evap:(self.N_evap + self.N_adiabatic), :, 0:2] = True
        vapour_mask[0:self.N_evap, 0, 1] = True
        vapour_mask[(self.N_evap + self.N_adiabatic):, 0, 1] = True
        cooling_mask[(self.N_evap + self.N_adiabatic):, -1, 0] = True

        # Apply masks
        alpha_tensor[adiabatic_mask] = 0
        alpha_tensor[vapour_mask] = surface_tensor[vapour_mask]
        alpha_tensor[cooling_mask] = surface_tensor[cooling_mask]

        return alpha_tensor

    def generate_matrix_form_temperature_bc(self, alpha: np.ndarray, k: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Number of physical grid nodes + extra vapor node (stored at index -1).
        N_phys = self.N_R * self.N_Z
        N = N_phys + 1
        stride = self.N_R 

        M = np.zeros((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # -----------------------
        # Bulk elements.
        for z in range(1, self.N_Z - 1):
            for r in range(1, self.N_R - 1):
                T_idx = (stride * z) + r

                M[T_idx][T_idx] = -k[z][r] * (
                    alpha[z][r][0] +
                    alpha[z][r][1] +
                    alpha[z][r][2] +
                    alpha[z][r][3]
                )

                M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
                M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
                M[T_idx][T_idx + stride]  = k[z][r] * alpha[z][r][2]  
                M[T_idx][T_idx - stride]  = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Insulated wall at z = 0
        z = 0
        for r in range(1, self.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx][T_idx] = -k[z][r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][2]
            )

            M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride]  = k[z][r] * alpha[z][r][2]  

        # -----------------------
        # Insulated wall at z = N_Z - 1
        z = self.N_Z - 1
        for r in range(1, self.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx][T_idx] = -k[z][r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][3]
            )

            M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx - stride]  = k[z][r] * alpha[z][r][3]  


        # -----------------------
        # Evaporator outer BC elements, no corners.
        r = self.N_R - 1
        for z in range(1, self.N_Z-1): #
            T_idx = z * stride + r 

            M[T_idx][T_idx] = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2] + alpha[z][r][3])
            M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][0]

            M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            if z < self.N_evap:
                C[T_idx] = -self.Q[z]
            else:
                C[T_idx] = -h[z][r] * alpha[z][r][0] * self.T_cond

        # -----------------------
        # Wick BC elements against vapor, no corners.
        r = 0
        for z in range(0, self.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2] + alpha[z][r][3])
            M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

            M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
            M[T_idx][-1]             = h[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Corner next to evaporator entrance (z = 0, r = N_R-1).
        z = 0
        r = self.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx] = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2])

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 

        C[T_idx] = -self.Q[z]

        # -----------------------
        # Corner next to condenser outlet (z = N_Z - 1, r = N_R - 1).
        z = self.N_Z - 1
        r = self.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][3])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0] 

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]  

        C[T_idx] = -h[z][r] * alpha[z][r][0] * self.T_cond 

        # -----------------------
        # Corner next to evaporator vapor inlet (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1] 

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][-1]             = h[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2]

        # -----------------------
        # Corner next to condenser vapor outlet/inlet (z = N_Z - 1, r = 0).
        z = self.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][3])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][-1]             = h[z][r] * alpha[z][r][1]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Vapor elements. 

        for z in range(0, self.N_evap):
            r = 0
            T_idx = z * stride + r 

            M[-1][-1   ] -= h[z][r] * alpha[z][r][1]
            M[-1][T_idx] += h[z][r] * alpha[z][r][1]

        for z in range(self.N_Z - self.N_cond, self.N_Z):
            r = 0
            T_idx = z * stride + r 

            M[-1][-1   ] -= h[z][r] * alpha[z][r][1]
            M[-1][T_idx] += h[z][r] * alpha[z][r][1]

        # -----------------------
        # Return matrix and vector

        return M, C

    def generate_matrix_form_heat_bc(self, alpha: np.ndarray, k: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Number of physical grid nodes + extra vapour node (stored at index -1).
        N_phys = self.N_R * self.N_Z
        N = N_phys
        stride = self.N_R 

        M = np.zeros((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # -----------------------
        # Bulk elements.
        for z in range(1, self.N_Z - 1):
            for r in range(1, self.N_R - 1):
                T_idx = (stride * z) + r

                M[T_idx][T_idx] = -k[z][r] * (
                    alpha[z][r][0] +
                    alpha[z][r][1] +
                    alpha[z][r][2] +
                    alpha[z][r][3]
                )

                M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
                M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
                M[T_idx][T_idx + stride]  = k[z][r] * alpha[z][r][2]  
                M[T_idx][T_idx - stride]  = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Insulated wall at z = 0
        z = 0
        for r in range(1, self.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx][T_idx] = -k[z][r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][2]
            )

            M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride]  = k[z][r] * alpha[z][r][2]  

        # -----------------------
        # Insulated wall at z = N_Z - 1
        z = self.N_Z - 1
        for r in range(1, self.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx][T_idx] = -k[z][r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][3]
            )

            M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx - stride]  = k[z][r] * alpha[z][r][3]  


        # -----------------------
        # Wall BC elements, no corners.
        r = self.N_R - 1
        for z in range(1, self.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx] = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2] + alpha[z][r][3])
            M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][0]

            M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -self.Q[z]

        # -----------------------
        # Wick BC elements against vapour, no corners.
        r = 0
        for z in range(0, self.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2] + alpha[z][r][3])
            M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

            M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
            # M[T_idx][-1]             = h[z][r] * alpha[z][r][1] ......................
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -h[z][r] * alpha[z][r][1] * self.T_op

        # -----------------------
        # Corner next to evaporator entrance (z = 0, r = N_R-1).
        z = 0
        r = self.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx] = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2])

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 

        C[T_idx] = -self.Q[z]

        # -----------------------
        # Corner next to condenser outlet (z = N_Z - 1, r = N_R - 1).
        z = self.N_Z - 1
        r = self.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][3])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0]

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]  

        C[T_idx] = -self.Q[z]

        # -----------------------
        # Corner next to evaporator vapour inlet (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1] 

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        # M[T_idx][-1]             = h[z][r] * alpha[z][r][1] .....................
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2]

        C[T_idx] = -h[z][r] * alpha[z][r][1] * self.T_op

        # -----------------------
        # Corner next to condenser vapour inlet (z = N_Z - 1, r = 0).
        z = self.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][3])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        # M[T_idx][-1]             = h[z][r] * alpha[z][r][1] .............................
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]

        C[T_idx] = -h[z][r] * alpha[z][r][1] * self.T_op

        # -----------------------
        # Vapour elements. 

        # for z in range(0, self.N_evap):
        #     r = 0
        #     T_idx = z * stride + r 

        #     M[-1][-1   ] -= h[z][r] * alpha[z][r][1]
        #     M[-1][T_idx] += h[z][r] * alpha[z][r][1]

        # for z in range(self.N_Z - self.N_cond, self.N_Z):
        #     r = 0
        #     T_idx = z * stride + r 

        #     M[-1][-1   ] -= h[z][r] * alpha[z][r][1]
        #     M[-1][T_idx] += h[z][r] * alpha[z][r][1] ........................

        # -----------------------
        # Return matrix and vector

        return M, C