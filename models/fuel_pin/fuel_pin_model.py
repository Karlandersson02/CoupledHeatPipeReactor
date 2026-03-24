import numpy as np

class fuelPin:
    def __init__(self):

        # Geometry
        self.N_R = 100
        self.N_Z = 100
        self.r = 1
        self.l = 1

        self.N_clad = 1
        self.N_gas = 1
        self.N_fuel = self.N_R - self.N_clad - self.N_gas

        self.Delta_Z = self.l / self.N_Z * np.ones(self.N_Z, dtype=float)

        # Neutron flux properties
        self.N_G = 8
        self.phi_g = np.ones((self.N_Z, self.N_G))

        # Neutron material properties
        self.Sigma_f = np.ones(self.N_G)
        self.kappa = np.ones(self.N_G)

        # Material properties
        self.k_fuel = 50
        self.k_clad = 25
        self.h_gas = 1e6

        self.T_moderator = np.ones(self.N_Z)


    def solve(self):
        self.initialize_discretization()

        surface_tensor = self.calculate_surfaces()

        k_matrix = self.generate_k_matrix()
        h_matrix = self.generate_h_matrix()

        alpha = self.calculate_alpha(surface_tensor)
        
        M, C = self.generate_matrix_form(alpha, k_matrix, h_matrix)
    
        T = np.linalg.solve(M, C)

        self.T = T


    def initialize_discretization(self):     
        R         = np.zeros(2 * self.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.N_R, dtype=float)
        
        # Calculating the radii of the half-elements
        R[0] = np.sqrt(self.r**2 / (self.N_R * 2))

        for i in range(1, 2 * self.N_R):
            R[i] = np.sqrt(R[i-1]**2 + R[0]**2)

        # Calculating the differences in the radius of the half-elements
        delta_R[0] = R[0]
        for i in range(1, 2 * self.N_R):
            delta_R[i] = R[i] - R[i - 1]

        delta_Rm = delta_R[0::2]
        delta_Rp = delta_R[1::2]

        self.delta_Rm = delta_Rm
        self.delta_Rp = delta_Rp

        self.Delta_V = np.pi * (delta_Rp[0] + delta_Rm[0])**2 * self.Delta_Z 


    def calculate_surfaces(self):
        delta_R = self.delta_Rp + self.delta_Rm
        Rp = np.cumsum(delta_R)
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi
        S_rm = Rm * 2*np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.N_Z, axis=0)
        surface_tensor[..., 0:2] *= 2 * self.Delta_Z[:, None, None]

        return surface_tensor
    
    
    def generate_k_matrix(self):
        k_matrix = np.zeros((self.N_Z, self.N_R))
        k_matrix[:, :self.N_fuel] = self.k_fuel
        k_matrix[:, self.N_fuel:] = self.k_clad
        self.k_matrix = k_matrix


    def generate_h_matrix(self):
        h_matrix = np.zeros((self.N_Z, self.N_R))
        h_matrix[:, self.N_fuel:(self.N_fuel + self.N_gas)] = self.h_gas
        self.h_matrix = h_matrix


    def calculate_alpha(self, surface_tensor):
        alpha_tensor = np.zeros_like(surface_tensor)

        for i in range(alpha_tensor.shape[0]):
            for j in range(alpha_tensor.shape[1]):
                if not (j == alpha_tensor.shape[1] - 1):
                    alpha_tensor[i, j, 0] = (surface_tensor[i, j, 0] * self.k_matrix[i, j+1]) / (self.k_matrix[i, j]*self.delta_Rm[j+1] + self.k_matrix[i, j+1]*self.delta_Rp[j])
                if not (j == 0):
                    alpha_tensor[i, j, 1] = (surface_tensor[i, j, 1] * self.k_matrix[i, j-1]) / (self.k_matrix[i, j]*self.delta_Rp[j-1] + self.k_matrix[i, j-1]*self.delta_Rm[j])
                if not (i == alpha_tensor.shape[0] - 1):
                    alpha_tensor[i, j, 2] = (surface_tensor[i, j, 2] * self.k_matrix[i+1, j]) / (self.k_matrix[i, j]*self.Delta_Z + self.k_matrix[i+1, j]*self.Delta_Z)
                if not (i == 0):
                    alpha_tensor[i, j, 3] = (surface_tensor[i, j, 3] * self.k_matrix[i-1, j]) / (self.k_matrix[i, j]*self.Delta_Z + self.k_matrix[i-1, j]*self.Delta_Z)

        # Init mask
        cooling_mask = np.zeros_like(alpha_tensor, dtype=bool)
                                     
        # Configure masks
        cooling_mask[:, -1, 0] = True

        # Apply masks
        # Behöver vi verkligen en mask här?
        alpha_tensor[cooling_mask] = surface_tensor[cooling_mask]

        return alpha_tensor
    
    
    def generate_matrix_form(self, alpha, k, h):
        N = self.N_R * self.N_Z
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
        # Insulated wall at z = 0, no corners.
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
        # Insulated wall at z = N_Z - 1, no corners.
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
        # Cladding BC elements, no corners.
        r = self.N_R - 1
        for z in range(1, self.N_Z-1): #
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (
                alpha[z][r][1] + 
                alpha[z][r][2] + 
                alpha[z][r][3]
            )
            M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0]

            M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -h[z][r] * alpha[z][r][0] * self.T_moderator[z]

        # -----------------------
        # Fuel inner BC elements, no corners.
        r = 0
        for z in range(0, self.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (
                alpha[z][r][0] + 
                alpha[z][r][2] + 
                alpha[z][r][3]
            )

            M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Lowermost cladding outer corner (z = 0, r = N_R-1).
        z = 0
        r = self.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0]

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 

        C[T_idx] = -h[z][r] * alpha[z][r][0] * self.T_moderator[z]

        # -----------------------
        # Uppermost cladding outer corner (z = N_Z - 1, r = N_R - 1).
        z = self.N_Z - 1
        r = self.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][3])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0] 

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]  

        C[T_idx] = -h[z][r] * alpha[z][r][0] * self.T_moderator[z]

        # -----------------------
        # Lowermost fuel inner corner (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2])

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2]

        # -----------------------
        # Uppermost fuel inner corner (z = N_Z - 1, r = 0).
        z = self.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][3])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Fuel heat production
        for z in range(0, self.N_Z):
            for r in range(1, self.N_fuel):
                T_idx = (stride * z) + r
                
                for g in range(self.N_G):

                    C[T_idx] += self.phi_g[z][g] * self.Sigma_f[g] * self.kappa * self.Delta_V
        

        return M, C

