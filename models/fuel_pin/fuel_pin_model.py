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

        self.l_clad = self.l / self.N_clad
        self.l_gas = self.l / self.N_gas
        self.l_fuel = self.l / self.N_fuel

        self.Delta_Z = self.l / self.N_Z

        # Material properties

        self.k_fuel = 50
        self.k_clad = 25
        self.h_gas = 1e6

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

    def calculate_surfaces(self, delta_Rp, delta_Rm, delta_Z):
        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R)
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi
        S_rm = Rm * 2*np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.N_Z, axis=0)
        surface_tensor[..., 0:2] *= 2*delta_Z[:, None, None]

        return surface_tensor
    
    def assemble_k_matrix(self):
        k_matrix = np.zeros((self.N_Z, self.N_R))
        k_matrix[:, :self.N_fuel] = self.k_fuel
        k_matrix[:, :self.N_fuel] = self.k_clad
        self.k_matrix = k_matrix

    def assemble_h_matrix(self):
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

        