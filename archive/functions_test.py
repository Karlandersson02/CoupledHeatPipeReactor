import numpy as np

class testClass:

    def __init__(self) -> None:
          self.N_Z = 10
          self.N_R = 5

          self.N_evap = 3
          self.N_adiabatic = 2
          self.N_cond = 5

          self.N_wick = 2
          self.N_wall = 3

          self.k_wall = 1
          self.k_wick = 2

          self.delta_Z = 1.5

    def calculate_surfaces(self, delta_Rp: np.ndarray, delta_Rm: np.ndarray) -> np.ndarray:
        """
        Calculates a surface tensor representing the areas in the positive and negative radial and axial directions at every discrete element. \\
        The order is (positive radial, negative radial, positive axial, negative axial).
        
        :param self:
        :param delta_Rp: shape (N_R /2 ,)
        :type delta_Rp: np.ndarray
        :param delta_Rm: shape (N_R /2 ,)
        :type delta_Rm: np.ndarray
        :param delta_Z:
        :type delta_Z: float
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
        :param k_matrix: shape (N_z, N_r)
        :type k_matrix: np.ndarray
        :return: shape (N_z, N_r, 4)
        :rtype: ndarray[Any, Any]
        """
        alpha_tensor = np.zeros_like(surface_tensor)

        for i in range(1, alpha_tensor.shape[0] - 1):
            for j in range(1, alpha_tensor.shape[1] - 1):
                alpha_tensor[i, j, 0] = (surface_tensor[i, j, 0] * k_matrix[i+1, j]) / (k_matrix[i, j]*delta_Rm[j] - k_matrix[i+1, j]*delta_Rp[j])
                alpha_tensor[i, j, 1] = (surface_tensor[i, j, 1] * k_matrix[i-1, j]) / (k_matrix[i, j]*delta_Rp[j] - k_matrix[i-1, j]*delta_Rm[j])
                alpha_tensor[i, j, 2] = (surface_tensor[i, j, 2] * k_matrix[i, j+1]) / (k_matrix[i, j]*delta_Rm[j+1] - k_matrix[i, j+1]*delta_Rp[j])
                alpha_tensor[i, j, 3] = (surface_tensor[i, j, 3] * k_matrix[i, j-1]) / (k_matrix[i, j]*delta_Rp[j-1] - k_matrix[i, j-1]*delta_Rm[j])

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

        cooling_mask[(self.N_evap + self.N_adiabatic):, -1, 0] = True

        # Apply masks
        alpha_tensor[boundary_mask] = 0
        alpha_tensor[adiabatic_mask] = 0
        alpha_tensor[vapour_mask] = surface_tensor[vapour_mask]
        alpha_tensor[cooling_mask] = surface_tensor[cooling_mask]

        return alpha_tensor


classinstance = testClass()

deltaR = np.arange(classinstance.N_R*2)
delta_Rp = deltaR[1::2]
delta_Rm = deltaR[0::2]

surface_tensor = classinstance.calculate_surfaces(delta_Rp, delta_Rm)
k_matrix = classinstance.generate_k_matrix()
# print(classinstance.calculate_alpha(surface_tensor, delta_Rm, delta_Rp, k_matrix).shape)
# delta_Rs = np.arange(np.random.randint(5, 25)*2)
# delta_Rm = delta_Rs[0::2]
# delta_Rp = delta_Rs[1::2]
# print(len(delta_Rp), len(delta_Rp))