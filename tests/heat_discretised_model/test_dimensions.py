import numpy as np
import unittest

from models.heat_discretised_model import heatpipe_discretised

class Test_randomised_discretised_model(unittest.TestCase):

    def setUp(self):
        self.data_discretised = {
            "r_outer": np.random.rand()*4 + 4,
            "delta_wick": np.random.rand() + 0.2,
            "delta_wall": np.random.rand() + 0.2,
            "l_evap": np.random.rand() + 0.2,
            "l_adiabatic": np.random.rand() + 0.2,
            "l_cond": np.random.rand() + 0.2,
            "N_wick": np.random.randint(20) + 4,
            "N_wall": np.random.randint(20) + 4,
            "N_evap": np.random.randint(20) + 4,
            "N_adiabatic": np.random.randint(20) + 4,
            "N_cond": np.random.randint(20) + 4,
            "h_vap": np.random.rand()*100 + 10,
            "h_cond": np.random.rand()*100 + 10,
            "T_cond": np.random.rand()*200 + 200,
            "k_wall": np.random.rand()*50 + 10,
            "k_wick": np.random.rand()*50 + 10,
            "Q": np.array([0])
        }

        self.N_R = self.data_discretised.get("N_wick") + self.data_discretised.get("N_wall")
        self.N_Z = self.data_discretised.get("N_evap") + self.data_discretised.get("N_adiabatic") + self.data_discretised.get("N_cond")
        self.delta_Z = (self.data_discretised.get("l_evap") + self.data_discretised.get("l_adiabatic") + self.data_discretised.get("l_cond")) / self.N_Z
        self.r_vapour = self.data_discretised.get("r_outer") - self.data_discretised.get("delta_wall") - self.data_discretised.get("delta_wick")

        self.heatpipe = heatpipe_discretised(self.data_discretised)

    def test_calculate_surfaces(self):
        delta_Rs = np.random.rand(self.N_R*2)*4
        delta_Rm = delta_Rs[0::2]
        delta_Rp = delta_Rs[1::2]
        surface_tensor = self.heatpipe.calculate_surfaces(delta_Rm, delta_Rp)

        # Testing shape
        self.assertEqual(surface_tensor.shape, (self.N_Z, len(delta_Rm), 4), "shape mismatch")

        # Testing random value
        i, j = np.random.randint(self.N_Z), np.random.randint(self.N_R)
        R = self.r_vapour
        for jx in range(j):
            R += delta_Rm[jx] + delta_Rp[jx]
        
        S_rp = (R + delta_Rp[j]) * 2*np.pi * self.delta_Z
        S_rm = (R - delta_Rm[j]) * 2*np.pi * self.delta_Z
        S_Z = np.pi*((R + delta_Rp[j])**2 - (R - delta_Rm[j])**2)

        self.assertEqual(surface_tensor[i, j, 0], S_rp, "S_rp failed")
        self.assertEqual(surface_tensor[i, j, 1], S_rm, "S_rm failed")
        self.assertEqual(surface_tensor[i, j, 2], S_Z, "S_Z failed")
        self.assertEqual(surface_tensor[i, j, 3], S_Z, "S_Z failed")

    def test_calculate_alpha(self):
        surface_tensor = np.random.rand(self.N_Z, self.N_R, 4) * 4
        delta_Rs = np.random.rand(self.N_R*2)*4
        delta_Rm = delta_Rs[0::2]
        delta_Rp = delta_Rs[1::2]
        k_matrix = np.random.rand(self.N_Z, self.N_R)*4

        alpha_tensor = self.heatpipe.calculate_alpha(surface_tensor, delta_Rm, delta_Rp, k_matrix)
        
        # Testing shape
        self.assertEqual(alpha_tensor.shape, (self.N_Z, len(delta_Rm), 4), "shape mismatch")

        # Testing random value
        i, j = np.random.randint(1, self.N_Z-1), np.random.randint(1, self.N_R-1)
        alpha_Rp = (surface_tensor[i, j, 0] * k_matrix[i+1, j]) / (k_matrix[i, j]*delta_Rm[j] - k_matrix[i+1, j]*delta_Rp[j])
        alpha_Rm = (surface_tensor[i, j, 1] * k_matrix[i-1, j]) / (k_matrix[i, j]*delta_Rp[j] - k_matrix[i-1, j]*delta_Rm[j])
        alpha_Zp = (surface_tensor[i, j, 2] * k_matrix[i, j+1]) / (k_matrix[i, j]*delta_Rm[j+1] - k_matrix[i, j+1]*delta_Rp[j])
        alpha_Zm = (surface_tensor[i, j, 3] * k_matrix[i, j-1]) / (k_matrix[i, j]*delta_Rp[j-1] - k_matrix[i, j-1]*delta_Rm[j])

        self.assertEqual(alpha_tensor[i, j, 0], alpha_Rp, "alpha_Rp failed")
        self.assertEqual(alpha_tensor[i, j, 1], alpha_Rm, "alpha_Rm failed")
        self.assertEqual(alpha_tensor[i, j, 2], alpha_Zp, "alpha_Zp failed")
        self.assertEqual(alpha_tensor[i, j, 3], alpha_Zm, "alpha_Zm failed")

        self.assertTrue(np.all(alpha_tensor[0, :, 2] == 0), "Zero boundary incorrect")
        self.assertTrue(np.all(alpha_tensor[-1, :, 3] == 0), "Zero boundary incorrect")

        self.assertTrue(np.all(alpha_tensor[:(self.data_discretised.get("N_evap")), 0, 1] == surface_tensor[:(self.data_discretised.get("N_evap")), 0, 1]), "Conduction boundary incorrect")
        self.assertTrue(np.all(alpha_tensor[(self.data_discretised.get("N_evap") + self.data_discretised.get("N_adiabatic")):, 0, 1] == surface_tensor[(self.data_discretised.get("N_evap") + self.data_discretised.get("N_adiabatic")):, 0, 1]), "Conduction boundary incorrect")
        self.assertTrue(np.all(alpha_tensor[(self.data_discretised.get("N_evap") + self.data_discretised.get("N_adiabatic")):, -1, 0] == surface_tensor[(self.data_discretised.get("N_evap") + self.data_discretised.get("N_adiabatic")):, -1, 0]), "Conduction boundary incorrect")

        self.assertTrue(np.all(alpha_tensor[self.data_discretised.get("N_evap"):self.data_discretised.get("N_adiabatic"), :, 0:2] == 0), "Radial heat transfer omission in adiabatic section incorrect")
