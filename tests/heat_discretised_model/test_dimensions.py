import numpy as np
import unittest

from models.heat_discretised_model import heatpipe_discretised

class Test_dimensions_discretised_model(unittest.TestCase):

    def setUp(self):
        data_discretised = {
            "r_outer": 0,
            "delta_wick": 0,
            "delta_wall": 0,
            "l_evap": 0,
            "l_adiabatic": 0,
            "l_cond": 0,
            "N_wick": 0,
            "N_wall": 0,
            "N_evap": 0,
            "N_adiabatic": 0,
            "N_cond": 0,
            "h_vap": 0,
            "h_cond": 0,
            "T_cond": 0,
            "k_wall": 0,
            "k_wick": 0,
            "Q": np.array([0])
        }

        self.heatpipe = heatpipe_discretised(data_discretised)

    def test_dimensions_calculate_surfaces(self):
        delta_Rs = np.arange(np.random.randint(5, 25)) * 2
        delta_Rm = delta_Rs[0::2]
        delta_Rp = delta_Rs[1::2]

        # Testing shape
        self.assertEqual(self.heatpipe.calculate_surfaces(delta_Rm, delta_Rp).shape, (len(delta_Rm), len(delta_Rp), 4))

        # Testing delta_Rm
        self.assertEqual 
        # 
