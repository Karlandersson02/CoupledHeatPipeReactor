import numpy as np
import unittest

from models.heat_discretised_model import heatpipe_discretised

class Test_dimensions_discretised_model(unittest.TestCase):

    def setUp(self):
        data_discretised = {
            "r_outer": 1.,
            "delta_wick": 1.,
            "delta_wall": 1.,
            "l_evap": 1.,
            "l_adiabatic": 1.,
            "l_cond": 1.,
            "N_wick": 1.,
            "N_wall": 1.,
            "N_evap": 1.,
            "N_adiabatic": 1.,
            "N_cond": 1.,
            "h_vap": 1.,
            "h_cond": 1.,
            "T_cond": 1.,
            "k_wall": 1.,
            "k_wick": 1.,
            "Q": np.array([1.])
        }

        self.heatpipe = heatpipe_discretised(data_discretised)

    def test_dimensions_calculate_surfaces(self):
        ...
        # delta_Rs = np.arange(np.random.randint(5, 25)) * 2
        # delta_Rm = delta_Rs[0.::2]
        # delta_Rp = delta_Rs[1::2]

        # Testing shape
        #self.assertEqual(self.heatpipe.calculate_surfaces(delta_Rm, delta_Rp).shape, (len(delta_Rm), len(delta_Rp), 4))
