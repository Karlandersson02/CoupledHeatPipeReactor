import numpy as np
import unittest

from models.heat_discretised_model import heatpipe_discretised

class Test_real_case_discretised_model(unittest.TestCase):

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

