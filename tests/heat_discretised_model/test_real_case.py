import numpy as np
import unittest

from models.heat_discretised_model import heatpipe_discretised

class Test_real_case_discretised_model(unittest.TestCase):

    def setUp(self):
        data_discretised = {
            "r_outer": 10.,
            "delta_wick": 1.,
            "delta_wall": 1.,
            "l_evap": 2.,
            "l_adiabatic": 2.,
            "l_cond": 2.,
            "N_wick": 2,
            "N_wall": 2,
            "N_evap": 2,
            "N_adiabatic": 2,
            "N_cond": 2,
            "h_vap": 1.,
            "h_cond": 2.,
            "T_cond": 100.,
            "k_wall": 3.,
            "k_wick": 4.,
            "Q": np.array([1., 1.])
        }

        self.heatpipe = heatpipe_discretised(data_discretised)

