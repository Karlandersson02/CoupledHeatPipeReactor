import numpy as np
import unittest

from models.vapor_discretised_model import vapor_discretised

class Test_simple_case_vapor_discretised_model(unittest.TestCase):

    def setUp(self):
        data = {
            "r_outer": 1. + 0.001 + 0.001,
            "delta_wick": 0.001,
            "delta_wall": 0.001,
            "l_evap": 0.2,
            "l_adiabatic": 0.4,
            "l_cond": 0.2,
            "N_wick": 1,
            "N_wall": 1,
            "N_evap": 4,
            "N_adiabatic": 8,
            "N_cond": 4,
            "h_vap": 1.,
            "h_cond": 39.,
            "T_cond": 300.,
            "k_wall": 21.7,
            "k_wick": 45,
            "viscosity_Na": 1.8e-5,
            "T_C": 856,
            "P_C": 2476,
            "T_HP": [850., 851., 850., 851., 850., 851., 850., 851., 849., 849., 849., 849., 849., 849., 849., 849., 849., 849., 849., 849., 849., 849., 849., 849., 848., 847., 848., 847., 848., 847., 848., 847., 849.]
        }
        self.vapour = vapor_discretised(data)

    def test_calculate_mass_flow_and_latent_heat(self):
        Gamma, h_fg_Na = self.vapour.calculate_mass_flow_and_latent_heat()
        
        np.testing.assert_array_almost_equal(len(Gamma.tolist()), 16)

        q_surface = 1. * 1.
        a_W = (1. * 2 * 0.2 / 4) / (1.**2 * np.pi * 0.2 / 4) 

        Gamma_expected = np.zeros(16)
        Gamma_expected[:4] = np.ones(4) * q_surface * a_W / h_fg_Na 
        Gamma_expected[4:-4] = 0 
        Gamma_expected[-4:] = -np.ones(4) * q_surface * a_W / h_fg_Na 

        np.testing.assert_array_almost_equal(Gamma, Gamma_expected)

        # Numbers taken for the value of the vaporisation enthalpy is from the same article as the formula. 
        self.assertGreater(h_fg_Na, 4112e3)
        self.assertLess(h_fg_Na, 4197e3)

    def test_calculate_pressure_profile(self):
        T_v = 856.
        Gamma, h_fg_Na = self.vapour.calculate_mass_flow_and_latent_heat()
        P_v = self.vapour.calculate_pressure_profile(T_v=np.array([T_v]), h_fg_Na=h_fg_Na)

        P_v_expected = 2476
        np.testing.assert_approx_equal(P_v[0], P_v_expected)

        T_v = 850.
        Gamma, h_fg_Na = self.vapour.calculate_mass_flow_and_latent_heat()
        P_v = self.vapour.calculate_pressure_profile(T_v=np.array([T_v]), h_fg_Na=h_fg_Na)

        P_v_expected = 2260
        np.testing.assert_approx_equal(P_v[0], P_v_expected, significant=3)

        

