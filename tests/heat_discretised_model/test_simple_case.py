import numpy as np
import unittest

from models.heat_discretised_model import heatpipe_discretised

class Test_simple_case_discretised_model(unittest.TestCase):

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

    def test_output_initialize_discretiszation(self):
        R, delta_Rp, delta_Rm, Z = self.heatpipe.initialize_discretization()

        R_expected = np.array([8.276473,  
                               8.544004,  
                               8.803408,  
                               9.055385,  
                               9.300538,  
                               9.539392, 
                               9.77241 , 
                               10.  ])
        
        np.testing.assert_array_almost_equal(R, R_expected)

        delta_Rm_expected = np.array([
            8.276473 - self.heatpipe.r_vapour, 
            8.803408 - 8.544004, 
            9.300538 - 9.055385, 
            9.772410 - 9.539392])
        
        np.testing.assert_array_almost_equal(delta_Rm, delta_Rm_expected)

        delta_Rp_expected = np.array([
            8.544004 - 8.276473, 
            9.055385 - 8.803408, 
            9.539392 - 9.300538, 
            10.      - 9.772410])
        
        np.testing.assert_array_almost_equal(delta_Rm, delta_Rm_expected)

        Z_expected = np.array([
            0 + 1./2., 
            1 + 1./2., 
            2 + 1./2., 
            3 + 1./2., 
            4.+ 1./2., 
            5 + 1./2.])
        
        np.testing.assert_array_almost_equal(Z, Z_expected)

    def test_output_calculate_surfaces(self):
        R, delta_Rp, delta_Rm, Z = self.heatpipe.initialize_discretization()

        surface_areas = self.heatpipe.calculate_surfaces(delta_Rp, delta_Rm) 

        radial_strip_surface_areas_1_expected = [50.2654825, 53.6835588, 56.8966629, 59.9377677]
        radial_strip_surface_areas_0_expected = [53.6835588, 56.8966629, 59.9377677, 62.8318531]
        radial_strip_surface_areas_2_expected = [28.2743339, 28.2743339, 28.2743339, 28.2743339]

        np.testing.assert_almost_equal(surface_areas[0, :, 0], radial_strip_surface_areas_0_expected)
        np.testing.assert_almost_equal(surface_areas[0, :, 1], radial_strip_surface_areas_1_expected)
        np.testing.assert_almost_equal(surface_areas[0, :, 2], radial_strip_surface_areas_2_expected)

        np.testing.assert_almost_equal(surface_areas[2,1,0]**2 - surface_areas[2,0,0]**2, surface_areas[2,3,0]**2 - surface_areas[2,2,0]**2)
        np.testing.assert_almost_equal(surface_areas[2,1,1]**2 - surface_areas[2,0,1]**2, surface_areas[2,3,1]**2 - surface_areas[2,2,1]**2)
        np.testing.assert_almost_equal(surface_areas[2,1,2]**2 - surface_areas[2,0,2]**2, surface_areas[2,3,2]**2 - surface_areas[2,2,2]**2)

    def test_output_generate_k_matrix(self):
        k = self.heatpipe.generate_k_matrix()

        k_expected = np.array([[4., 4., 3., 3.],
                               [4., 4., 3., 3.],
                               [4., 4., 3., 3.],
                               [4., 4., 3., 3.],
                               [4., 4., 3., 3.],
                               [4., 4., 3., 3.]])
        
        np.testing.assert_array_almost_equal(k, k_expected)

    def test_output_generate_h_matrix(self):
        h = self.heatpipe.generate_h_matrix()

        h_expected = np.array([[1., 0., 0., 0.],
                               [1., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [1., 0., 0., 2.],
                               [1., 0., 0., 2.]])
        
        np.testing.assert_array_almost_equal(h, h_expected)

    def test_output_calculate_alpha(self):
        R, delta_Rp, delta_Rm, Z = self.heatpipe.initialize_discretization()
        surface_areas = self.heatpipe.calculate_surfaces(delta_Rp, delta_Rm)
        k = self.heatpipe.generate_k_matrix()

        alpha = self.heatpipe.calculate_alpha(surface_areas, delta_Rm, delta_Rp, k)

        # Testing bulk elements
        alpha110_expected = surface_areas[1][1][0] * k[1][2] / (k[1][1] * delta_Rm[2] + k[1][2] * delta_Rp[1])
        alpha111_expected = surface_areas[1][1][1] * k[1][0] / (k[1][1] * delta_Rp[0] + k[1][0] * delta_Rm[1])
        alpha112_expected = surface_areas[1][1][2] * k[2][1] / (k[1][1] * self.heatpipe.delta_Z + k[2][1] * self.heatpipe.delta_Z)
        alpha113_expected = surface_areas[1][1][3] * k[0][1] / (k[1][1] * self.heatpipe.delta_Z + k[0][1] * self.heatpipe.delta_Z)

        print(alpha)

        np.testing.assert_almost_equal(alpha[1][1][0], alpha110_expected)
        np.testing.assert_almost_equal(alpha[1][1][1], alpha111_expected)
        np.testing.assert_almost_equal(alpha[1][1][2], alpha112_expected)
        np.testing.assert_almost_equal(alpha[1][1][3], alpha113_expected)

        # Testing insulated surface element

        # Testing vapor surface element

        # Testing condenstor surface element  
