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
        R, delta_Rp, delta_Rm, Z, delta_Z = self.heatpipe.initialize_discretization()

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
        R, delta_Rp, delta_Rm, Z, delta_Z = self.heatpipe.initialize_discretization()

        surface_areas = self.heatpipe.calculate_surfaces(delta_Rp, delta_Rm, delta_Z) 

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
        R, delta_Rp, delta_Rm, Z, delta_Z = self.heatpipe.initialize_discretization()
        surface_areas = self.heatpipe.calculate_surfaces(delta_Rp, delta_Rm, delta_Z)
        k = self.heatpipe.generate_k_matrix()

        alpha = self.heatpipe.calculate_alpha(surface_areas, delta_Rm, delta_Rp, k)

        N_Z = self.heatpipe.N_Z

        # Testing bulk elements
        alpha110_expected = surface_areas[1][1][0] * k[1][2] / (k[1][1] * delta_Rm[2] + k[1][2] * delta_Rp[1])
        alpha111_expected = surface_areas[1][1][1] * k[1][0] / (k[1][1] * delta_Rp[0] + k[1][0] * delta_Rm[1])
        alpha112_expected = surface_areas[1][1][2] * k[2][1] / (k[1][1] * self.heatpipe.delta_Z + k[2][1] * self.heatpipe.delta_Z)
        alpha113_expected = surface_areas[1][1][3] * k[0][1] / (k[1][1] * self.heatpipe.delta_Z + k[0][1] * self.heatpipe.delta_Z)

        np.testing.assert_almost_equal(alpha[1][1][0], alpha110_expected)
        np.testing.assert_almost_equal(alpha[1][1][1], alpha111_expected)
        np.testing.assert_almost_equal(alpha[1][1][2], alpha112_expected)
        np.testing.assert_almost_equal(alpha[1][1][3], alpha113_expected)

        # Testing insulated surface element at z = 0
        alpha010_expected = surface_areas[0][1][0] * k[1][2] / (k[0][1] * delta_Rm[2] + k[1][2] * delta_Rp[1])
        alpha011_expected = surface_areas[0][1][1] * k[1][0] / (k[0][1] * delta_Rp[0] + k[1][0] * delta_Rm[1])
        alpha012_expected = surface_areas[0][1][2] * k[2][1] / (k[0][1] * self.heatpipe.delta_Z + k[2][1] * self.heatpipe.delta_Z)
        alpha013_expected = 0.

        np.testing.assert_almost_equal(alpha[0][1][0], alpha010_expected)
        np.testing.assert_almost_equal(alpha[0][1][1], alpha011_expected)
        np.testing.assert_almost_equal(alpha[0][1][2], alpha012_expected)
        np.testing.assert_almost_equal(alpha[0][1][3], alpha013_expected)

        # Testing insulated surface element at z = l_tot
        alphaNZ10_expected = surface_areas[N_Z - 1][1][0] * k[1][2] / (k[1][1] * delta_Rm[2] + k[1][2] * delta_Rp[1])
        alphaNZ11_expected = surface_areas[N_Z - 1][1][1] * k[1][0] / (k[1][1] * delta_Rp[0] + k[1][0] * delta_Rm[1])
        alphaNZ12_expected = 0.
        alphaNZ13_expected = surface_areas[N_Z - 1][1][3] * k[0][1] / (k[1][1] * self.heatpipe.delta_Z + k[0][1] * self.heatpipe.delta_Z)

        np.testing.assert_almost_equal(alpha[N_Z - 1][1][0], alphaNZ10_expected)
        np.testing.assert_almost_equal(alpha[N_Z - 1][1][1], alphaNZ11_expected)
        np.testing.assert_almost_equal(alpha[N_Z - 1][1][2], alphaNZ12_expected)
        np.testing.assert_almost_equal(alpha[N_Z - 1][1][3], alphaNZ13_expected)

        # Testing vapor surface element
        alpha100_expected = surface_areas[1][0][0] * k[1][1] / (k[1][0] * delta_Rm[1] + k[1][0] * delta_Rp[0])
        alpha101_expected = surface_areas[1][0][1] 
        alpha102_expected = surface_areas[1][0][2] * k[2][0] / (k[1][0] * self.heatpipe.delta_Z + k[2][0] * self.heatpipe.delta_Z)
        alpha103_expected = surface_areas[1][0][3] * k[0][0] / (k[1][0] * self.heatpipe.delta_Z + k[0][0] * self.heatpipe.delta_Z)

        np.testing.assert_almost_equal(alpha[1][1][0], alpha110_expected)
        np.testing.assert_almost_equal(alpha[1][1][1], alpha111_expected)
        np.testing.assert_almost_equal(alpha[1][1][2], alpha112_expected)
        np.testing.assert_almost_equal(alpha[1][1][3], alpha113_expected)

        # Testing condenstor surface element  
        alpha430_expected = surface_areas[4][3][0]
        alpha431_expected = surface_areas[4][3][1] * k[4][2] / (k[4][3] * delta_Rp[2] + k[4][2] * delta_Rm[3])
        alpha432_expected = surface_areas[4][3][2] * k[4][2] / (k[4][3] * self.heatpipe.delta_Z + k[5][3] * self.heatpipe.delta_Z)
        alpha433_expected = surface_areas[4][3][3] * k[4][2] / (k[4][3] * self.heatpipe.delta_Z + k[3][3] * self.heatpipe.delta_Z)

        np.testing.assert_almost_equal(alpha[4][3][0], alpha430_expected)
        np.testing.assert_almost_equal(alpha[4][3][1], alpha431_expected)
        np.testing.assert_almost_equal(alpha[4][3][2], alpha432_expected)
        np.testing.assert_almost_equal(alpha[4][3][3], alpha433_expected)

    def test_output_generate_matrix_form(self):
        R, delta_Rp, delta_Rm, Z, delta_Z = self.heatpipe.initialize_discretization()
        surface_areas = self.heatpipe.calculate_surfaces(delta_Rp, delta_Rm, delta_Z)
        k = self.heatpipe.generate_k_matrix()
        h = self.heatpipe.generate_h_matrix()

        alpha = self.heatpipe.calculate_alpha(surface_areas, delta_Rm, delta_Rp, k)

        M, C = self.heatpipe.generate_matrix_form(alpha, k, h)

        stride = self.heatpipe.N_R

        # Testing bulk element
        T_idx11 = (stride * 1) + 1
        M11_diag_expected = -k[1][1] * (alpha[1][1][0] + alpha[1][1][1] + alpha[1][1][2] + alpha[1][1][3])
        M11_rp_expected   =  k[1][1] * alpha[1][1][0]
        M11_rm_expected   =  k[1][1] * alpha[1][1][1]
        M11_zp_expected   =  k[1][1] * alpha[1][1][2]
        M11_zm_expected   =  k[1][1] * alpha[1][1][3]

        np.testing.assert_almost_equal(M[T_idx11][T_idx11]         , M11_diag_expected)
        np.testing.assert_almost_equal(M[T_idx11][T_idx11 + 1]     , M11_rp_expected)
        np.testing.assert_almost_equal(M[T_idx11][T_idx11 - 1]     , M11_rm_expected)
        np.testing.assert_almost_equal(M[T_idx11][T_idx11 + stride], M11_zp_expected)
        np.testing.assert_almost_equal(M[T_idx11][T_idx11 - stride], M11_zm_expected)

        C11_expected = 0.
        np.testing.assert_almost_equal(C[T_idx11], C11_expected)

        # Testing insulated surface element at z = 0
        T_idx01 = (stride * 0) + 1
        M01_diag_expected = -k[0][1] * (alpha[0][1][0] + alpha[0][1][1] + alpha[0][1][2])
        M01_rp_expected   =  k[0][1] *  alpha[0][1][0]
        M01_rm_expected   =  k[0][1] *  alpha[0][1][1]
        M01_zp_expected   =  k[0][1] *  alpha[0][1][2]
        M01_zm_expected   =  k[0][1] *  0.0

        np.testing.assert_almost_equal(M[T_idx01][T_idx01]         , M01_diag_expected)
        np.testing.assert_almost_equal(M[T_idx01][T_idx01 + 1]     , M01_rp_expected)
        np.testing.assert_almost_equal(M[T_idx01][T_idx01 - 1]     , M01_rm_expected)
        np.testing.assert_almost_equal(M[T_idx01][T_idx01 + stride], M01_zp_expected)

        C01_expected = 0.
        np.testing.assert_almost_equal(C[T_idx01], C01_expected)

        # Testing insulated surface element at z = l_tot
        T_idx51 = (stride * 5) + 1
        M51_diag_expected = -k[5][1] * (alpha[5][1][0] + alpha[5][1][1] + alpha[5][1][3])
        M51_rp_expected   =  k[5][1] *  alpha[5][1][0]
        M51_rm_expected   =  k[5][1] *  alpha[5][1][1]
        M51_zm_expected   =  k[5][1] *  alpha[5][1][3]

        np.testing.assert_almost_equal(M[T_idx51][T_idx51]         , M51_diag_expected)
        np.testing.assert_almost_equal(M[T_idx51][T_idx51 + 1]     , M51_rp_expected)
        np.testing.assert_almost_equal(M[T_idx51][T_idx51 - 1]     , M51_rm_expected)
        np.testing.assert_almost_equal(M[T_idx51][T_idx51 - stride], M51_zm_expected)

        C51_expected = 0.
        np.testing.assert_almost_equal(C[T_idx51], C51_expected)

        # Testing vapor surface element
        T_idx10 = (stride * 1) + 0
        M10_diag_expected = -k[1][0] * (alpha[1][0][1] + alpha[1][0][2] + alpha[1][0][3])
        M10_diag_expected -= h[1][0] * alpha[1][0][0]
        M10_rp_expected   =  k[1][0] *  alpha[1][0][0]
        M10_rm_expected   =  h[1][0] *  alpha[1][0][1]
        M10_zp_expected   =  k[1][0] *  alpha[1][0][2]
        M10_zm_expected   =  k[1][0] *  alpha[1][0][3]

        np.testing.assert_almost_equal(M[T_idx10][T_idx10]         , M10_diag_expected)
        np.testing.assert_almost_equal(M[T_idx10][T_idx10 + 1]     , M10_rp_expected)
        np.testing.assert_almost_equal(M[T_idx10][-1         ]     , M10_rm_expected)
        np.testing.assert_almost_equal(M[T_idx10][T_idx10 + stride], M10_zp_expected)
        np.testing.assert_almost_equal(M[T_idx10][T_idx10 - stride], M10_zm_expected)

        C10_expected = 0.
        np.testing.assert_almost_equal(C[T_idx10], C10_expected)

        # Testing condenstor surface corner element  
        T_idx53 = (stride * 5) + 3
        M53_diag_expected = -k[5][3] * (alpha[5][3][1] + alpha[5][3][3])
        M53_diag_expected -= h[5][3] *  alpha[5][3][0]
        M53_rm_expected   =  k[5][3] *  alpha[5][3][1]
        M53_zm_expected   =  k[5][3] *  alpha[5][3][3]

        np.testing.assert_almost_equal(M[T_idx53][T_idx53]         , M53_diag_expected)
        np.testing.assert_almost_equal(M[T_idx53][T_idx53 - 1]     , M53_rm_expected)
        np.testing.assert_almost_equal(M[T_idx53][T_idx53 - stride], M53_zm_expected)

        C53_expected = h[5][3] * alpha[5][3][0] * self.heatpipe.T_cond
        np.testing.assert_almost_equal(C[T_idx53], C53_expected)

        # Testing insultade wick edge r = 0
        T_idx30 = (stride * 3) + 0
        M30_diag_expected = -k[3][0] * (alpha[1][1][2] + alpha[1][1][3])
        M30_zp_expected   =  k[3][0] * alpha[1][1][2]
        M30_zm_expected   =  k[3][0] * alpha[1][1][3]

        np.testing.assert_almost_equal(M[T_idx30][T_idx30]         , M30_diag_expected)
        np.testing.assert_almost_equal(M[T_idx30][T_idx30 + stride], M30_zp_expected)
        np.testing.assert_almost_equal(M[T_idx30][T_idx30 - stride], M30_zm_expected)

        C30_expected = 0.
        np.testing.assert_almost_equal(C[T_idx30], C30_expected)

        # Testing insultade wall edge r = N_R - 1
        T_idx33 = (stride * 3) + 3 - 1
        M33_diag_expected = -k[3][3] * (alpha[3][3][2] + alpha[3][3][3])
        M33_zp_expected   =  k[3][3] *  alpha[3][3][2]
        M33_zm_expected   =  k[3][3] *  alpha[3][3][3]

        np.testing.assert_almost_equal(M[T_idx33][T_idx33]         , M33_diag_expected)
        np.testing.assert_almost_equal(M[T_idx33][T_idx33 + stride], M33_zp_expected)
        np.testing.assert_almost_equal(M[T_idx33][T_idx33 - stride], M33_zm_expected)
        self.assertFalse(M[T_idx33][T_idx33 - stride] == 0.)

        C33_expected = 0.
        np.testing.assert_almost_equal(C[T_idx33], C33_expected)
