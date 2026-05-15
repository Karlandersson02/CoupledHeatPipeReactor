import numpy as np

import data.dataclass as d_class
import utils.material_properties as m_props

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from models.component import Component


class HeatpipeDiscretised(Component):
    def __init__(self, config: d_class.HeatpipeConfigResolved):

        self.cfg = config

        self._initialize_discretization()

        self.variable_k = False
        self.k_temperature = 800     # used if variable_k is False

    def initial_guess(self):
        X_initial = np.full(self.cfg.mesh.N_Z * self.cfg.mesh.N_R + 1, 800)
        if not self.cfg.bc.Temperature_BC:
            X_initial = X_initial[:-1]

        return X_initial

    def assemble(self):
        surface_areas, delta_Rp, delta_Rm, delta_Z = self._initialize_discretization()

        self.k_matrix = self._generate_k_matrix()
        self.h_matrix = self._generate_h_matrix()

        self.alpha = self._generate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z, self.k_matrix)

    def get_residuals(self, X):
        T, T_v = self.unpack(X)
        T = T.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)

        surface_areas, delta_Rp, delta_Rm, delta_Z = self._initialize_discretization()
        if self.variable_k:
            k = self._generate_k_matrix(T)
        else:
            k = self._generate_k_matrix()
        h = self._generate_h_matrix()
        alpha = self._generate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z, k)

        N_Z = self.cfg.mesh.N_Z
        N_R = self.cfg.mesh.N_R
        N_evap = self.cfg.mesh.N_evap
        N_cond = self.cfg.mesh.N_cond

        res = np.zeros((N_Z, N_R), dtype=float)

        # Bulk
        if N_Z > 2 and N_R > 2:
            kc = k[1:-1, 1:-1]
            ac = alpha[1:-1, 1:-1]

            res[1:-1, 1:-1] = (
                -kc * (ac[..., 0] + ac[..., 1] + ac[..., 2] + ac[..., 3]) * T[1:-1, 1:-1]
                + kc * ac[..., 0] * T[1:-1, 2:]
                + kc * ac[..., 1] * T[1:-1, :-2]
                + kc * ac[..., 2] * T[2:, 1:-1]
                + kc * ac[..., 3] * T[:-2, 1:-1]
            )

        # Top insulated boundary, z = 0, excluding corners
        if N_R > 2:
            z = 0
            kc = k[z, 1:-1]
            ac = alpha[z, 1:-1]

            res[z, 1:-1] = (
                -kc * (ac[:, 0] + ac[:, 1] + ac[:, 2]) * T[z, 1:-1]
                + kc * ac[:, 0] * T[z, 2:]
                + kc * ac[:, 1] * T[z, :-2]
                + kc * ac[:, 2] * T[z + 1, 1:-1]
            )

        # Bottom insulated boundary, z = N_Z - 1, excluding corners
        if N_R > 2:
            z = N_Z - 1
            kc = k[z, 1:-1]
            ac = alpha[z, 1:-1]

            res[z, 1:-1] = (
                -kc * (ac[:, 0] + ac[:, 1] + ac[:, 3]) * T[z, 1:-1]
                + kc * ac[:, 0] * T[z, 2:]
                + kc * ac[:, 1] * T[z, :-2]
                + kc * ac[:, 3] * T[z - 1, 1:-1]
            )

        # Outer wall boundary, r = N_R - 1, excluding corners
        if N_Z > 2:
            r = N_R - 1
            kc = k[1:-1, r]
            ac = alpha[1:-1, r]
            hc = h[1:-1, r]

            res[1:-1, r] = (
                (-kc * (ac[:, 1] + ac[:, 2] + ac[:, 3]) - hc * ac[:, 0]) * T[1:-1, r]
                + kc * ac[:, 1] * T[1:-1, r - 1]
                + kc * ac[:, 2] * T[2:, r]
                + kc * ac[:, 3] * T[:-2, r]
                + hc * ac[:, 0] * self.cfg.bc.T_cond
            )

            if N_evap > 1:
                res[1:N_evap, r] += self.cfg.bc.Q[1:N_evap]

        # Inner wick-vapor boundary, r = 0, excluding corners
        if N_Z > 2:
            r = 0
            kc = k[1:-1, r]
            ac = alpha[1:-1, r]
            hc = h[1:-1, r]

            res[1:-1, r] = (
                (-kc * (ac[:, 0] + ac[:, 2] + ac[:, 3]) - hc * ac[:, 1]) * T[1:-1, r]
                + kc * ac[:, 0] * T[1:-1, r + 1]
                + hc * ac[:, 1] * T_v
                + kc * ac[:, 2] * T[2:, r]
                + kc * ac[:, 3] * T[:-2, r]
            )

        # Corner: z = 0, r = N_R - 1
        z = 0
        r = N_R - 1
        res[z, r] = (
            -k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2]) * T[z, r]
            + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
            + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
            + self.cfg.bc.Q[z]
        )

        # Corner: z = N_Z - 1, r = N_R - 1
        z = N_Z - 1
        r = N_R - 1
        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 1] + alpha[z, r, 3]) - h[z, r] * alpha[z, r, 0]) * T[z, r]
            + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
            + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
            + h[z, r] * alpha[z, r, 0] * self.cfg.bc.T_cond
        )

        # Corner: z = 0, r = 0
        z = 0
        r = 0
        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2]) - h[z, r] * alpha[z, r, 1]) * T[z, r]
            + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
            + h[z, r] * alpha[z, r, 1] * T_v
            + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
        )

        # Corner: z = N_Z - 1, r = 0
        z = N_Z - 1
        r = 0
        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 0] + alpha[z, r, 3]) - h[z, r] * alpha[z, r, 1]) * T[z, r]
            + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
            + h[z, r] * alpha[z, r, 1] * T_v
            + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
        )

        # Vapour node
        beta = h[:, 0] * alpha[:, 0, 1]
        res_v = np.sum(beta * (T[:, 0] - T_v))

        return np.r_[res.reshape(-1), res_v]

    def linear_solve(self):
        self.assemble()
        
        if self.cfg.bc.Temperature_BC:
            self.M, self.C = self._generate_matrix_form_temperature_bc(self.alpha, self.k_matrix, self.h_matrix)
        else:
            self.M, self.C = self._generate_matrix_form_heat_bc(self.alpha, self.k_matrix, self.h_matrix)

        T = np.array(spsolve(self.M, self.C))

        if not self.cfg.bc.Temperature_BC:
            T = np.concatenate([T, np.array([self.cfg.bc.T_op])])

        return T

    def post_process(self, X):
        X_tuple = (X[:-1].reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R), X[-1])
        return X_tuple
    
    def pre_process(self, X_tuple):
        X = np.r_[X_tuple[0].reshape(self.cfg.mesh.N_Z * self.cfg.mesh.N_R), X_tuple[1]]
        return X
    
    def unpack(self, X):
        return (X[:-1], X[-1])
    
    def pack(self, X_tuple):
        X = np.r_[X_tuple[0], X_tuple[1]]
        return X

    def _initialize_discretization(self):     
        R         = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        delta_R_m = np.zeros(self.cfg.mesh.N_R, dtype=float)
        delta_R_p = np.zeros(self.cfg.mesh.N_R, dtype=float)
        Z         = np.zeros(2 * self.cfg.mesh.N_Z, dtype=float)
        delta_Z   = np.zeros(self.cfg.mesh.N_Z, dtype=float)

        self.R = R[0::2]
        self.delta_R = delta_R
        
        # Calculating the radii of the half-elements
        R[0] = np.sqrt((self.cfg.geometry.r_outer**2 - self.cfg.geometry.r_vapour**2) / (self.cfg.mesh.N_R * 2) + self.cfg.geometry.r_vapour**2)

        for i in range(1, 2 * self.cfg.mesh.N_R):
            R[i] = np.sqrt(R[i-1]**2 + R[0]**2 - self.cfg.geometry.r_vapour**2)

        # Calculating the differences in the radius of the half-elements
        delta_R[0] = R[0] - self.cfg.geometry.r_vapour
        for i in range(1, 2 * self.cfg.mesh.N_R):
            delta_R[i] = R[i] - R[i - 1]

        delta_R_m = delta_R[0::2]
        delta_R_p = delta_R[1::2]

        # Calculating the Z-position of the bulk and edges of the elements 
        for i in range(self.cfg.mesh.N_evap*2):
            Z[i] = (i/2 + 1/2) * self.cfg.geometry.l_evap/self.cfg.mesh.N_evap

        for i in range(self.cfg.mesh.N_adiabatic*2):
            Z[i + self.cfg.mesh.N_evap*2] = (i/2 + 1/2) * self.cfg.geometry.l_adiabatic/self.cfg.mesh.N_adiabatic + self.cfg.geometry.l_evap

        for i in range(self.cfg.mesh.N_cond*2):
            Z[i+ self.cfg.mesh.N_evap*2 + self.cfg.mesh.N_adiabatic*2] = (i/2 + 1/2) * self.cfg.geometry.l_cond/self.cfg.mesh.N_cond + (self.cfg.geometry.l_evap + self.cfg.geometry.l_adiabatic)

        for i in range(self.cfg.mesh.N_Z):
            delta_Z[i] = Z[2*i + 1] - Z[2*i]

        self.Z = Z[0::2]

        # Generate surface tensors
        surface_areas = self._generate_surfaces(delta_R_p, delta_R_m, delta_Z)

        return surface_areas, delta_R_p, delta_R_m, delta_Z

    def _generate_surfaces(self, delta_Rp, delta_Rm, delta_Z):
        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R) + self.cfg.geometry.r_vapour
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi
        S_rm = Rm * 2*np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.cfg.mesh.N_Z, axis=0)
        surface_tensor[..., 0:2] *= 2 * delta_Z[:, None, None] # This *2 should be there due to Deltz_Z is half the distance between cell centers!!!

        return surface_tensor

    def _generate_k_matrix(self, T=None):
        if T is None:
            T = np.full(((self.cfg.mesh.N_Z, self.cfg.mesh.N_R)), self.k_temperature)

        def eval_material_prop(prop, T_slice, scale=1.0):
            value = prop(T_slice) if callable(prop) else prop
            return value * scale    # type: ignore
        wick_k = lambda T: m_props.HP_wick_k(T, self.cfg.wick.porosity)
        
        wick_slice = slice(0, self.cfg.mesh.N_wick)
        gap_slice  = slice(self.cfg.mesh.N_wick, self.cfg.mesh.N_wick + self.cfg.mesh.N_gap)
        wall_slice = slice(self.cfg.mesh.N_R - self.cfg.mesh.N_wall, self.cfg.mesh.N_R)
        
        T_wick = T[:, wick_slice]
        T_gap  = T[:, gap_slice]
        T_wall = T[:, wall_slice]

        k_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R), dtype=float)

        # k_matrix[:, wick_slice] = eval_material_prop(self.cfg.material.k_wick, T_wick)
        # k_matrix[:, gap_slice]  = eval_material_prop(self.cfg.material.k_gap , T_gap)
        # k_matrix[:, wall_slice] = eval_material_prop(self.cfg.material.k_wall, T_wall)

        k_matrix[:, wick_slice] = eval_material_prop(wick_k, T_wick)
        k_matrix[:, gap_slice]  = eval_material_prop(m_props.HP_gap_k, T_gap)
        k_matrix[:, wall_slice] = eval_material_prop(m_props.HP_wall_k, T_wall)

        return k_matrix
    
    def _generate_h_matrix(self) -> np.ndarray:

        h_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R))
        
        # Heat transfer coefficient for the vapor section.
        # h_matrix[:self.cfg.mesh.N_evap, 0]                      = self.cfg.material.h_vap
        # h_matrix[(self.cfg.mesh.N_Z - self.cfg.mesh.N_cond):self.cfg.mesh.N_Z, 0] = self.cfg.material.h_vap
        h_matrix[:, 0] = self.cfg.material.h_vap

        # Heat transfer coefficient for the condensator section.
        h_matrix[(self.cfg.mesh.N_Z - self.cfg.mesh.N_cond):self.cfg.mesh.N_Z, self.cfg.mesh.N_R - 1] = self.cfg.material.h_cond

        return h_matrix

    def _generate_alpha(self, surface_tensor, delta_Rm, delta_Rp, delta_Z, k_matrix):
        alpha = np.zeros_like(surface_tensor)

        alpha[:, :-1, 0] = (
            surface_tensor[:, :-1, 0] * k_matrix[:, 1:]
            / (k_matrix[:, :-1] * delta_Rm[1:] + k_matrix[:, 1:] * delta_Rp[:-1])
        )

        alpha[:, 1:, 1] = (
            surface_tensor[:, 1:, 1] * k_matrix[:, :-1]
            / (k_matrix[:, 1:] * delta_Rp[:-1] + k_matrix[:, :-1] * delta_Rm[1:])
        )

        alpha[:-1, :, 2] = (
            surface_tensor[:-1, :, 2] * k_matrix[1:, :]
            / (k_matrix[:-1, :] * delta_Z[1:, None] + k_matrix[1:, :] * delta_Z[:-1, None])
        )

        alpha[1:, :, 3] = (
            surface_tensor[1:, :, 3] * k_matrix[:-1, :]
            / (k_matrix[1:, :] * delta_Z[:-1, None] + k_matrix[:-1, :] * delta_Z[1:, None])
        )

        # Init masks
        vapour_mask = np.zeros_like(alpha, dtype=bool)
        cooling_mask = np.zeros_like(alpha, dtype=bool)
        
        # Configure masks
        vapour_mask[0:self.cfg.mesh.N_evap, 0, 1] = True
        vapour_mask[(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic):, 0, 1] = True
        cooling_mask[(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic):, -1, 0] = True

        # Apply masks
        alpha[vapour_mask] = surface_tensor[vapour_mask]
        alpha[cooling_mask] = surface_tensor[cooling_mask]

        # # Adiabatic middle boundaries
        # adiabatic_mask = np.zeros_like(alpha, dtype=bool)
        # adiabatic_mask[self.cfg.mesh.N_evap:(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic), :, 0:2] = True
        # alpha[adiabatic_mask] = 0
        return alpha

    def _generate_matrix_form_temperature_bc(self, alpha, k, h):
        # Number of physical grid nodes + extra vapor node (stored at index -1).
        N_phys = self.cfg.mesh.N_R * self.cfg.mesh.N_Z
        N = N_phys + 1
        stride = self.cfg.mesh.N_R 

        M = lil_matrix((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # -----------------------
        # Bulk elements.
        for z in range(1, self.cfg.mesh.N_Z - 1):
            for r in range(1, self.cfg.mesh.N_R - 1):
                T_idx = (stride * z) + r

                M[T_idx, T_idx] = -k[z, r] * (
                    alpha[z, r, 0] +
                    alpha[z, r, 1] +
                    alpha[z, r, 2] +
                    alpha[z, r, 3]
                )

                M[T_idx, T_idx + 1]       = k[z, r] * alpha[z, r, 0]
                M[T_idx, T_idx - 1]       = k[z, r] * alpha[z, r, 1]
                M[T_idx, T_idx + stride]  = k[z, r] * alpha[z, r, 2]  
                M[T_idx, T_idx - stride]  = k[z, r] * alpha[z, r, 3] 

        # -----------------------
        # Insulated wall at z = 0
        z = 0
        for r in range(1, self.cfg.mesh.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx, T_idx] = -k[z, r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][2]
            )

            M[T_idx, T_idx + 1]       = k[z, r] * alpha[z, r, 0]
            M[T_idx, T_idx - 1]       = k[z, r] * alpha[z, r, 1]
            M[T_idx, T_idx + stride]  = k[z, r] * alpha[z, r, 2]  

        # -----------------------
        # Insulated wall at z = N_Z - 1
        z = self.cfg.mesh.N_Z - 1
        for r in range(1, self.cfg.mesh.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx, T_idx] = -k[z, r] * (
                alpha[z, r, 0] +
                alpha[z, r, 1] +
                alpha[z, r, 3]
            )

            M[T_idx, T_idx + 1]       = k[z, r] * alpha[z, r, 0]
            M[T_idx, T_idx - 1]       = k[z, r] * alpha[z, r, 1]
            M[T_idx, T_idx - stride]  = k[z, r] * alpha[z, r, 3]  


        # -----------------------
        # Wall BC elements, no corners.
        r = self.cfg.mesh.N_R - 1
        for z in range(1, self.cfg.mesh.N_Z-1): #
            T_idx = z * stride + r 

            M[T_idx, T_idx] = -k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2] + alpha[z, r, 3])
            M[T_idx, T_idx] -=  h[z, r] * alpha[z, r, 0]

            M[T_idx, T_idx - 1]      = k[z, r] * alpha[z, r, 1]
            M[T_idx, T_idx + stride] = k[z, r] * alpha[z, r, 2] 
            M[T_idx, T_idx - stride] = k[z, r] * alpha[z, r, 3] 

            if z < self.cfg.mesh.N_evap:
                C[T_idx] = -self.cfg.bc.Q[z]
            else:
                C[T_idx] = -h[z, r] * alpha[z, r, 0] * self.cfg.bc.T_cond

        # -----------------------
        # Wick BC elements against vapor, no corners.
        r = 0
        for z in range(0, self.cfg.mesh.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx, T_idx]  = -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2] + alpha[z, r, 3])
            M[T_idx, T_idx] -=  h[z, r] * alpha[z, r, 1]

            M[T_idx, T_idx + 1]      = k[z, r] * alpha[z, r, 0]
            M[T_idx, -1]             = h[z, r] * alpha[z, r, 1]
            M[T_idx, T_idx + stride] = k[z, r] * alpha[z, r, 2] 
            M[T_idx, T_idx - stride] = k[z, r] * alpha[z, r, 3] 

        # -----------------------
        # Corner next to evaporator entrance (z = 0, r = N_R-1).
        z = 0
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx, T_idx] = -k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2])

        M[T_idx, T_idx - 1]      = k[z, r] * alpha[z, r, 1]
        M[T_idx, T_idx + stride] = k[z, r] * alpha[z, r, 2] 

        C[T_idx] = -self.cfg.bc.Q[z]

        # -----------------------
        # Corner next to condenser outlet (z = N_Z - 1, r = N_R - 1).
        z = self.cfg.mesh.N_Z - 1
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx, T_idx]  = -k[z, r] * (alpha[z, r, 1] + alpha[z, r, 3])
        M[T_idx, T_idx] += -h[z, r] * alpha[z, r, 0] 

        M[T_idx, T_idx - 1]      = k[z, r] * alpha[z, r, 1]
        M[T_idx, T_idx - stride] = k[z, r] * alpha[z, r, 3]  

        C[T_idx] = -h[z][r] * alpha[z][r][0] * self.cfg.bc.T_cond 

        # -----------------------
        # Corner next to evaporator vapor inlet (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx, T_idx]  = -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2])
        M[T_idx, T_idx] -=  h[z, r] * alpha[z, r, 1] 

        M[T_idx, T_idx + 1]      = k[z, r] * alpha[z, r, 0]
        M[T_idx, -1]             = h[z][r] * alpha[z, r, 1]
        M[T_idx, T_idx + stride] = k[z, r] * alpha[z, r, 2]

        # -----------------------
        # Corner next to condenser vapor outlet/inlet (z = N_Z - 1, r = 0).
        z = self.cfg.mesh.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx, T_idx]  = -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 3])
        M[T_idx, T_idx] -=  h[z, r] * alpha[z, r, 1]

        M[T_idx, T_idx + 1]      = k[z, r] * alpha[z, r, 0]
        M[T_idx, -1]             = h[z, r] * alpha[z, r, 1]
        M[T_idx, T_idx - stride] = k[z, r] * alpha[z, r, 3] 

        # -----------------------
        # Vapor elements. 

        for z in range(0, self.cfg.mesh.N_evap):
            r = 0
            T_idx = z * stride + r 

            M[-1, -1   ] -= h[z][r] * alpha[z, r, 1]
            M[-1, T_idx] += h[z][r] * alpha[z, r, 1]

        for z in range(self.cfg.mesh.N_Z - self.cfg.mesh.N_cond, self.cfg.mesh.N_Z):
            r = 0
            T_idx = z * stride + r 

            M[-1, -1   ] -= h[z, r] * alpha[z, r, 1]
            M[-1, T_idx] += h[z, r] * alpha[z, r, 1]

        # -----------------------
        # Return matrix and vector

        M = M.tocsr()

        return M, C

    def _generate_matrix_form_heat_bc(self, alpha: np.ndarray, k: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Number of physical grid nodes + extra vapour node (stored at index -1).
        N_phys = self.cfg.mesh.N_R * self.cfg.mesh.N_Z
        N = N_phys
        stride = self.cfg.mesh.N_R 

        M = np.zeros((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # -----------------------
        # Bulk elements.
        for z in range(1, self.cfg.mesh.N_Z - 1):
            for r in range(1, self.cfg.mesh.N_R - 1):
                T_idx = (stride * z) + r

                M[T_idx][T_idx] = -k[z][r] * (
                    alpha[z][r][0] +
                    alpha[z][r][1] +
                    alpha[z][r][2] +
                    alpha[z][r][3]
                )

                M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
                M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
                M[T_idx][T_idx + stride]  = k[z][r] * alpha[z][r][2]  
                M[T_idx][T_idx - stride]  = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Insulated wall at z = 0
        z = 0
        for r in range(1, self.cfg.mesh.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx][T_idx] = -k[z][r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][2]
            )

            M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride]  = k[z][r] * alpha[z][r][2]  

        # -----------------------
        # Insulated wall at z = N_Z - 1
        z = self.cfg.mesh.N_Z - 1
        for r in range(1, self.cfg.mesh.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx][T_idx] = -k[z][r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][3]
            )

            M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx - stride]  = k[z][r] * alpha[z][r][3]  


        # -----------------------
        # Wall BC elements, no corners.
        r = self.cfg.mesh.N_R - 1
        for z in range(1, self.cfg.mesh.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx] = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2] + alpha[z][r][3])

            M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -self.cfg.bc.Q[z]

        # -----------------------
        # Wick BC elements against vapour, no corners.
        r = 0
        for z in range(0, self.cfg.mesh.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2] + alpha[z][r][3])
            M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

            M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -h[z][r] * alpha[z][r][1] * self.cfg.bc.T_op
            

        # -----------------------
        # Corner next to evaporator entrance (z = 0, r = N_R-1).
        z = 0
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx] = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2])

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 

        C[T_idx] = -self.cfg.bc.Q[z]

        # -----------------------
        # Corner next to condenser outlet (z = N_Z - 1, r = N_R - 1).
        z = self.cfg.mesh.N_Z - 1
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][3])

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]  

        C[T_idx] = -self.cfg.bc.Q[z]

        # -----------------------
        # Corner next to evaporator vapour inlet (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1] 

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2]

        C[T_idx] = -h[z][r] * alpha[z][r][1] * self.cfg.bc.T_op

        # -----------------------
        # Corner next to condenser vapour inlet (z = N_Z - 1, r = 0).
        z = self.cfg.mesh.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][3])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]

        C[T_idx] = -h[z][r] * alpha[z][r][1] * self.cfg.bc.T_op

        return M, C

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    
    from utils.solver import Solver

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]
    
    geom = d_class.HeatpipeGeometry(**data["geometry"])
    mesh = d_class.HeatpipeMesh(N_Z=30, N_R=30)
    mat = d_class.HeatpipeMaterial(**data["material"])
    wick = d_class.HeatpipeWick(**data["wick"])
    bc = d_class.HeatpipeBC(**data["bc"])
    cfg = d_class.HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg = cfg.resolve_geometry()
    
    heatpipe = HeatpipeDiscretised(cfg)
    solver = Solver([heatpipe])
    solver.newton_krylov()

    T_solid, T_vap = solver.solution # type: ignore

    N_Z = cfg.mesh.N_Z
    N_R = cfg.mesh.N_R

    T_linear = heatpipe.linear_solve()

    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Computer modern"
    plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize = (16, 9))

    ax = fig.add_subplot(111)
    r_centers = heatpipe.R

    delta_R = heatpipe.delta_R[0::2] + heatpipe.delta_R[1::2]
    r_faces = np.r_[heatpipe.cfg.geometry.r_vapour, heatpipe.cfg.geometry.r_vapour + np.cumsum(delta_R)]

    r_wick_disc = r_faces[heatpipe.cfg.mesh.N_wick]
    r_gap_disc  = r_faces[heatpipe.cfg.mesh.N_wick + heatpipe.cfg.mesh.N_gap]

    ax.plot(r_centers, T_solid[0], lw=4, label="non-linear")
    ax.plot(r_centers, T_linear[:-1].reshape(N_Z, N_R)[0], ls="--", lw=4, label="linear")
    ax.vlines([r_wick_disc, r_gap_disc], np.min(T_solid[0]), np.max(T_solid[0]), colors="black")
    
    plt.legend()
    plt.show()