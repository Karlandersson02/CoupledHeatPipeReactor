import numpy as np

import utils.material_properties as m_props
import data.dataclass as d_class

from models.component import Component

class FuelPin(Component):
    def __init__(self, config: d_class.FuelPinConfigResolved):

        self.cfg = config

        self.phi_g   = np.tile(np.array([4.16e18, 5.47e17], dtype=float), (self.cfg.mesh.N_Z, 1))
        self.Sigma_f = np.array([9.4e-2, 5.48e1], dtype=float)
        self.kappa   = 3.204e-11
        self.T_mod   = np.ones(self.cfg.mesh.N_Z) * 1000.0

        self.Delta_Z = self.cfg.geometry.l / self.cfg.mesh.N_Z
        
        self.calculate_qr()

        self.variable_k = False
        self.k_temperature = 1400        # used if variable_k is False

    def initial_guess(self):
        X_initial = np.ones((self.cfg.mesh.N_Z * self.cfg.mesh.N_R, ))
        return X_initial

    def assemble(self):
        self.initialize_discretization()

        k = self.generate_k_matrix()
        h = self.generate_h_matrix()

        alpha = self.calculate_alpha_with_old_boundary(k)

        self.M, self.C = self.generate_matrix_form(alpha, self.qr, self.T_mod, k, h)

    # def calculate_qr(self):
    #     self.initialize_discretization()
    #     self.qr = np.sum(self.phi_g * self.Sigma_f * self.kappa * self.Delta_V, axis = 1)

    # def calculate_qr(self):
    #     self.initialize_discretization()

    #     q_vol_z = np.sum(
    #         self.phi_g * self.Sigma_f * self.kappa,
    #         axis=1,
    #     )

    #     fuel_slice = slice(0, self.cfg.mesh.N_fuel)

    #     self.qr = q_vol_z * np.sum(self.Delta_V[fuel_slice])

    def calculate_qr(self):
        self.initialize_discretization()

        q_vol_z = np.sum(
            self.phi_g * self.Sigma_f * self.kappa,
            axis=1,
        )

        V_fuel = self.Delta_V[:self.cfg.mesh.N_fuel]

        if not np.allclose(V_fuel, V_fuel[0]):
            raise ValueError(
                "Expected equal fuel-cell volumes. "
                "Check the quadratic fuel-region discretisation."
            )

        self.qr = q_vol_z * V_fuel[0]

    # def get_residuals(self, X):
    #     T = X.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)

    #     qr = self.qr
    #     T_mod = self.T_mod

    #     N_Z = self.cfg.mesh.N_Z
    #     N_R = self.cfg.mesh.N_R
    #     N_fuel = self.cfg.mesh.N_fuel

    #     if self.variable_k:
    #         k = self.generate_k_matrix(T)
    #     else:
    #         k = self.generate_k_matrix()
    #     h = self.generate_h_matrix()
    #     alpha = self.calculate_alpha(k)

    #     res = np.zeros((N_Z, N_R), dtype=float)

    #     # -----------------------
    #     # Bulk elements
    #     if N_Z > 2 and N_R > 2:
    #         kc = k[1:-1, 1:-1]
    #         ac = alpha[1:-1, 1:-1]

    #         res[1:-1, 1:-1] = (
    #             -kc * (ac[..., 0] + ac[..., 1] + ac[..., 2] + ac[..., 3]) * T[1:-1, 1:-1]
    #             + kc * ac[..., 0] * T[1:-1, 2:]
    #             + kc * ac[..., 1] * T[1:-1, :-2]
    #             + kc * ac[..., 2] * T[2:, 1:-1]
    #             + kc * ac[..., 3] * T[:-2, 1:-1]
    #         )

    #     # -----------------------
    #     # Insulated wall at z = 0, excluding corners
    #     if N_R > 2:
    #         z = 0
    #         kc = k[z, 1:-1]
    #         ac = alpha[z, 1:-1]

    #         res[z, 1:-1] = (
    #             -kc * (ac[:, 0] + ac[:, 1] + ac[:, 2]) * T[z, 1:-1]
    #             + kc * ac[:, 0] * T[z, 2:]
    #             + kc * ac[:, 1] * T[z, :-2]
    #             + kc * ac[:, 2] * T[z + 1, 1:-1]
    #         )

    #     # -----------------------
    #     # Insulated wall at z = N_Z - 1, excluding corners
    #     if N_R > 2:
    #         z = N_Z - 1
    #         kc = k[z, 1:-1]
    #         ac = alpha[z, 1:-1]

    #         res[z, 1:-1] = (
    #             -kc * (ac[:, 0] + ac[:, 1] + ac[:, 3]) * T[z, 1:-1]
    #             + kc * ac[:, 0] * T[z, 2:]
    #             + kc * ac[:, 1] * T[z, :-2]
    #             + kc * ac[:, 3] * T[z - 1, 1:-1]
    #         )

    #     # -----------------------
    #     # Cladding BC elements, excluding corners
    #     if N_Z > 2:
    #         r = N_R - 1
    #         kc = k[1:-1, r]
    #         ac = alpha[1:-1, r]
    #         hc = h[1:-1, r]

    #         res[1:-1, r] = (
    #             (-kc * (ac[:, 1] + ac[:, 2] + ac[:, 3]) - hc * ac[:, 0]) * T[1:-1, r]
    #             + kc * ac[:, 1] * T[1:-1, r - 1]
    #             + kc * ac[:, 2] * T[2:, r]
    #             + kc * ac[:, 3] * T[:-2, r]
    #             + hc * ac[:, 0] * T_mod[1:-1]
    #         )

    #     # -----------------------
    #     # Fuel inner BC elements, excluding corners
    #     if N_Z > 2:
    #         r = 0
    #         kc = k[1:-1, r]
    #         ac = alpha[1:-1, r]

    #         res[1:-1, r] = (
    #             -kc * (ac[:, 0] + ac[:, 2] + ac[:, 3]) * T[1:-1, r]
    #             + kc * ac[:, 0] * T[1:-1, r + 1]
    #             + kc * ac[:, 2] * T[2:, r]
    #             + kc * ac[:, 3] * T[:-2, r]
    #         )

    #     # -----------------------
    #     # Lowermost cladding outer corner: z = 0, r = N_R - 1
    #     z = 0
    #     r = N_R - 1
    #     res[z, r] = (
    #         (-k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2]) - h[z, r] * alpha[z, r, 0]) * T[z, r]
    #         + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
    #         + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
    #         + h[z, r] * alpha[z, r, 0] * T_mod[z]
    #     )

    #     # -----------------------
    #     # Uppermost cladding outer corner: z = N_Z - 1, r = N_R - 1
    #     z = N_Z - 1
    #     r = N_R - 1
    #     res[z, r] = (
    #         (-k[z, r] * (alpha[z, r, 1] + alpha[z, r, 3]) - h[z, r] * alpha[z, r, 0]) * T[z, r]
    #         + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
    #         + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
    #         + h[z, r] * alpha[z, r, 0] * T_mod[z]
    #     )

    #     # -----------------------
    #     # Lowermost fuel inner corner: z = 0, r = 0
    #     z = 0
    #     r = 0
    #     res[z, r] = (
    #         -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2]) * T[z, r]
    #         + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
    #         + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
    #     )

    #     # -----------------------
    #     # Uppermost fuel inner corner: z = N_Z - 1, r = 0
    #     z = N_Z - 1
    #     r = 0
    #     res[z, r] = (
    #         -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 3]) * T[z, r]
    #         + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
    #         + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
    #     )

    #     # -----------------------
    #     # Fuel heat production
    #     res[:, :N_fuel] += qr[:, None]

    #     # res_norm_denom = (
    #     #     (np.sum(qr) * self.Delta_Z / self.cfg.geometry.l)
    #     #     / (np.pi * self.cfg.geometry.r**2)
    #     # )

    #     total_power = np.sum(qr) * N_fuel

    #     res_norm_denom = (
    #         (total_power * self.Delta_Z / self.cfg.geometry.l)
    #         / (np.pi * self.cfg.geometry.r**2)
    #     )

    #     return res.reshape(-1) / res_norm_denom

    def get_residuals(self, X):
        T = X.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)

        qr = self.qr
        T_mod = self.T_mod

        N_Z = self.cfg.mesh.N_Z
        N_R = self.cfg.mesh.N_R
        N_fuel = self.cfg.mesh.N_fuel

        if self.variable_k:
            k = self.generate_k_matrix(T)
        else:
            k = self.generate_k_matrix()

        h = self.generate_h_matrix()
        alpha = self.calculate_alpha(k)

        res = np.zeros((N_Z, N_R), dtype=float)

        # Outer cladding boundary conductance:
        # cell center -> cylindrical conduction through half-cell -> convection to moderator
        r_outer = self.cfg.geometry.r
        r_center_outer = self.R[-1]

        R_cond_outer = np.log(r_outer / r_center_outer) / (
            2.0 * np.pi * k[:, -1] * self.Delta_Z
        )

        R_conv_outer = 1.0 / (
            h[:, -1] * 2.0 * np.pi * r_outer * self.Delta_Z
        )

        G_mod = 1.0 / (R_cond_outer + R_conv_outer)

        # -----------------------
        # Bulk elements
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

        # -----------------------
        # Insulated wall at z = 0, excluding corners
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

        # -----------------------
        # Insulated wall at z = N_Z - 1, excluding corners
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

        # -----------------------
        # Cladding outer boundary elements, excluding corners
        if N_Z > 2:
            r = N_R - 1
            kc = k[1:-1, r]
            ac = alpha[1:-1, r]
            Gc = G_mod[1:-1]

            res[1:-1, r] = (
                (-kc * (ac[:, 1] + ac[:, 2] + ac[:, 3]) - Gc) * T[1:-1, r]
                + kc * ac[:, 1] * T[1:-1, r - 1]
                + kc * ac[:, 2] * T[2:, r]
                + kc * ac[:, 3] * T[:-2, r]
                + Gc * T_mod[1:-1]
            )

        # -----------------------
        # Fuel inner boundary elements, excluding corners
        if N_Z > 2:
            r = 0
            kc = k[1:-1, r]
            ac = alpha[1:-1, r]

            res[1:-1, r] = (
                -kc * (ac[:, 0] + ac[:, 2] + ac[:, 3]) * T[1:-1, r]
                + kc * ac[:, 0] * T[1:-1, r + 1]
                + kc * ac[:, 2] * T[2:, r]
                + kc * ac[:, 3] * T[:-2, r]
            )

        # -----------------------
        # Lowermost cladding outer corner: z = 0, r = N_R - 1
        z = 0
        r = N_R - 1
        G = G_mod[z]

        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2]) - G) * T[z, r]
            + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
            + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
            + G * T_mod[z]
        )

        # -----------------------
        # Uppermost cladding outer corner: z = N_Z - 1, r = N_R - 1
        z = N_Z - 1
        r = N_R - 1
        G = G_mod[z]

        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 1] + alpha[z, r, 3]) - G) * T[z, r]
            + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
            + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
            + G * T_mod[z]
        )

        # -----------------------
        # Lowermost fuel inner corner: z = 0, r = 0
        z = 0
        r = 0

        res[z, r] = (
            -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2]) * T[z, r]
            + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
            + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
        )

        # -----------------------
        # Uppermost fuel inner corner: z = N_Z - 1, r = 0
        z = N_Z - 1
        r = 0

        res[z, r] = (
            -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 3]) * T[z, r]
            + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
            + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
        )

        # -----------------------
        # Fuel heat production
        res[:, :N_fuel] += qr[:, None]

        total_power = np.sum(qr) * N_fuel

        res_norm_denom = (
            (total_power * self.Delta_Z / self.cfg.geometry.l)
            / (np.pi * self.cfg.geometry.r**2)
        )

        return res.reshape(-1) / res_norm_denom

    def post_process(self, X):
        return (X.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R), )

    def unpack(self, X):
        return (X, )
    
    def pack(self, X_tuple):
        X = X_tuple[0].reshape(self.cfg.mesh.N_Z * self.cfg.mesh.N_R)
        return X

    def linear_solve(self):
        self.calculate_qr()
        self.assemble()

        T = np.linalg.solve(self.M, self.C)

        self.T = T

    # def initialize_discretization(self):     
    #     R         = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
    #     delta_R   = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        
    #     # Calculating the radii of the half-elements
    #     R[0] = np.sqrt(self.cfg.geometry.r**2 / (self.cfg.mesh.N_R * 2))

    #     for i in range(1, 2 * self.cfg.mesh.N_R):
    #         R[i] = np.sqrt(R[i-1]**2 + R[0]**2)

    #     # Calculating the differences in the radius of the half-elements
    #     delta_R[0] = R[0]
    #     for i in range(1, 2 * self.cfg.mesh.N_R):
    #         delta_R[i] = R[i] - R[i - 1]

    #     delta_Rm = delta_R[0::2]
    #     delta_Rp = delta_R[1::2]

    #     surface_tensor = self.calculate_surfaces(delta_Rp, delta_Rm)

    #     self.R = R[0::2]
    #     self.Z = np.arange(0, self.cfg.geometry.l - self.Delta_Z, self.Delta_Z) + self.Delta_Z
    #     self.delta_Rm = delta_Rm
    #     self.delta_Rp = delta_Rp
    #     self.Delta_V = np.pi * (delta_Rp[0] + delta_Rm[0])**2 * self.Delta_Z 
    #     self.surface_tensor = surface_tensor

    def initialize_discretization(self):
        cfg = self.cfg

        N_fuel = cfg.mesh.N_fuel
        N_gap  = cfg.mesh.N_gap
        N_wall = cfg.mesh.N_wall
        N_R    = cfg.mesh.N_R

        r_fuel = cfg.geometry.r_fuel
        r_gap_outer = r_fuel + cfg.geometry.delta_gap
        r_outer = cfg.geometry.r

        if N_fuel + N_gap + N_wall != N_R:
            raise ValueError(
                "Inconsistent radial mesh: "
                f"N_fuel + N_gap + N_wall = {N_fuel + N_gap + N_wall}, "
                f"but N_R = {N_R}."
            )

        R_half = []

        # Fuel: quadratic / equal-area mesh from 0 to r_fuel
        if N_fuel > 0:
            i = np.arange(1, 2 * N_fuel + 1, dtype=float)
            R_fuel = r_fuel * np.sqrt(i / (2 * N_fuel))
            R_half.append(R_fuel)

        # Gap: linear mesh from r_fuel to r_gap_outer
        if N_gap > 0:
            i = np.arange(1, 2 * N_gap + 1, dtype=float)
            R_gap = r_fuel + i * cfg.geometry.delta_gap / (2 * N_gap)
            R_half.append(R_gap)

        # Wall: linear mesh from r_gap_outer to r_outer
        if N_wall > 0:
            i = np.arange(1, 2 * N_wall + 1, dtype=float)
            R_wall = r_gap_outer + i * cfg.geometry.delta_wall / (2 * N_wall)
            R_half.append(R_wall)

        R_half = np.concatenate(R_half)

        if len(R_half) != 2 * N_R:
            raise ValueError(
                "Internal discretisation error: "
                f"expected {2 * N_R} half-radii, got {len(R_half)}."
            )

        delta_R = np.empty_like(R_half)
        delta_R[0] = R_half[0]
        delta_R[1:] = R_half[1:] - R_half[:-1]

        delta_Rm = delta_R[0::2]
        delta_Rp = delta_R[1::2]

        R = R_half[0::2]

        self.R = R
        self.Z = np.arange(0, self.cfg.geometry.l - self.Delta_Z, self.Delta_Z) + self.Delta_Z / 2
        self.delta_Rm = delta_Rm
        self.delta_Rp = delta_Rp

        self.Delta_V = (
            np.pi
            * ((R + delta_Rp)**2 - (R - delta_Rm)**2)
            * self.Delta_Z
        )

        self.surface_tensor = self.calculate_surfaces(delta_Rp, delta_Rm)


    def calculate_surfaces(self, delta_Rp, delta_Rm):
        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R)
        Rm = Rp - delta_R

        S_rp = Rp * 2 * np.pi
        S_rm = Rm * 2 * np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.cfg.mesh.N_Z, axis=0)
        surface_tensor[..., 0:2] *= self.Delta_Z

        return surface_tensor

    # def calculate_surfaces_from_edges(self, r_edges):
    #     Rp = r_edges[1:]
    #     Rm = r_edges[:-1]

    #     S_rp = 2.0 * np.pi * Rp
    #     S_rm = 2.0 * np.pi * Rm
    #     S_z = np.pi * (Rp**2 - Rm**2)

    #     surface_tensor = np.concatenate(
    #         [
    #             S_rp[:, None],
    #             S_rm[:, None],
    #             S_z[:, None],
    #             S_z[:, None],
    #         ],
    #         axis=1,
    #     )

    #     surface_tensor = np.repeat(
    #         surface_tensor[None],
    #         self.cfg.mesh.N_Z,
    #         axis=0,
    #     )

    #     surface_tensor[..., 0:2] *= self.Delta_Z

    #     return surface_tensor
    
    
    def generate_k_matrix(self, T=None):
        if T is None:
            T = np.full((self.cfg.mesh.N_Z, self.cfg.mesh.N_R), self.k_temperature)

        k_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R), dtype=float)

        def eval_material_prop(prop, T_slice, scale=1.0):
            value = prop(T_slice) if callable(prop) else prop
            return value * scale     # type: ignore

        fuel_slice = slice(0, self.cfg.mesh.N_fuel)
        gap_slice  = slice(self.cfg.mesh.N_fuel, self.cfg.mesh.N_fuel + self.cfg.mesh.N_gap)
        clad_slice = slice(self.cfg.mesh.N_fuel + self.cfg.mesh.N_gap, self.cfg.mesh.N_R)

        T_fuel = T[:, fuel_slice]
        T_gap  = T[:, gap_slice ]
        T_clad = T[:, clad_slice]

        qp_gap = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_gap))
        qp_gap[:, -1:] = self.qr[:, None] * self.surface_tensor[:, -1:, 1]
        gap_h = lambda T: m_props.FP_gap_h(T, qp_gap)

        # k_matrix[:, fuel_slice] = eval_material_prop(self.cfg.material.k_fuel, T_fuel)
        # k_matrix[:, gap_slice ] = eval_material_prop(self.cfg.material.h_gap, T_gap, scale=self.cfg.geometry.delta_gap)
        # k_matrix[:, clad_slice] = eval_material_prop(self.cfg.material.k_clad, T_clad)
    
        k_matrix[:, fuel_slice] = eval_material_prop(m_props.FP_fuel_k, T_fuel)
        k_matrix[:, gap_slice ] = eval_material_prop(gap_h            , T_gap, scale=self.cfg.geometry.delta_gap)
        k_matrix[:, clad_slice] = eval_material_prop(m_props.FP_clad_k, T_clad)

        return k_matrix


    def generate_h_matrix(self, T=800.):
        h_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R))

        h_matrix[:, -1] = self.cfg.material.h_mod

        return h_matrix


    # def calculate_alpha(self, k_matrix):
    #     alpha = np.zeros_like(self.surface_tensor)

    #     alpha[:, :-1, 0] = (
    #         self.surface_tensor[:, :-1, 0] * k_matrix[:, 1:]
    #         / (k_matrix[:, :-1] * self.delta_Rm[1:] + k_matrix[:, 1:] * self.delta_Rp[:-1])
    #     )

    #     alpha[:, 1:, 1] = (
    #         self.surface_tensor[:, 1:, 1] * k_matrix[:, :-1]
    #         / (k_matrix[:, 1:] * self.delta_Rp[:-1] + k_matrix[:, :-1] * self.delta_Rm[1:])
    #     )

    #     alpha[:-1, :, 2] = (
    #         self.surface_tensor[:-1, :, 2] * k_matrix[1:, :]
    #         / (k_matrix[:-1, :] * self.Delta_Z + k_matrix[1:, :] * self.Delta_Z)
    #     )

    #     alpha[1:, :, 3] = (
    #         self.surface_tensor[1:, :, 3] * k_matrix[:-1, :]
    #         / (k_matrix[1:, :] * self.Delta_Z + k_matrix[:-1, :] * self.Delta_Z)
    #     )
    #     # Init mask
    #     cooling_mask = np.zeros_like(alpha, dtype=bool)
                                     
    #     # Configure masks
    #     cooling_mask[:, -1, 0] = True

    #     # Apply masks
    #     alpha[cooling_mask] = self.surface_tensor[cooling_mask]

    #     return alpha

    def calculate_alpha(self, k_matrix):
        alpha = np.zeros_like(self.surface_tensor)

        alpha[:, :-1, 0] = (
            self.surface_tensor[:, :-1, 0] * k_matrix[:, 1:]
            / (
                k_matrix[:, :-1] * self.delta_Rm[1:]
                + k_matrix[:, 1:] * self.delta_Rp[:-1]
            )
        )

        alpha[:, 1:, 1] = (
            self.surface_tensor[:, 1:, 1] * k_matrix[:, :-1]
            / (
                k_matrix[:, 1:] * self.delta_Rp[:-1]
                + k_matrix[:, :-1] * self.delta_Rm[1:]
            )
        )

        alpha[:-1, :, 2] = (
            self.surface_tensor[:-1, :, 2] * k_matrix[1:, :]
            / (
                k_matrix[:-1, :] * self.Delta_Z
                + k_matrix[1:, :] * self.Delta_Z
            )
        )

        alpha[1:, :, 3] = (
            self.surface_tensor[1:, :, 3] * k_matrix[:-1, :]
            / (
                k_matrix[1:, :] * self.Delta_Z
                + k_matrix[:-1, :] * self.Delta_Z
            )
        )

        return alpha

    def calculate_alpha_with_old_boundary(self, k_matrix):
        alpha = self.calculate_alpha(k_matrix)

        cooling_mask = np.zeros_like(alpha, dtype=bool)
        cooling_mask[:, -1, 0] = True
        alpha[cooling_mask] = self.surface_tensor[cooling_mask]

        return alpha
    
    
    def generate_matrix_form(self, alpha, qr, T_mod, k, h):
        N = self.cfg.mesh.N_R * self.cfg.mesh.N_Z
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
        # Insulated wall at z = 0, no corners.
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
        # Insulated wall at z = N_Z - 1, no corners.
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
        # Cladding BC elements, no corners.
        r = self.cfg.mesh.N_R - 1
        for z in range(1, self.cfg.mesh.N_Z-1): #
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (
                alpha[z][r][1] + 
                alpha[z][r][2] + 
                alpha[z][r][3]
            )
            M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0]

            M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -h[z][r] * alpha[z][r][0] * T_mod[z]

        # -----------------------
        # Fuel inner BC elements, no corners.
        r = 0
        for z in range(1, self.cfg.mesh.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (
                alpha[z][r][0] + 
                alpha[z][r][2] + 
                alpha[z][r][3]
            )

            M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Lowermost cladding outer corner (z = 0, r = N_R-1).
        z = 0
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0]

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 

        C[T_idx] = -h[z][r] * alpha[z][r][0] * T_mod[z]

        # -----------------------
        # Uppermost cladding outer corner (z = N_Z - 1, r = N_R - 1).
        z = self.cfg.mesh.N_Z - 1
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][3])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0] 

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]  

        C[T_idx] = -h[z][r] * alpha[z][r][0] * T_mod[z]

        # -----------------------
        # Lowermost fuel inner corner (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2])

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2]

        # -----------------------
        # Uppermost fuel inner corner (z = N_Z - 1, r = 0).
        z = self.cfg.mesh.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][3])

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Fuel heat production
        for z in range(0, self.cfg.mesh.N_Z):
            for r in range(0, self.cfg.mesh.N_fuel):
                T_idx = (stride * z) + r
                
                C[T_idx] += -qr[z]
        
        return M, C

if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    
    from utils.solver import Solver

    with open("./data/reactor_data.json", "r") as f:
        data = json.load(f)["FuelPin"]

    N_Z = 50
    N_R = 30

    phi_ng  = np.tile(np.array([4.16e18, 5.47e17], dtype=float), (N_Z, 1))
    Sigma_f = np.array([9.4e-2, 5.48e1], dtype=float)
    kappa   = 3.204e-11
    T_mod   = np.ones(N_Z) * 1000.0

    geom   = d_class.FuelPinGeometry(**data["geometry"])
    mesh   = d_class.FuelPinMesh(N_Z=N_Z, N_R=N_R)
    energy = d_class.FuelPinEnergy(**data["energy"])
    mat    = d_class.FuelPinMaterial(**data["material"])
    cfg    = d_class.FuelPinConfig(geom, mesh, energy, mat)
    cfg = cfg.resolve_geometry()

    fuel_pin = FuelPin(cfg)
    solver = Solver([fuel_pin])
    solver.newton_krylov()

    T, = solver.solution # type: ignore

    fuel_pin.linear_solve()

    plt.plot(fuel_pin.R, T[0], label="non-linear")
    plt.plot(fuel_pin.R, fuel_pin.T.reshape(cfg.mesh.N_Z, cfg.mesh.N_R)[0], ls="--", label="linear")
    
    plt.legend()
    plt.show()