import numpy as np

import data.dataclass as d_class
import utils.material_properties as m_props

from scipy.optimize import fsolve

from models.heatpipe.heatpipe import Heatpipe
from models.neutronics.neutronics import Neutronics
from models.fuel_pin.fuel_pin import FuelPin
from models.model import AbstractModel

from utils.solver import Solver


class Reactor(AbstractModel):

    def __init__(self, cfg_R: d_class.ReactorConfigResolved):

        self.cfg_R = cfg_R

        self.cfg_HP: d_class.HeatpipeConfigResolved = cfg_R.HP
        self.cfg_FP: d_class.FuelPinConfigResolved = cfg_R.FP
        self.cfg_N: d_class.NeutronicsConfigResolved = cfg_R.N

        self.heatpipe   = Heatpipe(self.cfg_HP)
        self.fuel_pin   = FuelPin(self.cfg_FP)
        self.neutronics = Neutronics(self.cfg_N)
        
        self.T_cond = 300.
        self.T_vap_ref = 2000
        self.u_v_ref = 50

        # heat transfer HP variables: N_R * N_Z + 2 * N_Z - 1, heat transfer FP variables: N_R * N_Z + 1, neutron flux variables: N_Z + 1
        self.N_T_HP = self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z
        self.N_u_v  = self.cfg_HP.mesh.N_Z - 1
        self.N_T_v  = self.cfg_HP.mesh.N_Z
        self.N_HP   = self.N_T_HP + self.N_u_v + self.N_T_v

        self.N_FP   = self.cfg_FP.mesh.N_R  * self.cfg_FP.mesh.N_Z
        self.N_N    = self.cfg_N.mesh.N_Z  * self.cfg_N.energy.N_G + 1

        self.N_var = self.N_HP + self.N_FP + self.N_N

        # variable r_eff
        self.variable_r_eff    = False
        self.r_eff_temperature = 1100           # used if variable_r_eff is False

    def set_variable_k(self, cond: bool):
        self.heatpipe.set_variable_k(cond)
        self.fuel_pin.variable_k = cond
        self.variable_r_eff = cond

    def set_variable_neutron_data(self, cond, T=900):
        self.neutronics.set_variable_neutron_data(cond, T)

    def initial_guess(self):
        i = np.arange(self.cfg_HP.mesh.N_evap, dtype=float)
        cosine_weights = 0.2 + 0.8 * np.cos(np.pi * (i - (self.cfg_HP.mesh.N_evap - 1) / 2) / (self.cfg_HP.mesh.N_evap - 1))
        cosine_weights /= cosine_weights.sum()
        Q = cosine_weights * self.cfg_N.energy.power * 24 / 7

        print("Solving Heat pipe initial values")
        self.heatpipe.solid.cfg.bc.Q = Q
        solver_HP = Solver([self.heatpipe])
        solver_HP.fsolve()
        (T_HP_guess, mach_guess, T_v_guess) = solver_HP.solutions[0]
        u_guess = self.heatpipe._calculate_c_s(T_v_guess) * mach_guess
        u_guess = (u_guess[:-1] + u_guess[1:]) / 2

        X_initial = np.ones(self.N_var)

        X_initial[:self.N_T_HP] = T_HP_guess.reshape(-1) / self.T_cond 
        X_initial[self.N_T_HP:(self.N_T_HP + self.N_u_v)] = u_guess / self.u_v_ref 
        X_initial[(self.N_T_HP + self.N_u_v):self.N_HP] = T_v_guess / self.T_vap_ref

        # neutronics weightings
        X_initial[-self.N_N:-1] = np.repeat(cosine_weights, self.cfg_N.energy.N_G)

        return X_initial

    def assemble(self):
        return
    
    def post_process(self, X):
        X_HP, T_FP, phi_ng_hat_and_k = self.unpack(X)

        T_HP = X_HP[:self.N_T_HP]
        u_v  = X_HP[self.N_T_HP:(self.N_T_HP + self.N_u_v)]
        T_v  = X_HP[(self.N_T_HP + self.N_u_v):]

        phi_ng_hat = phi_ng_hat_and_k[:-1]
        k = phi_ng_hat_and_k[-1]

        T_HP *= self.T_cond
        u_v  *= self.u_v_ref
        T_v  *= self.T_vap_ref
        T_FP *= self.T_cond

        T_solid = T_HP.reshape(
            self.cfg_HP.mesh.N_Z,
            self.cfg_HP.mesh.N_R,
        )

        T_FP_flat = T_FP

        T_FP_reshaped = T_FP_flat.reshape(
            self.cfg_FP.mesh.N_Z,
            self.cfg_FP.mesh.N_R,
        )

        _, _, _, Sigma_f, _, _, kappa = self.neutronics.get_material_data(
            T_FP_flat
        )

        phi_n_g_hat = phi_ng_hat.reshape(
            self.cfg_N.mesh.N_Z,
            self.cfg_N.energy.N_G,
        )

        power_density = kappa * Sigma_f * phi_n_g_hat

        power = (
            np.sum(power_density)
            * self.cfg_N.mesh.cross_sectional_area
            * self.cfg_N.mesh.delta_Z
        )

        phi_n_g = phi_n_g_hat * self.cfg_N.energy.power / power

        return ((T_solid, u_v, T_v), T_FP_reshaped, (phi_n_g, k))
    
    def unpack(self, X):
        X_HP             = X[:self.N_HP].copy()
        T_FP             = X[self.N_HP:(self.N_HP + self.N_FP)].copy()
        phi_ng_hat_and_k = X[-self.N_N:].copy()
        return X_HP, T_FP, phi_ng_hat_and_k

    def pack(self, X_tuple):
        X = np.r_[*X_tuple]
        return X

    def _validate_temperature_state(self, T_HP, T_FP, T_v, T_mod=None):
        arrays = {
            "T_HP": np.asarray(T_HP, dtype=float),
            "T_FP": np.asarray(T_FP, dtype=float),
            "T_v": np.asarray(T_v, dtype=float),
        }
        if T_mod is not None:
            arrays["T_mod"] = np.asarray(T_mod, dtype=float)

        invalid = []
        for name, values in arrays.items():
            if not np.all(np.isfinite(values)):
                invalid.append(f"{name} contains non-finite values")
            elif np.min(values) <= 0.0:
                invalid.append(
                    f"{name} reached a non-physical minimum of {np.min(values):.3f} K"
                )

        if invalid:
            ranges = ", ".join(
                f"{name}=[{np.nanmin(values):.3f}, {np.nanmax(values):.3f}]"
                for name, values in arrays.items()
            )
            raise ValueError(
                "Reactor encountered an invalid temperature state before "
                f"neutronics interpolation: {'; '.join(invalid)}. {ranges}"
            )
    
    def get_residuals(self, X):
        X_HP, T_FP, phi_ng_hat_and_k = self.unpack(X)

        T_HP        = X_HP[:self.N_T_HP]
        u_v         = X_HP[self.N_T_HP:(self.N_T_HP + self.N_u_v)]
        T_v         = X_HP[(self.N_T_HP + self.N_u_v):self.N_HP]
        phi_ng_hat  = phi_ng_hat_and_k[:-1]

        T_HP *= self.T_cond
        u_v  *= self.u_v_ref
        T_v  *= self.T_vap_ref
        T_FP *= self.T_cond

        self._validate_temperature_state(T_HP, T_FP, T_v)
        Q_HP, T_mod = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)
        self._validate_temperature_state(T_HP, T_FP, T_v, T_mod)

        T_HP_ave  = np.mean(
            T_HP[:self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap],
            dtype=float,
        )
        T_mod_ave = np.mean(T_mod)

        self.neutronics.T_FP = T_FP
        self.neutronics.T_M  = T_mod_ave
        self.neutronics.T_HP = T_HP_ave

        qr = self.calculate_qr(T_FP, phi_ng_hat)

        self.heatpipe.cfg.bc.Q = Q_HP
        res_cond_HP = self.heatpipe.get_residuals(X_HP)

        self.fuel_pin.qr = qr
        self.fuel_pin.T_mod = T_mod
        res_cond_FP = self.fuel_pin.get_residuals(T_FP)

        res_flux = self.neutronics.get_residuals(phi_ng_hat_and_k)

        return np.r_[res_cond_HP, res_cond_FP, res_flux]
    

    # def calculate_qr(self, T_FP, phi_ng_hat):
    #     _, _, _, Sigma_f, _, _, kappa = self.neutronics.get_material_data(T_FP)

    #     qr_rel = np.sum(phi_ng_hat.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G) * Sigma_f * kappa * self.fuel_pin.Delta_V, axis=1) # W
    #     power_rel = np.sum(qr_rel)

    #     return qr_rel * self.cfg_N.energy.power / (power_rel * self.cfg_FP.mesh.N_fuel)

    def calculate_qr(self, T_FP, phi_ng_hat):
        _, _, _, Sigma_f, _, _, kappa = self.neutronics.get_material_data(T_FP)

        phi_ng_hat = phi_ng_hat.reshape(
            self.cfg_N.mesh.N_Z,
            self.cfg_N.energy.N_G,
        )

        q_vol_z = np.sum(
            phi_ng_hat * Sigma_f * kappa,
            axis=1,
        )

        V_fuel = self.fuel_pin.Delta_V[:self.cfg_FP.mesh.N_fuel]

        if not np.allclose(V_fuel, V_fuel[0]):
            raise ValueError(
                "Expected equal fuel-cell volumes. "
                "Check the quadratic fuel-region discretisation."
            )

        qr_rel = q_vol_z * V_fuel[0]
        power_rel = np.sum(q_vol_z) * np.sum(V_fuel)

        return qr_rel * self.cfg_N.energy.power / power_rel
            

    def calculate_HP_FP_boundary_cond(self, T_FP, T_HP):
        T_fp = T_FP.reshape(self.cfg_FP.mesh.N_Z, self.cfg_FP.mesh.N_R)
        T_hp_solid = T_HP.reshape(self.cfg_HP.mesh.N_Z, self.cfg_HP.mesh.N_R)

        evap = slice(0, self.cfg_HP.mesh.N_evap)

        T_edge_FP = T_fp[:, -1]
        T_edge_HP = T_hp_solid[evap, -1]

        if self.fuel_pin.variable_k:
            k_fp_edge = self.fuel_pin.generate_k_matrix(T_fp)[:, -1]
        else:
            k_fp_edge = self.fuel_pin.generate_k_matrix()[:, -1]

        if self.heatpipe.solid.variable_k:
            k_hp_edge = self.heatpipe.solid._generate_k_matrix(T_hp_solid)[evap, -1]
        else:
            k_hp_edge = self.heatpipe.solid._generate_k_matrix()[evap, -1]

        r_fp_outer = self.cfg_FP.geometry.r
        r_hp_outer = self.cfg_HP.geometry.r_outer

        r_fp_center_outer = self.fuel_pin.R[-1]
        r_hp_center_outer = self.heatpipe.solid.R[-1]

        Delta_z_fp = self.fuel_pin.Delta_Z
        Delta_z_hp = self.cfg_HP.geometry.l_evap / self.cfg_HP.mesh.N_evap

        R_fp_cond = np.log(r_fp_outer / r_fp_center_outer) / (
            2.0 * np.pi * k_fp_edge * Delta_z_fp
        )
        R_hp_cond = np.log(r_hp_outer / r_hp_center_outer) / (
            2.0 * np.pi * k_hp_edge * Delta_z_hp
        )

        T_mod_ref = (T_edge_FP + T_edge_HP) / 2.0

        if self.variable_r_eff:
            R_mod = m_props.moderator_R_eff(T_mod_ref) / Delta_z_hp
        else:
            R_mod = m_props.moderator_R_eff(self.r_eff_temperature) / Delta_z_hp

        R_total = R_fp_cond + R_mod + R_hp_cond

        Q_FP = (T_edge_FP - T_edge_HP) / R_total
        Q_HP = Q_FP * 24 / 7

        R_fp_conv = 1.0 / (
            self.cfg_FP.material.h_mod
            * 2.0
            * np.pi
            * r_fp_outer
            * Delta_z_fp
        )

        G_fp = 1.0 / (R_fp_cond + R_fp_conv)
        T_mod = T_edge_FP - Q_FP / G_fp

        return Q_HP, T_mod
    
    # -------------------------------------------------------------------------
    # Scaling helpers
    # -------------------------------------------------------------------------
    def unpack_HP_scaled(self, X_HP_scaled):
        X_HP_scaled = np.asarray(X_HP_scaled, dtype=float).copy()

        T_HP = X_HP_scaled[:self.N_T_HP] * self.T_cond

        u_v = (
            X_HP_scaled[self.N_T_HP:(self.N_T_HP + self.N_u_v)]
            * self.u_v_ref
        )

        T_v = (
            X_HP_scaled[(self.N_T_HP + self.N_u_v):self.N_HP]
            * self.T_vap_ref
        )

        return T_HP, u_v, T_v

    def unpack_FP_N_scaled(self, X_FP_N_scaled):
        X_FP_N_scaled = np.asarray(X_FP_N_scaled, dtype=float).copy()

        T_FP = X_FP_N_scaled[:self.N_FP] * self.T_cond
        phi_ng_hat_and_k = X_FP_N_scaled[self.N_FP:]

        return T_FP, phi_ng_hat_and_k

    # -------------------------------------------------------------------------
    # Block 1: Heat pipe + vapour, with fuel pin temperature fixed
    # -------------------------------------------------------------------------
    def get_residuals_HP_vapour_block(self, X_HP_scaled, T_FP_scaled):
        T_HP, u_v, T_v = self.unpack_HP_scaled(X_HP_scaled)

        T_FP = np.asarray(T_FP_scaled, dtype=float).copy() * self.T_cond

        Q_HP, _ = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)

        self.heatpipe.cfg.bc.Q = Q_HP

        X_HP_physical = np.r_[T_HP, u_v, T_v]

        return self.heatpipe.get_residuals(X_HP_physical)

    # -------------------------------------------------------------------------
    # Block 2: Fuel pin + neutronics, with heat pipe state fixed
        # -------------------------------------------------------------------------
    def get_residuals_neutronics_block(self, phi_ng_hat_and_k, X_HP_scaled, T_FP_scaled):
        """
        Neutronics-only block.

        Unknowns:
            phi_ng_hat_and_k

        Fixed:
            X_HP_scaled
            T_FP_scaled
        """
        T_HP, _, _ = self.unpack_HP_scaled(X_HP_scaled)
        T_FP = np.asarray(T_FP_scaled, dtype=float).copy() * self.T_cond

        _, T_mod = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)

        T_HP_ave = np.mean(
            T_HP[:self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap],
            dtype=float,
        )

        self.neutronics.T_FP = T_FP
        self.neutronics.T_M  = np.mean(T_mod)
        self.neutronics.T_HP = T_HP_ave

        return self.neutronics.get_residuals(phi_ng_hat_and_k)
    
    def get_residuals_fuel_pin_block(self, T_FP_scaled, X_HP_scaled, phi_ng_hat_and_k):
        """
        Fuel-pin-temperature-only block.

        Unknowns:
            T_FP_scaled

        Fixed:
            X_HP_scaled
            phi_ng_hat_and_k
        """
        T_HP, _, _ = self.unpack_HP_scaled(X_HP_scaled)

        T_FP = np.asarray(T_FP_scaled, dtype=float).copy() * self.T_cond
        phi_ng_hat = phi_ng_hat_and_k[:-1]

        _, T_mod = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)
        qr = self.calculate_qr(T_FP, phi_ng_hat)

        self.fuel_pin.qr = qr
        self.fuel_pin.T_mod = T_mod

        return self.fuel_pin.get_residuals(T_FP)
    # -------------------------------------------------------------------------
    # Picard solve
    # -------------------------------------------------------------------------
    def solve_picard(self, X0=None, max_iter=30, picard_tol=1e-6, full_res_tol=None, block_tol=1e-8, relaxation_HP=1., relaxation_N=1., relaxation_FP=1., maxfev_HP=10000, maxfev_N=10000, maxfev_FP=10000, verbose=True, raise_on_block_fail=False):
        """
        Three-block Picard iteration using fsolve:

            1. Solve heat pipe + vapour with T_FP fixed.
            2. Solve neutronics with T_FP and HP fixed.
            3. Solve fuel pin temperature with phi and HP fixed.
            4. Under-relax all block updates.
        """

        if X0 is None:
            X0 = self.initial_guess()

        X_HP, T_FP_scaled, phi_ng_hat_and_k = self.unpack(X0)

        history = []

        for it in range(max_iter):
            X_old = np.r_[X_HP, T_FP_scaled, phi_ng_hat_and_k]

            # ================================================================
            # Block 3: Fuel pin temperature only
            # ================================================================
            T_FP_scaled_new, info_FP, ier_FP, msg_FP = fsolve(
                func=lambda x: self.get_residuals_fuel_pin_block(
                    x,
                    X_HP,
                    phi_ng_hat_and_k,
                ),
                x0=T_FP_scaled,
                xtol=block_tol,
                maxfev=maxfev_FP,
                full_output=True,
            )

            FP_success = ier_FP == 1

            if (not FP_success) and verbose:
                print(f"[Picard {it:03d}] Fuel pin block did not fully converge:")
                print(f"    ier = {ier_FP}")
                print(f"    {msg_FP}")

            if (not FP_success) and raise_on_block_fail:
                raise RuntimeError(
                    f"Fuel pin block failed at Picard iteration {it}: {msg_FP}"
                )

            T_FP_scaled = (
                (1.0 - relaxation_FP) * T_FP_scaled
                + relaxation_FP * T_FP_scaled_new
            )


            # ================================================================
            # Block 1: Heat pipe + vapour
            # ================================================================
            X_HP_new, info_HP, ier_HP, msg_HP = fsolve(
                func=lambda x: self.get_residuals_HP_vapour_block(
                    x,
                    T_FP_scaled,
                ),
                x0=X_HP,
                xtol=block_tol,
                maxfev=maxfev_HP,
                full_output=True,
            )

            HP_success = ier_HP == 1

            if (not HP_success) and verbose:
                print(f"[Picard {it:03d}] HP block did not fully converge:")
                print(f"    ier = {ier_HP}")
                print(f"    {msg_HP}")

            if (not HP_success) and raise_on_block_fail:
                raise RuntimeError(
                    f"HP block failed at Picard iteration {it}: {msg_HP}"
                )

            X_HP = (
                (1.0 - relaxation_HP) * X_HP
                + relaxation_HP * X_HP_new
            )

            # ================================================================
            # Block 2: Neutronics only
            # ================================================================
            phi_ng_hat_and_k_new, info_N, ier_N, msg_N = fsolve(
                func=lambda x: self.get_residuals_neutronics_block(
                    x,
                    X_HP,
                    T_FP_scaled,
                ),
                x0=phi_ng_hat_and_k,
                xtol=block_tol,
                maxfev=maxfev_N,
                full_output=True,
            )

            N_success = ier_N == 1

            if (not N_success) and verbose:
                print(f"[Picard {it:03d}] Neutronics block did not fully converge:")
                print(f"    ier = {ier_N}")
                print(f"    {msg_N}")

            if (not N_success) and raise_on_block_fail:
                raise RuntimeError(
                    f"Neutronics block failed at Picard iteration {it}: {msg_N}"
                )

            phi_ng_hat_and_k = (
                (1.0 - relaxation_N) * phi_ng_hat_and_k
                + relaxation_N * phi_ng_hat_and_k_new
            )

            # ================================================================
            # Diagnostics
            # ================================================================
            X = np.r_[X_HP, T_FP_scaled, phi_ng_hat_and_k]

            step_abs = np.linalg.norm(X - X_old)
            step_rel = step_abs / (np.linalg.norm(X_old) + 1e-14)

            full_res = self.get_residuals(X)
            full_res_rms = np.linalg.norm(full_res) / np.sqrt(full_res.size)

            history.append(
                {
                    "iteration": it,
                    "step_abs": step_abs,
                    "step_rel": step_rel,
                    "full_res_rms": full_res_rms,

                    "HP_success": HP_success,
                    "N_success": N_success,
                    "FP_success": FP_success,

                    "HP_nfev": info_HP["nfev"],
                    "N_nfev": info_N["nfev"],
                    "FP_nfev": info_FP["nfev"],

                    "HP_ier": ier_HP,
                    "N_ier": ier_N,
                    "FP_ier": ier_FP,
                }
            )

            if verbose:
                print(
                    f"[Picard {it:03d}] "
                    f"step_rel={step_rel:.3e}, "
                    f"full_res_rms={full_res_rms:.3e}, "
                    f"HP_nfev={info_HP['nfev']}, "
                    f"N_nfev={info_N['nfev']}, "
                    f"FP_nfev={info_FP['nfev']}"
                )

            if full_res_tol is None:
                converged = step_rel < picard_tol
            else:
                converged = step_rel < picard_tol and full_res_rms < full_res_tol

            if converged:
                if verbose:
                    print(f"Picard converged after {it + 1} iterations.")
                break

        X = np.r_[X_HP, T_FP_scaled, phi_ng_hat_and_k]
        solution = self.post_process(X)

        return X, solution, history

if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    from utils.solver import Solver

    with open("./data/reactor_data.json", "r") as f:
        data = json.load(f)

    N_R_HP, N_R_FP, N_Z = 15, 65, 40

    # Heat pipe config
    geom   = d_class.HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh   = d_class.HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
    mat    = d_class.HeatpipeMaterial(**data["HeatPipe"]["material"])
    wick   = d_class.HeatpipeWick(**data["HeatPipe"]["wick"])
    bc     = d_class.HeatpipeBC(**data["HeatPipe"]["bc"])
    cfg_HP = d_class.HeatpipeConfig(geom, mesh, mat, wick, bc)

    # Fuel pin config
    geom_FP   = d_class.FuelPinGeometry(**data["FuelPin"]["geometry"])
    mesh_FP   = d_class.FuelPinMesh(N_R=N_R_FP, N_Z=9*N_Z//20)
    energy_FP = d_class.FuelPinEnergy(**data["FuelPin"]["energy"])
    mat_FP    = d_class.FuelPinMaterial(**data["FuelPin"]["material"])
    cfg_FP    = d_class.FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)

    # Neutronics config
    mesh_N = d_class.NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = 9*N_Z//20,
        l   = data["FuelPin"]["geometry"]["l"]
    )
    energy = d_class.NeutronicsEnergy(
        N_G   = cfg_FP.energy.N_G,
        power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
    )
    cfg_N = d_class.NeutronicsConfig(mesh_N, energy)

    # Reactor config
    cfg_R = d_class.ReactorConfig(cfg_HP, cfg_FP, cfg_N)
    cfg_R = cfg_R.resolve_mesh()

    vapour_reactor = Reactor(cfg_R)
    vapour_reactor.set_variable_k(True)

    solver = Solver([vapour_reactor])
    solver.fsolve()

    # plot_reactor_temperature_schematic_with_vapour(solver)

    ((T_solid, u_v, T_v), T_FP, (phi_n_g, k)) = solver.solution # type: ignore

    u_full = np.r_[0, u_v, 0]
    u_bar  = (u_full[:-1] + u_full[1:]) / 2

    z_full = vapour_reactor.heatpipe.solid.Z
    z_evap = vapour_reactor.heatpipe.solid.Z[:cfg_R.HP.mesh.N_evap]

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    axs[0, 0].plot(z_full, T_solid[:, 0])
    axs[0, 1].plot(z_full, u_bar)
    axs[0, 2].plot(z_full, T_v)

    axs[1, 0].plot(z_evap, T_FP[:cfg_R.HP.mesh.N_evap, -1])
    axs[1, 1].plot(z_evap, phi_n_g[:cfg_R.HP.mesh.N_evap, 0])
    axs[1, 2].plot(z_evap, phi_n_g[:cfg_R.HP.mesh.N_evap, -1])

    axs[0, 0].set_title("Solid Axial Temperature")
    axs[0, 1].set_title("Vapour Velocity")
    axs[0, 2].set_title("Vapour Temperature")
    axs[1, 0].set_title("Fuel Axial Pin Temperature")
    axs[1, 1].set_title("Neutron Flux (Group 0)")
    axs[1, 2].set_title("Neutron Flux (Group -1)")

    plt.show()
